import os
import base64
import re
import numpy as np
import cv2
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

# Cloud fallback URL — set this if you want to route to Modal when local model unavailable
MODAL_CLOUD_URL = os.getenv("MODAL_CLOUD_URL", "").rstrip("/")

# Lightweight OCR for question detection (CPU-only, ~50ms)
try:
    import pytesseract
    HAS_TESSERACT = True
    
    # Auto-configure tesseract path for Windows
    import sys, os
    if sys.platform == 'win32':
        tess_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(tess_path):
            pytesseract.pytesseract.tesseract_cmd = tess_path
            
except ImportError:
    print("Warning: pytesseract not installed. Question detection disabled — will trigger on any text.")
    HAS_TESSERACT = False

# Import the EdgeAgent from inference.py
try:
    from inference import EdgeAgent
except ImportError:
    print("Warning: Could not import EdgeAgent. Ensure you are in the ml-fastvlm directory.")
    EdgeAgent = None

# Import MLX engine (only available on Apple Silicon)
MLXEdgeAgent = None
try:
    from mlx_engine import MLXEdgeAgent
except ImportError:
    pass  # Not on macOS or mlx-vlm not installed — that's fine

app = FastAPI(title="LiveQA Vision API")

# Allow Next.js frontend on any port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = None
boot_time = time.time()


# ─── Health Check ───

@app.get("/")
async def root():
    return {
        "status": "ready" if agent is not None else "no_model",
        "agent_initialized": agent is not None,
        "uptime_seconds": round(time.time() - boot_time),
    }


@app.get("/health")
async def health():
    """Frontend calls this to check if backend is alive and model is loaded."""
    hw_info = {}
    if agent and hasattr(agent, "hw"):
        hw = agent.hw
        hw_info = {
            "backend": hw.backend,
            "device_name": hw.device_name,
            "vram_gb": hw.vram_gb,
            "optimizations": hw.optimizations,
        }
    
    # Determine inference mode
    if agent is not None:
        mode = "local"
    elif MODAL_CLOUD_URL:
        mode = "cloud_fallback"
    else:
        mode = "unavailable"
    
    return JSONResponse({
        "status": "ok",
        "model_loaded": agent is not None,
        "mode": mode,
        "cloud_url": MODAL_CLOUD_URL if mode == "cloud_fallback" else None,
        "device": getattr(agent, "device", None),
        "dtype": str(getattr(agent, "dtype", None)),
        "hardware": hw_info,
    })


# ─── Startup ───

@app.on_event("startup")
async def startup_event():
    global agent
    
    # ─── Smart Engine Selection ───
    # Priority: MLX (Apple Silicon) > PyTorch (CUDA/MPS/CPU)
    from hardware_detector import detect_hardware
    hw = detect_hardware()
    
    # Try MLX engine first on Apple Silicon (sub-100ms TTFT)
    if hw.backend == "mlx" and MLXEdgeAgent is not None:
        print("\n[STARTUP] Apple Silicon detected — using MLX engine")
        mlx_model = os.getenv("MLX_MODEL_PATH", "mlx-community/FastVLM-0.5B-bf16")
        try:
            agent = MLXEdgeAgent(model_path=mlx_model)
            print("[STARTUP] MLX engine initialized successfully")
            return
        except Exception as e:
            print(f"[STARTUP] MLX engine failed, falling back to PyTorch: {e}")
            import traceback
            traceback.print_exc()
    
    # Fall back to PyTorch EdgeAgent (CUDA / MPS / DirectML / CPU)
    if EdgeAgent is not None:
        model_path = os.getenv("MODEL_PATH", "checkpoints/llava-fastvithd_0.5b_stage3")
        print(f"\n[STARTUP] Using PyTorch engine with model: {model_path}")
        try:
            agent = EdgeAgent(model_path=model_path)
        except Exception as e:
            print(f"[STARTUP] Failed to initialize EdgeAgent: {e}")
            import traceback
            traceback.print_exc()
            agent = None
    else:
        print("[STARTUP] No inference engine available. Check imports.")


# ─── Question Detection Heuristics (CPU-only) ───

QUESTION_MARK_RE = re.compile(r'\?')

QUESTION_WORD_RE = re.compile(
    r'\b(?:what|how|why|when|where|who|which|is|are|can|could|would|'
    r'do|does|did|will|shall|should|has|have|had)\b',
    re.IGNORECASE
)

MATH_TASK_RE = re.compile(
    r'(?:solve|calculate|find|compute|evaluate|simplify|prove|determine|'
    r'factor|derive|integrate|differentiate|convert|express)',
    re.IGNORECASE
)

MATH_EXPR_RE = re.compile(
    r'\d+\s*[\+\-\*\/\=\^]',  # digits followed by math operators
    re.IGNORECASE
)


def detect_question_in_text(ocr_text: str) -> bool:
    """Check if OCR output contains question-like patterns.
    
    Errs on the side of triggering — a missed question is worse
    than an occasional false positive.
    """
    if not ocr_text or len(ocr_text.strip()) < 3:
        return False
    
    # 1. Explicit question mark — strongest signal
    if QUESTION_MARK_RE.search(ocr_text):
        return True
    
    # 2. Starts with a question word
    if QUESTION_WORD_RE.search(ocr_text):
        return True
    
    # 3. Contains math task words (solve, find, calculate...)
    if MATH_TASK_RE.search(ocr_text):
        return True
    
    # 4. Contains math expressions (numbers + operators)
    if MATH_EXPR_RE.search(ocr_text):
        return True
    
    return False


# ─── Quick Check: Lightweight Text + Question Detection (Gatekeeper) ───

class QuickCheckRequest(BaseModel):
    image_base64: str

@app.post("/api/quick_check")
async def quick_check(request: QuickCheckRequest):
    """
    Lightweight text + question detection.
    Stage 1: OpenCV contour analysis (~15ms, CPU) — detects text presence.
    Stage 2: Tesseract OCR + heuristics (~50ms, CPU) — detects question patterns.
    No GPU usage. Returns has_text, has_question, and ocr_preview.
    """
    try:
        frame = decode_base64_image(request.image_base64)
        if frame is None:
            return JSONResponse({"has_text": False, "has_question": False, "reason": "bad_image"})
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Adaptive threshold to find dark marks on light background (pen on paper)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 5
        )
        
        # 2. Find contours (text strokes)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Filter for text-like contours (small, clustered, non-circular)
        text_contours = 0
        for c in contours:
            area = cv2.contourArea(c)
            if 10 < area < 8000:  # Handwriting can produce very thin or wide strokes
                x, y, w, h = cv2.boundingRect(c)
                aspect = w / max(h, 1)
                if 0.1 < aspect < 10:  # Not too extreme aspect ratio
                    text_contours += 1
        
        # 4. Mean brightness (logged for debugging, not used as a gate)
        mean_brightness = float(np.mean(gray))
        
        # 5. Decision: enough text-like contours confirms text presence
        #    Increased from 4 to 15 to prevent false triggers on human faces/hair
        has_text = text_contours > 15
        
        # 6. If text detected, run OCR to check for question patterns
        has_question = False
        ocr_text = ""
        if has_text:
            if HAS_TESSERACT:
                try:
                    # Superior webcam preprocessing: 
                    # 1. Upscale 2x (Tesseract needs tall characters)
                    # 2. Adaptive thresholding (handles camera shadows/uneven lighting perfectly)
                    large = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    binary = cv2.adaptiveThreshold(
                        large, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 21, 15
                    )
                    ocr_text = pytesseract.image_to_string(
                        binary,
                        config='--psm 6 --oem 3'
                    ).strip()
                    has_question = detect_question_in_text(ocr_text)
                except Exception as ocr_err:
                    # OCR failed — fall back to triggering (missed question > false trigger)
                    has_question = True
                    ocr_text = f"ERROR: {ocr_err}"
                    print(f"  ⚠️  OCR failed ({ocr_err}), falling back to trigger")
            else:
                # No Tesseract available — fall back to Phase 1 behavior (trigger on any text)
                has_question = True
                ocr_text = "ERROR: pytesseract module not installed"
        
        # ─── Debug Log ───
        print(f"\n🔍 GATEKEEPER CHECK:")
        print(f"   Contours: {text_contours} | Brightness: {mean_brightness:.0f}")
        print(f"   ▸ has_text: {has_text}")
        if has_text:
            print(f"   ▸ OCR read: \"{ocr_text[:80]}\"" if ocr_text else "   ▸ OCR read: (empty)")
            print(f"   ▸ has_question: {has_question}")
            if has_question:
                reasons = []
                if QUESTION_MARK_RE.search(ocr_text): reasons.append("found '?'")
                if QUESTION_WORD_RE.search(ocr_text): reasons.append("question word")
                if MATH_TASK_RE.search(ocr_text): reasons.append("math task word")
                if MATH_EXPR_RE.search(ocr_text): reasons.append("math expression")
                print(f"   ▸ reason: {', '.join(reasons) if reasons else 'OCR fallback'}")
        
        return JSONResponse({
            "has_text": has_text,
            "has_question": has_question,
            "ocr_preview": ocr_text[:100] if ocr_text else "",
            "text_contours": text_contours,
            "brightness": round(mean_brightness, 1)
        })
    except Exception as e:
        return JSONResponse({"has_text": False, "has_question": False, "reason": str(e)})


# ─── Request/Response Models ───

class AnalyzeRequest(BaseModel):
    image_base64: str   # data:image/jpeg;base64,... OR raw base64
    prompt: Optional[str] = ""  # Optional verbal question or typed text


# ─── Image Decoder ───

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode a base64 image string to an OpenCV BGR numpy array."""
    # Remove data URI header if present
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_cv2


# ─── SSE Streaming Endpoint ───

import asyncio
import json


async def _proxy_to_cloud(request):
    """
    Proxy the analyze request to the Modal cloud backend.
    Used as fallback when local model is unavailable (e.g. CPU too slow, no GPU, no model weights).
    """
    import httpx
    
    cloud_url = f"{MODAL_CLOUD_URL}/api/analyze"
    print(f"[CLOUD FALLBACK] Proxying to {cloud_url}")
    
    payload = {"image_base64": request.image_base64}
    if request.prompt:
        payload["prompt"] = request.prompt
    
    async def cloud_stream():
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", cloud_url, json=payload) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            yield line + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({
                "status": "error",
                "message": f"Cloud fallback failed: {e}"
            }) + "\n\n"
    
    return StreamingResponse(cloud_stream(), media_type="text/event-stream")

@app.post("/api/analyze")
async def analyze_image(request: AnalyzeRequest):
    """
    Main inference endpoint. Accepts a camera frame + optional text prompt.
    Returns a Server-Sent Events stream with live token output.
    """
    if agent is None:
        # ─── Cloud Fallback ───
        if MODAL_CLOUD_URL:
            return await _proxy_to_cloud(request)
        raise HTTPException(status_code=503, detail="Vision engine is not initialized and no cloud fallback configured.")
    
    def event_generator():
        import threading
        stop_event = threading.Event()
        
        try:
            # Decode image
            frame = decode_base64_image(request.image_base64)
            if frame is None:
                yield "data: " + json.dumps({"status": "error", "message": "Could not decode image"}) + "\n\n"
                return
            
            # Acknowledge receipt
            yield "data: " + json.dumps({
                "status": "thinking",
                "message": "Vision engine analyzing..."
            }) + "\n\n"
            
            # Run inference — pass the verbal/typed prompt through
            prompt = request.prompt.strip() if request.prompt else ""
            stream = agent.generate_stream(
                image=frame,
                prompt=prompt,
                stop_event=stop_event
            )
            
            full_response = ""
            for chunk in stream:
                full_response += chunk
                yield "data: " + json.dumps({
                    "status": "answering",
                    "chunk": chunk,
                    "full_text": full_response
                }) + "\n\n"
            
            yield "data: " + json.dumps({
                "status": "complete",
                "full_text": full_response.strip()
            }) + "\n\n"
            
        except GeneratorExit:
            stop_event.set()
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield "data: " + json.dumps({
                "status": "error",
                "message": str(e)
            }) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("  LiveQA Vision Server")
    print("  Starting on http://0.0.0.0:8000")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
