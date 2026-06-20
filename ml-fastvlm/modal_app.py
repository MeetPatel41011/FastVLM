import modal
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# 1. Define the Modal App and the Volume for the PyTorch weights
app = modal.App("fastvlm-unified")
vol = modal.Volume.from_name("fastvlm-weights", create_if_missing=True)

# 2. Build the Docker Image required for the PyTorch Backend
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("tesseract-ocr")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "tokenizers",
        "sentencepiece",
        "shortuuid",
        "accelerate",
        "peft",
        "bitsandbytes",
        "pydantic",
        "numpy",
        "einops",
        "einops-exts",
        "timm",
        "Pillow",
        "opencv-python-headless",
        "pytesseract",
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "requests",
        "beautifulsoup4",
        "sse-starlette"
    )
    .add_local_dir("../web-fastvlm/out", remote_path="/root/out", copy=True) # Copy the Next.js frontend into image
    .add_local_dir(".", remote_path="/root/app", ignore=["checkpoints", "__pycache__"], copy=True) # Copy local python files into image
    .run_commands("pip install -e /root/app --no-deps") # Install the local llava package so `from llava.xxx` works
)

# 3. Define the Web Endpoint
@app.function(
    image=image,
    gpu="T4",
    volumes={"/root/app/checkpoints": vol}, # Mount the weights where inference.py expects them
    secrets=[modal.Secret.from_name("tavily-secret", environment_missing_ok=True)], # Securely pass the Tavily API key
    timeout=600, # 10 minute timeout
    scaledown_window=60, # Keep container alive for 1 min after request to prevent cold starts on follow-up questions
)
@modal.asgi_app()
def web():
    # We must add the mounted directory to Python's path so it can find server.py
    import sys
    import os
    if "/root/app" not in sys.path:
        sys.path.insert(0, "/root/app")
    
    # Also change directory so relative file reads work if any
    os.chdir("/root/app")
    
    # Import your exact local server
    import server
    from inference import EdgeAgent
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def lifespan(app):
        # ── Startup: Load the AI model onto the GPU ──
        print("🚀 Modal container starting — loading FastVLM model onto GPU...")
        model_path = os.getenv("MODEL_PATH", "checkpoints/llava-fastvithd_0.5b_stage3")
        try:
            server.agent = EdgeAgent(model_path=model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
        yield
        # ── Shutdown ──
        print("🛑 Modal container shutting down.")
    
    # Create a new top-level FastAPI app with the lifespan handler
    web_app = FastAPI(lifespan=lifespan)
    
    # Mount your PyTorch API EXACTLY where the frontend expects it
    web_app.mount("/api/backend", server.app)
    
    # Mount the Next.js static website at the root!
    web_app.mount("/", StaticFiles(directory="/root/out", html=True), name="static")
    
    return web_app
