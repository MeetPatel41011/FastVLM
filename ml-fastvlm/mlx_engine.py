"""
MLX Vision Engine for LiveQA — Apple Silicon Optimized.

Uses Apple's MLX framework with mlx-vlm to run FastVLM inference
directly on Apple Silicon (M1/M2/M3/M4) using the Neural Engine + GPU.

Expected TTFT: ~50-180ms (vs ~1-3s on cloud T4)

Requires: pip install mlx mlx-vlm
Model:    mlx-community/FastVLM-0.5B-bf16 (auto-downloaded on first run)

This engine exposes the same generate_stream() API as EdgeAgent (inference.py),
so server.py can swap between them transparently.
"""

import time
import re
import numpy as np
from typing import Generator, Optional
from threading import Event
from PIL import Image

# ─── Shared tool routing (same as inference.py) ───
from inference import detect_tool, extract_math_expression, is_garbage_response, MODEL_NAME

# Default MLX model on HuggingFace (auto-downloaded and cached)
DEFAULT_MLX_MODEL = "mlx-community/FastVLM-0.5B-bf16"


class MLXEdgeAgent:
    """
    MLX-based Vision-Language Model agent for Apple Silicon.
    
    Drop-in replacement for EdgeAgent (inference.py) with the same API.
    Uses mlx-vlm for FastVLM inference on Apple Neural Engine + GPU.
    """

    def __init__(self, model_path: str = DEFAULT_MLX_MODEL):
        try:
            from mlx_vlm import load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import stream_generate
        except ImportError:
            raise ImportError(
                "MLX VLM not installed. Run:\n"
                "  pip install mlx mlx-vlm\n"
                "This only works on macOS with Apple Silicon (M1+)."
            )
        
        print(f"\n{'='*50}")
        print(f"  [MLX] LiveQA Vision Engine")
        print(f"  Model:   {model_path}")
        print(f"  Runtime: Apple MLX (Neural Engine + GPU)")
        print(f"{'='*50}\n")
        
        print("  [MLX] Loading model (first run downloads ~1GB)...")
        t0 = time.perf_counter()
        
        self.model, self.processor = load(model_path)
        
        load_time = time.perf_counter() - t0
        print(f"  [OK]  Model loaded in {load_time:.1f}s")
        
        # Store references for streaming
        self._stream_generate = stream_generate
        self._apply_chat_template = apply_chat_template
        self.model_path = model_path
        
        # Device info for compatibility with server.py
        self.device = "mlx"
        self.dtype = "bfloat16"
        
        # Hardware info stub for health endpoint
        from hardware_detector import detect_hardware
        self.hw = detect_hardware()
        
        self._warmup()
    
    def _warmup(self):
        """Quick warmup to prime MLX compilation caches."""
        print("  [MLX] Warming up...")
        try:
            # Create a tiny test image
            dummy_img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            dummy_img_path = "/tmp/_mlx_warmup.jpg"
            dummy_img.save(dummy_img_path)
            
            prompt = self._apply_chat_template(
                self.processor, 
                config=self.model.config,
                prompt="Describe.",
                images=[dummy_img_path],
            )
            
            # Run a short generation to compile MLX graphs
            output = ""
            for token in self._stream_generate(
                self.model, self.processor, prompt,
                image=[dummy_img_path],
                max_tokens=8,
                temp=0.0,
            ):
                output += token.text if hasattr(token, 'text') else str(token)
            
            print("  [OK]  MLX warmup complete")
        except Exception as e:
            print(f"  [WARN] MLX warmup: {e}")
        print("[OK] MLX Engine ready.\n")
    
    def generate_stream(
        self, image: np.ndarray, prompt: str = "", stop_event: Optional[Event] = None
    ) -> Generator[str, None, None]:
        """
        Core inference pipeline — same API as EdgeAgent.generate_stream().
        
        Args:
            image:  BGR numpy array from camera
            prompt: Optional text prompt
            stop_event: Threading event to cancel generation
            
        Yields:
            Text chunks as they are generated, followed by tool results.
        """
        from tools import AVAILABLE_TOOLS
        import tempfile, os
        
        if stop_event is None:
            stop_event = Event()
        
        t0 = time.perf_counter()
        perf = {"preprocess": 0.0, "ttft": 0.0, "total": 0.0}
        
        # ─── Phase 1: Image Preprocessing ───
        import cv2
        from PIL import ImageEnhance
        
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Enhance for OCR (same as PyTorch engine)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(2.0)
        if max(pil_img.size) > 336:
            pil_img.thumbnail((336, 336), Image.Resampling.LANCZOS)
        
        # Save to temp file (mlx-vlm expects file path)
        tmp_path = os.path.join(tempfile.gettempdir(), "_mlx_inference.jpg")
        pil_img.save(tmp_path, quality=95)
        
        perf["preprocess"] = time.perf_counter() - t0
        
        # ─── Phase 2: Build Prompt ───
        user_prompt = prompt.strip() if prompt else ""
        
        if user_prompt:
            instruction = (
                f'The user asks: "{user_prompt}"\n'
                "Look at this image. If there is any text written in the image, read it first.\n"
                "Then answer the user's question. Be concise and factual."
            )
        else:
            instruction = (
                "Read ALL the text written in this image carefully and exactly.\n"
                "State what the text says. If it contains a question, answer it.\n"
                "Format: First state what you read, then provide the answer."
            )
        
        formatted_prompt = self._apply_chat_template(
            self.processor,
            config=self.model.config,
            prompt=instruction,
            images=[tmp_path],
        )
        
        # ─── Phase 3: Stream Generation ───
        max_tokens = 128 if not user_prompt else 512
        
        full_text = ""
        try:
            for token_result in self._stream_generate(
                self.model, self.processor, formatted_prompt,
                image=[tmp_path],
                max_tokens=max_tokens,
                temp=0.0,
            ):
                if stop_event.is_set():
                    break
                
                # Extract text from token result
                new_text = token_result.text if hasattr(token_result, 'text') else str(token_result)
                
                # Record TTFT
                if not full_text:
                    perf["ttft"] = time.perf_counter() - t0
                
                full_text += new_text
                
                # Clean stop tokens
                clean = new_text
                for tok in ['<|im_end|>', '</s>']:
                    clean = clean.replace(tok, '')
                if clean:
                    yield clean
                
                if any(tok in new_text for tok in ['<|im_end|>', '</s>']):
                    break
        except GeneratorExit:
            if stop_event:
                stop_event.set()
            raise
        
        # Final cleanup
        for tok in ['<|im_end|>', '</s>']:
            full_text = full_text.replace(tok, '')
        full_text = full_text.strip()
        
        perf["total"] = time.perf_counter() - t0
        
        # ─── Phase 4: Deterministic Tool Routing ───
        # (Same logic as EdgeAgent — reused from inference.py)
        text_for_routing = user_prompt if user_prompt else full_text
        tool = detect_tool(text_for_routing)
        
        model_label = "FastVLM-0.5B-MLX"
        
        if tool != "none" and tool in AVAILABLE_TOOLS:
            yield "\n\n"
            
            if tool == "calculator":
                expr = extract_math_expression(text_for_routing)
                yield f"Calculator -> `{expr}`\n"
                try:
                    result = AVAILABLE_TOOLS["calculator"](expr)
                    yield f"= {result}\n"
                except Exception as e:
                    yield f"Calculator error: {e}\n"
                yield f"_Source: {model_label} + Calculator Tool_"
                    
            elif tool == "matrix":
                yield "Matrix Solver\n"
                try:
                    result = AVAILABLE_TOOLS["matrix"](text_for_routing)
                    yield f"```\n{result}\n```\n"
                except Exception as e:
                    yield f"Matrix error: {e}\n"
                yield f"_Source: {model_label} + NumPy Matrix Tool_"
                
            elif tool == "web_search":
                query = text_for_routing
                yield f'Searching: "{query}"\n'
                try:
                    result = AVAILABLE_TOOLS["web_search"](query)
                    yield f"Synthesizing answer...\n\n"
                    
                    # Use MLX model for synthesis (text-only, no image)
                    synth_prompt_text = (
                        f"Based on this web search result:\n{result}\n\n"
                        f"Answer this question: {query}\nBe concise and clear."
                    )
                    
                    synth_formatted = self._apply_chat_template(
                        self.processor,
                        config=self.model.config,
                        prompt=synth_prompt_text,
                    )
                    
                    for token_result in self._stream_generate(
                        self.model, self.processor, synth_formatted,
                        max_tokens=256,
                        temp=0.0,
                    ):
                        if stop_event.is_set():
                            break
                        chunk = token_result.text if hasattr(token_result, 'text') else str(token_result)
                        for tok in ['<|im_end|>', '</s>']:
                            chunk = chunk.replace(tok, '')
                        if chunk:
                            yield chunk
                        
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    yield f"Search error: {e}\n"
                yield f"\n\n_Source: {model_label} + Web Search_"
        else:
            yield f"\n\n_Source: {model_label} (local inference)_"
        
        # ─── Performance Footer ───
        perf["total"] = time.perf_counter() - t0
        yield (
            f"\n\n---\n"
            f"Pre: {perf['preprocess']:.2f}s "
            f"| TTFT: {perf['ttft']:.2f}s "
            f"| Total: {perf['total']:.2f}s"
        )
        
        # Console performance log
        print(f"\n{'='*50}")
        print(f"  [MLX] PERFORMANCE REPORT")
        print(f"  Preprocessing:  {perf['preprocess']:.3f}s")
        print(f"  TTFT:           {perf['ttft']:.3f}s")
        print(f"  Total:          {perf['total']:.3f}s")
        print(f"  Tool:           {tool}")
        print(f"  Output:         {full_text[:80]}...")
        print(f"{'='*50}\n")
        
        # Cleanup temp file
        try:
            os.remove(tmp_path)
        except:
            pass
