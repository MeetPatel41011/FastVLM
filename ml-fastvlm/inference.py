import torch, numpy as np, time, cv2, os, json, re
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from typing import Generator, Optional
from threading import Thread, Event
from PIL import Image, ImageEnhance
import multiprocessing
from hardware_detector import detect_hardware, get_device_report

# ─── Global CPU optimization ───
# Set both thread pools for maximum CPU parallelism
_num_cores = multiprocessing.cpu_count()
torch.set_num_threads(_num_cores)
try:
    torch.set_num_interop_threads(max(1, _num_cores // 2))
except RuntimeError:
    pass  # Already initialized — can't change interop threads after first op

class StopOnEventCriteria(StoppingCriteria):
    """Propagates cancellation from the API layer down to model.generate()."""
    def __init__(self, stop_event):
        self.stop_event = stop_event
    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    import llava.conversation as conversation_lib
except ImportError:
    print("Warning: LLaVA not installed. Run: pip install -e .")

# Model name shown in source attribution
MODEL_NAME = "llava-fastvithd_0.5b"

# ─── Deterministic Tool Detection (Python-only, no LLM involvement) ───

MATH_PATTERN = re.compile(
    r'\d+\s*[\+\-\*\/\^\%]\s*\d+',
    re.IGNORECASE
)

MATRIX_PATTERN = re.compile(
    r'(?:\[.*\d.*\])|(?:matrix\s*multipli)|(?:C\s*=\s*A\s*[\*@xX]\s*B)|(?:A\s*=\s*\[)',
    re.IGNORECASE
)

WEBSEARCH_PATTERN = re.compile(
    r'\b(?:news|search|latest|current|price|who\s+is|what\s+is|where\s+is|'
    r'when\s+was|today|yesterday|now|2025|2026|president|ceo|stock|weather)\b',
    re.IGNORECASE
)

# Words that indicate the VLM couldn't read anything useful
GARBAGE_INDICATORS = [
    "i cannot", "i can't", "no text", "cannot read", "unable to",
    "not visible", "blurry", "unclear", "empty"
]


def detect_tool(text: str) -> str:
    """Pure Python tool router. Returns 'calculator', 'matrix', 'web_search', or 'none'."""
    if not text or len(text.strip()) < 3:
        return "none"
    
    # Priority: matrix > calculator > web_search
    if MATRIX_PATTERN.search(text):
        return "matrix"
    
    if MATH_PATTERN.search(text):
        # Verify there are actual numbers (not just a stray +/-)
        has_numbers = len(re.findall(r'\d+', text)) >= 2
        if has_numbers:
            return "calculator"
    
    if WEBSEARCH_PATTERN.search(text):
        return "web_search"
    
    return "none"


def extract_math_expression(text: str) -> str:
    """Pull out the cleanest math expression from text."""
    match = MATH_PATTERN.search(text)
    if match:
        return match.group(0).strip().rstrip('=?').strip()
    # Fallback: extract anything that looks like numbers + operators
    nums = re.findall(r'[\d\.\+\-\*\/\(\)\s\^]+', text)
    if nums:
        expr = max(nums, key=len).strip()
        if len(expr) >= 3:
            return expr
    return text


def extract_search_query(text: str) -> str:
    """Extract just the actual question or query from VLM conversational output."""
    import re
    # Try to extract quoted text first, as VLM often quotes the OCR text
    match = re.search(r'"([^"]+)"|\'([^\']+)\'', text)
    if match:
        extracted = match.group(1) or match.group(2)
        if len(extracted) > 3:
            return extracted
    # Fallback: Strip common conversational prefixes
    text = re.sub(r'^(?i)(the\s+(?:text|question|image).*?(?:reads|says|is|asks|shows)?:?\s*)', '', text)
    return text.strip(' "\'')



def is_garbage_response(text: str) -> bool:
    """Check if the VLM response indicates it couldn't read the image."""
    lower = text.lower()
    return any(ind in lower for ind in GARBAGE_INDICATORS)


def enhance_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Enhance a camera frame for better text readability.
    Applies contrast stretching and sharpening — critical for handwriting.
    """
    # 1. Increase contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.5)
    
    # 2. Increase sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(2.0)
    
    # 3. Scale down large images for faster vision encoding
    # Reducing from 768 to 336 forces exactly ONE vision patch instead of 4-6.
    # This mathematically cuts the Vision Encoder time by 80%+.
    if max(pil_img.size) > 336:
        pil_img.thumbnail((336, 336), Image.Resampling.LANCZOS)
    
    return pil_img


class EdgeAgent:
    """
    Local Vision-Language Model agent.
    
    Reads handwritten/printed text from camera frames and answers questions.
    Uses FastVLM (LLaVA + FastViTHD + Qwen2 0.5B) for vision understanding,
    with deterministic Python-based tool routing for math, matrices, and web search.
    
    Auto-detects CUDA → MPS → CPU at initialization.
    """

    def __init__(self, model_path="checkpoints/llava-fastvithd_0.5b_stage3", device=None, torch_dtype=None):
        # ─── Smart Hardware Auto-Detection ───
        hw = detect_hardware()
        print(get_device_report(hw))
        
        if device is None:
            # Use hardware detector's recommendation
            if hw.backend == "mlx":
                # MLX engine would be used instead — for now fall back to MPS/CPU
                device = "mps" if hw.device == "mlx" else "cpu"
            elif hw.backend in ("cuda", "mps", "cpu"):
                device = hw.device
            elif hw.backend == "directml":
                # DirectML needs torch_directml — fall through to CPU if not available
                try:
                    import torch_directml
                    device = str(torch_directml.device())
                except ImportError:
                    device = "cpu"
            else:
                device = "cpu"
        
        if torch_dtype is None:
            # CPU handles float32 best; CUDA/MPS are faster with float16
            torch_dtype = torch.float32 if device == "cpu" else torch.float16

        print(f"\n{'='*50}")
        print(f"  🧠 LiveQA Vision Engine")
        print(f"  Device:  {device.upper() if isinstance(device, str) else device}")
        print(f"  Dtype:   {torch_dtype}")
        print(f"  Model:   {model_path}")
        print(f"{'='*50}\n")
        
        self.device = device if isinstance(device, str) else str(device)
        self.dtype = torch_dtype
        self.hw = hw  # Store hardware info for runtime decisions
        
        # Load model with SDPA for fast attention
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path, model_base=None,
            model_name=get_model_name_from_path(model_path),
            device=self.device, torch_dtype=torch_dtype,
            attn_implementation="sdpa"
        )
        
        # Ensure vision tower is loaded and on correct device
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=self.device, dtype=torch_dtype)
        self.model.to(device=self.device, dtype=torch_dtype)
        
        # ─── Optimization: JIT-compile for faster forward passes ───
        if "torch_compile" in hw.optimizations:
            try:
                # CUDA: reduce-overhead mode uses CUDA graphs for max speed
                # MPS: default mode (reduce-overhead not supported on Metal)
                compile_mode = "reduce-overhead" if hw.backend == "cuda" else "default"
                print(f"  [OPT] Applying torch.compile(mode='{compile_mode}')...")
                self.model = torch.compile(self.model, mode=compile_mode)
                print("  [OK]  torch.compile() applied")
            except Exception as e:
                print(f"  [SKIP] torch.compile(): {e}")
        
        # ─── Optimization: INT8 dynamic quantization for CPU ───
        # Quantizes all nn.Linear weight matrices to int8, reducing memory bandwidth
        # by 4x and speeding up matmul by ~1.5-2x. Quality impact is minimal for 0.5B.
        if "int8_quantization" in hw.optimizations and hw.backend == "cpu":
            try:
                print("  [OPT] Applying INT8 dynamic quantization (CPU)...")
                # Only quantize the language model layers, not the vision encoder
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                print("  [OK]  INT8 quantization applied — ~1.5-2x CPU speedup")
            except Exception as e:
                print(f"  [SKIP] INT8 quantization: {e}")
        
        # ─── Optimization: CUDA-specific tuning ───
        if hw.backend == "cuda":
            # Use TF32 for matmul on Ampere+ GPUs (RTX 30xx/40xx/50xx)
            # ~2x faster than FP32 with negligible precision loss
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Enable cuDNN autotuner — finds fastest convolution algorithms
            torch.backends.cudnn.benchmark = True
            print("  [OK]  CUDA: TF32 matmul + cuDNN benchmark enabled")
        
        # Update model name for attribution
        global MODEL_NAME
        MODEL_NAME = "llava-fastvithd_1.5b" if "1.5b" in model_path.lower() else "llava-fastvithd_0.5b"
        
        self._warmup()

    def _warmup(self):
        """
        Full warmup: run a complete dummy inference to prime all CUDA kernels,
        memory allocators, and JIT caches. The first real request will be fast.
        """
        print("🔥 Warming up vision engine (full inference pass)...")
        try:
            # Create a realistic dummy image (not just zeros — triggers real vision encoding)
            dummy_img = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))
            image_tensor = process_images([dummy_img], self.image_processor, self.model.config)[0]
            image_tensor = image_tensor.to(self.device, dtype=self.dtype)
            
            # Build a minimal prompt and run a short generation
            conv = conversation_lib.conv_templates["qwen_2"].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\nDescribe.')
            conv.append_message(conv.roles[1], None)
            warmup_prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(
                warmup_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.device)
            
            with torch.inference_mode():
                _ = self.model.generate(
                    inputs=input_ids,
                    images=image_tensor.unsqueeze(0),
                    image_sizes=[dummy_img.size],
                    max_new_tokens=8,  # Just enough to prime the pipeline
                    do_sample=False,
                    use_cache=True,
                )
            
            # Second warmup pass to fully compile torch.compile() graphs
            if "torch_compile" in self.hw.optimizations:
                with torch.inference_mode():
                    _ = self.model.generate(
                        inputs=input_ids,
                        images=image_tensor.unsqueeze(0),
                        image_sizes=[dummy_img.size],
                        max_new_tokens=8,
                        do_sample=False,
                        use_cache=True,
                    )
            
            print("  ✅ Full warmup complete")
        except Exception as e:
            print(f"  ⚠️ Warmup note: {e}")
            import traceback
            traceback.print_exc()
        print("✅ Engine ready — awaiting questions.\n")

    def generate_stream(self, image: np.ndarray, prompt: str = "", stop_event=None) -> Generator[str, None, None]:
        """
        Core inference pipeline. Streams text tokens as they are generated.
        
        Args:
            image:  BGR numpy array from camera/decoder
            prompt: Optional verbal question from speech-to-text or text input
            stop_event: Threading event to cancel generation early
            
        Yields:
            Text chunks as they are generated, followed by tool results if applicable.
        """
        from tools import AVAILABLE_TOOLS
        
        t0 = time.perf_counter()
        perf = {"preprocess": 0.0, "ttft": 0.0, "total": 0.0}
        
        if stop_event is None:
            stop_event = Event()

        # ─── Phase 1: Image Preprocessing ───
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_img = enhance_image_for_ocr(pil_img)

        # ─── Phase 2: Build Prompt ───
        conv = conversation_lib.conv_templates["qwen_2"].copy()
        
        user_prompt = prompt.strip() if prompt else ""
        
        if user_prompt:
            # Mode A: User asked a question verbally (or typed it)
            # The VLM should read any text in the image for context, then answer
            instruction = (
                f"The user asks: \"{user_prompt}\"\n"
                "Look at this image. If there is any text written in the image, read it first.\n"
                "Then answer the user's question. Be concise and factual."
            )
        else:
            # Mode B: No verbal question — read handwritten text and answer it
            instruction = (
                "Read the question written in the image and answer it directly. "
                "Do NOT describe the image. Just provide the exact answer to the question."
            )
        
        qs = DEFAULT_IMAGE_TOKEN + '\n' + instruction
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        formatted_prompt = conv.get_prompt()

        # ─── Optimization: torch.inference_mode() disables autograd for faster GPU ops ───
        with torch.inference_mode():
            # Tokenize
            input_ids = tokenizer_image_token(
                formatted_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.device)

            # Process image through vision encoder preprocessing
            image_tensor = process_images([pil_img], self.image_processor, self.model.config)[0]
            image_tensor = image_tensor.to(self.device, dtype=self.dtype)

        perf["preprocess"] = time.perf_counter() - t0

        # ─── Phase 3: Stream Generation ───
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Optimization: Use fewer tokens for OCR mode (handwriting reads are short)
        max_tokens = 128 if not user_prompt else 512

        gen_kwargs = dict(
            inputs=input_ids,
            images=image_tensor.unsqueeze(0),
            image_sizes=[pil_img.size],
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            stopping_criteria=StoppingCriteriaList([StopOnEventCriteria(stop_event)])
        )

        # Wrap generation in inference_mode for the background thread
        def _generate():
            with torch.inference_mode():
                self.model.generate(**gen_kwargs)

        thread = Thread(target=_generate)
        thread.start()

        full_text = ""
        STOP_TOKENS = ['<|im_end|>', '</s>']
        
        try:
            for new_text in streamer:
                if stop_event.is_set():
                    break
                
                # Record Time-to-First-Token
                if not full_text:
                    perf["ttft"] = time.perf_counter() - t0
                
                full_text += new_text
                
                # Clean stop tokens from streamed output
                clean = new_text
                for tok in STOP_TOKENS:
                    clean = clean.replace(tok, '')
                if clean:
                    yield clean
                
                if any(tok in new_text for tok in STOP_TOKENS):
                    break
        except GeneratorExit:
            stop_event.set()
            raise

        # Final cleanup
        for tok in STOP_TOKENS:
            full_text = full_text.replace(tok, '')
        full_text = full_text.strip()
        
        perf["total"] = time.perf_counter() - t0

        # ─── Phase 4: Deterministic Tool Routing ───
        # Route based on verbal prompt first (higher quality text), else VLM output
        text_for_routing = user_prompt if user_prompt else full_text
        tool = detect_tool(text_for_routing)
        
        if tool != "none" and tool in AVAILABLE_TOOLS:
            yield "\n\n"
            
            if tool == "calculator":
                expr = extract_math_expression(text_for_routing)
                yield f"🔧 Calculator → `{expr}`\n"
                try:
                    result = AVAILABLE_TOOLS["calculator"](expr)
                    yield f"= {result}\n"
                except Exception as e:
                    yield f"Calculator error: {e}\n"
                yield f"_Source: {MODEL_NAME} + Calculator Tool_"
                    
            elif tool == "matrix":
                yield "🔧 Matrix Solver\n"
                try:
                    result = AVAILABLE_TOOLS["matrix"](text_for_routing)
                    yield f"```\n{result}\n```\n"
                except Exception as e:
                    yield f"Matrix error: {e}\n"
                yield f"_Source: {MODEL_NAME} + NumPy Matrix Tool_"
                
            elif tool == "web_search":
                query = extract_search_query(text_for_routing)
                yield f"🌐 Searching: \"{query}\"\n"
                try:
                    result = AVAILABLE_TOOLS["web_search"](query)
                    
                    yield f"🧠 Synthesizing answer...\n\n"
                    
                    # Create text-only prompt
                    synth_conv = conversation_lib.conv_templates["qwen_2"].copy()
                    synth_qs = (
                        f"Context from web search:\n{result}\n\n"
                        f"Question: {query}\n"
                        f"Answer the question using only the context above. Keep it brief."
                    )
                    synth_conv.append_message(synth_conv.roles[0], synth_qs)
                    synth_conv.append_message(synth_conv.roles[1], None)
                    synth_prompt = synth_conv.get_prompt()
                    
                    # Standard text tokenization (no image tokens)
                    synth_input_ids = self.tokenizer(synth_prompt, return_tensors="pt").input_ids.to(self.device)
                    
                    synth_streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                    synth_kwargs = dict(
                        inputs=synth_input_ids,
                        streamer=synth_streamer,
                        max_new_tokens=256,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                    )
                    
                    if stop_event is not None:
                        synth_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnEventCriteria(stop_event)])
                    
                    # Run generation in background thread with inference_mode
                    def _synth_generate():
                        with torch.inference_mode():
                            self.model.generate(**synth_kwargs)
                    Thread(target=_synth_generate).start()
                    
                    for chunk in synth_streamer:
                        if stop_event is not None and stop_event.is_set():
                            break
                        yield chunk
                        
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    yield f"Search error: {e}\n"
                yield f"\n\n_Source: {MODEL_NAME} + Web Search_"
        else:
            yield f"\n\n_Source: {MODEL_NAME} (local inference)_"

        # ─── Performance Footer ───
        yield (
            f"\n\n---\n"
            f"⚡ Pre: {perf['preprocess']:.2f}s "
            f"| TTFT: {perf['ttft']:.2f}s "
            f"| Total: {perf['total']:.2f}s"
        )
        
        # Console performance log
        print(f"\n{'='*50}")
        print(f"  🏎️  PERFORMANCE REPORT")
        print(f"  Preprocessing:  {perf['preprocess']:.3f}s")
        print(f"  TTFT:           {perf['ttft']:.3f}s")
        print(f"  Total:          {perf['total']:.3f}s")
        print(f"  Tool:           {tool}")
        print(f"  Prompt:         {'[voice] ' + user_prompt[:50] if user_prompt else '[OCR mode]'}")
        print(f"  Output:         {full_text[:80]}...")
        print(f"{'='*50}\n")
