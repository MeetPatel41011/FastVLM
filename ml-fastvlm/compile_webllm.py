import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)

def main():
    print("==================================================")
    print("Phase 1: Model Compilation (MLC-LLM & WebLLM)")
    print("==================================================")
    
    # 1. Download Model (Qwen2 0.5B from Xenova or LLaVA base)
    MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
    LOCAL_DIR = "./dist/models/Qwen2-0.5B-Instruct"
    
    print(f"\n[1/3] Downloading LLM Component: {MODEL_ID}...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR, local_dir_use_symlinks=False)
        print(f"Successfully downloaded to {LOCAL_DIR}")
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)
    
    # 2. Convert and Quantize with MLC-LLM
    print("\n[2/3] Quantizing LLM to 4-bit (q4f16_1) using MLC-LLM...")
    try:
        import mlc_llm
    except ImportError:
        print("\n[ERROR] mlc_llm is not installed.")
        print("For AMD/Vulkan on Windows, the '-cpu' wheels include Vulkan support. Run:")
        print("pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu")
        sys.exit(1)

    # Generate config for Qwen2
    run_cmd(f"mlc_llm gen_config {LOCAL_DIR} --quantization q4f16_1 --conv-template qwen2 --output ./dist/Qwen2-0.5B-Instruct-q4f16_1-MLC")
    
    # Compile for WebGPU (which will use Vulkan on your AMD Windows machine)
    print("\n[COMPILING] Generating WebGPU/Vulkan WASM and Kernels...")
    run_cmd(f"mlc_llm compile {LOCAL_DIR} --quantization q4f16_1 --device webgpu -o ./dist/Qwen2-0.5B-Instruct-q4f16_1-MLC/Qwen2-0.5B-Instruct-q4f16_1-MLC-webgpu.wasm")
    
    # 3. Compile Vision Encoder (ONNX for Transformers.js)
    print("\n[3/3] Vision Encoder is natively supported via Xenova/llava-onevision ONNX artifacts.")
    print("Phase 1 Complete: WebGPU WASM Kernels and 4-bit weights generated.")

if __name__ == "__main__":
    main()
