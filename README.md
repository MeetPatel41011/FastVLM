# FastVLM: LiveQandA Agentic Vision System

FastVLM is a powerful, real-time Agentic Vision System built on top of the FastVLM Vision-Language Model. It allows you to point a camera at a document, mathematical problem, or real-world object, and get immediate, intelligent answers. The system is split into a Next.js web frontend and a Python/FastAPI backend for ML inference.

## 🌟 Key Features

* **Real-time Vision**: Automatically detects when a document or question is held up to the camera using a lightweight contour and text-detection "Gatekeeper".
* **Agentic Reasoning**: Uses the FastVLM hybrid vision encoder (FastViTHD) to process high-resolution images rapidly, then uses a Large Language Model (Qwen2-0.5B) to generate answers.
* **Tool Calling**: The ML engine can intelligently call external tools, such as the Tavily API, to search the web for real-time information if the model doesn't know the answer.
* **Hardware Accelerated**: Automatically detects your hardware (NVIDIA CUDA, Apple Silicon MPS, or CPU) and optimizes inference.
* **Low Latency**: Engineered for sub-1000ms Time-to-First-Token (TTFT) by using SDPA attention and aggressive preprocessing.

## 🏗️ Architecture

The project consists of two main folders:

1. **`web-fastvlm/`**: The modern Next.js frontend UI. It captures the camera feed, handles the user interface, and displays the streamed results from the backend.
2. **`ml-fastvlm/`**: The Python FastAPI backend. It loads the PyTorch FastVLM model into your GPU/CPU, processes the image, calls the LLM, and streams the text response back via Server-Sent Events (SSE).

## 🚀 Getting Started Locally

### Prerequisites
* **Node.js** (v18+) for the frontend
* **Python 3.10 or 3.11** for the ML backend
* **Git** and **git-lfs** (for downloading model weights)
* (Optional but recommended) **NVIDIA GPU** or **Apple Silicon Mac** for fast inference

### 1. Setup the ML Backend (`ml-fastvlm`)

The backend requires downloading the FastVLM model weights and setting up a Python environment.

```bash
cd ml-fastvlm

# Create and activate a Python virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

# Install dependencies
pip install -e .
pip install opencv-python numpy==1.26.4 tavily-python python-dotenv fastapi uvicorn
```

*(Note for Windows Users: `numpy==1.26.4` is strictly required to prevent crashes).*

#### Download the Model
You need to download the FastVLM PyTorch checkpoint. From inside the `ml-fastvlm` directory:
```bash
bash get_models.sh
```
*(This will download the `fastvlm_0.5b_stage3` model to the `ml-fastvlm/checkpoints/` directory).*

#### Configure Environment Variables
Create a `.env` file in the `ml-fastvlm` folder with your Tavily search key:
```env
TAVILY_API_KEY=your_tavily_api_key_here
```

#### Run the Server
```bash
python server.py
```
The FastAPI backend will start on `http://localhost:8000`.

### 2. Setup the Web Frontend (`web-fastvlm`)

Open a new terminal window.

```bash
cd web-fastvlm

# Install Node dependencies
npm install

# Configure Environment Variables
# Create a .env file inside web-fastvlm:
TAVILY_API_KEY=your_tavily_api_key_here

# Run the Next.js development server
npm run dev
```

The frontend will start on `http://localhost:3000`. Open this in your browser, grant camera permissions, and hold up a handwritten or printed question to the camera!

## ☁️ Cloud Deployment (Modal)

If you don't have a local GPU, you can deploy the backend (or the unified app) to [Modal](https://modal.com) for serverless GPU execution.

1. Install the Modal CLI: `pip install modal`
2. Authenticate: `modal setup`
3. Upload the model weights: Run `./upload_weights.ps1` (or `modal volume put fastvlm-weights ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3 /llava-fastvithd_0.5b_stage3`)
4. Deploy the app: Run `./deploy_modal.ps1` (or `modal deploy ml-fastvlm/modal_app.py`)

## 📝 License

This project includes code derived from the official FastVLM implementation. Please review the `LICENSE` and `LICENSE_MODEL` files inside the `ml-fastvlm` directory for specific usage terms regarding the models and the training code.
