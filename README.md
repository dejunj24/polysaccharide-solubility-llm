# 🧪 Polysaccharide Solubility Predictor (LLM Pipeline)

This repository contains an end-to-end pipeline for predicting the solubility of polysaccharides in various solvents using a **Parameter-Efficient Fine-Tuned (PEFT / LoRA) Large Language Model**.

## 🌟 Key Features
- **Domain-Specific Dataset:** Automated generation of SMILES and solvent pairs.
- **Hardware Optimized:** Fine-tunes a `1.3B` parameter model (`facebook/opt-1.3b`) using under **4GB of VRAM**.
- **Environment Stable:** Bypasses `GLIBCXX` and legacy NVIDIA driver (v470) conflicts.
- **Production Ready:** Includes FastAPI serving script and Dockerfile.

## 🚀 Step-by-Step Reproduction Guide

### Step 1. Clone & Setup
```bash
git clone https://github.com/dejunj24/polysaccharide-solubility-llm.git
cd polysaccharide-solubility-llm
conda create -n poly_ai python=3.10 -y
conda activate poly_ai
pip install "setuptools<70"
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Step 2. Data Generation
```bash
python src/create_dataset.py
python src/build_jsonl.py
```

### Step 3. Train & Evaluate
```bash
CUDA_VISIBLE_DEVICES=1 python src/train_lora.py
CUDA_VISIBLE_DEVICES=1 python src/evaluate.py
```

### Step 4. Run API Server
```bash
CUDA_VISIBLE_DEVICES=1 uvicorn src.app:app --host 0.0.0.0 --port 8000
```

## 🐳 Docker Deployment
```bash
docker build -t poly-solubility-api .
docker run --gpus all -p 8000:8000 -d poly-solubility-api
```