import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "facebook/opt-1.3b"
ADAPTER_DIR = "/home/hoseo/polysaccharide_ai/models/opt_lora_solubility"

app = FastAPI(
    title="Polysaccharide Solubility Predictor API",
    description="Fine-tuned OPT-1.3B with LoRA for predicting polysaccharide solubility based on SMILES.",
    version="1.0.0"
)

# 전역 모델 & 토크나이저 컨텍스트
context = {}

class PredictionRequest(BaseModel):
    smiles: str = Field(..., example="[*]OC1C([*])OC(CO)C(O)C1O", description="Polymer repeat unit SMILES")
    solvent: str = Field(..., example="water", description="Target solvent name")

class PredictionResponse(BaseModel):
    polysaccharide_smiles: str
    solvent: str
    solubility: str
    is_soluble: bool

@app.on_event("startup")
def load_model():
    print("Loading base model & LoRA weights into GPU...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    context["tokenizer"] = tokenizer
    context["model"] = model
    print("Model loaded and ready for serving.")

@app.get("/health")
def health_check():
    return {"status": "ok", "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024**2)}

@app.post("/predict", response_model=PredictionResponse)
def predict_solubility(payload: PredictionRequest):
    tokenizer = context.get("tokenizer")
    model = context.get("model")

    if not tokenizer or not model:
        raise HTTPException(status_code=503, detail="Model is still initializing.")

    prompt = (
        f"### System:\nYou are an expert materials chemist specialized in polymer solubility prediction. "
        f"Given the repeating unit SMILES of a polysaccharide and a target solvent, "
        f"predict whether the polysaccharide is soluble in that solvent. "
        f"Respond strictly with either 'yes' or 'no'.\n\n"
        f"### User:\nIs the polysaccharide with the following repeating unit SMILES soluble in {payload.solvent}?\n"
        f"SMILES: {payload.smiles}\n\n"
        f"### Answer:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_pred = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()

    label = "yes" if "yes" in raw_pred else "no"

    return PredictionResponse(
        polysaccharide_smiles=payload.smiles,
        solvent=payload.solvent,
        solubility=label,
        is_soluble=(label == "yes")
    )
