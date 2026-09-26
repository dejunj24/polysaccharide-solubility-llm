import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

BASE_MODEL_ID = "facebook/opt-1.3b"
ADAPTER_DIR = "/home/hoseo/polysaccharide_ai/models/opt_lora_solubility"
TEST_FILE = "/home/hoseo/polysaccharide_ai/data/test.jsonl"

def main():
    print("Loading base model & fine-tuned LoRA adapter...")
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

    y_true, y_pred = [], []
    test_samples = []

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            test_samples.append(json.loads(line))

    print(f"\nEvaluating on {len(test_samples)} test samples...")
    for item in tqdm(test_samples):
        messages = item["messages"]
        ground_truth = messages[2]["content"].strip().lower()

        prompt = (
            f"### System:\n{messages[0]['content']}\n\n"
            f"### User:\n{messages[1]['content']}\n\n"
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
        prediction_raw = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()

        pred_label = "yes" if "yes" in prediction_raw else "no"

        y_true.append(ground_truth)
        y_pred.append(pred_label)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix (rows: true, cols: pred):")
    print(confusion_matrix(y_true, y_pred, labels=["yes", "no"]))
    print("=" * 50)

if __name__ == "__main__":
    main()
