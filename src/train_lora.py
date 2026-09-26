import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# transformers 4.30.2와 표준 MHA로 100% 호환되는 안정적인 모델
MODEL_ID = "facebook/opt-1.3b"
OUTPUT_DIR = "/home/hoseo/polysaccharide_ai/models/opt_lora_solubility"
TRAIN_FILE = "/home/hoseo/polysaccharide_ai/data/train.jsonl"

def load_data(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)

def main():
    print(f"Loading tokenizer & model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1080 Ti FP16 가속
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA 어댑터 설정 (OPT 계열의 q_proj, v_proj 타겟)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_raw = load_data(TRAIN_FILE)

    def preprocess_function(examples):
        input_ids_list, labels_list = [], []
        for messages in examples["messages"]:
            # OPT 프롬프트 포맷
            prompt = (
                f"### System:\n{messages[0]['content']}\n\n"
                f"### User:\n{messages[1]['content']}\n\n"
                f"### Answer:\n"
            )
            target = f"{messages[2]['content']}</s>"
            
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]

            # Prompt 영역은 loss 계산에서 제외 (-100)
            input_ids = prompt_tokens + target_tokens
            labels = [-100] * len(prompt_tokens) + target_tokens

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "labels": labels_list}

    tokenized_train = train_raw.map(preprocess_function, batched=True, remove_columns=["messages"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=5,
        fp16=True,
        logging_steps=2,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    )

    print("Starting LoRA fine-tuning on GPU...")
    trainer.train()

    print(f"Saving fine-tuned adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
