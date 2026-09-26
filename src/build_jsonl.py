import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

SYSTEM_PROMPT = (
    "You are an expert materials chemist specialized in polymer solubility prediction. "
    "Given the repeating unit SMILES of a polysaccharide and a target solvent, "
    "predict whether the polysaccharide is soluble in that solvent. "
    "Respond strictly with either 'yes' or 'no'."
)

def format_row_to_message(row):
    user_content = (
        f"Is the polysaccharide with the following repeating unit SMILES soluble in {row['solvent']}?\n"
        f"SMILES: {row['smiles']}"
    )
    assistant_content = str(row['solubility']).strip().lower()
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def convert_to_jsonl():
    csv_path = "/home/hoseo/polysaccharide_ai/data/polysaccharide_solubility.csv"
    df = pd.read_csv(csv_path)
    
    # 80:20 Train / Test 분할 (계층적 샘플링으로 yes/no 비율 유지)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['solubility'])
    
    def save_jsonl(dataframe, filename):
        filepath = os.path.join("/home/hoseo/polysaccharide_ai/data", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for _, row in dataframe.iterrows():
                f.write(json.dumps(format_row_to_message(row), ensure_ascii=False) + "\n")
        print(f"Saved {len(dataframe)} samples to {filepath}")

    save_jsonl(train_df, "train.jsonl")
    save_jsonl(test_df, "test.jsonl")

if __name__ == "__main__":
    convert_to_jsonl()
