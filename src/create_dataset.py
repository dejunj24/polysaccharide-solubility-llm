import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

# 1. 다당류 반복단위 SMILES 사전 (대표 다당류 10종)
# [*] 표기는 고분자 중합 연결 위치(Attachment point)를 의미함
POLYSACCHARIDES = {
    "Cellulose": "[*]OC1C(O)C(O)C([*])OC1CO",
    "Amylose (Starch)": "[*]OC1C([*])OC(CO)C(O)C1O",
    "Chitin": "[*]OC1C(NC(=O)C)C(O)C([*])OC1CO",
    "Chitosan": "[*]OC1C(N)C(O)C([*])OC1CO",
    "Dextran": "[*]OCC1OC([*])C(O)C(O)C1O",
    "Pullulan": "[*]OC1C(CO)OC([*])C(O)C1O",
    "Agarose": "[*]OC1CC2OC1C(CO)OC2[*]",
    "Alginate": "[*]OC1C(C(=O)O)OC([*])C(O)C1O",
    "Hyaluronic Acid": "[*]OC1C(O)C([*])OC(C(=O)O)C1OC2C(CO)OC(O)C(NC(=O)C)C2O",
    "Curdlan": "[*]OC1C(O)C([*])OC(CO)C1O"
}

# 2. 용매 목록
SOLVENTS = ["water", "DMSO", "ethanol", "acetone", "THF", "chloroform", "DMF"]

# 3. 화학적 실제 용해도 매핑 규칙 (도메인 지식 반영)
# (다당류, 용매) -> "yes" (가용) / "no" (불용)
SOLUBILITY_RULES = {
    ("Cellulose", "water"): "no",
    ("Cellulose", "DMSO"): "no",
    ("Cellulose", "ethanol"): "no",
    ("Cellulose", "acetone"): "no",
    ("Cellulose", "THF"): "no",
    ("Cellulose", "chloroform"): "no",
    ("Cellulose", "DMF"): "no",
    
    ("Amylose (Starch)", "water"): "yes",
    ("Amylose (Starch)", "DMSO"): "yes",
    ("Amylose (Starch)", "ethanol"): "no",
    ("Amylose (Starch)", "acetone"): "no",
    ("Amylose (Starch)", "THF"): "no",
    ("Amylose (Starch)", "chloroform"): "no",
    ("Amylose (Starch)", "DMF"): "yes",
    
    ("Chitin", "water"): "no",
    ("Chitin", "DMSO"): "no",
    ("Chitin", "ethanol"): "no",
    ("Chitin", "acetone"): "no",
    ("Chitin", "THF"): "no",
    ("Chitin", "chloroform"): "no",
    ("Chitin", "DMF"): "no",
    
    ("Chitosan", "water"): "no",  # 순수 물(중성)에서는 불용, 산성수용액에서 가용
    ("Chitosan", "DMSO"): "no",
    ("Chitosan", "ethanol"): "no",
    ("Chitosan", "acetone"): "no",
    ("Chitosan", "THF"): "no",
    ("Chitosan", "chloroform"): "no",
    ("Chitosan", "DMF"): "no",
    
    ("Dextran", "water"): "yes",
    ("Dextran", "DMSO"): "yes",
    ("Dextran", "ethanol"): "no",
    ("Dextran", "acetone"): "no",
    ("Dextran", "THF"): "no",
    ("Dextran", "chloroform"): "no",
    ("Dextran", "DMF"): "yes",
    
    ("Pullulan", "water"): "yes",
    ("Pullulan", "DMSO"): "yes",
    ("Pullulan", "ethanol"): "no",
    ("Pullulan", "acetone"): "no",
    ("Pullulan", "THF"): "no",
    ("Pullulan", "chloroform"): "no",
    ("Pullulan", "DMF"): "yes",
    
    ("Agarose", "water"): "yes",  # 온수에서 가용
    ("Agarose", "DMSO"): "yes",
    ("Agarose", "ethanol"): "no",
    ("Agarose", "acetone"): "no",
    ("Agarose", "THF"): "no",
    ("Agarose", "chloroform"): "no",
    ("Agarose", "DMF"): "no",
    
    ("Alginate", "water"): "yes",
    ("Alginate", "DMSO"): "no",
    ("Alginate", "ethanol"): "no",
    ("Alginate", "acetone"): "no",
    ("Alginate", "THF"): "no",
    ("Alginate", "chloroform"): "no",
    ("Alginate", "DMF"): "no",
    
    ("Hyaluronic Acid", "water"): "yes",
    ("Hyaluronic Acid", "DMSO"): "no",
    ("Hyaluronic Acid", "ethanol"): "no",
    ("Hyaluronic Acid", "acetone"): "no",
    ("Hyaluronic Acid", "THF"): "no",
    ("Hyaluronic Acid", "chloroform"): "no",
    ("Hyaluronic Acid", "DMF"): "no",
    
    ("Curdlan", "water"): "no",
    ("Curdlan", "DMSO"): "yes",
    ("Curdlan", "ethanol"): "no",
    ("Curdlan", "acetone"): "no",
    ("Curdlan", "THF"): "no",
    ("Curdlan", "chloroform"): "no",
    ("Curdlan", "DMF"): "no",
}

def generate_dataset():
    records = []
    for poly_name, smiles in POLYSACCHARIDES.items():
        for solvent in SOLVENTS:
            solubility = SOLUBILITY_RULES.get((poly_name, solvent), "no")
            records.append({
                "polysaccharide_name": poly_name,
                "smiles": smiles,
                "solvent": solvent,
                "solubility": solubility
            })
            
    df = pd.DataFrame(records)
    os.makedirs("/home/hoseo/polysaccharide_ai/data", exist_ok=True)
    df.to_csv("/home/hoseo/polysaccharide_ai/data/polysaccharide_solubility.csv", index=False)
    print(f"Total dataset generated: {len(df)} samples saved to CSV.")
    return df

if __name__ == "__main__":
    generate_dataset()
