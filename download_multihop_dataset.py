import os
import pandas as pd
from datasets import load_dataset
import json

def prepare_multihop_dataset():
    output_dir = "./local_models/MultiHopRAG"
    os.makedirs(output_dir, exist_ok=True)
    
    # Tải tập dữ liệu từ HuggingFace
    dataset = load_dataset("yixuantt/MultiHopRAG", split="test")
    df = pd.DataFrame(dataset)
    
    # Chuẩn bị file corpus (dataset.jsonl)
    corpus_data = []
    for contexts in df['contexts'].tolist():
        for ctx in contexts:
            corpus_data.append({
                "title": ctx.get("title", "Unknown"),
                "text": ctx.get("body", "")
            })
    
    corpus_df = pd.DataFrame(corpus_data).drop_duplicates(subset=['text'])
    with open(f"{output_dir}/dataset.jsonl", "w", encoding="utf-8") as f:
        for record in corpus_df.to_dict(orient="records"):
            f.write(json.dumps(record) + "\n")
            
    # Chuẩn bị file truy vấn (dataset.csv)
    query_df = df[['query', 'answer']].copy()
    query_df.rename(columns={'query': 'Question', 'answer': 'Gold Answer'}, inplace=True)
    query_df.to_csv(f"{output_dir}/dataset.csv", index=False, encoding="utf-8")
    
    print("Quá trình chuẩn bị dữ liệu MultiHopRAG hoàn tất.")

if __name__ == "__main__":
    prepare_multihop_dataset()