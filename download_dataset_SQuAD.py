import os
import argparse
import pandas as pd
from datasets import load_dataset

def prepare_squad_dataset(base_dir: str, max_samples: int = 100) -> None:
    """
    Tải và tiền xử lý tập dữ liệu SQuAD v2, định dạng lại để tương thích với MiniRAG.
    
    Args:
        base_dir (str): Thư mục gốc chứa dữ liệu đầu ra.
        max_samples (int): Kích thước mẫu tối đa cần trích xuất.
    """
    print("[THÔNG BÁO] Bắt đầu tải tập dữ liệu SQuAD v2...")
    
    data_dir = os.path.join(base_dir, "data")
    qa_dir = os.path.join(base_dir, "qa")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(qa_dir, exist_ok=True)
    
    try:
        dataset = load_dataset("rajpurkar/squad_v2", split="validation")
    except Exception as e:
        print(f"[LỖI] Không thể kết nối hoặc tải dữ liệu từ HuggingFace: {e}")
        return

    questions, gold_answers = [], []
    seen_contexts = set()
    count = 0
    
    print("[THÔNG BÁO] Đang phân tích và trích xuất ngữ cảnh...")
    for item in dataset:
        if not item["answers"]["text"]: 
            continue
        
        q_text = item["question"]
        gold_ans = item["answers"]["text"][0]
        context_text = item["context"]
        
        # Lưu ngữ cảnh thành các tệp văn bản độc lập để MiniRAG lập chỉ mục
        if context_text not in seen_contexts:
            seen_contexts.add(context_text)
            doc_path = os.path.join(data_dir, f"squad_doc_{len(seen_contexts)}.txt")
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(context_text)
                
        questions.append(q_text)
        gold_answers.append(gold_ans)
        
        count += 1
        if max_samples and count >= max_samples:
            break
            
    df_qa = pd.DataFrame({"Question": questions, "Gold Answer": gold_answers})
    csv_path = os.path.join(qa_dir, "query_set.csv")
    df_qa.to_csv(csv_path, index=False, encoding="utf-8")
    
    print(f"[THÀNH CÔNG] Đã trích xuất {count} mẫu truy vấn.")
    print(f" - Dữ liệu văn bản (Contexts): {os.path.abspath(data_dir)}")
    print(f" - Tập truy vấn (Queries): {os.path.abspath(csv_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu SQuAD")
    parser.add_argument("--output_dir", type=str, default="./dataset/SQuAD", help="Thư mục xuất dữ liệu")
    parser.add_argument("--samples", type=int, default=100, help="Số lượng mẫu cần xử lý")
    args = parser.parse_args()
    
    prepare_squad_dataset(args.output_dir, args.samples)