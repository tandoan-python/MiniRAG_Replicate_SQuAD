import sys
import os
import argparse
import csv
import traceback
import ast
from tqdm import trange

# Thêm thư mục MiniRAG vào sys.path để import minirag
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MINIRAG_ROOT = os.path.dirname(FILE_DIR)
sys.path.append(MINIRAG_ROOT)

from minirag import MiniRAG, QueryParam
from minirag.llm import hf_model_complete, hf_embed
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG QA Step")
    parser.add_argument("--model", type=str, default="PHI")
    parser.add_argument("--outputpath", type=str, required=True, help="Path to output CSV")
    parser.add_argument("--workingdir", type=str, required=True, help="Path where VectorDB is saved")
    parser.add_argument("--querypath", type=str, required=True, help="Path to question CSV")
    return parser.parse_args()

args = get_args()

# Thiết lập đường dẫn TUYỆT ĐỐI cho Local Models từ thư mục root
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MINIRAG_ROOT = os.path.dirname(FILE_DIR)
WORKSPACE_ROOT = os.path.dirname(MINIRAG_ROOT) # Lùi 2 cấp về root project

EMBEDDING_MODEL = os.path.join(WORKSPACE_ROOT, "local_models", "all-MiniLM-L6-v2")

if args.model.upper() == "PHI":
    LLM_MODEL = os.path.join(WORKSPACE_ROOT, "local_models", "Phi-3.5-mini-instruct")
elif args.model.lower() == "qwen":
    LLM_MODEL = os.path.join(WORKSPACE_ROOT, "local_models", "Qwen2.5-3B-Instruct")
else:
    print(f"[ERROR] Invalid model name: {args.model}")
    sys.exit(1)

WORKING_DIR = os.path.abspath(args.workingdir)
QUERY_PATH = os.path.abspath(args.querypath)
OUTPUT_PATH = os.path.abspath(args.outputpath)

print("="*50)
print(f"[STEP 1] STARTING QA EVALUATION")
print(f"USING LLM: {LLM_MODEL}")
print(f"WORKING DIR: {WORKING_DIR}")
print(f"OUTPUT PATH: {OUTPUT_PATH}")
print("="*50)

try:
    rag = MiniRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_max_token_size=3000,
        llm_model_name=LLM_MODEL,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=1000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
                embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
            ),
        ),
    )
except Exception as e:
    print(f"[CRITICAL ERROR] Lỗi khởi tạo RAG lúc QA: {e}")
    sys.exit(1)

def _extract_question(r):
    try:
        keys = list(r.keys())
        for k in keys:
            k_clean = str(k).lower().strip().replace('\ufeff', '')
            if any(x in k_clean for x in ['quest', 'quer', 'prom']): return str(r[k])
        if keys: return str(r[keys[0]])
    except: pass
    return "UNKNOWN_Q"

def _extract_answer(r):
    try:
        keys = list(r.keys())
        ans = None
        for k in keys:
            k_clean = str(k).lower().strip().replace('\ufeff', '')
            if any(x in k_clean for x in ['answ', 'gold', 'resp']): 
                ans = str(r[k])
                break
                
        if not ans:
            if len(keys) > 1: ans = str(r[keys[1]])
            elif keys: ans = str(r[keys[0]])
            else: ans = "UNKNOWN_A"
            
        # Xử lý trường hợp MultiHop-RAG trả về mảng chuỗi kiểu "['ans1', 'ans2']"
        if ans.startswith('[') and ans.endswith(']'):
            try:
                parsed_ans = ast.literal_eval(ans)
                if isinstance(parsed_ans, list):
                    ans = " | ".join(str(x) for x in parsed_ans)
            except:
                pass
        return ans
    except: pass
    return "UNKNOWN_A"

QUESTION_LIST = []
GA_LIST = []
with open(QUERY_PATH, mode="r", encoding="utf-8") as question_file:
    reader = csv.DictReader(question_file)
    for row in reader:
        QUESTION_LIST.append(_extract_question(row))
        GA_LIST.append(_extract_answer(row))

def run_experiment(output_path):
    headers = ["Question", "Gold Answer", "minirag"]
    q_already = set()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, mode="r", encoding="utf-8") as question_file:
            reader = csv.DictReader(question_file)
            for row in reader:
                q_already.add(_extract_question(row))

    row_count = len(q_already)
    print(f"[INFO] Bỏ qua {row_count} câu hỏi đã có sẵn. Resume tiến trình...")

    with open(output_path, mode="a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if row_count == 0:
            writer.writerow(headers)

        for QUESTIONid in trange(len(QUESTION_LIST)):
            QUESTION = QUESTION_LIST[QUESTIONid]
            Gold_Answer = GA_LIST[QUESTIONid]
            
            # Skip if already answered
            if QUESTION in q_already:
                continue

            print(f"\n[Q]: {QUESTION}")
            
            try:
                # Gửi Query tới RAG
                ans = rag.query(QUESTION, param=QueryParam(mode="mini"))
                minirag_answer = ans.replace("\n", " ").replace("\r", "") if ans else "[Empty Answer]"
            except Exception as e:
                print(f"[ERROR] Lỗi khi sinh câu trả lời: {e}")
                minirag_answer = "Error"
                
            print(f"[MiniRAG]: {minirag_answer}")

            writer.writerow([QUESTION, Gold_Answer, minirag_answer])
            log_file.flush() # Flush ngay xuống đĩa để tránh mất dữ liệu nếu bị Crash

    print(f"\n[SUCCESS] Hoàn thành QA. File log: {output_path}")

if __name__ == "__main__":
    run_experiment(OUTPUT_PATH)