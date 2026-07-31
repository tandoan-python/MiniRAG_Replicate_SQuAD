import sys
import os
import argparse
import traceback
import json

# Thêm thư mục MiniRAG vào sys.path để import minirag
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MINIRAG_ROOT = os.path.dirname(FILE_DIR)
sys.path.append(MINIRAG_ROOT)

from minirag import MiniRAG
from minirag.llm import hf_model_complete, hf_embed
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG Indexing Step")
    parser.add_argument("--model", type=str, default="PHI")
    parser.add_argument("--workingdir", type=str, required=True, help="Path to save VectorDB")
    parser.add_argument("--datapath", type=str, required=True, help="Path to raw .txt or .jsonl files")
    parser.add_argument("--max_files", type=int, default=0, help="Max documents/files to index (for quick testing)")
    return parser.parse_args()

args = get_args()

# Thiết lập đường dẫn TUYỆT ĐỐI cho Local Models từ thư mục root
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MINIRAG_ROOT = os.path.dirname(FILE_DIR)
WORKSPACE_ROOT = os.path.dirname(MINIRAG_ROOT) # Lùi thêm 1 cấp về root project (CS2202_NEW)

EMBEDDING_MODEL = os.path.join(WORKSPACE_ROOT, "local_models", "all-MiniLM-L6-v2")

if args.model.upper() == "PHI":
    LLM_MODEL = os.path.join(WORKSPACE_ROOT, "local_models", "Phi-3.5-mini-instruct")
elif args.model.lower() == "qwen":
    LLM_MODEL = os.path.join(WORKSPACE_ROOT, "local_models", "Qwen2.5-3B-Instruct")
else:
    print(f"[ERROR] Invalid model name: {args.model}")
    sys.exit(1)

WORKING_DIR = os.path.abspath(args.workingdir)
DATA_PATH = os.path.abspath(args.datapath)

print("="*50)
print(f"[STEP 0] STARTING INDEXING")
print(f"USING LLM: {LLM_MODEL}")
print(f"WORKING DIR: {WORKING_DIR}")
print(f"DATA PATH: {DATA_PATH}")
print("="*50)

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)

try:
    print("[SYSTEM] Đang load mô hình vào VRAM...")
    rag = MiniRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_max_token_size=3000, # Giới hạn token để tránh quá tải cho SLM
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
    print(f"[CRITICAL ERROR] Không thể khởi tạo MiniRAG/Load Model: {e}")
    traceback.print_exc()
    sys.exit(1)

print(f"\n[SYSTEM] Bắt đầu quét dữ liệu tại: {DATA_PATH}")
file_count = 0
success_count = 0

target_files = []
if os.path.isfile(DATA_PATH):
    target_files.append(DATA_PATH)
else:
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith((".txt", ".jsonl")):
                target_files.append(os.path.join(root, file))

for file_path in target_files:
    file_name = os.path.basename(file_path)
    file_count += 1
    
    # Dừng sớm file loop nếu đã vượt max_files
    if args.max_files > 0 and success_count >= args.max_files:
        break
        
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if len(content) > 10:
                    chunk_text = f"[Timeline: {file_name}]\n{content}"
                    print(f"   -> Đang Index file ({success_count+1}): {file_name}")
                    rag.insert(chunk_text)
                    success_count += 1
                    
        elif file_path.endswith(".jsonl"):
            print(f"   -> Đang quét file JSONL: {file_name}")
            with open(file_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    if args.max_files > 0 and success_count >= args.max_files:
                        break
                        
                    line = line.strip()
                    if not line: continue
                    
                    try:
                        data = json.loads(line)
                        # Dataset JSONL thường chứa dữ liệu thực ở các key bên dưới
                        content = data.get("text", data.get("context", data.get("passage", data.get("body", ""))))
                        
                        if not content:
                            # Fallback: Ghép tất cả các giá trị string lại
                            content = " \n ".join([str(v) for k, v in data.items() if isinstance(v, str)])
                            
                        if len(content) > 10:
                            chunk_text = f"[Source: {file_name} - Dòng {line_idx}]\n{content}"
                            rag.insert(chunk_text)
                            success_count += 1
                            
                    except json.JSONDecodeError:
                        print(f"[WARNING] Lỗi cú pháp JSON ở dòng {line_idx} trong {file_name}")
                        
    except Exception as e:
        print(f"[WARNING] Bỏ qua file lỗi {file_path}. Nguyên nhân: {e}")

if success_count > 0:
    print(f"\n[SUCCESS] Đã tạo VectorDB thành công cho {success_count} documents!")
else:
    print(f"\n[CRITICAL ERROR] Không có file nào được nạp vào VectorDB thành công. Vui lòng kiểm tra DataPath.")
    sys.exit(1)