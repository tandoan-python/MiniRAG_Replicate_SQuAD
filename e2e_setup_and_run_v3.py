import os
import subprocess
import sys
import time
import gc
import csv
import logging
from typing import Dict

# --- CONFIGURATION V6 (CLEAN ARCHITECTURE) ---
MAX_EVAL_ROWS = 15     # Đổi thành 15: Chỉ test sinh câu trả lời cho 15 câu hỏi đầu tiên
MAX_INDEX_FILES = 30   # Giới hạn chỉ đọc 30 file text đầu tiên để tạo VectorDB nhanh chóng
TARGET_MODELS = ["PHI", "qwen"]

# Config paths absolute to ensure no relative path issues during sub-processing
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MINIRAG_DIR = os.path.join(CURRENT_DIR, "MiniRAG") # Khai báo biến trỏ tới thư mục con MiniRAG

DATASETS_CONFIG = {
    # "LiHua-World": {
    #     "working_dir": os.path.join(MINIRAG_DIR, "dataset", "LiHua-World", "LiHua-World"),
    #     "data_path": os.path.join(MINIRAG_DIR, "dataset", "LiHua-World", "data", "LiHua-World"),
    #     "query_path": os.path.join(MINIRAG_DIR, "dataset", "LiHua-World", "qa", "query_set.csv")
    # },
    "MultiHop-RAG": {
        "working_dir": os.path.join(MINIRAG_DIR, "dataset", "MultiHopRAG", "VectorDB"), 
        "data_path": os.path.join(MINIRAG_DIR, "dataset", "MultiHopRAG"), # Trỏ vào thư mục chứa file dataset.jsonl
        "query_path": os.path.join(MINIRAG_DIR, "dataset", "MultiHopRAG", "dataset.csv") # File bộ câu hỏi test
    }
}

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class MiniRAGEvaluatorV6:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.python_exe = sys.executable # Use current python env
        
        self.env = os.environ.copy()
        self.env.pop("HF_HUB_OFFLINE", None)
        # Tối ưu RAM/VRAM cho SLMs
        self.env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.env["PYTHONUTF8"] = "1"
        self.env["TOKENIZERS_PARALLELISM"] = "false"

    def _release_resources(self):
        """Dọn dẹp RAM và VRAM sau mỗi tiến trình nặng."""
        gc.collect()
        time.sleep(5)

    def _run_indexing(self, model: str, config: Dict[str, str]) -> bool:
        """Kiểm tra và chạy Step 0 (Indexing) nếu VectorDB chưa tồn tại."""
        # MiniRAG thường tạo file kv_store_doc_status.json hoặc vdb_*.json
        check_file = os.path.join(config["working_dir"], "kv_store_doc_status.json")
        
        if os.path.exists(check_file) and os.path.getsize(check_file) > 100:
            logger.info(f"VectorDB hợp lệ đã tồn tại tại {config['working_dir']}. Bỏ qua Indexing.")
            return True

        logger.warning(f"VectorDB rỗng hoặc chưa hoàn thiện. Bắt đầu Indexing cho {model}...")
        idx_cmd = [
            self.python_exe, os.path.join(MINIRAG_DIR, "reproduce", "Step_0_index_v3.py"), 
            "--model", model,
            "--workingdir", config["working_dir"],
            "--datapath", config["data_path"],
            "--max_files", str(MAX_INDEX_FILES) # Thêm tham số giới hạn file đã fix
        ]
        
        # Chạy file Step 0 độc lập
        result = subprocess.run(idx_cmd, cwd=self.base_dir, env=self.env, text=True)
        self._release_resources()
        
        if result.returncode != 0 or not os.path.exists(check_file):
            logger.error(f"Tạo VectorDB thất bại cho model {model}. Quá trình dừng lại.")
            return False
            
        return True

    def _check_early_stop(self, file_path: str) -> str:
        """Kiểm tra log QA để quyết định Resume, Restart hay Complete."""
        if not os.path.exists(file_path): 
            return "RESTART"
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
            if len(reader) == 0: 
                return "RESTART"
            if len(reader) >= MAX_EVAL_ROWS: 
                return "COMPLETE"
            
            # Check lỗi logic ngớ ngẩn của SLM trong 5 dòng đầu
            check_limit = min(5, len(reader))
            error_count = 0
            for row in reader[:check_limit]:
                # Chỉ kiểm tra riêng cột trả lời của minirag, tránh nhầm từ khóa ở Câu hỏi
                ans = str(row.get("minirag", "")).lower()
                if any(x in ans for x in ["sorry", "error", "[no-context]"]):
                    error_count += 1
            
            if error_count >= 3:
                logger.warning(f"Tỷ lệ lỗi ảo giác quá cao ({error_count}/{check_limit}). Yêu cầu DỪNG NGAY.")
                return "FAILED"
                
            return "RESUME"
        except Exception as e:
            logger.error(f"Lỗi đọc file CSV: {e}")
            return "RESTART"

    def run_pipeline(self):
        """Thực thi end-to-end toàn bộ pipeline."""
        logger.info("Bắt đầu Pipeline đánh giá.")
        
        for dataset_name, config in DATASETS_CONFIG.items():
            print(f"\n{'='*60}")
            logger.info(f"BẮT ĐẦU ĐÁNH GIÁ: {dataset_name.upper()}")
            print(f"{'='*60}")
            
            log_dir = os.path.join(self.base_dir, "logs", dataset_name)
            os.makedirs(log_dir, exist_ok=True)
            
            for model in TARGET_MODELS:
                print(f"\n--- CHẠY MÔ HÌNH: {model} ---")
                
                # 1. Chạy Indexing (Step 0)
                if not self._run_indexing(model, config):
                    continue
                    
                # 2. Chuẩn bị chạy QA (Step 1)
                output_log_name = f"output_{model}_minirag.csv"
                output_log_absolute = os.path.join(log_dir, output_log_name)
                
                status = self._check_early_stop(output_log_absolute)
                if status == "COMPLETE":
                    logger.info(f"Đã hoàn thành đánh giá cho {model}. Bỏ qua.")
                    continue
                elif status == "FAILED":
                    logger.error(f"Đánh giá trước đó cho {model} đã thất bại do lỗi 'Sorry/Error'. Bỏ qua chạy lại.")
                    continue
                elif status == "RESTART" and os.path.exists(output_log_absolute):
                    os.remove(output_log_absolute)
                    
                qa_cmd = [
                    self.python_exe, os.path.join(MINIRAG_DIR, "reproduce", "Step_1_QA_v3.py"),
                    "--model", model,
                    "--workingdir", config["working_dir"],
                    "--querypath", config["query_path"],
                    "--outputpath", output_log_absolute
                ]
                
                logger.info(f"Bắt đầu QA sinh câu trả lời cho {model}...")
                proc = subprocess.Popen(qa_cmd, cwd=self.base_dir, env=self.env)
                
                # --- CƠ CHẾ GIÁM SÁT EARLY-STOP LIÊN TỤC ---
                while proc.poll() is None: # Trong khi tiến trình con vẫn đang chạy
                    time.sleep(3) # Cứ 3 giây kiểm tra file log 1 lần
                    current_status = self._check_early_stop(output_log_absolute)
                    if current_status == "FAILED":
                        logger.error(f"❌ DỪNG KHẨN CẤP: Mô hình {model} không thể sinh câu trả lời hợp lệ (Sorry/Error)! Hủy tiến trình.")
                        proc.terminate() # Ép buộc kill tiến trình QA ngay lập tức
                        break
                        
                proc.wait() # Đảm bảo tiến trình đã thực sự đóng
                self._release_resources()

if __name__ == "__main__":
    pipeline = MiniRAGEvaluatorV6(CURRENT_DIR)
    pipeline.run_pipeline()