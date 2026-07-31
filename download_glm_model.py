import os
import argparse
from huggingface_hub import snapshot_download

def download_model(repo_id: str, local_dir: str, use_mirror: bool = False) -> None:
    """
    Tải mô hình từ HuggingFace Hub về máy cục bộ.
    
    Args:
        repo_id (str): Định danh của kho lưu trữ trên HuggingFace (VD: THUDM/glm-edge-1.5b-chat).
        local_dir (str): Đường dẫn thư mục đích trên máy tính.
        use_mirror (bool): Sử dụng máy chủ dự phòng (mirror) để tránh lỗi kết nối mạng.
    """
    print(f"[THÔNG BÁO] Đang tiến hành tải mô hình '{repo_id}'...")
    os.makedirs(local_dir, exist_ok=True)
    
    if use_mirror:
        print("[THÔNG BÁO] Đã kích hoạt máy chủ phản chiếu HF Mirror (hf-mirror.com)...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    try:
        # Đã loại bỏ 'resume_download' và 'local_dir_use_symlinks' vì đây là mặc định của bản mới
        # Thêm max_workers=2 để tránh lỗi WinError 10054 (đứt kết nối do quá nhiều luồng)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            max_workers=2
        )
        print(f"[THÀNH CÔNG] Mô hình đã được lưu tại: {os.path.abspath(local_dir)}")
    except Exception as e:
        print(f"[LỖI] Quá trình tải thất bại: {e}")
        print("\n[GỢI Ý] Vui lòng chạy lại lệnh với cờ --use_mirror (VD: python download_glm_model.py --use_mirror)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải mô hình LLM từ HuggingFace")
    parser.add_argument("--repo_id", type=str, default="THUDM/glm-edge-1.5b-chat", help="HuggingFace Repository ID")
    parser.add_argument("--save_dir", type=str, default="./models/glm-edge-1.5b-chat", help="Thư mục lưu trữ cục bộ")
    parser.add_argument("--use_mirror", action="store_true", help="Kích hoạt hf-mirror để khắc phục lỗi mạng")
    args = parser.parse_args()
    
    download_model(args.repo_id, args.save_dir, args.use_mirror)