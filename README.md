# Thực nghiệm Replicate MiniRAG

Dự án này thực hiện tái hiện và đánh giá hệ thống MiniRAG (được phát triển bởi HKUDS) dựa trên mã nguồn mở. Quá trình thực nghiệm được thiết kế để kiểm tra khả năng lưu trữ, lập chỉ mục đồ thị (graph indexing) và truy vấn của MiniRAG bằng cách sử dụng các mô hình ngôn ngữ lớn (LLM) và mô hình nhúng (Embedding) trên các tập dữ liệu chuẩn (như SQuAD, MultiHop-RAG).

Dự án cung cấp hai phương pháp để tái thực nghiệm: chạy trực tuyến thông qua Jupyter Notebook (Kaggle/Colab) hoặc triển khai toàn diện trên môi trường máy chủ/cá nhân cục bộ (Local).

---

## 1. Yêu cầu hệ thống và Cài đặt chung

**Phần cứng khuyến nghị:**
- Bộ nhớ RAM: Tối thiểu 16GB (Khuyến nghị 32GB nếu chạy hoàn toàn trên CPU).
- GPU (Tùy chọn nhưng khuyến nghị): NVIDIA GPU với tối thiểu 8GB VRAM (để chạy các mô hình 1.5B - 3B như GLM, QWEN, PHI).
- Ổ cứng: Trống tối thiểu 20GB để lưu trữ dữ liệu và các checkpoint mô hình.

**Phần mềm:**
- Hệ điều hành: Linux/Windows/macOS.
- Python: Phiên bản 3.12 trở lên.
- Các thư viện Python cốt lõi: `transformers`, `accelerate`, `bitsandbytes`, `sentence-transformers`, `nano-vectordb`, `datasets`, `pandas`, `tqdm`, `modelscope`, `requests`.

---

## 2. Lựa chọn 1: Thực thi trực tuyến trên Kaggle/Colab

Mã nguồn thực nghiệm trực tuyến được gộp trong một notebook duy nhất: `CS2202_MiniRAG_GLM-Edge_QWEN_SQuAD.ipynb`. 

**Hướng dẫn thực thi:**
1. Mở tệp notebook này trên môi trường Kaggle hoặc Google Colab.
2. Notebook đã tích hợp sẵn toàn bộ các quy trình bao gồm:
   - Tự động clone mã nguồn MiniRAG từ kho lưu trữ gốc, cài đặt thư viện và thiết lập mã nguồn nhúng.
   - Tải dataset SQuAD v2 từ HuggingFace và định dạng thành các tệp văn bản.
   - Chạy tập lệnh Lập chỉ mục Vector (`Step_0_index.py`).
   - Chạy tập lệnh Truy vấn (`Step_1_QA.py`).
3. Vòng lặp sẽ tự động thực hiện cả hai quá trình cho các mô hình mục tiêu (GLM, QWEN) và in ra báo cáo tổng kết độ chính xác cuối cùng trực tiếp trên màn hình.

*(Lưu ý: Môi trường notebook này đánh giá mức độ chính xác cơ bản dựa trên thuật toán so khớp chuỗi - substring matching).*

---

## 3. Lựa chọn 2: Thực thi cục bộ (Local Execution)

Lựa chọn này dành cho những hệ thống triển khai local, quy trình được module hóa rõ ràng, hỗ trợ tải mô hình ngoại tuyến (offline) và đánh giá nâng cao sử dụng phương pháp "LLM-as-a-judge" qua Ollama.

### 3.1. Chuẩn bị mã nguồn
Hệ thống yêu cầu mã nguồn gốc của MiniRAG cùng thư mục tái thực nghiệm (`reproduce`) của dự án này.
1. Clone mã nguồn gốc của MiniRAG:
   ```bash
   git clone https://github.com/HKUDS/MiniRAG.git
   ```
2. Thay thế (hoặc chép đè) thư mục `reproduce` bên trong thư mục `MiniRAG` vừa clone về bằng thư mục `reproduce` được cung cấp trong dự án này. Điều này đảm bảo các file logic thực thi (`Step_0_index_v3.py`, `Step_1_QA_v3.py`) đã được tùy chỉnh tương thích với quy trình của chúng ta.

### 3.2. Chuẩn bị dữ liệu và tải mô hình
Các tệp kịch bản `download_*.py` dùng để thiết lập dữ liệu và tải trọng số của các mô hình về thư mục nội bộ (để tránh lỗi ngắt kết nối mạng khi chạy End-to-End):
- **Tải Mô hình LLM:** Chạy `download_models.py` (và `download_glm_model.py` nếu cần). Kịch bản này sử dụng ModelScope Hub để kéo các mô hình SLM (như Phi-3.5-mini, Qwen2.5-3B) một cách bền bỉ và lưu vào thư mục `local_models`.
- **Tải Dữ liệu:** Chạy `download_dataset_SQuAD.py` và `download_multihop_dataset.py`. Scripts này sẽ xử lý các bộ dữ liệu thành tập các văn bản `.txt` thô và các tệp `.csv` truy vấn chuẩn bị cho quá trình RAG.

### 3.3. Thực thi quy trình toàn trình (End-to-End Pipeline)
Kịch bản `e2e_setup_and_run_v3.py` đóng vai trò là bộ điều phối chính:
```bash
python e2e_setup_and_run_v3.py
```
- **Bước Lập chỉ mục (Indexing - Step 0):** Script quét các file text đã tải, sử dụng thư viện `nano-vectordb` phân rã văn bản và trích xuất thực thể, quan hệ. Chỉ mục (VectorDB) được tạo và lưu trữ tự động tại thư mục quy định.
- **Bước Truy vấn (QA - Step 1):** Hệ thống gọi SLMs để sinh câu trả lời cho các truy vấn dựa trên kho đồ thị vừa tạo. Hệ thống tích hợp cơ chế giải phóng tài nguyên tự động (Early-Stop / RAM Cleanup) để xử lý tình trạng ảo giác hoặc quá tải bộ nhớ trên SLM.
- Toàn bộ kết quả truy vấn sinh ra được kết xuất vào file log định dạng `.csv`.

### 3.4. Đánh giá chất lượng sinh sử dụng Ollama LLM-as-a-judge
Thay vì dùng các API đắt đỏ (như GPT-4o), dự án sử dụng LLM nội bộ qua **Ollama** để chấm điểm câu trả lời sinh ra, dựa trên định dạng Prompt chuẩn được tham chiếu từ bài báo gốc của tác giả.
- **Yêu cầu:** Bạn cần cài đặt [Ollama](https://ollama.com/) trên máy và tải sẵn mô hình giám khảo (Ví dụ: `ollama run llama3`).
- **Thực thi:**
  ```bash
  python evaluate_ollama.py --input /duong/dan/toi/file/output_model.csv --output /duong/dan/toi/ketqua_danhgia.csv --judge_model llama3
  ```
- **Mô tả chi tiết Output:** Script gọi cục bộ đến API của Ollama, thiết lập `temperature=0` nhằm cố định kết quả JSON trả về. Mỗi câu trả lời của mô hình sẽ được đánh giá điểm số thuộc tập {-1, 0, 1}. Báo cáo tổng kết sẽ in ra màn hình Độ chính xác (Accuracy) và Tỷ lệ lỗi (Error Rate) của từng mô hình sinh.
