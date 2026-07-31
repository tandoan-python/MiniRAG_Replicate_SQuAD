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

### 1. Môi trường và Thư viện
Mã nguồn thực nghiệm được thiết kế để chạy trên môi trường Jupyter Notebook (cụ thể tương thích với Kaggle/Colab).
Các thư viện phụ thuộc chính bao gồm:
- `transformers`, `accelerate`, `bitsandbytes` (Hỗ trợ chạy LLM)
- `sentence-transformers` (Mô hình nhúng - Embedding)
- `nano-vectordb` (Cơ sở dữ liệu vector cục bộ)
- `datasets`, `pandas`, `tqdm` (Xử lý dữ liệu)

### 2. Tập dữ liệu (Dataset)
- **Tên tập dữ liệu:** SQuAD v2 (`rajpurkar/squad_v2`) - tập `validation`.
- **Đặc tả dữ liệu:** Dữ liệu được trích xuất ngữ cảnh (context), câu hỏi (question) và câu trả lời tham chiếu (gold answer).
- **Quy mô thực nghiệm:** Tập kiểm thử được giới hạn ở 100 mẫu dữ liệu (`MAX_TEST_SAMPLES = 100`) nhằm phục vụ mục đích kiểm chứng tính khả thi của hệ thống trong thời gian giới hạn. Các đoạn văn bản (context) được lưu dưới định dạng `.txt` để hệ thống MiniRAG đọc và lập chỉ mục.

### 3. Các mô hình được sử dụng
Thực nghiệm sử dụng hai họ mô hình ngôn ngữ khác nhau để đối chiếu hiệu năng nội suy và tổng hợp câu trả lời dựa trên Retrieval-Augmented Generation (RAG):
- **Mô hình 1 (GLM):** `THUDM/glm-edge-1.5b-chat`
- **Mô hình 2 (QWEN):** `Qwen/Qwen2.5-3B-Instruct`
- **Mô hình Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (sử dụng với độ dài chuỗi tối đa 1000 token, chiều vector nhúng: 384).

### 4. Quy trình thực nghiệm
Thực nghiệm được chia thành các giai đoạn tuần tự như sau:

#### Giai đoạn 0: Lập chỉ mục Vector (Vector Indexing)
- **Tập lệnh:** `Step_0_index.py`
- **Mô tả:** Hệ thống MiniRAG quét toàn bộ các file tài liệu văn bản (`.txt`) từ tập SQuAD đã được chuẩn bị.
- Thông qua hàm `rag.insert()`, văn bản được phân rã thành các đoạn (chunks), trích xuất các thực thể (entities), mối quan hệ (relationships) và lập chỉ mục thành đồ thị kiến thức cục bộ, được lưu bằng `nano-vectordb`. 
- Quá trình này chạy độc lập cho từng mô hình LLM.

#### Giai đoạn 1: Truy vấn và Đánh giá (Query & Evaluation)
- **Tập lệnh:** `Step_1_QA.py`
- **Mô tả:** Hệ thống đọc danh sách câu hỏi kiểm thử từ file `query_set.csv`. Đối với mỗi câu hỏi, hệ thống gọi hàm `rag.query()` ở chế độ `mode="mini"` để sinh câu trả lời.
- Kết quả sinh từ hệ thống (minirag_answer) được lưu song song cùng với câu trả lời chuẩn (Gold Answer) vào file báo cáo đánh giá `.csv`.

#### Đánh giá hiệu suất
Kết quả đầu ra được kiểm tra tự động thông qua việc so khớp chuỗi (substring matching): Nếu câu trả lời sinh ra chứa nội dung của câu trả lời chuẩn (Gold Answer), mẫu đó được tính là hợp lệ (Correct). Kết quả cuối cùng báo cáo độ chính xác tương đối (Relative Accuracy) cho từng mô hình (GLM và QWEN).

#### 5. Cấu trúc mã nguồn thực nghiệm
Thực nghiệm được gộp trong một notebook chính (`cs2202-project-minirag-glm-edge.ipynb`), chứa các ô mã lệnh (cells) sau:
1. **Thiết lập môi trường:** Clone mã nguồn MiniRAG từ kho lưu trữ GitHub gốc, cài đặt các thư viện cần thiết và thiết lập mã nguồn LLM nhúng cơ bản (patch file).
2. **Kịch bản Lập chỉ mục (`Step_0_index.py`):** Viết script xử lý quá trình nạp tài liệu.
3. **Kịch bản Truy vấn (`Step_1_QA.py`):** Viết script xử lý truy vấn QA và lưu log.
4. **Chuẩn bị Dữ liệu:** Tải dataset SQuAD v2 từ HuggingFace và định dạng thành các tệp văn bản.
5. **Thực thi và Đánh giá:** Vòng lặp tự động chạy cả hai quá trình lập chỉ mục và truy vấn cho danh sách các mô hình mục tiêu (GLM, QWEN), đồng thời in ra báo cáo tổng kết độ chính xác cuối cùng.


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
