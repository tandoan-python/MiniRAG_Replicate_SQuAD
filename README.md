# Thực nghiệm Replicate MiniRAG

Dự án này thực hiện tái hiện và đánh giá hệ thống MiniRAG (được phát triển bởi HKUDS) dựa trên mã nguồn mở. Quá trình thực nghiệm được thiết kế để kiểm tra khả năng lưu trữ, lập chỉ mục đồ thị (graph indexing) và truy vấn của MiniRAG bằng cách sử dụng các mô hình ngôn ngữ lớn (LLM) và mô hình nhúng (Embedding) trên tập dữ liệu chuẩn.

## 1. Môi trường và Thư viện
Mã nguồn thực nghiệm được thiết kế để chạy trên môi trường Jupyter Notebook (cụ thể tương thích với Kaggle/Colab).
Các thư viện phụ thuộc chính bao gồm:
- `transformers`, `accelerate`, `bitsandbytes` (Hỗ trợ chạy LLM)
- `sentence-transformers` (Mô hình nhúng - Embedding)
- `nano-vectordb` (Cơ sở dữ liệu vector cục bộ)
- `datasets`, `pandas`, `tqdm` (Xử lý dữ liệu)

## 2. Tập dữ liệu (Dataset)
- **Tên tập dữ liệu:** SQuAD v2 (`rajpurkar/squad_v2`) - tập `validation`.
- **Đặc tả dữ liệu:** Dữ liệu được trích xuất ngữ cảnh (context), câu hỏi (question) và câu trả lời tham chiếu (gold answer).
- **Quy mô thực nghiệm:** Tập kiểm thử được giới hạn ở 100 mẫu dữ liệu (`MAX_TEST_SAMPLES = 100`) nhằm phục vụ mục đích kiểm chứng tính khả thi của hệ thống trong thời gian giới hạn. Các đoạn văn bản (context) được lưu dưới định dạng `.txt` để hệ thống MiniRAG đọc và lập chỉ mục.

## 3. Các mô hình được sử dụng
Thực nghiệm sử dụng hai họ mô hình ngôn ngữ khác nhau để đối chiếu hiệu năng nội suy và tổng hợp câu trả lời dựa trên Retrieval-Augmented Generation (RAG):
- **Mô hình 1 (GLM):** `THUDM/glm-edge-1.5b-chat`
- **Mô hình 2 (QWEN):** `Qwen/Qwen2.5-3B-Instruct`
- **Mô hình Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (sử dụng với độ dài chuỗi tối đa 1000 token, chiều vector nhúng: 384).

## 4. Quy trình thực nghiệm
Thực nghiệm được chia thành các giai đoạn tuần tự như sau:

### Giai đoạn 0: Lập chỉ mục Vector (Vector Indexing)
- **Tập lệnh:** `Step_0_index.py`
- **Mô tả:** Hệ thống MiniRAG quét toàn bộ các file tài liệu văn bản (`.txt`) từ tập SQuAD đã được chuẩn bị.
- Thông qua hàm `rag.insert()`, văn bản được phân rã thành các đoạn (chunks), trích xuất các thực thể (entities), mối quan hệ (relationships) và lập chỉ mục thành đồ thị kiến thức cục bộ, được lưu bằng `nano-vectordb`. 
- Quá trình này chạy độc lập cho từng mô hình LLM.

### Giai đoạn 1: Truy vấn và Đánh giá (Query & Evaluation)
- **Tập lệnh:** `Step_1_QA.py`
- **Mô tả:** Hệ thống đọc danh sách câu hỏi kiểm thử từ file `query_set.csv`. Đối với mỗi câu hỏi, hệ thống gọi hàm `rag.query()` ở chế độ `mode="mini"` để sinh câu trả lời.
- Kết quả sinh từ hệ thống (minirag_answer) được lưu song song cùng với câu trả lời chuẩn (Gold Answer) vào file báo cáo đánh giá `.csv`.

### Đánh giá hiệu suất
Kết quả đầu ra được kiểm tra tự động thông qua việc so khớp chuỗi (substring matching): Nếu câu trả lời sinh ra chứa nội dung của câu trả lời chuẩn (Gold Answer), mẫu đó được tính là hợp lệ (Correct). Kết quả cuối cùng báo cáo độ chính xác tương đối (Relative Accuracy) cho từng mô hình (GLM và QWEN).

## 5. Cấu trúc mã nguồn thực nghiệm
Thực nghiệm được gộp trong một notebook chính (`cs2202-project-minirag-glm-edge.ipynb`), chứa các ô mã lệnh (cells) sau:
1. **Thiết lập môi trường:** Clone mã nguồn MiniRAG từ kho lưu trữ GitHub gốc, cài đặt các thư viện cần thiết và thiết lập mã nguồn LLM nhúng cơ bản (patch file).
2. **Kịch bản Lập chỉ mục (`Step_0_index.py`):** Viết script xử lý quá trình nạp tài liệu.
3. **Kịch bản Truy vấn (`Step_1_QA.py`):** Viết script xử lý truy vấn QA và lưu log.
4. **Chuẩn bị Dữ liệu:** Tải dataset SQuAD v2 từ HuggingFace và định dạng thành các tệp văn bản.
5. **Thực thi và Đánh giá:** Vòng lặp tự động chạy cả hai quá trình lập chỉ mục và truy vấn cho danh sách các mô hình mục tiêu (GLM, QWEN), đồng thời in ra báo cáo tổng kết độ chính xác cuối cùng.
