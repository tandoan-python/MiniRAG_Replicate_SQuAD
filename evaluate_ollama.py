import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# Cấu hình hệ thống ghi log để theo dõi quá trình thực nghiệm
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Giữ nguyên bản prompt của tác giả từ Phụ lục trang 21.
# Sử dụng cơ chế nội suy chuỗi (string interpolation) ở phần cuối để truyền dữ liệu.
AUTHOR_ORIGINAL_PROMPT = """Now, I'll give you a question, a gold answer to this question, and three answers provided by different students.
Determine the answer according to the following rules:
If the answer is correct, get 1 point.
If the answer is irrelevant to the question, it will receive 0 points.
If the answer is incorrect, get -1 point.
Return your answer in JSON mode.
For example:
Question: When does Li Hua arrive to the city?
Gold Answer: 20260105
Answer1: LiHua arrived on the afternoon of January 5th
Answer2: Sorry, there is no information about LiHua's arrival in the information you provided
Answer3: There is no accurate answer in the information you provided, but according to the first information found, LiHua arrived on April 17th
output: {"Score1": 1, "Score2": 0, "Score3": -1}
Real data:
Question: {question}
Gold Answer: {ga}
Answer1: {answer1}
Answer2: {answer2}
Answer3: {answer3}
output:"""

def call_ollama_api(prompt: str, model_name: str, api_url: str = "http://localhost:11434/api/generate") -> str:
    """
    Gửi yêu cầu HTTP POST đến API cục bộ của Ollama để thực hiện đánh giá.
    
    Args:
        prompt (str): Chuỗi prompt đã được điền dữ liệu thực tế.
        model_name (str): Tên mô hình LLM đang chạy trên Ollama (ví dụ: 'llama3').
        api_url (str): Địa chỉ endpoint của Ollama.
        
    Returns:
        str: Chuỗi JSON thô được trả về từ mô hình.
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "format": "json",  # Ép buộc định dạng đầu ra là JSON để tối ưu hóa quá trình phân tích (parsing)
        "stream": False,
        "options": {
            "temperature": 0.0  # Đặt nhiệt độ về 0 để đảm bảo tính tất định (deterministic) trong đánh giá
        }
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi giao tiếp với Ollama API: {e}")
        return "{}"

def parse_llm_response(response_text: str) -> int:
    """
    Trích xuất và chuẩn hóa điểm số từ phản hồi của LLM.
    Xử lý các trường hợp nhiễu do LLM mã nguồn mở đôi khi sinh ra JSON không hoàn hảo.
    
    Args:
        response_text (str): Chuỗi phản hồi từ Ollama.
        
    Returns:
        int: Điểm số của model (-1, 0, hoặc 1). Trả về 0 nếu xảy ra lỗi ngoại lệ (fallback).
    """
    try:
        # Sử dụng biểu thức chính quy để trích xuất khối JSON trong trường hợp LLM sinh thêm văn bản giải thích
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
        else:
            data = json.loads(response_text)
            
        # Lấy giá trị Score1 (đại diện cho model cần đánh giá). Mặc định là 0 nếu không tìm thấy.
        score = data.get("Score1", 0)
        
        # Đảm bảo điểm số nằm trong không gian hợp lệ được định nghĩa trong 논문 (paper)
        if score not in [-1, 0, 1]:
            return 0
        return int(score)
        
    except json.JSONDecodeError:
        logging.warning(f"Lỗi phân tích cú pháp JSON từ chuỗi: {response_text}. Chuyển về giá trị mặc định (0).")
        return 0

def evaluate_csv(input_csv: str, target_model: str, output_csv: str) -> Tuple[float, float]:
    """
    Thực thi quy trình đánh giá LLM-as-a-judge trên toàn bộ tập dữ liệu.
    
    Args:
        input_csv (str): Đường dẫn đến file chứa kết quả dự đoán (ví dụ: output_PHI_minirag.csv).
        target_model (str): Tên mô hình giám khảo trên Ollama.
        output_csv (str): Đường dẫn lưu trữ kết quả sau khi chấm điểm.
        
    Returns:
        Tuple[float, float]: Bộ giá trị chứa (Accuracy, Error Rate).
    """
    logging.info(f"Khởi chạy tiến trình đánh giá tệp: {input_csv}")
    
    df = pd.read_csv(input_csv)
    
    # Kiểm tra tính toàn vẹn của cấu trúc dữ liệu đầu vào
    required_columns = ["Question", "Gold Answer", "Model Response"]
    for col in required_columns:
        if col not in df.columns:
            # Linh hoạt ánh xạ cột nếu tên có sự sai lệch nhỏ
            if col == "Model Response" and "Answer" in df.columns:
                df.rename(columns={"Answer": "Model Response"}, inplace=True)
            else:
                raise ValueError(f"Thiếu trường dữ liệu bắt buộc: {col}")

    scores = []
    
    # Sử dụng tqdm để hiển thị tiến trình trong quá trình gọi API tuần tự
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Tiến độ đánh giá"):
        question = str(row["Question"])
        gold_answer = str(row["Gold Answer"])
        model_answer = str(row["Model Response"])
        
        # Xây dựng prompt. Gán "N/A" cho Answer2 và Answer3 để bảo toàn cấu trúc prompt gốc.
        prompt = AUTHOR_ORIGINAL_PROMPT.format(
            question=question,
            ga=gold_answer,
            answer1=model_answer,
            answer2="N/A",
            answer3="N/A"
        )
        
        raw_response = call_ollama_api(prompt, target_model)
        score = parse_llm_response(raw_response)
        scores.append(score)

    df["Judge_Score"] = scores
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    # Tính toán các độ đo thống kê theo phương pháp mô tả trong bài báo
    total_samples = len(df)
    accuracy = (df["Judge_Score"] == 1).sum() / total_samples * 100
    error_rate = (df["Judge_Score"] == -1).sum() / total_samples * 100
    
    logging.info("Hoàn tất đánh giá. Phân phối điểm số:")
    logging.info(f"Correct (1): {(df['Judge_Score'] == 1).sum()}")
    logging.info(f"Irrelevant/Abstain (0): {(df['Judge_Score'] == 0).sum()}")
    logging.info(f"Incorrect (-1): {(df['Judge_Score'] == -1).sum()}")
    
    return accuracy, error_rate

def main():
    parser = argparse.ArgumentParser(description="Công cụ đánh giá độc lập mô hình MiniRAG sử dụng Ollama LLM-as-a-judge.")
    parser.add_argument("--input", type=str, required=True, help="Đường dẫn đến tệp CSV kết quả cần đánh giá.")
    parser.add_argument("--output", type=str, required=True, help="Đường dẫn lưu tệp CSV kết quả sau khi chấm điểm.")
    parser.add_argument("--judge_model", type=str, default="llama3", help="Tên mô hình triển khai trên Ollama (mặc định: llama3).")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Không tìm thấy tệp đầu vào tại: {input_path}")
        return
        
    accuracy, error_rate = evaluate_csv(str(input_path), args.judge_model, args.output)
    
    logging.info("="*50)
    logging.info(f"BÁO CÁO KẾT QUẢ ĐÁNH GIÁ - FILE: {input_path.name}")
    logging.info(f"Mô hình giám khảo (Ollama): {args.judge_model}")
    logging.info(f"Độ chính xác (Accuracy): {accuracy:.2f}%")
    logging.info(f"Tỷ lệ lỗi (Error Rate): {error_rate:.2f}%")
    logging.info("="*50)

if __name__ == "__main__":
    main()