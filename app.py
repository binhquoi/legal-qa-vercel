# app.py - Phiên bản tối ưu nhất, không dùng LangChain chains
import os
import sys
import json
import streamlit as st
import google.generativeai as genai
from astrapy.db import AstraDB
import numpy as np

# --- Cấu hình và Khởi tạo ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_COLLECTION_NAME = "legal_documents"

# Hàm để khởi tạo các kết nối một cách an toàn
def initialize_services():
    """Khởi tạo và trả về các đối tượng kết nối tới Google và AstraDB."""
    if not all([GOOGLE_API_KEY, ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN]):
        st.error("Lỗi: Vui lòng thiết lập đầy đủ các biến môi trường trên Vercel.")
        return None, None
    try:
        # Cấu hình Google AI
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Kết nối tới AstraDB
        astra_db = AstraDB(
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN
        )
        collection = astra_db.collection(ASTRA_DB_COLLECTION_NAME)
        return genai, collection
    except Exception as e:
        print(f"ERROR_INITIALIZE: {e}", file=sys.stderr)
        st.error(f"Lỗi khi khởi tạo dịch vụ. Vui lòng kiểm tra lại API keys.")
        return None, None

# --- Logic xử lý chính ---
def user_input(user_question, genai_service, db_collection):
    """Xử lý câu hỏi của người dùng."""
    if not all([genai_service, db_collection]):
        return

    try:
        # 1. Tạo embedding cho câu hỏi của người dùng
        with st.spinner("Đang phân tích câu hỏi..."):
            question_embedding_result = genai.embed_content(
                model="models/text-embedding-004",
                content=user_question,
                task_type="RETRIEVAL_QUERY"
            )
            question_vector = question_embedding_result['embedding']

        # 2. Tìm kiếm các tài liệu liên quan trong AstraDB
        with st.spinner("Đang tìm kiếm tài liệu liên quan..."):
            # Sử dụng vector_find để tìm kiếm tương đồng
            relevant_docs = db_collection.vector_find(
                vector=question_vector,
                limit=5,
                fields={"text", "_id"} # Chỉ lấy trường text và id
            )
            context = "\n\n".join([doc['text'] for doc in relevant_docs])

        # 3. Tạo prompt và gọi LLM để có câu trả lời
        prompt_template = f"""Dựa vào ngữ cảnh được cung cấp dưới đây, hãy trả lời câu hỏi một cách chi tiết và chính xác bằng tiếng Việt.
Nếu câu trả lời không có trong ngữ cảnh, hãy nói "Tôi không tìm thấy thông tin trả lời trong tài liệu được cung cấp".

Ngữ cảnh:
{context}

Câu hỏi:
{user_question}

Câu trả lời:"""
        
        with st.spinner("AI đang tổng hợp câu trả lời..."):
            llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = llm_model.generate_content(prompt_template)
            st.write("### Câu trả lời:", response.text)

    except Exception as e:
        print(f"ERROR_PROCESS_QUERY: {e}", file=sys.stderr)
        st.error(f"Đã xảy ra lỗi trong quá trình xử lý câu hỏi. Vui lòng thử lại.")

# --- Giao diện Streamlit ---
st.set_page_config(page_title="Hỏi Đáp Pháp Luật", page_icon="⚖️")
st.header("⚖️ Trợ lý Hỏi Đáp Pháp Luật AI")

# Khởi tạo dịch vụ
genai_service, db_collection = initialize_services()

user_question = st.text_input("Hãy đặt câu hỏi của bạn về các văn bản pháp luật đã cung cấp:", key="user_question")

if st.button("Hỏi đáp", key="ask_button"):
    if user_question and genai_service and db_collection:
        user_input(user_question, genai_service, db_collection)
    elif not user_question:
        st.warning("Vui lòng nhập câu hỏi của bạn.")

st.sidebar.title("Về ứng dụng")
st.sidebar.info(
    "Ứng dụng này sử dụng Google Gemini và Astra DB để trả lời câu hỏi dựa trên kho tài liệu pháp luật của bạn. "
    "Triển khai bởi Vercel."
)
```

### Các bước bạn cần làm:

1.  **Cập nhật file `requirements.txt`:** Thay thế toàn bộ nội dung file cũ bằng nội dung trong Canvas `requirements.txt (Siêu nhẹ)`.
2.  **Cập nhật file `app.py`:** Thay thế toàn bộ nội dung file cũ bằng nội dung trong Canvas `app.py (Tối ưu không dùng LangChain)`.
3.  **Kiểm tra file `.vercelignore`:** Đảm bảo file này vẫn tồn tại và có nội dung đúng để bỏ qua thư mục `data/`.
4.  **Lưu, Commit và Đẩy lên GitHub:**
    ```bash
    git add .
    git commit -m "Refactor: Remove LangChain to fix bundle size error"
    git push origin main
    ```

Phương pháp này là sự thay đổi triệt để nhất, trực tiếp giải quyết vấn đề dung lượng. Tôi tin rằng đây sẽ là lần cuối cùng bạn cần sửa lỗi n