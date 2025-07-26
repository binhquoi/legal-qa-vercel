# app.py - Mã nguồn cho ứng dụng web (Phiên bản ổn định hơn)
import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_astradb import AstraDBVectorStore
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Lấy các biến môi trường đã thiết lập trên Vercel
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

# --- Hàm khởi tạo và kiểm tra ---
@st.cache_resource
def get_vector_store():
    """Kết nối và lấy vector store từ Astra DB một cách an toàn."""
    if not all([GOOGLE_API_KEY, ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN]):
        st.error("Lỗi: Vui lòng thiết lập đầy đủ các biến môi trường (API keys) trong phần Settings > Environment Variables trên Vercel.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
        vector_store = AstraDBVectorStore(
            embedding=embeddings,
            collection_name="legal_documents",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
        )
        return vector_store
    except Exception as e:
        st.error(f"Lỗi khi kết nối tới cơ sở dữ liệu vector: {e}")
        return None

@st.cache_resource
def get_conversational_chain():
    """Tạo chuỗi xử lý câu hỏi."""
    prompt_template = """
    Dựa vào ngữ cảnh được cung cấp dưới đây, hãy trả lời câu hỏi một cách chi tiết và chính xác bằng tiếng Việt.
    Nếu câu trả lời không có trong ngữ cảnh, hãy nói "Tôi không tìm thấy thông tin trả lời trong tài liệu được cung cấp".
    
    Ngữ cảnh:\n {context}\n
    Câu hỏi: \n{question}\n
    Câu trả lời:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo mô hình AI: {e}")
        return None

# --- Logic chính ---
def user_input(user_question):
    """Xử lý câu hỏi của người dùng và trả về câu trả lời."""
    vector_store = get_vector_store()
    if vector_store is None:
        return

    chain = get_conversational_chain()
    if chain is None:
        return

    try:
        with st.spinner("AI đang tìm kiếm tài liệu liên quan..."):
            docs = vector_store.similarity_search(user_question, k=5)
        
        with st.spinner("AI đang tổng hợp câu trả lời..."):
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        st.write("### Câu trả lời:", response["output_text"])
    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình xử lý câu hỏi: {e}")

# --- Giao diện Streamlit ---
st.set_page_config(page_title="Hỏi Đáp Pháp Luật", page_icon="⚖️")
st.header("⚖️ Trợ lý Hỏi Đáp Pháp Luật AI")

user_question = st.text_input("Hãy đặt câu hỏi của bạn về các văn bản pháp luật đã cung cấp:", key="user_question")

if st.button("Hỏi đáp", key="ask_button"):
    if user_question:
        user_input(user_question)
    else:
        st.warning("Vui lòng nhập câu hỏi của bạn.")

st.sidebar.title("Về ứng dụng")
st.sidebar.info(
    "Ứng dụng này sử dụng Google Gemini và Astra DB để trả lời câu hỏi dựa trên kho tài liệu pháp luật của bạn. "
    "Triển khai bởi Vercel."
)
