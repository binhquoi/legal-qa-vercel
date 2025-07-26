# setup.py - Chạy file này trên máy tính của bạn MỘT LẦN
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore

# Tải các biến môi trường từ file .env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

def setup_vector_database():
    """Đọc tài liệu, xử lý và đẩy lên Astra DB."""
    if not all([GOOGLE_API_KEY, ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN]):
        print("Vui lòng thiết lập các biến môi trường cần thiết trong file .env")
        return

    print("Bắt đầu xử lý dữ liệu...")
    # 1. Tải tài liệu từ thư mục 'data'
    loader = DirectoryLoader('./data/', glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()

    # 2. Phân mảnh văn bản
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Đã chia thành {len(chunks)} đoạn văn bản.")

    # 3. Khởi tạo model embedding
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

    # 4. Khởi tạo và kết nối tới Astra DB
    vector_store = AstraDBVectorStore(
        embedding=embeddings,
        collection_name="legal_documents", # Tên bộ sưu tập
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
    )
    
    print("Đang thêm tài liệu vào Astra DB...")
    # 5. Thêm các chunks vào Astra DB (quá trình này có thể mất vài phút)
    inserted_ids = vector_store.add_documents(chunks)
    print(f"\nĐã thêm thành công {len(inserted_ids)} tài liệu vào Astra DB.")

if __name__ == "__main__":
    # Trước khi chạy, hãy tạo file .env và điền các key của bạn vào
    # GOOGLE_API_KEY="..."
    # ASTRA_DB_API_ENDPOINT="..."
    # ASTRA_DB_APPLICATION_TOKEN="..."
    setup_vector_database()
