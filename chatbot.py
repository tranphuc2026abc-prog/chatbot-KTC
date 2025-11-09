# Chạy bằng lệnh: streamlit run chatbot.py
# ‼️ Yêu cầu cài đặt: 
# pip install groq streamlit
# (Lưu ý: Các thư viện RAG như pypdf, langchain, scikit-learn đã được tắt)

import streamlit as st
from groq import Groq
import os
import glob
import time
# --- RAG ĐÃ TẮT: Vô hiệu hóa các thư viện không cần thiết ---
# from pypdf import PdfReader 
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np 
# --- KẾT THÚC TẮT RAG ---

# --- BƯỚC 1: LẤY API KEY ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Lỗi: Không tìm thấy GROQ_API_KEY. Vui lòng thêm vào Secrets trên Streamlit Cloud.")
    st.stop()
    
# --- BƯỚC 2: THIẾT LẬP VAI TRÒ (SYSTEM_INSTRUCTION) ---
SYSTEM_INSTRUCTION = """
---
BỐI CẢNH VAI TRÒ (ROLE CONTEXT)
---
Bạn là “Chatbook”, một Cố vấn Học tập Tin học AI toàn diện.
Vai trò của bạn được mô phỏng theo một **Giáo viên Tin học dạy giỏi cấp Quốc gia**: tận tâm, hiểu biết sâu rộng, và luôn kiên nhẫn.
Mục tiêu của bạn là đồng hành, hỗ trợ học sinh THCS và THPT (từ lớp 6 đến lớp 12) nắm vững kiến thức, phát triển năng lực Tin học theo **Chuẩn chương trình Giáo dục Phổ thông 2018** của Việt Nam.

---
📚 NỀN TẢNG TRI THỨC CỐT LÕI (CORE KNOWLEDGE BASE) - BẮT BUỘC
---
Bạn **PHẢI** nắm vững và sử dụng thành thạo toàn bộ hệ thống kiến thức trong Sách giáo khoa Tin học từ lớp 6 đến lớp 12 của **CẢ BA BỘ SÁCH HIỆN HÀNH**:
1.  **Kết nối tri thức với cuộc sống (KNTT)**
2.  **Cánh Diều (CD)**
3.  **Chân trời sáng tạo (CTST)**

Khi giải thích khái niệm hoặc hướng dẫn kỹ năng, bạn phải ưu tiên cách tiếp cận, thuật ngữ, và ví dụ được trình bày trong các bộ sách này để đảm bảo tính thống nhất và bám sát chương trình, tránh nhầm lẫn.

*** DỮ LIỆU MỤC LỤC CHUYÊN BIỆT (KHẮC PHỤC LỖI) ***
Khi học sinh hỏi về mục lục sách (ví dụ: Tin 12 KNTT), bạn PHẢI cung cấp thông tin sau:

* **Sách Tin học 12 – KẾT NỐI TRI THỨC VỚI CUỘC SỐNG (KNTT)** (ĐÃ CẬP NHẬT) gồm 5 Chủ đề chính:
    1.  **Chủ đề 1:** Máy tính và Xã hội tri thức
    2.  **Chủ đề 2:** Mạng máy tính và Internet
    3.  **Chủ đề 3:** Đạo đức, pháp luật và văn hoá trong môi trường số
    4.  **Chủ đề 4:** Giải quyết vấn đề với sự trợ giúp của máy tính
    5.  **Chủ đề 5:** Hướng nghiệp với Tin học

* **Sách Tin học 12 – CHÂN TRỜI SÁNG TẠO (CTST)** (GIỮ NGUYÊN) gồm các Chủ đề chính:
    1.  **Chủ đề 1:** Máy tính và cộng đồng
    2.  **Chủ đề 2:** Tổ chức và lưu trữ dữ liệu
    3.  **Chủ đề 3:** Đạo đức, pháp luật và văn hóa trong môi trường số
    4.  **Chủ đề 4:** Giải quyết vấn đề với sự hỗ trợ của máy tính
    5.  **Chủ đề 5:** Mạng máy tính và Internet

* **Sách Tin học 12 – CÁNH DIỀU (CD)** (GIỮ NGUYÊN) gồm các Chủ đề chính:
    1.  **Chủ đề 1:** Máy tính và Xã hội
    2.  **Chủ đề 2:** Mạng máy tính và Internet
    3.  **Chủ đề 3:** Thuật toán và Lập trình
    4.  **Chủ đề 4:** Dữ liệu và Hệ thống thông tin
    5.  **Chủ đề 5:** Ứng dụng Tin học
*** KẾT THÚC DỮ LIỆU CHUYÊN BIỆT ***

---
🌟 6 NHIỆM VỤ CỐT LÕI (CORE TASKS)
---
#... (Giữ nguyên các nhiệm vụ từ 1 đến 6) ...

**1. 👨‍🏫 Gia sư Chuyên môn (Specialized Tutor):**
    - Giải thích các khái niệm (ví dụ: thuật toán, mạng máy tính, CSGD, CSDL) một cách trực quan, sư phạm, sử dụng ví dụ gần gũi với lứa tuổi học sinh.
    - Luôn kết nối lý thuyết với thực tiễn, giúp học sinh thấy được "học cái này để làm gì?".
    - Bám sát nội dung Sách giáo khoa (KNTT, CD, CTST) và yêu cầu cần đạt của Ctr 2018.
#... (Giữ nguyên các nhiệm vụ còn lại) ...
#... (Giữ nguyên phần QUY TẮC ỨNG XỬ & PHONG CÁCH) ...
#... (Giữ nguyên phần XỬ LÝ THÔNG TIN TRA CỨU) ...
#... (Bỏ qua RAG) ...
#... (Giữ nguyên phần LỚP TƯ DUY PHẢN BIỆN AI) ...
#... (Giữ nguyên phần MỤC TIÊU CUỐI CÙNG) ...
"""

# --- BƯỚC 3: KHỞI TẠO CLIENT VÀ CHỌN MÔ HÌNH ---
try:
    client = Groq(api_key=api_key) 
except Exception as e:
    st.error(f"Lỗi khi cấu hình API Groq: {e}")
    st.stop()

MODEL_NAME = 'llama-3.1-405b-instruct'
# PDF_DIR = "./PDF_KNOWLEDGE" # <-- RAG ĐÃ TẮT

# --- BƯỚC 4: CẤU HÌNH TRANG VÀ CSS ---
st.set_page_config(page_title="Chatbot Tin học 2018", page_icon="✨", layout="centered")
st.markdown("""
<style>
    /* ... (Toàn bộ CSS của thầy giữ nguyên) ... */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    [data-testid="stSidebar"] {
        background-color: #f8f9fa; border-right: 1px solid #e6e6e6;
    }
    .main .block-container { 
        max-width: 850px; padding-top: 2rem; padding-bottom: 5rem;
    }
    .welcome-message { font-size: 1.1em; color: #333; }
</style>
""", unsafe_allow_html=True)


# --- BƯỚC 4.5: THANH BÊN (SIDEBAR) ---
with st.sidebar:
    st.title("🤖 Chatbot KTC")
    st.markdown("---")
    
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = []
        # --- RAG ĐÃ TẮT (Không cần xóa cache RAG) ---
        # st.session_state.pop("rag_system", None) 
        # st.cache_resource.clear() 
        st.rerun()

    st.markdown("---")
    st.markdown(
        "Giáo viên hướng dẫn:\n"
        "**Thầy Nguyễn Thế Khanh** (GV Tin học)\n\n"
        "Học sinh thực hiện:\n"
        "*(Bùi Tá Tùng)*\n"
        "*(Cao Sỹ Bảo Chung)*"
    )
    st.markdown("---")
    st.caption(f"Model: {MODEL_NAME}")


# --- BƯỚC 4.6: CÁC HÀM RAG (ĐÃ TẮT) --- #
# <-- TOÀN BỘ CÁC HÀM RAG ĐÃ BỊ VÔ HIỆU HÓA ---

# @st.cache_resource(ttl=3600) 
# def initialize_rag_system(pdf_directory):
#     """
#     (Toàn bộ hàm RAG đã được tắt)
#     """
#     print("--- HÀM initialize_rag_system (ĐÃ TẮT) ---")
#     return None, None, None
# 
# def find_relevant_knowledge(query, vectorizer, tfidf_matrix, all_chunks, num_chunks=3):
#     """
#     (Toàn bộ hàm RAG đã được tắt)
#     """
#     print("--- HÀM find_relevant_knowledge (ĐÃ TẮT) ---")
#     return None

# --- KẾT THÚC VÔ HIỆU HÓA RAG ---


# --- BƯỚC 5: KHỞI TẠO LỊCH SỬ CHAT VÀ "SỔ TAY" PDF --- #
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- RAG ĐÃ BỊ TẮT --- # 
# (Toàn bộ logic khởi tạo RAG, đọc file, lập chỉ mục đã bị vô hiệu hóa)

# 1. Bỏ qua việc tải PDF và khởi tạo RAG
# if "rag_system" not in st.session_state:
#     with st.spinner("Đang khởi tạo và lập chỉ mục 'sổ tay' PDF (RAG)..."):
#         # Hàm này trả về (vectorizer, tfidf_matrix, all_chunks)
#         rag_components = initialize_rag_system(PDF_DIR)
#         # Lưu cả 3 vào một biến session state
#         st.session_state.rag_system = rag_components
        
# 2. Ép RAG về trạng thái "không hoạt động"
vectorizer, tfidf_matrix, all_chunks = None, None, None
print("RAG (Đọc sổ tay PDF) đã được tắt. Chatbot sẽ trả lời bằng kiến thức nền.")
# --- KẾT THÚC VÔ HIỆU HÓA RAG ---


# --- BƯỚC 6: HIỂN THỊ LỊCH SỬ CHAT ---
for message in st.session_state.messages:
    avatar = "✨" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- BƯỚC 7: MÀN HÌNH CHÀO MỪNG VÀ GỢI Ý ---
logo_path = "LOGO.jpg" 
col1, col2 = st.columns([1, 5])
with col1:
    try:
        st.image(logo_path, width=80)
    except Exception as e:
        st.error(f"Lỗi: Không tìm thấy file logo tên là '{logo_path}'. Vui lòng kiểm tra lại tên file trên GitHub.")
        st.stop()
with col2:
    st.title("KTC. Chatbot hỗ trợ môn Tin Học")

def set_prompt_from_suggestion(text):
    st.session_state.prompt_from_button = text

if not st.session_state.messages:
    st.markdown(f"<div class='welcome-message'>Xin chào! Thầy/em cần hỗ trợ gì về môn Tin học (Chương trình 2018)?</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # ... (Toàn bộ các nút bấm gợi ý của thầy giữ nguyên) ...
    col1_btn, col2_btn = st.columns(2)
    with col1_btn:
        st.button(
            "Giải thích về 'biến' trong lập trình?",
            on_click=set_prompt_from_suggestion, args=("Giải thích về 'biến' trong lập trình?",),
            use_container_width=True
        )
        st.button(
            "Trình bày về an toàn thông tin?",
            on_click=set_prompt_from_suggestion, args=("Trình bày về an toàn thông tin?",),
            use_container_width=True
        )
    with col2_btn:
        st.button(
            "Sự khác nhau giữa RAM và ROM?",
            on_click=set_prompt_from_suggestion, args=("Sự khác nhau giữa RAM và ROM?",),
            use_container_width=True
        )
        st.button(
            "Các bước chèn ảnh vào word",
            on_click=set_prompt_from_suggestion, args=("Các bước chèn ảnh vào word?",),
            use_container_width=True
        )


# --- BƯỚC 8: XỬ LÝ INPUT (RAG ĐÃ BỊ TẮT) --- # 
prompt_from_input = st.chat_input("Mời thầy hoặc các em đặt câu hỏi về Tin học...")
prompt_from_button = st.session_state.pop("prompt_from_button", None)
prompt = prompt_from_button or prompt_from_input

if prompt:
    # 1. Thêm câu hỏi của user vào lịch sử và hiển thị
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Gửi câu hỏi đến Groq
    try:
        with st.chat_message("assistant", avatar="✨"):
            placeholder = st.empty()
            bot_response_text = ""

            # --- LOGIC RAG ĐÃ BỊ VÔ HIỆU HÓA --- #
            
            # 2.1. Lấy các thành phần RAG (luôn là None)
            # (biến vectorizer, tfidf_matrix, all_chunks đã được gán = None ở BƯỚC 5)
            
            # 2.2. Tìm kiếm (sẽ bị bỏ qua vì all_chunks là None)
            retrieved_context = None
            # if all_chunks: 
            #     (code này sẽ không chạy)
            
            # 2.3. Chuẩn bị list tin nhắn gửi cho AI
            messages_to_send = [
                {"role": "system", "content": SYSTEM_INSTRUCTION}
            ]

            # 2.4. Xây dựng prompt
            # Vì retrieved_context luôn là None, code sẽ LUÔN chạy vào nhánh 'else'
            if retrieved_context:
                # (code này sẽ không bao giờ chạy)
                pass 
            else:
                # RAG không tìm thấy gì, hoặc RAG bị tắt
                print("RAG đã tắt. Trả lời bình thường.")
                # Gửi toàn bộ lịch sử chat (hoặc N tin nhắn cuối)
                messages_to_send.extend(st.session_state.messages[-10:]) # Gửi 10 tin nhắn cuối

            # --- KẾT THÚC LOGIC RAG --- #

            # 2.5. Gọi API Groq
            stream = client.chat.completions.create(
                messages=messages_to_send, # Gửi list tin nhắn không có RAG
                model=MODEL_NAME,
                stream=True,
                max_tokens=4096 
            )
            
            # 2.6. Lặp qua từng "mẩu" (chunk) API trả về
            for chunk in stream:
                if chunk.choices[0].delta.content is not None: 
                    bot_response_text += chunk.choices[0].delta.content
                    placeholder.markdown(bot_response_text + "▌")
                    time.sleep(0.005) 
            
            placeholder.markdown(bot_response_text) # Xóa dấu ▌ khi hoàn tất

    except Exception as e:
        with st.chat_message("assistant", avatar="✨"):
            st.error(f"Xin lỗi, đã xảy ra lỗi khi kết nối Groq: {e}")
        bot_response_text = ""

    # 3. Thêm câu trả lời của bot vào lịch sử
    if bot_response_text:
        st.session_state.messages.append({"role": "assistant", "content": bot_response_text})

    # 4. Rerun nếu bấm nút
    if prompt_from_button:
        st.rerun()


