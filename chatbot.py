# Chạy bằng lệnh: streamlit run web_chatboot.py
# ‼️ Yêu cầu cài đặt: pip install google-generativeai streamlit

import streamlit as st
import google.generativeai as genai  # ĐỔI 1: Import thư viện Gemini
import os
import time # Thêm thư viện time để xử lý lỗi API (nếu cần)

# --- BƯỚC 1: LẤY API KEY (Đổi sang Gemini) ---
# Lấy key từ https://aistudio.google.com/app/apikey
# ‼️ QUAN TRỌNG: Thầy cần tạo 1 biến môi trường tên là "GOOGLE_API_KEY"
# (Thầy có thể dùng file .env hoặc cài đặt bí mật (secret) của Streamlit)

try:
    # ĐỔI 2: Lấy GOOGLE_API_KEY từ biến môi trường
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
except KeyError:
    st.error("Lỗi: Không tìm thấy GOOGLE_API_KEY.")
    st.info("Vui lòng thiết lập biến môi trường GOOGLE_API_KEY (hoặc Streamlit Secret) để sử dụng Gemini API.")
    st.stop()


# --- BƯỚC 2: THIẾT LẬP VAI TRÒ (SYSTEM INSTRUCTION) ---
# (Giữ nguyên, Gemini hỗ trợ tốt)
SYSTEM_INSTRUCTION = (
    "Bạn là 'Chatbook' - một Cố vấn Học tập Tin học AI toàn diện, với kiến thức cốt lõi của một "
    "giáo viên Tin học dạy giỏi cấp quốc gia, nắm vững chương trình GDPT 2018. "
    "Nhiệm vụ của bạn là hỗ trợ học sinh THCS/THPT một cách toàn diện. "
    "Bạn có 6 nhiệm vụ chính: "
    
    "1. **Gia sư Chuyên môn (Lý thuyết):** Giải đáp mọi thắc mắc về lý thuyết môn Tin học "
    "(từ lớp 6-12) một cách chính xác, sư phạm, và đi thẳng vào vấn đề. "
    
    "2. **Mentor Lập trình (Thực hành Code):** Cung cấp code mẫu (Python, C++, Pascal), "
    "hỗ trợ gỡ lỗi (debug), tối ưu hóa code, và giải thích các thuật toán "
    "hoặc cấu trúc dữ liệu một cách trực quan, dễ hiểu. "
    
    "3. **Người hướng dẫn Dự án (Sáng tạo):** Giúp học sinh lên ý tưởng, xây dựng kế hoạch "
    "cho các dự án học tập (ví dụ: làm bài tập lớn, dự án KHKT). "
    "Chủ động gợi ý các chủ đề liên quan đến bài học. "
    
    "4. **Chuyên gia Tin học Văn phòng (Ứng dụng):** Hướng dẫn chi tiết cách sử dụng "
    "các công cụ Microsoft Office (Word, Excel, PowerPoint) để áp dụng vào bài học, "
    "làm báo cáo, thuyết trình, hay xử lý dữ liệu. "
    
    "5. **Trợ lý Ôn tập (Củng cố):** Khi được yêu cầu, bạn sẽ tạo các câu hỏi trắc nghiệm, "
    "câu hỏi tự luận, tóm tắt bài học, hoặc giải thích các dạng bài tập "
    "để giúp học sinh củng cố kiến thức trước kỳ thi. "
    
    "6. **Cố vấn Định hướng (Tương lai):** Cung cấp thông tin cơ bản về các ngành nghề "
    "trong lĩnh vực CNTT, các kỹ năng quan trọng cần có, và gợi ý lộ trình "
    "học tập để các em có cái nhìn sớm về tương lai. "
    
    "Khi tương tác, hãy luôn giữ giọng văn chuyên nghiệp nhưng thân thiện, "
    "tập trung 100% vào nội dung chương trình 2018 và các ứng dụng thực tế của nó."
)

# --- BƯỚC 3: KHỞI TẠO CLIENT VÀ CHỌN MÔ HÌNH (Đổi sang Gemini) ---

try:
    # ĐỔI 3: Cấu hình API key cho thư viện Gemini
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Lỗi khi cấu hình Google AI (Kiểm tra API Key): {e}")
    st.stop()

# ĐỔI 4: Chọn mô hình Gemini (gemini-1.5-flash là lựa chọn nhanh và tiết kiệm)
MODEL_NAME = 'gemini-1.5-flash-latest' # Hoặc 'gemini-1.5-pro-latest'

# Cấu hình an toàn (Tùy chọn, nhưng nên có để tránh bị chặn)
# Đặt ở mức thấp nhất để ưu tiên System Instruction
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# ĐỔI 5: Khởi tạo mô hình Gemini
try:
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION, # Gán vai trò hệ thống trực tiếp
        safety_settings=safety_settings
    )
except Exception as e:
    st.error(f"Lỗi khi khởi tạo mô hình Gemini ('{MODEL_NAME}'): {e}")
    st.stop()


# --- BƯỚC 4: CẤU HÌNH TRANG VÀ CSS ---
# (Giữ nguyên)
st.set_page_config(page_title="Chatbot Tin học 2018", page_icon="✨", layout="centered")
st.markdown("""
<style>
    /* Ẩn thanh menu của Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Tùy chỉnh thanh bên */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa; /* Màu nền sidebar */
        border-right: 1px solid #e6e6e6;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background-color: #FFFFFF;
        border: 1px solid #e0e0e0;
        color: #333; /* Màu chữ nút */
        border-radius: 8px;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background-color: #f0f0f0;
        border: 1px solid #d0d0d0;
        color: #000;
    }

    /* Khu vực chat chính */
    .main .block-container { 
        max-width: 850px; 
        padding-top: 2rem;
        padding-bottom: 5rem; /* Thêm khoảng đệm ở dưới */
    }
    .stButton>button { border: 1px solid #dfe1e5; }
    [data-testid="chatAvatarIcon-user"] { background-color: #C0C0C0; }
    .welcome-message {
        font-size: 1.1em;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


# --- BƯỚC 4.5: THANH BÊN (SIDEBAR) ---
# (Giữ nguyên)
with st.sidebar:
    st.title("🤖 Chatbot KTC")
    st.markdown("---")
    
    # Nút "Cuộc trò chuyện mới"
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
        # Xóa lịch sử chat hiện tại
        st.session_state.messages = []
        # Tải lại trang (để xóa prompt khỏi nút bấm nếu có)
        st.session_state.pop("prompt_from_button", None) 
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


# --- BƯỚC 5: KHỞI TẠO LỊCH SỬ CHAT ---
# (Giữ nguyên)
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- BƯỚC 6: HIỂN THỊ LỊCH SỬ CHAT ---
# (Giữ nguyên)
for message in st.session_state.messages:
    avatar = "✨" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- BƯỚC 7: MÀN HÌNH CHÀO MỪNG VÀ GỢI Ý ---

# (XÓA BỎ: Dòng 'import streamlit as st' lặp lại đã được xóa)

# --- PHẦN TIÊU ĐỀ VÀ LOGO (LUÔN HIỂN THỊ) ---
# (Giữ nguyên)
logo_path = r"C:\Users\Khanh Nguyen\Documents\logo.jpg"
col1, col2 = st.columns([1, 5]) 

with col1:
    st.image(logo_path, width=80) 

with col2:
    st.title("KCT. Chat boot hỗ trợ môn Tin Học")

# --- HÀM HỖ TRỢ NÚT BẤM (ĐÃ CHUẨN) ---
# (Giữ nguyên)
def set_prompt_from_suggestion(text):
    st.session_state.prompt_from_button = text

# --- HIỂN THỊ MÀN HÌNH CHÀO VÀ GỢI Ý (CHỈ KHI MỚI VÀO) ---
# (Giữ nguyên)
if not st.session_state.messages:
    
    st.markdown(f"<div class='welcome-message'>Xin chào! Thầy/em cần hỗ trợ gì về môn Tin học (Chương trình 2018)?</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1_btn, col2_btn = st.columns(2)
    with col1_btn:
        st.button(
            "Giải thích về 'biến' trong lập trình?",
            on_click=set_prompt_from_suggestion,
            args=("Giải thích về 'biến' trong lập trình?",),
            use_container_width=True
        )
        st.button(
            "Trình bày về an toàn thông tin?",
            on_click=set_prompt_from_suggestion,
            args=("Trình bày về an toàn thông tin?",),
            use_container_width=True
        )
    with col2_btn:
        st.button(
            "Sự khác nhau giữa RAM và ROM?",
            on_click=set_prompt_from_suggestion,
            args=("Sự khác nhau giữa RAM và ROM?",),
            use_container_width=True
        )
        st.button(
            "Tóm tắt bài 6 Tin 12 (KNTT)?",
            on_click=set_prompt_from_suggestion,
            args=("Tóm tắt bài 6 Tin 12 (KNTT)?",),
            use_container_width=True
        )

# --- BƯỚC 8: XỬ LÝ INPUT (Đã sửa lỗi) ---
prompt_from_input = st.chat_input("Mời thầy hoặc các em đặt câu hỏi về Tin học...")
prompt_from_button = st.session_state.pop("prompt_from_button", None)
prompt = prompt_from_button or prompt_from_input

if prompt:
    # 1. Thêm câu hỏi của user vào lịch sử và hiển thị
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Gửi câu hỏi đến Gemini (ĐÂY LÀ PHẦN THAY ĐỔI LỚN NHẤT)
    try:
        with st.chat_message("assistant", avatar="✨"):
            placeholder = st.empty()
            bot_response_text = ""

            # ĐỔI 6: Chuyển đổi lịch sử chat sang định dạng Gemini
            # Gemini dùng "model" thay vì "assistant"
            gemini_history = []
            for msg in st.session_state.messages:
                role = "model" if msg["role"] == "assistant" else "user"
                gemini_history.append({"role": role, "parts