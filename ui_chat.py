import streamlit as st
import datetime
import pytz
import app  # memanggil logika utama dari app.py

# ================================
# 🌿 KONFIGURASI HALAMAN
# ================================
st.set_page_config(
    page_title="Chatbot Pintar UU Perlindungan Data Pribadi (PDP)",
    page_icon="Suhardi",
    layout="wide",
)

# ================================
# 🌿 CSS PROFESIONAL — GAYA CHATGPT (tema hijau-emas)
# ================================
st.markdown("""
<style>
/* Umum */
.stApp {
    background: linear-gradient(135deg, #0b3d2e 0%, #198754 100%);
    color: #f8f9fa !important;
    font-family: "Segoe UI", sans-serif;
}

/* Header */
h2 {
    text-align: center;
    font-weight: 700;
    color: #d1f7c4 !important;
    margin-bottom: 5px;
}
p.subtitle {
    text-align: center;
    font-size: 15px;
    color: #e8ffe0;
    margin-bottom: 25px;
}

/* Chat Container */
.chat-container {
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 20px;
    max-height: 70vh;
    overflow-y: auto;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* Chat Bubbles */
.chat-bubble-user {
    background: #d1e7dd;
    color: #0f5132;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin-bottom: 12px;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.chat-bubble-assistant {
    background: #f8f9fa;
    color: #0f5132;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin-bottom: 12px;
    max-width: 80%;
    margin-right: auto;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #fff176 0%, #fbc02d 100%);
    color: #1b4332;
}
.sidebar-header {
    text-align: center;
    font-weight: bold;
    font-size: 18px;
    margin-bottom: 15px;
}
.sidebar-item {
    background-color: rgba(255,255,255,0.5);
    padding: 8px 10px;
    border-radius: 10px;
    margin-bottom: 6px;
    color: #0f5132;
    font-size: 14px;
    transition: 0.2s;
}
.sidebar-item:hover {
    background-color: rgba(255,255,255,0.8);
    cursor: pointer;
}

/* 🔴 Semua tombol di sidebar */
[data-testid="stSidebar"] button {
    background-color: #ffffff !important; /* putih */
    color: #c1121f !important; /* teks merah */
    font-weight: bold !important;
    border-radius: 10px !important;
    border: 2px solid #c1121f !important; /* garis merah */
    transition: all 0.3s ease !important;
}
[data-testid="stSidebar"] button:hover {
    background-color: #c1121f !important; /* latar merah saat hover */
    color: #ffffff !important; /* teks putih saat hover */
    border: 2px solid #c1121f !important;
}
</style>
""", unsafe_allow_html=True)

# ================================
# 🌿 INISIALISASI SESSION STATE
# ================================
for key in ["messages", "chat_history", "pending_prompt"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "pending_prompt" else None

if "viewing_history_index" not in st.session_state:
    st.session_state.viewing_history_index = None

# ================================
# 🌿 SIDEBAR — RIWAYAT CHAT
# ================================
with st.sidebar:
    if st.button("MULAI CHAT BARU", use_container_width=True):
        if st.session_state.messages and st.session_state.viewing_history_index is None:
            st.session_state.chat_history.append(st.session_state.messages.copy())
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.session_state.viewing_history_index = None
        st.rerun()

    st.markdown("<div class='sidebar-header'>RIWAYAT PERTANYAAN ANDA:</div>", unsafe_allow_html=True)

    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
            first_msg = next((m["text"] for m in chat if m["role"] == "user"), "(tanpa isi)")
            short_preview = (first_msg[:60] + "...") if len(first_msg) > 60 else first_msg
            if st.button(f"{short_preview}", use_container_width=True, key=f"hist_{i}"):
                st.session_state.messages = chat.copy()
                st.session_state.viewing_history_index = len(st.session_state.chat_history) - i
                st.session_state.pending_prompt = None
                st.rerun()
    else:
        st.info('Belum ada riwayat pertanyaan tersimpan. Riwayat hanya akan tersimpan setelah Anda menekan tombol "MULAI CHAT BARU".')

# ================================
# 🌿 HEADER
# ================================
st.markdown("""
<h2>🤖 Chatbot UU Perlindungan Data Pribadi (Agentic RAG with Langchain)</h2>
<p class='subtitle'>
Tanyakan apa pun seputar <b>UU No. 27 Tahun 2022</b> tentang Perlindungan Data Pribadi (PDP).<br>
Chatbot ini dibuat oleh <b>SUHARDI</b> untuk memenuhi tugas akhir pembelajaran model bahasa besar (large language model).
</p>
""", unsafe_allow_html=True)

# ================================
# 💬 AREA CHAT
# ================================
chat_box = st.container()
with chat_box:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role, text, time = msg["role"], msg["text"], msg["time"]
        if role == "user":
            st.markdown(f"<div class='chat-bubble-user'><b>Anda ({time})</b><br>{text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-assistant'><b>Jawaban Chatbot Suhardi ({time})</b><br>{text}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ================================
# 💬 INPUT CHAT
# ================================
if st.session_state.viewing_history_index is None:
    prompt = st.chat_input("💬 Tuliskan pertanyaan Anda seputar UU Perlindungan Data Pribadi (PDP) No. 27 Tahun 2022 pada kolom ini...")
    if prompt:
        tz = pytz.timezone("Asia/Jakarta")
        current_time = datetime.datetime.now(tz).strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "text": prompt,
            "time": current_time
        })
        st.session_state.pending_prompt = prompt
        st.rerun()

    if st.session_state.pending_prompt:
        with st.spinner("🔍 Sedang menganalisis dengan Agentic RAG..."):
            try:
                state = {"question": st.session_state.pending_prompt}
                result = app.runnable_graph.invoke(state)
                answer = result.get("answer", "Tidak ada jawaban ditemukan.")
                reasoning = result.get("reasoning", "")
                response_text = f"{answer}\n\n🧠 <b>Analisis Tools:</b> {reasoning}"
            except Exception as e:
                response_text = f"⚠️ Terjadi kesalahan: {e}"

        tz = pytz.timezone("Asia/Jakarta")
        current_time = datetime.datetime.now(tz).strftime("%H:%M:%S")

        st.session_state.messages.append({
            "role": "Asisten",
            "text": response_text,
            "time": current_time
        })

        st.session_state.pending_prompt = None
        st.rerun()
else:
    st.info("🔒 Anda sedang melihat riwayat chat lama. Klik 'MULAI CHAT BARU' untuk memulai percakapan baru.")
