import os
import streamlit as st
from typing import TypedDict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.tools import Tool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langsmith import traceable

# ================================
# 🔧 Konfigurasi Awal
# ================================
os.environ["TAVILY_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__YourLangSmithKeyHere"
os.environ["LANGCHAIN_PROJECT"] = "UU-PDP-AgenticRAG"

# ================================
# 🔮 Setup Google Gemini
# ================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    google_api_key="AIzaSyBI6YdES2PyWC3JU2_eDtTW1ipi6Z07DcE"
)

# ================================
# 🧰 Tools Bahasa Indonesia
# ================================
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="id"))
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
tavily_tool_instance = TavilySearchResults(k=3)

tools = {
    "Wikipedia": Tool(
        name="Wikipedia",
        func=wikipedia_tool.run,
        description="Gunakan untuk menemukan konsep hukum umum, sejarah, dan hal-hal lain yang berkaitan dengan UU Perlindungan Data Pribadi (PDP) dalam Bahasa Indonesia!"
    ),
    "arXiv": Tool(
        name="arXiv",
        func=arxiv_tool.run,
        description="Gunakan untuk referensi akademik tentang teori atau penelitian berkaitan UU Perlindungan Data Pribadi!"
    ),
    "TavilySearch": Tool(
        name="TavilySearch",
        func=tavily_tool_instance.run,
        description="Gunakan untuk berita UU Perlindungan Data Pribadi terbaru, peraturan Indonesia, atau putusan pengadilan!"
    )
}

# ================================
# 📚 Load Dokumen UU PDP
# ================================
loader = TextLoader("uu_pdp.txt", encoding='utf-8')
documents = loader.load()
uu_text = " ".join([d.page_content for d in documents])

# Fungsi untuk mencari isi pasal relevan dari teks UU
def cari_pasal_relevan(pertanyaan: str, max_hasil: int = 3) -> List[str]:
    """
    Mencari pasal-pasal relevan dari dokumen uu_pdp.txt berdasarkan kata kunci dari pertanyaan.
    Mengambil kalimat/pasal yang mengandung kata kunci secara langsung.
    """
    hasil = []
    q_lower = pertanyaan.lower()
    baris = uu_text.split("\n")

    for line in baris:
        if any(kata in line.lower() for kata in q_lower.split()):
            if line.strip() and len(line.strip()) > 20:
                hasil.append(line.strip())

    # Hilangkan duplikat dan batasi hasil
    hasil_unik = list(dict.fromkeys(hasil))
    return hasil_unik[:max_hasil] if hasil_unik else ["(Tidak ditemukan pasal relevan dalam dokumen UU PDP)"]

# ================================
# 🧩 Define Agent State
# ================================
class AgentState(TypedDict):
    question: str
    docs: Optional[List[str]]
    external_docs: Optional[List[str]]
    answer: Optional[str]
    relevant: Optional[bool]
    answered: Optional[bool]
    selected_tools: Optional[List[str]]
    reasoning: Optional[str]

# ================================
# 🧠 Node: Tool Selection
# ================================
@traceable
def tool_selection_node(state: AgentState) -> AgentState:
    q = state["question"]
    prompt = f"""
    Kamu adalah asisten ahli UU Pelindungan Data Pribadi.
    Utamakan menggunakan dokumen UU PDP terlebih dahulu sebelum memakai tools eksternal.

    Pertanyaan: {q}

    Pilih tools yang paling sesuai:
    - Wikipedia → konsep umum hukum
    - arXiv → teori akademik
    - TavilySearch → berita hukum terbaru
    - Documents UU PDP → isi pasal per pasal

    Format jawaban:
    TOOLS: tool1,tool2
    REASONING: alasan
    """
    result = llm.invoke(prompt)
    lines = result.content.strip().split("\n")
    tools_selected, reasoning = [], ""
    for line in lines:
        if line.startswith("TOOLS:"):
            tools_selected = [t.strip() for t in line.replace("TOOLS:", "").split(",")]
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
    return {**state, "selected_tools": tools_selected, "reasoning": reasoning}

# ================================
# 🔍 Node: Multi Source Retrieval (diperbaiki agar ambil dari dokumen)
# ================================
@traceable
def multi_source_retrieve_node(state: AgentState) -> AgentState:
    q = state["question"]
    selected = state.get("selected_tools", [])

    # Ambil isi pasal relevan dari dokumen lokal
    internal_docs = cari_pasal_relevan(q)

    # Jalankan tools eksternal jika diperlukan
    external_docs = []
    for tool_name in selected:
        if tool_name in tools:
            try:
                hasil = tools[tool_name].run(q)
                external_docs.append(hasil)
            except Exception as e:
                external_docs.append(f"{tool_name} gagal: {str(e)}")

    return {**state, "docs": internal_docs, "external_docs": external_docs}

# ================================
# 🧮 Node: Grade Relevance
# ================================
@traceable
def enhanced_grade_node(state: AgentState) -> AgentState:
    q = state["question"]
    all_docs = state.get("docs", []) + state.get("external_docs", [])
    prompt = f"""
    Evaluasi relevansi isi dokumen berikut terhadap pertanyaan ini:

    Pertanyaan: {q}
    Dokumen: {all_docs}

    Apakah dokumen-dokumen ini cukup relevan untuk menjawab pertanyaan pengguna? (ya/tidak)
    """
    res = llm.invoke(prompt)
    return {**state, "relevant": "ya" in res.content.lower()}

# ================================
# 🧩 Node: Generate Final Answer
# ================================
@traceable
def enhanced_generation_node(state: AgentState) -> AgentState:
    q = state["question"]
    context = "\n".join(state.get("docs", []) + state.get("external_docs", []))

    prompt = f"""
    Kamu adalah asisten hukum ahli dalam UU Perlindungan Data Pribadi (PDP).
    Gunakan isi pasal yang ditemukan dari dokumen UU PDP di bawah ini untuk menjawab pertanyaan secara presisi.

    Pertanyaan: {q}
    Isi Dokumen Relevan:
    {context}

    Jawablah dalam Bahasa Indonesia formal dengan menyebutkan pasal atau ayat yang sesuai dari UU PDP.
    Jika tidak ada pasal relevan, berikan penjelasan umum dan tandai dengan "(tidak ditemukan dalam UU PDP)".
    """
    res = llm.invoke(prompt)
    return {**state, "answer": res.content.strip()}

# ================================
# 🔁 Node: Answer Check
# ================================
@traceable
def answer_check_node(state: AgentState) -> AgentState:
    q = state["question"]
    ans = state.get("answer", "")
    prompt = f"Apakah jawaban ini sudah sangat menjawab pertanyaan?\nPertanyaan: {q}\nJawaban: {ans}\nBalas hanya 'ya' atau 'tidak'."
    res = llm.invoke(prompt)
    return {**state, "answered": "ya" in res.content.lower()}

# ================================
# 🔧 Workflow Graph (LangGraph)
# ================================
workflow = StateGraph(AgentState)
workflow.add_node("ToolSelection", tool_selection_node)
workflow.add_node("Retrieve", multi_source_retrieve_node)
workflow.add_node("Grade", enhanced_grade_node)
workflow.add_node("Generate", enhanced_generation_node)
workflow.add_node("Evaluate", answer_check_node)

workflow.set_entry_point("ToolSelection")
workflow.add_edge("ToolSelection", "Retrieve")
workflow.add_edge("Retrieve", "Grade")
workflow.add_edge("Grade", "Generate")
workflow.add_edge("Generate", "Evaluate")
workflow.add_conditional_edges(
    "Evaluate",
    lambda s: "Yes" if s["answered"] else "No",
    {"Yes": END, "No": "Retrieve"}
)

runnable_graph = workflow.compile()
