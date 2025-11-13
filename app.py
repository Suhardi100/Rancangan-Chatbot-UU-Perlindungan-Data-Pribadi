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
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key="AIzaSyD7o4ouxDE0xb1Iz3ez9oOPb7Qcwyfqg3I"
)

# ================================
# 🧰 Tools
# ================================
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="id"))
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
tavily_tool_instance = TavilySearchResults(k=3)

tools = {
    "Wikipedia": Tool(
        name="Wikipedia",
        func=wikipedia_tool.run,
        description="Gunakan untuk konsep umum terkait UU PDP"
    ),
    "arXiv": Tool(
        name="arXiv",
        func=arxiv_tool.run,
        description="Gunakan untuk referensi akademik hukum data pribadi"
    ),
    "TavilySearch": Tool(
        name="TavilySearch",
        func=tavily_tool_instance.run,
        description="Gunakan untuk berita hukum terkini"
    )
}

# ================================
# 📚 Load Dokumen UU PDP
# ================================
loader = TextLoader("uu_pdp.txt", encoding='utf-8')
documents = loader.load()
uu_text = documents[0].page_content.lower()

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
    Tentukan tools untuk menjawab pertanyaan berikut:
    "{q}"

    Jika pertanyaan menyinggung isi pasal, ayat, atau isi langsung dari UU PDP,
    maka gunakan hanya dokumen lokal ("Documents UU PDP").
    Jika pertanyaan bersifat konseptual atau umum, boleh gunakan Wikipedia/arXiv/Tavily.

    Format:
    TOOLS: nama_tool1,nama_tool2
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
# 🔍 Node: Multi Source Retrieval
# ================================
@traceable
def multi_source_retrieve_node(state: AgentState) -> AgentState:
    q = state["question"].lower()
    selected = state.get("selected_tools", [])
    internal_docs, external_docs = [], []

    # ===== Cari isi pasal langsung dari dokumen =====
    import re
    pasal_pattern = re.search(r'pasal\s*(\d+)', q)
    if pasal_pattern:
        nomor_pasal = pasal_pattern.group(1)
        pattern = rf"pasal\s*{nomor_pasal}.*?(?=pasal\s*\d+|$)"
        hasil = re.search(pattern, uu_text, re.DOTALL)
        if hasil:
            internal_docs.append(hasil.group(0).strip())
        else:
            internal_docs.append("Pasal tidak ditemukan dalam dokumen UU PDP.")
    else:
        # Jika tidak menyebut pasal, cari berdasarkan kata kunci
        kalimat_terkait = [line for line in uu_text.splitlines() if any(word in line for word in q.split())]
        if kalimat_terkait:
            internal_docs.append("\n".join(kalimat_terkait[:10]))
        else:
            internal_docs.append("Tidak ditemukan bagian terkait dalam dokumen.")

    # ===== Tambahan (opsional) dari tools eksternal =====
    for tool_name in selected:
        if tool_name in tools:
            try:
                external_docs.append(tools[tool_name].run(q))
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
    Apakah dokumen berikut relevan menjawab pertanyaan '{q}'?
    Dokumen: {all_docs}
    Jawab dengan ya/tidak.
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

    # Jika sudah ada isi pasal dari dokumen, tampilkan langsung tanpa mengubah
    if "pasal" in q.lower() and "tidak ditemukan" not in context.lower():
        return {**state, "answer": f"📜 Berdasarkan dokumen UU Perlindungan Data Pribadi:\n\n{context.strip()}"}

    prompt = f"""
    Pertanyaan: {q}
    Konteks: {context}

    Jawablah dengan memprioritaskan isi dokumen UU PDP.
    Jika tidak ditemukan di dokumen, baru tambahkan sumber lain.
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
    prompt = f"Apakah jawaban ini sudah menjawab pertanyaan '{q}'?\n{ans}\nBalas hanya 'ya' atau 'tidak'."
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
