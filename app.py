import os
import sys
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

# ======================================
# 🧩 Setup Konfigurasi Dasar
# ======================================
sys.setrecursionlimit(100)
os.environ["TAVILY_API_KEY"] = "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"
os.environ["LANGCHAIN_API_KEY"] = "ls__YourLangSmithKeyHere"
os.environ["LANGCHAIN_PROJECT"] = "UU-PDP-AgenticRAG"

# ======================================
# 🔮 Setup Google Gemini
# ======================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,  # dibuat lebih rendah supaya tidak ngelantur
    google_api_key="AIzaSyCWV_230ec-t_xUZ2Vsj1XXJDSx57UaJlA"
)

# ======================================
# 📚 Load Dokumen Lokal UU PDP
# ======================================
loader = TextLoader("uu_pdp.txt", encoding='utf-8')
documents = loader.load()
local_text = "\n".join([d.page_content for d in documents])

# ======================================
# 🧰 Tools Eksternal
# ======================================
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="id"))
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
tavily_tool_instance = TavilySearchResults(k=3)

tools = {
    "Wikipedia": Tool(
        name="Wikipedia",
        func=wikipedia_tool.run,
        description="Gunakan untuk konsep hukum umum (Bahasa Indonesia)"
    ),
    "arXiv": Tool(
        name="arXiv",
        func=arxiv_tool.run,
        description="Gunakan untuk referensi akademik hukum atau data pribadi"
    ),
    "TavilySearch": Tool(
        name="TavilySearch",
        func=tavily_tool_instance.run,
        description="Gunakan untuk berita hukum terbaru di Indonesia"
    )
}

# ======================================
# 📊 Agent State
# ======================================
class AgentState(TypedDict):
    question: str
    docs: Optional[List[str]]
    external_docs: Optional[List[str]]
    answer: Optional[str]
    relevant: Optional[bool]
    answered: Optional[bool]
    selected_tools: Optional[List[str]]
    reasoning: Optional[str]


# ======================================
# 🧠 Node: Tool Selection
# ======================================
@traceable
def tool_selection_node(state: AgentState) -> AgentState:
    q = state["question"]

    # logika manual agar LLM tidak asal pilih tools
    if any(k in q.lower() for k in ["pasal", "bab", "ayat", "definisi", "pengertian", "tujuan"]):
        selected_tools = ["Documents"]
        reasoning = "Pertanyaan spesifik tentang isi UU PDP, jadi gunakan dokumen lokal terlebih dahulu."
    else:
        selected_tools = ["Documents", "Wikipedia"]
        reasoning = "Pertanyaan umum, gunakan dokumen lokal lalu referensi eksternal bila perlu."

    return {**state, "selected_tools": selected_tools, "reasoning": reasoning}


# ======================================
# 🔍 Node: Multi Source Retrieval
# ======================================
@traceable
def multi_source_retrieve_node(state: AgentState) -> AgentState:
    q = state["question"]
    selected = state.get("selected_tools", [])
    internal_docs, external_docs = [], []

    # cari dari dokumen lokal
    if "Documents" in selected:
        lines = [line for line in local_text.splitlines() if q.lower().split()[0] in line.lower()]
        internal_docs = lines[:5] if lines else [local_text[:1500]]

    # kalau perlu tambahan sumber luar
    for tool_name in selected:
        if tool_name in tools:
            try:
                external_docs.append(tools[tool_name].run(q))
            except Exception as e:
                external_docs.append(f"{tool_name} gagal: {str(e)}")

    return {**state, "docs": internal_docs, "external_docs": external_docs}


# ======================================
# 🎯 Node: Relevance Check
# ======================================
@traceable
def relevance_node(state: AgentState) -> AgentState:
    q = state["question"]
    all_docs = state.get("docs", []) + state.get("external_docs", [])
    prompt = f"""
    Evaluasi apakah dokumen berikut relevan menjawab pertanyaan UU Perlindungan Data Pribadi.

    Pertanyaan: {q}
    Dokumen: {all_docs[:1000]}

    Jawab hanya 'ya' jika relevan, 'tidak' jika tidak.
    """
    res = llm.invoke(prompt)
    return {**state, "relevant": "ya" in res.content.lower()}


# ======================================
# 💬 Node: Generate Final Answer
# ======================================
@traceable
def answer_generate_node(state: AgentState) -> AgentState:
    q = state["question"]
    context = "\n".join(state.get("docs", []) + state.get("external_docs", []))
    prompt = f"""
    Anda adalah pakar hukum yang menjelaskan isi Undang-Undang Perlindungan Data Pribadi (UU PDP) Indonesia.

    Pertanyaan: {q}
    Konteks: {context}

    Jawab secara lengkap, resmi, dan utamakan isi dari dokumen UU PDP.
    Jika tidak ditemukan di dokumen, barulah tambahkan sumber eksternal (Wikipedia, arXiv, Tavily).
    Sebutkan sumber di akhir jawaban.
    """

    res = llm.invoke(prompt)
    return {**state, "answer": res.content.strip()}


# ======================================
# 🧩 Node: Answer Validation
# ======================================
@traceable
def answer_check_node(state: AgentState) -> AgentState:
    q = state["question"]
    ans = state.get("answer", "")
    prompt = f"""
    Pertanyaan: {q}
    Jawaban: {ans}

    Apakah jawaban sudah menjawab dengan tepat? Jawab ya/tidak.
    """
    res = llm.invoke(prompt)
    return {**state, "answered": "ya" in res.content.lower()}


# ======================================
# 🔧 Workflow Graph
# ======================================
workflow = StateGraph(AgentState)
workflow.add_node("ToolSelection", tool_selection_node)
workflow.add_node("Retrieve", multi_source_retrieve_node)
workflow.add_node("Grade", relevance_node)
workflow.add_node("Generate", answer_generate_node)
workflow.add_node("Check", answer_check_node)

workflow.set_entry_point("ToolSelection")
workflow.add_edge("ToolSelection", "Retrieve")
workflow.add_edge("Retrieve", "Grade")
workflow.add_edge("Grade", "Generate")
workflow.add_edge("Generate", "Check")
workflow.add_conditional_edges(
    "Check",
    lambda s: "Yes" if s["answered"] else "No",
    {"Yes": END, "No": "Retrieve"}
)

runnable_graph = workflow.compile(config={"recursion_limit": 100})


# ======================================
# ▶️ Fungsi untuk dijalankan dari UI atau terminal
# ======================================
def jawab_pertanyaan(pertanyaan: str):
    state = {"question": pertanyaan}
    result = runnable_graph.invoke(state)
    return result.get("answer", "⚠️ Tidak ditemukan jawaban.")


if __name__ == "__main__":
    q = input("Tanyakan tentang UU PDP: ")
    print("\n🧠 Jawaban:\n", jawab_pertanyaan(q))
