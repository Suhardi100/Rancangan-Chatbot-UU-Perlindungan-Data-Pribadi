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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ================================
# 🔧 Konfigurasi Awal
# ================================
os.environ["TAVILY_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__YourLangSmithKeyHere"
os.environ["LANGCHAIN_PROJECT"] = "UU-CiptaKerja-AgenticRAG"

# ================================
# 🔮 Setup Google Gemini
# ================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.3,
    google_api_key="AIzaSyCWV_230ec-t_xUZ2Vsj1XXJDSx57UaJlA"
)

# ================================
# 📚 Load Dokumen UU PDP
# ================================
loader = TextLoader("uu_pdp.txt", encoding='utf-8')
documents = loader.load()

# Split dokumen panjang menjadi potongan untuk pencarian efisien
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
texts = splitter.split_documents(documents)

# Buat embedding untuk pencarian semantik di dalam UU PDP
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyCWV_230ec-t_xUZ2Vsj1XXJDSx57UaJlA")
vectorstore = FAISS.from_documents(texts, embedding_model)

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
        description="Gunakan untuk menemukan konsep umum tentang UU PDP (Bahasa Indonesia)."
    ),
    "arXiv": Tool(
        name="arXiv",
        func=arxiv_tool.run,
        description="Gunakan untuk referensi akademik tentang teori atau penelitian UU PDP."
    ),
    "TavilySearch": Tool(
        name="TavilySearch",
        func=tavily_tool_instance.run,
        description="Gunakan untuk berita atau hukum terbaru terkait UU PDP."
    )
}

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
    Kamu adalah asisten ahli UU Perlindungan Data Pribadi. 
    Prioritaskan pencarian dari dokumen UU PDP terlebih dahulu. 
    Jika tidak ditemukan di dokumen, baru gunakan tools lain.

    Pertanyaan: {q}

    Tools tersedia:
    - Documents UU PDP
    - Wikipedia
    - arXiv
    - TavilySearch

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
# 🔍 Node: Multi Source Retrieval
# ================================
@traceable
def multi_source_retrieve_node(state: AgentState) -> AgentState:
    q = state["question"]
    selected = state.get("selected_tools", [])

    # Cari dari dokumen UU PDP
    search_results = vectorstore.similarity_search(q, k=5)
    internal_docs = [doc.page_content for doc in search_results]

    external_docs = []
    for tool_name in selected:
        if tool_name != "Documents UU PDP" and tool_name in tools:
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
    Evaluasi relevansi teks berikut terhadap pertanyaan tentang UU PDP.

    Pertanyaan: {q}
    Dokumen: {all_docs}

    Jawab hanya 'ya' jika sangat relevan atau 'tidak' jika tidak relevan.
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
    Kamu adalah pakar hukum yang menjawab berdasarkan isi dokumen UU Perlindungan Data Pribadi (PDP).
    Utamakan isi pasal atau penjelasan langsung dari dokumen UU PDP terlebih dahulu. 
    Jika tidak ditemukan di dokumen, baru tambahkan konteks dari sumber luar.

    Pertanyaan: {q}
    Konteks dari dokumen dan sumber luar:
    {context}

    Jawablah secara lengkap dalam Bahasa Indonesia formal dan sebutkan sumber (UU PDP Pasal X, Wikipedia, dsb).
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
    prompt = f"""
    Apakah jawaban ini sudah menjawab pertanyaan secara tepat?
    Pertanyaan: {q}
    Jawaban: {ans}
    Jawab hanya 'ya' atau 'tidak'.
    """
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
