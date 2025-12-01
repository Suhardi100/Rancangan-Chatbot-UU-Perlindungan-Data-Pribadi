import os
import streamlit as st
from typing import TypedDict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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
os.environ["LANGCHAIN_PROJECT"] = "UU-CiptaKerja-AgenticRAG"

# ================================
# 🔮 Setup Google Gemini
# ================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    google_api_key="AIzaSyCWV_230ec-t_xUZ2Vsj1XXJDSx57UaJlA"
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
        description="Gunakan untuk menemukan konsep utama, sejarah, dan hal-hal lain yang berkaitan dengan UU Perlindungan Data Pribadi (PDP) dalam Bahasa Indonesia!"
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
    Kamu adalah asisten ahli UU Pelindungan Data Pribadi yang sangat cerdas setara 100 profesor. Sebelum menjawab wajib mengecek apakah pertanyaan tersebut berkaitan dengan UU Perlindungan Data Pribadi (PDP) atau tidak? Jika tidak, maka jangan mencoba menjawab. Namun jawablah dengan kata-kata yang sama persis dengan "saya tidak bisa menjawab pertanyaan Anda karena tidak berkaitan dengan UU Perlindungan Data Pribadi (PDP)". Utamakan mencari dulu sumber yang terdapat dalam documents. Baru setelah itu, tentukan tools terbaik untuk menjawab pertanyaan berikut:

    Pertanyaan: {q}

    Tools tersedia:
    1. Wikipedia - konsep hukum umum (Bahasa Indonesia)
    2. arXiv - penelitian hukum akademik
    3. TavilySearch - berita dan hukum terbaru di Indonesia
    4. Documents UU PDP - dokumen UU Perlindungan Data Pribadi

    Analisis:
    - Apakah ada referensi tentang UU Perlindungan Data Pribadi (PDP) terkini Indonesia? → TavilySearch
    - Apakah teori akademik yang berkenaan UU Perlindungan Data Pribadi (PDP) di Indonesia? → arXiv
    - Apakah konsep dasar UU Perlindungan Data Pribadi (PDP)? → Wikipedia
    - Apakah isi UU Perlindungan Data Pribadi (PDP) setiap pasalnya? → Documents UU PDP

    Format:
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
    
    # Gunakan konten aktual dari dokumen uu_pdp.txt
    internal_docs = [doc.page_content for doc in documents]
    external_docs = []

    for tool_name in selected:
        if tool_name in tools:
            try:
                result = tools[tool_name].run(q)
                # Handle case where TavilySearch returns a list
                if isinstance(result, list):
                    # Convert list of search results to string
                    result_str = ""
                    for i, item in enumerate(result):
                        if isinstance(item, dict):
                            result_str += f"Result {i+1}: {item.get('title', 'No title')} - {item.get('content', 'No content')}\n"
                        else:
                            result_str += f"Result {i+1}: {str(item)}\n"
                    external_docs.append(result_str.strip())
                else:
                    external_docs.append(str(result))
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
    Evaluasi relevansi dokumen berikut untuk pertanyaan UU Perlindungan Data Pribadi (PDP) ini dengan documents:

    Pertanyaan: {q}
    Dokumen: {all_docs}

    Apakah sangat relevan untuk menjawab pertanyaan dan sesuai dengan documents? (ya/tidak)
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
    Kamu adalah asisten ahli UU Perlindungan Data Pribadi (PDP) di Indonesia.
    Sebelum menjawab wajib mengecek apakah pertanyaan tersebut berkaitan dengan UU Perlindungan Data Pribadi (PDP) atau tidak? Jika tidak, maka jangan mencoba menjawab. 
    Namun jawablah dengan kata-kata yang sama persis dengan "saya tidak bisa menjawab pertanyaan Anda karena tidak berkaitan dengan UU Perlindungan Data Pribadi (PDP)".
    Utamakan mengambil dari documents lalu gabungkan informasi dari berbagai sumber berikut untuk menjawab pertanyaan secara komprehensif.

    Pertanyaan: {q}
    Konteks: {context}

    Jawablah dengan mengutamakan yang ada di dokumen tersebut dengan bahasa Indonesia formal, dan sebutkan sumber (UU, Wikipedia, Tavily, dll).
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
    lambda s: "Yes" if s.get("answered") else "No",
    {"Yes": END, "No": "Retrieve"}
)
runnable_graph = workflow.compile()
