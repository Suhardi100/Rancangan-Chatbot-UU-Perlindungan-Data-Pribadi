import os
import re
import streamlit as st
from typing import TypedDict, List, Optional, Tuple
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
# 🔮 Setup LLM (Google Gemini)
# ================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.0,
    google_api_key="AIzaSyBI6YdES2PyWC3JU2_eDtTW1ipi6Z07DcE"
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
        description="(sekunder) konsep umum hukum dalam Bahasa Indonesia"
    ),
    "arXiv": Tool(
        name="arXiv",
        func=arxiv_tool.run,
        description="(sekunder) referensi akademik"
    ),
    "TavilySearch": Tool(
        name="TavilySearch",
        func=tavily_tool_instance.run,
        description="(sekunder) berita/putusan terbaru Indonesia"
    )
}

# ================================
# 📚 Load Dokumen UU PDP
# ================================
loader = TextLoader("uu_pdp.txt", encoding="utf-8")
documents = loader.load()
raw_text = "\n".join([d.page_content for d in documents])

def parse_pasals(text: str) -> List[Tuple[str, str]]:
    text = text.replace("\r\n", "\n")
    pattern = re.compile(r'(?i)^(pasal\s+\d+[\s\.\-\:a-z0-9\(\)]*)', re.MULTILINE)
    matches = list(pattern.finditer(text))
    units = []
    if matches:
        for i, m in enumerate(matches):
            start = m.start()
            heading = m.group(1).strip()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            content = text[start:end].strip()
            units.append((heading, content))
    else:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        for idx, p in enumerate(paras, 1):
            units.append((f"Paragraf {idx}", p))
    return units

parsed_units = []
for idx, (heading, content) in enumerate(parse_pasals(raw_text), 1):
    parsed_units.append({
        "id": idx,
        "heading": heading,
        "content": content,
        "lower": content.lower()
    })

# ================================
# 🔎 Pencarian Pasal
# ================================
def search_units_by_question(question: str, top_k: int = 5) -> List[dict]:
    q = question.lower()
    tokens = [t for t in re.findall(r'\w+', q) if len(t) > 2]
    scores = []
    for u in parsed_units:
        score = 0
        for t in tokens:
            if t in u["lower"]:
                score += u["lower"].count(t)
        if score > 0:
            scores.append((score, u))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [u for s, u in scores[:top_k]]

def format_units(units: List[dict]) -> str:
    if not units:
        return "(tidak ditemukan pasal relevan dalam dokumen lokal UU PDP)"
    return "\n".join([f"=== {u['heading']} ===\n{u['content']}\n" for u in units])

# ================================
# 🧩 Agent State
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
# 🧠 Tool Selection
# ================================
@traceable
def tool_selection_node(state: AgentState) -> AgentState:
    question = state["question"]
    units = search_units_by_question(question, top_k=3)
    if units:
        selected = ["Documents"]
        reasoning = "Dokumen lokal (uu_pdp.txt) mengandung potensi jawaban sehingga diprioritaskan."
    else:
        selected = ["Documents", "TavilySearch", "Wikipedia", "arXiv"]
        reasoning = "Tidak ditemukan potensi relevansi di dokumen lokal; memperluas pencarian ke sumber eksternal."
    return {**state, "selected_tools": selected, "reasoning": reasoning}

# ================================
# 🔍 Multi Source Retrieval
# ================================
@traceable
def multi_source_retrieve_node(state: AgentState) -> AgentState:
    question = state["question"]
    selected = state.get("selected_tools", [])
    local_units = search_units_by_question(question, top_k=5)
    internal_docs = []
    if local_units:
        for u in local_units:
            internal_docs.append(f"{u['heading']}\n{u['content']}")
    else:
        internal_docs = ["(Tidak ditemukan pasal relevan dalam dokumen lokal UU PDP)"]

    external_docs = []
    if (not local_units) and any(t in ["TavilySearch", "Wikipedia", "arXiv"] for t in selected):
        for tool_name in selected:
            if tool_name in tools and tool_name != "Documents":
                try:
                    external_docs.append(f"{tool_name}:\n{tools[tool_name].run(question)}")
                except Exception as e:
                    external_docs.append(f"{tool_name}: Error - {str(e)}")
    return {**state, "docs": internal_docs, "external_docs": external_docs}

# ================================
# 🧮 Grade Relevance
# ================================
@traceable
def enhanced_grade_node(state: AgentState) -> AgentState:
    docs = state.get("docs", []) or []
    concatenated = "\n".join(docs)
    is_relevant = "(tidak ditemukan pasal relevan" not in concatenated.lower()
    if not is_relevant and state.get("external_docs"):
        is_relevant = True
    return {**state, "relevant": is_relevant}

# ================================
# 🧩 Generate Final Answer
# ================================
@traceable
def enhanced_generation_node(state: AgentState) -> AgentState:
    question = state["question"]
    docs = state.get("docs", []) or []
    external = state.get("external_docs", []) or []

    prompt = f"""
Kamu adalah asisten hukum spesialis UU Perlindungan Data Pribadi (PDP) di Indonesia.
Penting: **UTAMAKAN dan HANYA gunakan isi dari dokumen lokal 'uu_pdp.txt'** jika ada bagian yang relevan.
Jika dokumen lokal berisi potongan relevan, jawab dengan:
1) Kutip potongan dokumen persis (jangan parafrase),
2) Sebutkan sumbernya (contoh: "Sumber: Pasal 5 UU PDP"),
3) Jelaskan singkat keterkaitannya (2-3 kalimat).
Jika tidak ditemukan di dokumen lokal, nyatakan:
"Fallback: tidak ditemukan di UU PDP, informasi berasal dari sumber eksternal (Wikipedia/Arxiv/TavilySearch)."

Pertanyaan pengguna:
{question}

Bagian dokumen lokal:
{chr(10).join(docs)}

Sumber eksternal (jika ada):
{chr(10).join(external)}
    """
    resp = llm.invoke(prompt)
    answer_text = resp.content.strip()
    return {**state, "answer": answer_text}

# ================================
# 🔁 Answer Check
# ================================
@traceable
def answer_check_node(state: AgentState) -> AgentState:
    ans = state.get("answer", "") or ""
    answered = "(tidak ditemukan" not in ans.lower()
    return {**state, "answered": answered}

# ================================
# 🔧 Workflow Graph
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

# ================================
# 🧪 Quick Test
# ================================
if __name__ == "__main__":
    test_q = "Siapa yang bertanggung jawab atas pengolahan data menurut UU PDP?"
    state = {"question": test_q}
    res = runnable_graph.invoke(state)
    print("ANSWER:\n", res.get("answer"))
