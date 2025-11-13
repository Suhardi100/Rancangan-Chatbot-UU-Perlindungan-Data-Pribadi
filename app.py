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
# 🔧 Konfigurasi Awal (sesuaikan environment keys)
# ================================
os.environ["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY", "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")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY", "ls__YourLangSmithKeyHere")
os.environ["LANGCHAIN_PROJECT"] = "UU-PDP-AgenticRAG"

# ================================
# 🔮 Setup LLM (Google Gemini) - hanya digunakan sebagai fallback
# ================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.0,  # deterministik agar hasil konsisten
    google_api_key=os.environ.get("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")
)

# ================================
# 🧰 Tools eksternal
# ================================
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="id"))
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
tavily_tool_instance = TavilySearchResults(k=3)

tools = {
    "Wikipedia": Tool(name="Wikipedia", func=wikipedia_tool.run, description="(sekunder) konsep umum hukum (ID)"),
    "arXiv": Tool(name="arXiv", func=arxiv_tool.run, description="(sekunder) referensi akademik"),
    "TavilySearch": Tool(name="TavilySearch", func=tavily_tool_instance.run, description="(sekunder) berita/putusan terbaru")
}

# ================================
# 📚 Load Dokumen UU PDP (lokal)
# ================================
loader = TextLoader("uu_pdp.txt", encoding="utf-8")
documents = loader.load()
raw_text = "\n".join([d.page_content for d in documents])

# ================================
# 🔧 Parser Pasal (lebih toleran)
# ================================
def parse_pasals(text: str) -> List[Tuple[str, str]]:
    """
    Kembalikan list of (heading, content). Heading biasanya 'Pasal X' (bisa 'Pasal 1.', 'Pasal 1 -', dll).
    Jika regex tidak menemukan, fallback ke paragraf.
    """
    text = text.replace("\r\n", "\n")
    # Cari heading "Pasal <nomor>" termasuk jika ada 'Pasal 1.' 'Pasal 1 -' 'Pasal 1 ayat (1)'
    pattern = re.compile(r'(?i)(^|\n)\s*(pasal\s+\d+(?:\s*[^\n]*)?)', re.IGNORECASE)
    matches = list(pattern.finditer(text))

    units = []
    if matches:
        # buat list indeks (start) yang rapi
        indices = [m.start(2) for m in matches]
        headings = [m.group(2).strip() for m in matches]
        for i, start_idx in enumerate(indices):
            heading = headings[i]
            end = indices[i+1] if i+1 < len(indices) else len(text)
            content = text[start_idx:end].strip()
            units.append((heading, content))
    else:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        for idx, p in enumerate(paras, 1):
            units.append((f"Paragraf {idx}", p))
    return units

parsed_units = []
for idx, (heading, content) in enumerate(parse_pasals(raw_text), 1):
    parsed_units.append({"id": idx, "heading": heading, "content": content, "lower": content.lower()})

# ================================
# 🔎 Search helper (token match + direct pasal match)
# ================================
def extract_pasals_from_question(question: str) -> List[str]:
    """
    Coba ambil referensi 'Pasal <n>' dari pertanyaan (mis: 'pasal 1', 'pasal 4 ayat (1)').
    Kembalikan list heading normalized (lower).
    """
    found = []
    pattern = re.compile(r'(?i)pasal\s*\d+(?:\s*ayat\s*\(\d+\))?')
    for m in pattern.finditer(question):
        found.append(m.group(0).strip().lower())
    return found

def search_units_by_question(question: str, top_k: int = 5) -> List[dict]:
    q = question.lower()
    # 1) If question explicitly asks for Pasal N, try exact heading match
    explicit = extract_pasals_from_question(question)
    results = []
    if explicit:
        # try exact heading match in parsed_units (case-insensitive)
        for ex in explicit:
            for u in parsed_units:
                if u["heading"].lower().startswith(ex):
                    results.append(u)
        # return unique and preserve order
        seen = set()
        uniq = []
        for u in results:
            if u["id"] not in seen:
                uniq.append(u); seen.add(u["id"])
        if uniq:
            return uniq[:top_k]

    # 2) otherwise token match
    tokens = [t for t in re.findall(r'\w+', q) if len(t) > 2]
    scores = []
    for u in parsed_units:
        score = 0
        for t in tokens:
            score += u["lower"].count(t)
        if score > 0:
            scores.append((score, u))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [u for s,u in scores[:top_k]]

def format_units(units: List[dict]) -> str:
    if not units:
        return "(tidak ditemukan pasal relevan dalam dokumen lokal UU PDP)"
    out = []
    for u in units:
        out.append(f"=== {u['heading']} ===\n{u['content']}\n")
    return "\n".join(out)

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
# 🧠 Tool Selection (prioritize local)
# ================================
@traceable
def tool_selection_node(state: AgentState) -> AgentState:
    question = state["question"]
    units = search_units_by_question(question, top_k=3)
    if units:
        selected = ["Documents"]
        reasoning = "Dokumen lokal memiliki potensi jawaban; gunakan lokal."
    else:
        selected = ["Documents", "TavilySearch", "Wikipedia", "arXiv"]
        reasoning = "Tidak ditemukan potensi relevansi lokal; coba eksternal."
    return {**state, "selected_tools": selected, "reasoning": reasoning}

# ================================
# 🔍 Retrieval (ambil local selalu)
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
    # only call external if no local hits
    if (not local_units) and any(t in ["TavilySearch", "Wikipedia", "arXiv"] for t in selected):
        for tool_name in selected:
            if tool_name in tools and tool_name != "Documents":
                try:
                    external_docs.append(f"{tool_name}:\n{tools[tool_name].run(question)}")
                except Exception as e:
                    external_docs.append(f"{tool_name}: Error - {str(e)}")
    return {**state, "docs": internal_docs, "external_docs": external_docs}

# ================================
# 🧮 Grade relevance (simple)
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
# 🧩 Generate Final Answer (DETERMINISTIC FOR LOCAL HITS)
# ================================
@traceable
def enhanced_generation_node(state: AgentState) -> AgentState:
    question = state["question"]
    docs = state.get("docs", []) or []
    external = state.get("external_docs", []) or []

    # If local docs exist (not the '(tidak ditemukan...') => return deterministic quote
    concatenated = "\n".join(docs).strip()
    if concatenated and "(tidak ditemukan pasal relevan" not in concatenated.lower():
        # We will quote top docs (already formatted as heading + content)
        # Provide brief deterministic explanation (2 sentences max)
        parts = []
        for d in docs:
            # split first line as heading if present
            lines = d.splitlines()
            heading = lines[0] if lines else "Sumber: (tidak diketahui)"
            # content rest (limit length to avoid huge outputs)
            body = "\n".join(lines[1:]).strip()
            if len(body) > 1200:
                body_snip = body[:1200] + "... (truncated)"
            else:
                body_snip = body
            parts.append(f"{heading}\n{body_snip}\n")
        # create explanation automatically
        explanation = ("Bagian di atas diambil langsung dari dokumen lokal 'uu_pdp.txt'. "
                       "Sumber telah dicantumkan pada setiap kutipan. "
                       "Gunakan kutipan tersebut sebagai referensi hukum terkait pertanyaan Anda.")
        answer_text = "\n".join(parts) + "\nSumber: uu_pdp.txt\n\n" + explanation
        return {**state, "answer": answer_text}

    # Fallback: gunakan external + LLM
    prompt = f"""
Kamu adalah asisten hukum spesialis UU Perlindungan Data Pribadi (PDP) di Indonesia.
Gunakan sumber eksternal berikut untuk menjawab (hanya bila dokumen lokal tidak tersedia):
{chr(10).join(external)}

Pertanyaan pengguna:
{question}

Jawablah singkat dan sebutkan sumber eksternal yang digunakan.
"""
    resp = llm.invoke(prompt)
    return {**state, "answer": resp.content.strip()}

# ================================
# 🔁 Answer Check
# ================================
@traceable
def answer_check_node(state: AgentState) -> AgentState:
    ans = state.get("answer", "") or ""
    answered = bool(ans.strip()) and "(tidak ditemukan" not in ans.lower()
    return {**state, "answered": answered}

# ================================
# 🔧 Workflow Graph & runnable
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
# 🧪 Quick Test (CLI)
# ================================
if __name__ == "__main__":
    test_q = "Apa isi Pasal 1 UU Perlindungan Data Pribadi?"
    state = {"question": test_q}
    res = runnable_graph.invoke(state)
    print("==== ANSWER ====\n")
    print(res.get("answer"))
