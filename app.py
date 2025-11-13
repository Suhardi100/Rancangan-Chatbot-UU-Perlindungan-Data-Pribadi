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
# 🔧 Konfigurasi Awal (jgn lupa set key di environment jika diperlukan)
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
    temperature=0.0,  # deterministik agar hasil konsisten
    google_api_key="AIzaSyBI6YdES2PyWC3JU2_eDtTW1ipi6Z07DcE"
)

# ================================
# 🧰 Tools (tetap didefinisikan tapi TIDAK diprioritaskan)
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
# 📚 Load & Parse Dokumen UU PDP (uu_pdp.txt)
# ================================
loader = TextLoader("uu_pdp.txt", encoding="utf-8")
documents = loader.load()
raw_text = "\n".join([d.page_content for d in documents])

# Parser sederhana: pisahkan berdasarkan kata kunci "Pasal" / "Pasal X" / "Pasal nomor"
# Jika dokumen format lain, fungsi ini tetap mengambil potongan paragraf sebagai fallback.
def parse_pasals(text: str) -> List[Tuple[str, str]]:
    """
    Kembalikan list of (heading, content) di mana heading mis. 'Pasal 1', 'Pasal 2 ayat (1)', atau jika
    tidak ada heading, pembagi berdasarkan paragraf.
    """
    # Normalisasi \r\n -> \n
    text = text.replace("\r\n", "\n")
    # Temukan posisi heading seperti "Pasal 1", "Pasal 2.", "Pasal 3 -", case-insensitive
    # Gunakan regex untuk menemukan "Pasal" dan nomor setelahnya
    pattern = re.compile(r'(?i)^(pasal\s+\d+[\s\.\-\:a-z0-9\(\)]*)', re.MULTILINE)
    matches = list(pattern.finditer(text))

    units = []
    if matches:
        # Jika ada heading, potong berdasarkan heading
        for i, m in enumerate(matches):
            start = m.start()
            heading = m.group(1).strip()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            content = text[start:end].strip()
            units.append((heading, content))
    else:
        # Fallback: split by double newlines (paragraf)
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        for idx, p in enumerate(paras, 1):
            heading = f"Paragraf {idx}"
            units.append((heading, p))
    return units

# Buat index sederhana: list of (id, heading, content, lower_content)
parsed_units = []
for idx, (heading, content) in enumerate(parse_pasals(raw_text), 1):
    parsed_units.append({
        "id": idx,
        "heading": heading,
        "content": content,
        "lower": content.lower()
    })

# ================================
# 🔎 Fungsi pencarian pasal (prioritas dokumen lokal)
# ================================
def search_units_by_question(question: str, top_k: int = 5) -> List[dict]:
    """
    Cari unit (pasal/paragraf) yang paling relevan terhadap question.
    Metode sederhana: token match count (bisa ditingkatkan dengan embeddings).
    Mengembalikan list unit dict terurut by score desc.
    """
    q = question.lower()
    # tokenisasi kata kunci (hapus stopwords sederhana? di versi ini gunakan kata)
    tokens = [t for t in re.findall(r'\w+', q) if len(t) > 2]  # kata >=3 huruf
    scores = []
    for u in parsed_units:
        score = 0
        for t in tokens:
            if t in u["lower"]:
                score += u["lower"].count(t)
        if score > 0:
            scores.append((score, u))
    # Urutkan dan ambil top_k
    scores.sort(key=lambda x: x[0], reverse=True)
    top = [u for s,u in scores[:top_k]]
    return top

# Helper untuk format hasil pasal yang ditemukan
def format_units(units: List[dict]) -> str:
    if not units:
        return "(tidak ditemukan pasal relevan dalam dokumen lokal UU PDP)"
    out = []
    for u in units:
        out.append(f"=== {u['heading']} ===\n{u['content']}\n")
    return "\n".join(out)

# ================================
# 🧩 Define Agent State (TypedDict)
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
# 🧠 Node: Tool Selection (FORCE Documents primary)
# ================================
@traceable
def tool_selection_node(state: AgentState) -> AgentState:
    """
    FORCE memilih Documents (dokumen lokal) terlebih dahulu.
    Hanya pilih tools eksternal jika dokumen lokal tidak mengandung informasi relevan.
    """
    question = state["question"]
    # Cari apakah dokumen lokal memiliki setidaknya 1 hasil relevan
    units = search_units_by_question(question, top_k=3)
    if units:
        # Prioritaskan dokumen lokal
        selected = ["Documents"]
        reasoning = "Dokumen lokal (uu_pdp.txt) mengandung potensi jawaban sehingga diprioritaskan."
    else:
        # Tidak ada di dokumen lokal -> coba eksternal sebagai fallback
        selected = ["Documents", "TavilySearch", "Wikipedia", "arXiv"]
        reasoning = "Tidak ditemukan potensi relevansi di dokumen lokal; memperluas pencarian ke sumber eksternal."
    return {**state, "selected_tools": selected, "reasoning": reasoning}

# ================================
# 🔍 Node: Multi Source Retrieval (PRIORITIZE DOCUMENTS)
# ================================
@traceable
def multi_source_retrieve_node(state: AgentState) -> AgentState:
    question = state["question"]
    selected = state.get("selected_tools", [])

    # Ambil hasil dari dokumen lokal (HARUS)
    local_units = search_units_by_question(question, top_k=5)
    internal_docs = []
    if local_units:
        # Format tiap unit agar mudah dibaca oleh LLM
        for u in local_units:
            # sertakan heading seperti "Pasal X" agar LLM bisa menyebut sumber persis
            internal_docs.append(f"{u['heading']}\n{u['content']}")
    else:
        internal_docs = ["(Tidak ditemukan pasal relevan dalam dokumen lokal UU PDP)"]

    # External retrieval only if selected contains external tools AND no local hits
    external_docs = []
    if (not local_units) and any(t in ["TavilySearch", "Wikipedia", "arXiv"] for t in selected):
        for tool_name in selected:
            if tool_name in tools and tool_name != "Documents":
                try:
                    external_docs.append(f"{tool_name}:\n{tools[tool_name].run(question)}")
                except Exception as e:
                    external_docs.append(f"{tool_name}: Error - {str(e)}")
    # Return docs
    return {**state, "docs": internal_docs, "external_docs": external_docs}

# ================================
# 🧮 Node: Grade Relevance (mengandalkan dokumen lokal)
# ================================
@traceable
def enhanced_grade_node(state: AgentState) -> AgentState:
    question = state["question"]
    docs = state.get("docs", []) or []
    # Jika hanya berisi "(tidak ditemukan...)" maka mark not relevant
    concatenated = "\n".join(docs)
    is_relevant = "(tidak ditemukan pasal relevan" not in concatenated.lower()
    # Jika ada external_docs, anggap relevan juga (tapi hanya dipakai saat local tidak punya)
    if not is_relevant and state.get("external_docs"):
        is_relevant = True
    return {**state, "relevant": is_relevant}

# ================================
# 🧩 Node: Generate Final Answer (MUST use only local docs if available)
# ================================
@traceable
def enhanced_generation_node(state: AgentState) -> AgentState:
    question = state["question"]
    docs = state.get("docs", []) or []
    external = state.get("external_docs", []) or []

    # Build prompt that instructs model to only use local docs if they contain relevant info.
    # The model is REQUIRED to quote exact passages found in `docs` and to include heading (Pasal X).
    prompt = f"""
Kamu adalah asisten hukum spesialis UU Perlindungan Data Pribadi (PDP) di Indonesia.
Penting: **UTAMAKAN dan HANYA gunakan isi yang berasal dari dokumen lokal 'uu_pdp.txt'** jika ada bagian dokumen yang relevan.
Jika dokumen lokal berisi potongan yang relevan, jawab dengan:
1) Kutip potongan dokumen persis (jangan parafrase untuk bagian kutipan), 
2) Sebutkan sumbernya (contoh: "Sumber: Pasal 5 UU PDP"), 
3) Berikan penjelasan singkat yang menghubungkan kutipan dengan pertanyaan (maksimal 2-3 kalimat).
Jika dokumen lokal **TIDAK** mengandung jawaban, baru gunakan informasi eksternal sebagai fallback dan nyatakan "Fallback: tidak ditemukan di UU PDP, informasi dari {sumber}".

Pertanyaan pengguna:
{question}

Bagian dokumen lokal yang relevan (jika ada):
{chr(10).join(docs)}

Bagian sumber eksternal (jika digunakan sebagai fallback):
{chr(10).join(external)}

Jawab sekarang sesuai aturan di atas. Jika ada pasal/ayat sebutkan secara eksplisit (misal: "Pasal 4 ayat (1)").
Jika tidak ada referensi yang ditemukan di lokasimu, jawab jujur: "(tidak ditemukan dalam UU PDP)" dan berikan rekomendasi sumber eksternal singkat.
    """
    resp = llm.invoke(prompt)
    answer_text = resp.content.strip()
    return {**state, "answer": answer_text}

# ================================
# 🔁 Node: Answer Check
# ================================
@traceable
def answer_check_node(state: AgentState) -> AgentState:
    # Simpel: jika answer mengandung "(tidak ditemukan" => not answered else yes
    ans = state.get("answer", "") or ""
    answered = "(tidak ditemukan" not in ans.lower()
    return {**state, "answered": answered}

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

# Optional helper main for quick testing (tidak mengubah logika lain)
if __name__ == "__main__":
    test_q = "Siapa yang bertanggung jawab atas pengolahan data menurut UU PDP?"
    state = {"question": test_q}
    res = runnable_graph.invoke(state)
    print("ANSWER:\n", res.get("answer"))
