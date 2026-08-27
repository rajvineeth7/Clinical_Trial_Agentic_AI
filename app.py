import streamlit as st
import requests
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool

# ================================
# CTIA — Clinical Trial Agent (Simplified)
# ================================

st.set_page_config(page_title="CTIA | Clinical Trial Agent", page_icon="🧬", layout="wide")
st.title("🧬 Clinical Trial Intelligence Agent")
st.write("Ask about clinical trials, publications, or compounds.")

# -----------------------------
# API Key Handling
# -----------------------------
def get_groq_api_key() -> str:
    """Read Groq API key from Streamlit Secrets."""
    try:
        return st.secrets["GROQ_API_KEY"].strip()
    except Exception:
        return ""

groq_key = get_groq_api_key()
if not groq_key:
    st.error("❌ GROQ_API_KEY not found. Please add it in Streamlit Secrets.")
    st.stop()

# -----------------------------
# Tools
# -----------------------------
@tool
def search_clinical_trials(query: str, max_results: int = 5) -> str:
    """Search ClinicalTrials.gov for trials."""
    try:
        response = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={"query.term": query, "pageSize": max_results, "format": "json"},
            timeout=20,
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error: {e}"

@tool
def search_pubmed(query: str, max_results: int = 5) -> str:
    """Search PubMed for publications."""
    try:
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
            timeout=20,
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error: {e}"

@tool
def lookup_drug_or_compound(name: str) -> str:
    """Look up a compound in PubChem."""
    try:
        response = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/Title,MolecularFormula,MolecularWeight,CanonicalSMILES/JSON",
            timeout=20,
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error: {e}"

# -----------------------------
# Agent Setup
# -----------------------------
SYSTEM_PROMPT = """
You are CTIA, a Clinical Trial Intelligence Agent.
Use ClinicalTrials.gov for trials, PubMed for publications, and PubChem for compounds.
Always provide source links when available.
"""

def build_agent(api_key: str):
    model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, api_key=api_key)
    return create_agent(model=model, tools=[search_clinical_trials, search_pubmed, lookup_drug_or_compound], system_prompt=SYSTEM_PROMPT)

# -----------------------------
# UI
# -----------------------------
query = st.text_area("🔎 Enter your research question:", height=100)
run = st.button("🚀 Run Agent")

if run and query.strip():
    agent = build_agent(groq_key)
    with st.spinner("🤖 Agent is working..."):
        result = agent.invoke({"messages": [{"role": "user", "content": query.strip()}]})
    st.subheader("🧠 Agent Response")
    st.write(result.get("messages")[-1].content if result.get("messages") else "No response.")
