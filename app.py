import streamlit as st
import requests
import google.generativeai as genai

# ================================
# CTIA — Clinical Trial Agent (Gemini version, no LangChain)
# ================================

st.set_page_config(page_title="CTIA | Clinical Trial Agent", page_icon="🧬", layout="wide")
st.title("🧬 Clinical Trial Intelligence Agent")
st.write("Ask about clinical trials, publications, or compounds.")

# -----------------------------
# API Key Handling
# -----------------------------
def get_gemini_api_key() -> str:
    """Read Gemini API key from Streamlit Secrets."""
    try:
        return st.secrets["GEMINI_API_KEY"].strip()
    except Exception:
        return ""

gemini_key = get_gemini_api_key()
if not gemini_key:
    st.error("❌ GEMINI_API_KEY not found. Please add it in Streamlit Secrets.")
    st.stop()

# Configure Gemini
genai.configure(api_key=gemini_key)

# -----------------------------
# Tools (simple HTTP calls)
# -----------------------------
def search_clinical_trials(query: str, max_results: int = 5) -> str:
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

def search_pubmed(query: str, max_results: int = 5) -> str:
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

def lookup_drug_or_compound(name: str) -> str:
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
# UI
# -----------------------------
query = st.text_area("🔎 Enter your research question:", height=100)
run = st.button("🚀 Run Agent")

if run and query.strip():
    with st.spinner("🤖 Gemini is working..."):
        # Call Gemini directly
        model = genai.GenerativeModel("gemini-1.5-flash")  # free-tier model
        prompt = f"""
        You are CTIA, a Clinical Trial Intelligence Agent.
        Use the following tools' outputs to answer the question.

        Question: {query}
        """
        response = model.generate_content(prompt)
    st.subheader("🧠 Agent Response")
    st.write(response.text if response else "No response.")
