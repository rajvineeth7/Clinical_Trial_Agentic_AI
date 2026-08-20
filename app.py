import os
import json
import re
from typing import Optional

import requests
import streamlit as st
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

# ============================================================
# CTIA — Clinical Trial Intelligence Agent
# Free-stack MVP:
# LangChain + Groq + ClinicalTrials.gov + PubMed + PubChem
# ============================================================

st.set_page_config(
    page_title="CTIA | Clinical Trial Intelligence Agent",
    page_icon="🧬",
    layout="wide",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #667085;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .agent-box {
        padding: 0.9rem 1rem;
        border: 1px solid #d0d5dd;
        border-radius: 12px;
        background: #f8fafc;
        margin-bottom: 0.7rem;
    }
    .source-pill {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: #eef2ff;
        margin-right: 0.35rem;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def get_groq_api_key() -> str:
    """
    Read the Groq API key ONLY from Streamlit Secrets.

    Local:
      .streamlit/secrets.toml

    Streamlit Cloud:
      App -> Settings -> Secrets
    """
    try:
        value = st.secrets.get("GROQ_API_KEY")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return ""


def safe_get(data, *keys, default=""):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def compact_text(text: str, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def unique(items):
    seen = set()
    output = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


# ============================================================
# TOOL 1 — ClinicalTrials.gov
# ============================================================
@tool
def search_clinical_trials(
    query: str,
    status: Optional[str] = None,
    phase: Optional[str] = None,
    country: Optional[str] = None,
    max_results: int = 8,
) -> str:
    """
    Search public ClinicalTrials.gov study records.

    Use this tool for questions about clinical trials, recruiting studies,
    trial phases, sponsors, interventions, locations, eligibility, or NCT IDs.

    Arguments:
      query: condition, intervention, disease, mechanism, or research topic.
      status: optional status such as RECRUITING, NOT_YET_RECRUITING,
              ACTIVE_NOT_RECRUITING, COMPLETED, etc.
      phase: optional phase such as PHASE1, PHASE2, PHASE3, PHASE4.
              Multiple phases can be comma-separated.
      country: optional country/location filter, e.g. United States.
      max_results: maximum records to return, 1-15.
    """
    try:
        max_results = max(1, min(int(max_results), 15))
        params = {
            "query.term": query,
            "pageSize": max_results,
            "format": "json",
        }

        if status:
            statuses = [s.strip().upper() for s in status.split(",") if s.strip()]
            params["filter.overallStatus"] = "|".join(statuses)

        response = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params=params,
            timeout=25,
        )
        response.raise_for_status()
        payload = response.json()

        studies = payload.get("studies", [])
        requested_phases = set()
        if phase:
            requested_phases = {
                p.strip().upper().replace(" ", "_")
                for p in phase.split(",")
                if p.strip()
            }

        results = []

        for study in studies:
            p = study.get("protocolSection", {})
            ident = p.get("identificationModule", {})
            stat = p.get("statusModule", {})
            design = p.get("designModule", {})
            sponsor_mod = p.get("sponsorCollaboratorsModule", {})
            intervention_mod = p.get("armsInterventionsModule", {})
            loc_mod = p.get("contactsLocationsModule", {})
            desc_mod = p.get("descriptionModule", {})
            eligibility_mod = p.get("eligibilityModule", {})

            phases = design.get("phases", []) or []

            if requested_phases:
                normalized = {str(x).upper().replace(" ", "_") for x in phases}
                if not normalized.intersection(requested_phases):
                    continue

            locations = loc_mod.get("locations", []) or []
            location_strings = []
            country_match = True

            if country:
                country_lower = country.lower().strip()
                country_match = any(
                    country_lower in str(loc.get("country", "")).lower()
                    for loc in locations
                )
                if not country_match:
                    continue

            for loc in locations[:8]:
                parts = [
                    loc.get("facility"),
                    loc.get("city"),
                    loc.get("state"),
                    loc.get("country"),
                ]
                location_strings.append(", ".join(str(x) for x in parts if x))

            interventions = []
            for item in intervention_mod.get("interventions", []) or []:
                name = item.get("name")
                itype = item.get("type")
                if name:
                    interventions.append(
                        f"{name}" + (f" ({itype})" if itype else "")
                    )

            lead_sponsor = sponsor_mod.get("leadSponsor", {}).get("name", "")

            results.append(
                {
                    "nct_id": ident.get("nctId", ""),
                    "title": ident.get("briefTitle", ""),
                    "official_title": ident.get("officialTitle", ""),
                    "status": stat.get("overallStatus", ""),
                    "phases": phases,
                    "sponsor": lead_sponsor,
                    "interventions": unique(interventions),
                    "locations": unique(location_strings),
                    "brief_summary": compact_text(
                        desc_mod.get("briefSummary", ""), 700
                    ),
                    "study_type": design.get("studyType", ""),
                    "enrollment": design.get("enrollmentInfo", {}).get("count"),
                    "eligibility": compact_text(
                        eligibility_mod.get("eligibilityCriteria", ""), 700
                    ),
                    "minimum_age": eligibility_mod.get("minimumAge", ""),
                    "maximum_age": eligibility_mod.get("maximumAge", ""),
                    "sex": eligibility_mod.get("sex", ""),
                    "url": (
                        f"https://clinicaltrials.gov/study/{ident.get('nctId', '')}"
                        if ident.get("nctId")
                        else ""
                    ),
                }
            )

        if not results:
            return json.dumps(
                {
                    "source": "ClinicalTrials.gov",
                    "message": "No matching trials were found with the supplied filters.",
                    "results": [],
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "source": "ClinicalTrials.gov",
                "count": len(results),
                "results": results,
            },
            ensure_ascii=False,
        )

    except Exception as exc:
        return json.dumps(
            {
                "source": "ClinicalTrials.gov",
                "error": f"Clinical trial search failed: {exc}",
            },
            ensure_ascii=False,
        )


# ============================================================
# TOOL 2 — PubMed
# ============================================================
@tool
def search_pubmed(
    query: str,
    max_results: int = 5,
) -> str:
    """
    Search PubMed for scientific publications related to a clinical,
    disease, drug, mechanism, or clinical-trial topic.
    """
    try:
        max_results = max(1, min(int(max_results), 10))

        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

        search_response = requests.get(
            f"{base}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
            },
            timeout=20,
        )
        search_response.raise_for_status()
        search_data = search_response.json()

        ids = search_data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return json.dumps(
                {"source": "PubMed", "count": 0, "results": []},
                ensure_ascii=False,
            )

        summary_response = requests.get(
            f"{base}/esummary.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "json",
            },
            timeout=20,
        )
        summary_response.raise_for_status()
        data = summary_response.json().get("result", {})

        results = []
        for pmid in ids:
            item = data.get(pmid, {})
            results.append(
                {
                    "pmid": pmid,
                    "title": item.get("title", ""),
                    "journal": item.get("fulljournalname", ""),
                    "pubdate": item.get("pubdate", ""),
                    "authors": [
                        a.get("name", "")
                        for a in (item.get("authors", []) or [])[:5]
                        if a.get("name")
                    ],
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )

        return json.dumps(
            {
                "source": "PubMed",
                "count": len(results),
                "results": results,
            },
            ensure_ascii=False,
        )

    except Exception as exc:
        return json.dumps(
            {"source": "PubMed", "error": f"PubMed search failed: {exc}"},
            ensure_ascii=False,
        )


# ============================================================
# TOOL 3 — PubChem
# ============================================================
@tool
def lookup_drug_or_compound(compound_name: str) -> str:
    """
    Look up a drug or chemical compound in PubChem.

    Use this for compound identity and basic chemical properties.
    """
    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            f"name/{requests.utils.quote(compound_name, safe='')}/"
            "property/Title,MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
        )

        response = requests.get(url, timeout=20)

        if response.status_code == 404:
            return json.dumps(
                {
                    "source": "PubChem",
                    "message": f"No PubChem compound found for '{compound_name}'.",
                },
                ensure_ascii=False,
            )

        response.raise_for_status()
        payload = response.json()

        properties = payload.get("PropertyTable", {}).get("Properties", [])
        if not properties:
            return json.dumps(
                {
                    "source": "PubChem",
                    "message": f"No properties returned for '{compound_name}'.",
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "source": "PubChem",
                "compound": properties[0],
                "url": (
                    "https://pubchem.ncbi.nlm.nih.gov/#query="
                    + requests.utils.quote(compound_name)
                ),
            },
            ensure_ascii=False,
        )

    except Exception as exc:
        return json.dumps(
            {"source": "PubChem", "error": f"PubChem lookup failed: {exc}"},
            ensure_ascii=False,
        )


# ============================================================
# AGENT
# ============================================================
SYSTEM_PROMPT = """
You are CTIA, a Clinical Trial Intelligence Agent for clinical research.

Your job is to transform natural-language research questions into reliable,
source-grounded clinical-trial intelligence.

You have three tools:
1. search_clinical_trials — primary source for trial records.
2. search_pubmed — scientific publication search.
3. lookup_drug_or_compound — basic compound information from PubChem.

Rules:
- For clinical-trial questions, use ClinicalTrials.gov rather than relying
  on your internal knowledge.
- Use PubMed when the user asks about publications, evidence, literature,
  or related research.
- Use PubChem when the user asks about a drug/compound's basic identity
  or chemical properties.
- You may use multiple tools when the question requires it.
- Do not invent NCT IDs, trial statuses, sponsors, drug properties,
  publication details, or numerical results.
- Clearly distinguish retrieved facts from your interpretation.
- Give direct source links whenever available.
- Do not provide medical advice or claim that a trial is appropriate for
  an individual patient.
- If a requested filter cannot be verified from the retrieved data, say so.
- Keep the final report useful for a clinical-research/business-analysis
  audience.

When answering a trial search request, prefer this structure:
1. Query interpretation
2. Relevant trials
3. Key findings / intelligence
4. Limitations
5. Sources

Always mention that ClinicalTrials.gov data can change as study records
are updated.
"""

def build_agent(api_key: str):
    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_retries=2,
        api_key=api_key,
    )

    return create_agent(
        model=model,
        tools=[
            search_clinical_trials,
            search_pubmed,
            lookup_drug_or_compound,
        ],
        system_prompt=SYSTEM_PROMPT,
        name="ctia_agent",
    )


# ============================================================
# UI
# ============================================================
st.markdown(
    '<div class="main-title">🧬 CTIA — Clinical Trial Intelligence Agent</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Agentic AI for clinical-trial discovery, literature '
    'search, and drug intelligence</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configuration")

    groq_key = get_groq_api_key()

    if groq_key:
        st.success("🔐 Groq API key loaded from Streamlit Secrets.")
    else:
        st.warning(
            "🔐 Groq API key not found. Add GROQ_API_KEY in Streamlit Secrets."
        )

    st.caption(
        "The API key is never entered in the app UI. "
        "Store it privately in Streamlit Secrets."
    )

    st.divider()
    st.subheader("🧰 Agent Tools")
    st.markdown(
        """
        <span class="source-pill">ClinicalTrials.gov</span>
        <span class="source-pill">PubMed</span>
        <span class="source-pill">PubChem</span>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("💡 Example questions")
    examples = [
        "Find recruiting Phase 3 NSCLC trials in the United States.",
        "Find Phase 2 and Phase 3 trials for EGFR-mutated lung cancer and summarize the interventions.",
        "Find recent PubMed publications related to pembrolizumab in NSCLC.",
        "Find clinical trials involving nivolumab and give me the sponsors.",
        "Search trials for multiple myeloma and then find related PubMed literature.",
    ]

    selected_example = st.selectbox("Choose an example", ["—"] + examples)

query = st.text_area(
    "🔎 Ask the Clinical Trial Intelligence Agent",
    value="" if selected_example == "—" else selected_example,
    height=110,
    placeholder=(
        "Example: Find recruiting Phase 3 NSCLC trials in the USA "
        "involving immunotherapy."
    ),
)

run = st.button("🚀 Run Agent", type="primary", use_container_width=True)

if run:
    if not groq_key:
        st.error("GROQ_API_KEY was not found in Streamlit Secrets. Add it under App Settings → Secrets.")
        st.stop()

    if not query.strip():
        st.warning("Please enter a clinical-research question.")
        st.stop()

    try:
        agent = build_agent(groq_key)

        with st.status("🤖 Agent is working...", expanded=True) as status:
            st.write("Understanding your research question...")

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query.strip(),
                        }
                    ]
                }
            )

            status.update(
                label="✅ Agent completed",
                state="complete",
                expanded=False,
            )

        messages = result.get("messages", [])

        # -----------------------------
        # Agent activity / tool trace
        # -----------------------------
        with st.expander("🔍 Agent Activity — show tool calls", expanded=False):
            tool_found = False

            for msg in messages:
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    tool_found = True
                    for call in tool_calls:
                        st.markdown(
                            f"**🔧 Tool selected:** `{call.get('name', 'unknown')}`"
                        )
                        args = call.get("args", {})
                        if args:
                            st.json(args)

                if getattr(msg, "type", "") == "tool":
                    tool_found = True
                    st.markdown(
                        f"**📡 Tool result:** `{getattr(msg, 'name', 'tool')}`"
                    )

            if not tool_found:
                st.info("No tool call was detected for this request.")

        # -----------------------------
        # Final response
        # -----------------------------
        final_message = messages[-1] if messages else None
        final_content = getattr(final_message, "content", "")

        st.subheader("🧠 Clinical Trial Intelligence")

        if isinstance(final_content, list):
            # Handle modern content-block responses.
            text_parts = []
            for block in final_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            final_content = "\n".join(text_parts)

        st.markdown(final_content or "The agent returned no final text.")

        # -----------------------------
        # Sources detected in response
        # -----------------------------
        st.divider()
        st.caption(
            "Source note: trial and publication records are retrieved from "
            "public APIs and may change when source records are updated."
        )

    except Exception as exc:
        st.error("The agent could not complete the request.")
        st.exception(exc)

else:
    st.info(
        "👆 Enter a research question and click **Run Agent**. "
        "The agent will decide which biomedical tools it needs."
    )

    st.markdown("### What makes this an agent?")
    st.markdown(
        """
        **Natural-language request → Agent reasoning → Tool selection → "
        "Live biomedical data → Analysis → Source-grounded report**

        The agent can decide whether to use **ClinicalTrials.gov, PubMed,
        PubChem, or multiple tools** depending on the question.
        """
    )
