"""AI Troubleshooting Assistant for L1/L2 Support  (Streamlit frontend)

Flow: user describes issue -> retrieve top matches from the knowledge base (TF-IDF)
      -> if nothing relevant: ask for more info (no AI guess)
      -> else: AI (Claude) writes a grounded answer, or a template answer if no API key
      -> UI shows steps, escalation team and the source KB entries used.
"""
import json
import os
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = Path(__file__).parent / "data" / "knowledge_base.json"
DEFAULT_MODELS = {
    "Anthropic": "claude-sonnet-5-5",
    # if you get NotFoundError, pick a current ID from console.groq.com/docs/models
    "Groq": "openai/gpt-oss-120b",
}

EXAMPLES = [
    "The payment API is returning a 500 error",
    "Application is timing out while connecting to the database",
    "Server says no space left on device",
    "Users are not receiving emails",
    "My laptop screen is flickering",  # unsupported -> tests the fallback
]

SYSTEM_PROMPT = """You are an AI troubleshooting assistant for L1/L2 support engineers.
Answer ONLY using the knowledge base entries provided in <knowledge_base_entries>.

Rules:
- Never invent causes, commands, steps or resolutions that are not supported by the entries.
- If the entries do not clearly match the user's issue, say the knowledge base is insufficient
  and ask 2-3 specific clarifying questions (system, error message, when it started, recent changes).
- Otherwise respond in markdown with these sections:
  **Likely cause**, **Troubleshooting steps** (numbered, in a sensible order),
  **Known resolution**, **Escalate to** (team name from the entry, and when to escalate).
- End with a line: **Sources:** followed by the KB entry IDs you used.
- Be concise and practical."""


# ---------- Knowledge base + retrieval ----------
@st.cache_data
def load_kb():
    with open(KB_PATH, encoding="utf-8") as f:
        return json.load(f)


def entry_text(e):
    return " ".join(
        [e["title"], e["category"]] + e["symptoms"] +
        e["keywords"] + e["keywords"]
    )


@st.cache_resource
def build_index():
    kb = load_kb()
    vec = TfidfVectorizer(stop_words="english",
                          ngram_range=(1, 2), sublinear_tf=True)
    matrix = vec.fit_transform([entry_text(e) for e in kb])
    return kb, vec, matrix


def retrieve(query, top_k, threshold):
    kb, vec, matrix = build_index()
    scores = cosine_similarity(vec.transform([query]), matrix).ravel()
    ranked = sorted(zip(kb, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"entry": e, "score": float(s)} for e, s in ranked if s >= threshold]


# ---------- Answer generation ----------
def format_entry(e):
    checks = "\n".join(f"{i}. {c}" for i, c in enumerate(
        e["diagnostic_checks"], 1))
    return (
        f"[{e['id']}] {e['title']}\nCategory: {e['category']}\n"
        f"Symptoms: {'; '.join(e['symptoms'])}\nDiagnostic checks:\n{checks}\n"
        f"Resolution: {e['resolution']}\nEscalation team: {e['escalation_team']}"
    )


def ai_answer(provider, api_key, model, query, hits):
    context = "\n\n".join(format_entry(h["entry"]) for h in hits)
    user_msg = (
        f"<knowledge_base_entries>\n{context}\n</knowledge_base_entries>\n\n"
        f"Issue reported by support engineer: {query}"
    )
    if provider == "Groq":
        from groq import Groq

        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=2000,  # reasoning models spend part of this on thinking
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=900,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def template_answer(hits, note=True):
    """Used when no API key is set: grounded answer built straight from the KB."""
    e = hits[0]["entry"]
    steps = "\n".join(f"{i}. {c}" for i, c in enumerate(
        e["diagnostic_checks"], 1))
    return (
        f"**Likely issue:** {e['title']}\n\n**Troubleshooting steps**\n{steps}\n\n"
        f"**Known resolution:** {e['resolution']}\n\n"
        f"**Escalate to:** {e['escalation_team']} if the checks above do not resolve it.\n\n"
        f"**Sources:** {e['id']}"
        + ("\n\n_(Template answer. Add an API key in the sidebar for AI-written responses.)_" if note else "")
    )


INSUFFICIENT = (
    "I couldn't find a close match in the knowledge base, so I don't want to guess a fix.\n\n"
    "Could you share more detail?\n"
    "- Which system / application / service is affected?\n"
    "- The exact error message or code, and when it started\n"
    "- Any recent changes (deployment, config, upgrade)?\n\n"
    "If this is a new type of issue, please escalate to the relevant L2/L3 team "
    "and consider adding it to the knowledge base."
)


# ---------- UI ----------
st.set_page_config(page_title="AI Troubleshooting Assistant",
                   page_icon="🛠️", layout="wide")
st.title("🛠️ AI Troubleshooting Assistant")
st.caption("L1/L2 support helper: describe an issue and get grounded troubleshooting steps from the knowledge base.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "AI API key (optional)", type="password",
        value=os.getenv("GROQ_API_KEY", "") or os.getenv(
            "ANTHROPIC_API_KEY", ""),
        help="Keys starting with 'gsk_' are treated as Groq keys, others as Anthropic keys.",
    )
    api_key = api_key.strip()
    provider = "Groq" if api_key.startswith("gsk_") else "Anthropic"
    if api_key:
        st.caption(f"Provider detected: **{provider}**")
    model = st.text_input(
        "Model", value=DEFAULT_MODELS[provider], key=f"model_{provider}")
    top_k = st.slider("Entries to retrieve", 1, 3, 2)
    threshold = st.slider("Relevance threshold", 0.05, 0.50, 0.12, 0.01,
                          help="Below this score the assistant says the KB is insufficient.")
    st.divider()
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.divider()
    with st.expander(f"Browse knowledge base ({len(load_kb())} entries)"):
        for e in load_kb():
            st.markdown(f"**{e['id']}**: {e['title']}")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def show_sources(sources):
    if not sources:
        return
    st.markdown("**📚 Knowledge-base sources used**")
    for s in sources:
        e = s["entry"]
        with st.expander(f"{e['id']} · {e['title']}  (match {s['score']:.0%})"):
            st.markdown(f"**Category:** {e['category']}")
            st.markdown("**Symptoms:** " + "; ".join(e["symptoms"]))
            st.markdown("**Diagnostic checks:**")
            for i, c in enumerate(e["diagnostic_checks"], 1):
                st.markdown(f"{i}. {c}")
            st.markdown(f"**Resolution:** {e['resolution']}")
            st.markdown(f"**Escalation team:** {e['escalation_team']}")


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("team"):
            st.info(f"🔼 Likely escalation team: **{m['team']}**")
        show_sources(m.get("sources"))

prompt = st.chat_input(
    "Describe the technical issue, e.g. 'The payment API is returning a 500 error'")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        hits = retrieve(prompt, top_k, threshold)
        if not hits:
            reply, team = INSUFFICIENT, None
            st.markdown(reply)
        else:
            team = hits[0]["entry"]["escalation_team"]
            with st.spinner("Analysing knowledge base..."):
                try:
                    reply = ai_answer(
                        provider, api_key, model, prompt, hits) if api_key else template_answer(hits)
                except Exception as err:  # fall back so the demo never breaks
                    reply = template_answer(
                        hits, note=False) + f"\n\n_AI call failed ({type(err).__name__}: {str(err)[:200]}); showing KB answer instead._"
            st.markdown(reply)
            st.info(f"🔼 Likely escalation team: **{team}**")
            show_sources(hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "team": team, "sources": hits}
    )
