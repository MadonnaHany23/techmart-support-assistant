import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st
from rag_core import answer_question, summarize_conversation, PROMPT_VERSION, LOG_DIR

st.set_page_config(page_title="TechMart Support", page_icon="\U0001F6CD\uFE0F")
st.title("TechMart Customer Support")
st.caption("Grounded answers from approved TechMart policy documents")

api_key = st.secrets.get("GEMINI_API_KEY", None)  # falls back to the GEMINI_API_KEY env var inside rag_core

if "history" not in st.session_state:
    st.session_state.history = []  # list of {question, answer, status, category}

question = st.text_input("How can we help?", placeholder="How long does express shipping take?")
if st.button("Ask", type="primary") and question:
    with st.spinner("Searching support policies..."):
        result = answer_question(question, prompt_version=PROMPT_VERSION, api_key=api_key)

    st.caption(f"Category: {result['category']} \u00b7 Status: {result['status']} \u00b7 {result['latency_ms']} ms")
    if result["status"] == "error":
        st.error(result["answer"])
        st.caption(f"{result.get('error_type')}: {result.get('error_detail')}")
    else:
        st.write(result["answer"])
        if result["citations"]:
            st.subheader("Sources")
            for c in result["citations"]:
                st.write(f"- {c['source']} \u2014 page {c['page']} (relevance {c['relevance']})")
        with st.expander("Retrieved chunks"):
            for h in result["retrieved_chunks"]:
                st.markdown(f"**{h['source']} \u2014 page {h['page']}** (relevance {h['score']:.3f})")
                st.write(h["text"])

    st.session_state.history.append({
        "question": question, "answer": result["answer"],
        "status": result["status"], "category": result["category"],
    })

if st.session_state.history:
    with st.expander(f"Conversation so far ({len(st.session_state.history)} turns)"):
        for turn in st.session_state.history:
            st.markdown(f"**Customer:** {turn['question']}")
            st.markdown(f"**Assistant** ({turn['status']}, {turn['category']}): {turn['answer']}")
            st.divider()

    if st.button("Summarize conversation for agent handoff"):
        with st.spinner("Summarizing..."):
            summary_result = summarize_conversation(st.session_state.history, api_key=api_key)
        st.subheader("Conversation summary")
        st.write(summary_result["summary"])
        st.caption(f"{summary_result['turns']} turns \u00b7 {summary_result['latency_ms']} ms")

    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

with st.expander("Recent LLMOps logs (latency, tokens, cost)"):
    log_path = LOG_DIR / "events.jsonl"
    if log_path.exists():
        logs = pd.read_json(log_path, lines=True)
        cols = [c for c in ["timestamp_utc", "status", "category", "prompt_version", "event_type", "latency_ms", "input_tokens", "output_tokens", "cost_usd"] if c in logs.columns]
        st.dataframe(logs[cols].tail(10))
    else:
        st.caption("No requests logged yet.")
