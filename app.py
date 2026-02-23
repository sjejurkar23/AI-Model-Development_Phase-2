import streamlit as st
import json
from datetime import datetime
from pathlib import Path
from rag_pipeline_eval_v2 import retrieve, call_llm, build_prompt, extract_citations
import pandas as pd

st.set_page_config(page_title="Research Portal", layout="wide")

# Initialize session state for threads
if 'threads' not in st.session_state:
    st.session_state.threads = []
if 'current_thread' not in st.session_state:
    st.session_state.current_thread = {"queries": []}

# Sidebar navigation
page = st.sidebar.radio("", ["Search", "History", "Evaluation"])

# SEARCH PAGE
if page == "Search":
    st.title("🔍 Research Portal")
    question = st.text_input("Ask a question:")
    
    if st.button("Search") and question:
        # Retrieve and generate
        ids, docs, metas, sims = retrieve(question)
        
        if not ids or max(sims) < 0.35:
            st.warning("⚠️ The requested information is not part of the corpus. Try: rephrasing question or adding more documents.")
        else:
            prompt = build_prompt(question, ids, docs, metas, sims)
            answer = call_llm(prompt)
            cited = extract_citations(answer)
            
            # Save to thread
            st.session_state.current_thread["queries"].append({
                "question": question,
                "answer": answer,
                "sources": [{"chunk_id": cid, "text": doc, "source": meta["source_path"], "sim": sim} 
                           for cid, doc, meta, sim in zip(ids, docs, metas, sims)],
                "cited": list(cited)
            })
            
            # Display answer
            st.subheader("Answer")
            st.write(answer)
            
            # Display sources
            st.subheader("Sources")
            for cid, doc, meta, sim in zip(ids, docs, metas, sims):
                cited_marker = "✅" if cid in cited else ""
                with st.expander(f"{cited_marker} {meta['source_path']} - {cid} (sim: {sim:.3f})"):
                    st.text(doc[:500])
    
    # Export evidence table\\
    if st.button("📥 Generate Evidence Table"):
        rows = []
        for q in st.session_state.current_thread["queries"]:
            answer_text = q["answer"]
            
            # Split by sentences
            import re
            sentences = re.split(r'(?<=[.!?])\s+', answer_text)
            
            for sentence in sentences:
                # Find all citations in this sentence
                citations = re.findall(r'\[CHUNK_ID=([^\]]+)\]', sentence)
                # Remove citation tags to get clean claim
                claim = re.sub(r'\[CHUNK_ID=[^\]]+\]', '', sentence).strip()
                
                # Skip if too short
                if not claim or len(claim) < 20:
                    continue
                
                if citations:
                    # Has citations - add row for each
                    for cite_id in citations:
                        src = next((s for s in q["sources"] if s["chunk_id"] == cite_id), None)
                        if src:
                            # Show FULL chunk text (no truncation)
                            evidence = src["text"].strip()
                            
                            rows.append({
                                "Claim": claim,
                                "Evidence": evidence,  # Full text, no limit
                                "Source": src["source"].split('\\')[-1].split('/')[-1],
                                "Chunk_ID": cite_id,
                                "Confidence": f"{src['sim']:.3f}",
                                "Notes": ""  # Placeholder for user notes
                            })
                else:
                    # No citations - still show the claim with a warning
                    rows.append({
                        "Claim": claim,
                        "Evidence": "⚠️ No citation provided",
                        "Source": "N/A",
                        "Chunk_ID": "N/A",
                        "Confidence": "0.000",
                        "Notes": "This claim has no supporting evidence cited. Consider rephrasing or asking a more specific question."
                    })
        
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df)
            
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "evidence_table.csv", "text/csv")
            key = "evidence_csv_download" 
        else:
            st.warning("No claims found. Ask more questions first!")

# HISTORY PAGE
elif page == "History":
    st.title("📊 Research History")
    for i, q in enumerate(st.session_state.current_thread["queries"]):
        with st.expander(f"Q{i+1}: {q['question'][:80]}..."):
            st.write("**Answer:**", q['answer'])
            st.write("**Sources used:**", len(q['cited']))


# EVALUATION PAGE
elif page == "Evaluation":
    st.title("📈 Evaluation Dashboard")
    
    if not st.session_state.current_thread["queries"]:
        st.info("No queries yet. Go to the Search page and ask some questions first!")
    else:
        queries = st.session_state.current_thread["queries"]
        
        # Calculate metrics for all queries
        from rag_pipeline_eval_v2 import (
            compute_groundedness, 
            compute_citation_precision,
            compute_citation_recall,
            compute_evidence_strength,
            compute_retrieval_coverage,
            compute_overall_confidence
        )
        
        results = []
        for q in queries:
            cited = set(q["cited"])
            ids = [s["chunk_id"] for s in q["sources"]]
            sims = [s["sim"] for s in q["sources"]]
            
            groundedness = compute_groundedness(q["answer"], cited)
            citation_precision = compute_citation_precision(cited, ids)
            citation_recall = compute_citation_recall(cited, ids)
            evidence_strength = compute_evidence_strength(cited, ids, sims)
            retrieval_coverage = compute_retrieval_coverage(sims)
            overall_confidence = compute_overall_confidence(
                groundedness, citation_precision, citation_recall, 
                evidence_strength, retrieval_coverage
            )
            
            results.append({
                "question": q["question"],
                "answer": q["answer"],
                "confidence": overall_confidence["score"],
                "level": overall_confidence["level"],
                "groundedness": groundedness,
                "precision": citation_precision,
                "recall": citation_recall,
                "evidence": evidence_strength,
                "coverage": retrieval_coverage,
            })
        
        # Summary metrics
        st.subheader("📊 Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", len(results))
        with col2:
            avg_conf = sum(r["confidence"] for r in results) / len(results)
            st.metric("Avg Confidence", f"{avg_conf:.2f}")
        with col3:
            high_conf = sum(1 for r in results if r["confidence"] >= 0.70)
            st.metric("High Confidence", f"{high_conf}/{len(results)}")
        with col4:
            avg_groundedness = sum(r["groundedness"] for r in results) / len(results)
            st.metric("Avg Groundedness", f"{avg_groundedness:.2f}")
        
        # Metrics breakdown
        st.subheader("📈 Metrics Breakdown")
        
        metrics_df = pd.DataFrame([
            {
                "Question": r["question"][:50] + "...",
                "Confidence": f"{r['confidence']:.2f}",
                "Level": r["level"],
                "Groundedness": f"{r['groundedness']:.2f}",
                "Precision": f"{r['precision']:.2f}",
                "Recall": f"{r['recall']:.2f}",
            } for r in results
        ])
        st.dataframe(metrics_df)
        
        # Representative examples
        st.subheader("🌟 Representative Examples")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Best answer
            best = max(results, key=lambda x: x["confidence"])
            st.success(f"**Highest Confidence ({best['confidence']:.2f} - {best['level']})**")
            st.write(f"**Q:** {best['question']}")
            st.write(f"**Metrics:**")
            st.write(f"- Groundedness: {best['groundedness']:.2f}")
            st.write(f"- Precision: {best['precision']:.2f}")
            st.write(f"- Recall: {best['recall']:.2f}")
            with st.expander("See answer"):
                st.write(best['answer'])
        
        with col2:
            # Worst answer
            worst = min(results, key=lambda x: x["confidence"])
            st.warning(f"**Lowest Confidence ({worst['confidence']:.2f} - {worst['level']})**")
            st.write(f"**Q:** {worst['question']}")
            st.write(f"**Metrics:**")
            st.write(f"- Groundedness: {worst['groundedness']:.2f}")
            st.write(f"- Precision: {worst['precision']:.2f}")
            st.write(f"- Recall: {worst['recall']:.2f}")
            with st.expander("See answer"):
                st.write(worst['answer'])