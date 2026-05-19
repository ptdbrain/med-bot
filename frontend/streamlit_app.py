from __future__ import annotations

import os

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
RAG_MODES = ["naive", "hybrid", "rerank", "parent_child", "multi_query", "corrective", "agentic"]


st.set_page_config(page_title="MedTutor RAG Copilot", page_icon="🩺", layout="wide")
st.title("MedTutor Multimodal RAG Copilot")
st.caption("Medical education assistant. Not for diagnosis or prescriptions.")

tabs = st.tabs(["Chat", "Upload tài liệu", "Upload ảnh", "Upload audio", "Ingest mẫu"])

with tabs[0]:
    rag_mode = st.selectbox("RAG mode", RAG_MODES, index=0)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    question = st.text_area("Câu hỏi", placeholder="FAST trong đột quỵ gồm những dấu hiệu nào?")
    if st.button("Gửi", type="primary") and question.strip():
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"question": question, "rag_mode": rag_mode, "top_k": top_k},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        st.subheader("Answer")
        st.write(payload["answer"])
        if payload.get("safety_warning"):
            st.warning(payload["safety_warning"])
        st.subheader("Sources")
        st.json(payload.get("sources", []))
        st.subheader("Retrieved chunks")
        st.json(payload.get("retrieved_chunks", []))

with tabs[1]:
    doc_file = st.file_uploader("Tải lên .txt/.md/.pdf/.docx/.html", type=["txt", "md", "pdf", "docx", "html", "htm"])
    if st.button("Ingest tài liệu") and doc_file:
        response = requests.post(
            f"{BACKEND_URL}/ingest/file",
            files={"file": (doc_file.name, doc_file.getvalue(), doc_file.type or "application/octet-stream")},
            timeout=120,
        )
        response.raise_for_status()
        st.success("Đã ingest tài liệu")
        st.json(response.json())

with tabs[2]:
    image_file = st.file_uploader("Tải ảnh học tập", type=["png", "jpg", "jpeg"])
    if st.button("Ingest ảnh") and image_file:
        response = requests.post(
            f"{BACKEND_URL}/image/ingest",
            files={"file": (image_file.name, image_file.getvalue(), image_file.type or "image/png")},
            timeout=120,
        )
        response.raise_for_status()
        st.success("Đã ingest ảnh bằng mock OCR/caption")
        st.json(response.json())

with tabs[3]:
    audio_file = st.file_uploader("Tải audio", type=["wav", "mp3", "m4a", "ogg"])
    if st.button("Transcribe audio") and audio_file:
        response = requests.post(
            f"{BACKEND_URL}/voice/transcribe",
            files={"file": (audio_file.name, audio_file.getvalue(), audio_file.type or "audio/wav")},
            timeout=120,
        )
        response.raise_for_status()
        st.json(response.json())

with tabs[4]:
    st.write("Chạy script sample từ terminal:")
    st.code("python scripts/ingest_sample_data.py", language="bash")
