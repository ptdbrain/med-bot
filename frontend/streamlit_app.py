from __future__ import annotations

import os

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
RAG_MODES = [
    "naive",
    "hybrid",
    "rerank",
    "parent_child",
    "multi_query",
    "corrective",
    "agentic",
]


st.set_page_config(page_title="MedTutor RAG Copilot", page_icon="🩺", layout="wide")
st.title("MedTutor Multimodal RAG Copilot")
st.caption("Medical education assistant. Not for diagnosis or prescriptions.")

tabs = st.tabs(
    [
        "Chat",
        "Tài liệu",
        "URL",
        "Ảnh",
        "Audio",
        "Video",
        "Đánh giá",
    ]
)

with tabs[0]:
    rag_mode = st.selectbox("RAG mode", RAG_MODES, index=0)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    question = st.text_area("Câu hỏi", placeholder="FAST trong đột quỵ gồm những dấu hiệu nào?")
    if st.button("Gửi", type="primary") and question.strip():
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"question": question, "rag_mode": rag_mode, "top_k": top_k},
            timeout=300,
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
    public_url = st.text_input(
        "URL tài liệu public",
        placeholder="https://example.org/medical-guideline",
    )
    if st.button("Ingest URL") and public_url.strip():
        response = requests.post(
            f"{BACKEND_URL}/ingest/url",
            json={"url": public_url.strip()},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("changed", True):
            st.success("Đã cập nhật nội dung URL")
        else:
            st.info("Nội dung URL không thay đổi, giữ nguyên chỉ mục")
        st.json(payload)

with tabs[3]:
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

with tabs[4]:
    audio_file = st.file_uploader("Tải audio", type=["wav", "mp3", "m4a", "ogg"])
    audio_action = st.radio(
        "Xử lý audio",
        ["Transcribe", "Hỏi RAG", "Ingest bài giảng"],
        horizontal=True,
    )
    audio_rag_mode = st.selectbox(
        "RAG mode cho audio question",
        RAG_MODES,
        index=1,
    )
    audio_top_k = st.slider("Top K cho audio question", min_value=1, max_value=10, value=5)
    if audio_file:
        audio_bytes = audio_file.getvalue()
        st.audio(audio_bytes, format=audio_file.type or "audio/wav")
    if st.button("Thực hiện xử lý audio", type="primary") and audio_file:
        files = {"file": (audio_file.name, audio_bytes, audio_file.type or "audio/wav")}
        if audio_action == "Hỏi RAG":
            endpoint = "/voice/chat"
            params = {"rag_mode": audio_rag_mode, "top_k": audio_top_k}
        elif audio_action == "Ingest bài giảng":
            endpoint = "/voice/ingest"
            params = {}
        else:
            endpoint = "/voice/transcribe"
            params = {}
        response = requests.post(
            f"{BACKEND_URL}{endpoint}",
            files=files,
            params=params,
            timeout=300,
        )
        response.raise_for_status()
        st.json(response.json())

with tabs[5]:
    video_file = st.file_uploader(
        "Tải video bài giảng",
        type=["mp4", "mov", "mkv", "webm"],
    )
    if st.button("Ingest video") and video_file:
        response = requests.post(
            f"{BACKEND_URL}/video/ingest",
            files={
                "file": (
                    video_file.name,
                    video_file.getvalue(),
                    video_file.type or "video/mp4",
                )
            },
            timeout=300,
        )
        response.raise_for_status()
        st.success("Đã ingest transcript video")
        st.json(response.json())

with tabs[6]:
    eval_top_k = st.slider(
        "Top K evaluation",
        min_value=1,
        max_value=20,
        value=5,
    )
    if st.button("Chạy evaluation", type="primary"):
        response = requests.post(
            f"{BACKEND_URL}/eval/run",
            json={
                "dataset_path": "app/evaluation/eval_dataset.json",
                "top_k": eval_top_k,
            },
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        for metric in payload.get("metrics", []):
            st.metric(metric["name"], f"{metric['value'] * 100:.1f}%")
        st.dataframe(payload.get("details", []), use_container_width=True)
