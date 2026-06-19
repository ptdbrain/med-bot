FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG INSTALL_AUDIO=false
ARG INSTALL_RAG=false
ARG INSTALL_FRONTEND=false

COPY requirements-backend.txt requirements-audio.txt requirements-rag.txt ./
RUN pip install --no-cache-dir -r requirements-backend.txt
RUN if [ "$INSTALL_AUDIO" = "true" ]; then pip install --no-cache-dir -r requirements-audio.txt; fi
RUN if [ "$INSTALL_RAG" = "true" ]; then pip install --no-cache-dir -r requirements-rag.txt; fi
RUN if [ "$INSTALL_FRONTEND" = "true" ]; then pip install --no-cache-dir "streamlit>=1.34.0"; fi

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
