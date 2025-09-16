# /mnt/g/tool_ui/streamlit_vqe_test/Dockerfile
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# (선택) 빌드 속도 개선용 OS 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 복사 → 레이어 캐시
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# 소스는 런타임에 볼륨 마운트(=개발 중 핫리로드)
# COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
