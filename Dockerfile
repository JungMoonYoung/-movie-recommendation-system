# Movie Recommendation System - Dockerfile
# Python 3.11 기반 Streamlit 앱

FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 모델 디렉토리 생성
RUN mkdir -p models

# Streamlit 설정 파일 생성
RUN mkdir -p .streamlit && \
    echo "[server]\n\
headless = true\n\
port = 8501\n\
address = \"0.0.0.0\"\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
" > .streamlit/config.toml

# 포트 노출
EXPOSE 8501

# 헬스체크
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Streamlit 앱 실행
CMD ["streamlit", "run", "app.py"]
