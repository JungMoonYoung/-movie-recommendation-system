# Streamlit Web UI 실행 가이드

## 사전 준비

### 1. ML 모델 학습 (필수)

Hybrid 및 ML-based 추천을 사용하려면 먼저 모델을 학습해야 합니다:

```bash
python src/recommenders/ml_based.py
```

**출력 예시:**
```
============================================================
TRAINING ML-BASED RECOMMENDER (SVD)
============================================================
INFO - Loading training data from database...
INFO - Loaded 800,167 ratings
INFO - Creating rating matrix: 6,040 users × 3,706 movies
INFO - Training SVD model with 50 factors...
INFO - Performing SVD decomposition...
INFO - Training completed in 8.52 seconds
INFO - Model saved to models/svd_model.pkl
============================================================
TRAINING COMPLETED
============================================================
```

### 2. 의존성 설치

```bash
pip install streamlit scipy
```

또는 전체 requirements 설치:

```bash
pip install -r requirements.txt
```

## Streamlit 앱 실행

### 방법 1: 기본 실행

```bash
streamlit run app.py
```

### 방법 2: 포트 지정

```bash
streamlit run app.py --server.port 8501
```

### 방법 3: 외부 접속 허용

```bash
streamlit run app.py --server.address 0.0.0.0
```

## 앱이 자동으로 열립니다

실행하면 자동으로 브라우저가 열리며 다음 URL로 접속됩니다:
- **로컬:** http://localhost:8501
- **네트워크:** http://YOUR_IP:8501

## 앱 사용 방법

### 1. 사이드바 설정

- **User ID 선택**: 드롭다운에서 사용자 선택 (1~6040)
- **Algorithm 선택**: 추천 알고리즘 선택
  - Popularity: 전체 인기 기반
  - Genre-based: 사용자 선호 장르 기반
  - Item-based CF: 유사 영화 기반
  - ML-based (SVD): 행렬 분해 기반 (모델 필요)
  - Hybrid: 모든 알고리즘 결합 (권장)
- **추천 개수**: 5~50개 선택

### 2. 탭 사용

#### 🎯 Recommendations 탭
- 사용자 정보 확인 (성별, 나이, 직업, 평점 통계)
- "Get Recommendations" 버튼 클릭
- 추천 결과 테이블 확인

#### 🔍 Search Movies 탭
- 영화 제목 검색
- 검색 결과에서 "Similar Movies" 버튼 클릭
- 유사 영화 10개 확인

#### 📺 My Watch History 탭
- 사용자의 시청 기록 확인
- 평점별 필터링 (1.0~5.0)
- 최대 50개 영화 표시

## 기능 설명

### 캐싱 (성능 최적화)

앱은 자동으로 다음 데이터를 캐싱합니다:
- 사용자 정보 (1시간)
- 시청 기록 (1시간)
- 영화 검색 결과 (1시간)

캐시를 초기화하려면: **사이드바 상단 > "Clear cache"**

### 에러 처리

- **ML model not found**: 모델 학습 필요 (위 "사전 준비" 참조)
- **User not found**: 유효한 User ID 선택 필요
- **Database connection failed**: .env 파일 확인 및 PostgreSQL 실행 확인

## 문제 해결

### 1. "ML model not found" 오류

```bash
# 모델 학습
python src/recommenders/ml_based.py

# 모델 파일 확인
ls models/svd_model.pkl
```

### 2. "Database connection failed" 오류

```bash
# .env 파일 확인
cat .env

# PostgreSQL 상태 확인
psql -U postgres -d movielens -c "SELECT COUNT(*) FROM movies;"
```

### 3. Streamlit이 설치되지 않음

```bash
pip install streamlit==1.29.0
```

### 4. 포트가 이미 사용 중

```bash
# 다른 포트 사용
streamlit run app.py --server.port 8502
```

## 개발 모드

### Auto-reload 활성화

Streamlit은 기본적으로 파일 변경을 감지하고 자동으로 reload합니다.

### 디버깅

```bash
# verbose 모드
streamlit run app.py --logger.level=debug
```

## 배포

### Streamlit Cloud (무료)

1. GitHub에 프로젝트 푸시
2. https://share.streamlit.io 접속
3. Repository 연결
4. `app.py` 선택
5. "Deploy" 클릭

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# 빌드
docker build -t movie-recommender .

# 실행
docker run -p 8501:8501 movie-recommender
```

## 성능 팁

1. **캐싱 활용**: `@st.cache_data` 데코레이터 사용
2. **쿼리 최적화**: 인덱스 확인
3. **배치 크기 제한**: top_n을 50 이하로 유지
4. **ML 모델 미리 로딩**: 첫 실행 시 로딩 시간 발생

## 추가 기능 아이디어

- [ ] 영화 포스터 이미지 추가 (TMDB API)
- [ ] 추천 이유 설명 추가
- [ ] 사용자 프로필 편집
- [ ] 평점 입력 기능
- [ ] 알고리즘 성능 비교 차트
- [ ] 실시간 추천 업데이트

---

**문제가 발생하면:** GitHub Issues에 문의해주세요!
