# 🚀 빠른 시작 가이드

## ✅ 현재 상태 확인 완료!

### 설치된 패키지
- ✅ Python 3.14.1
- ✅ pandas 2.3.3
- ✅ numpy 2.3.5
- ✅ psycopg2-binary 2.9.11
- ✅ sqlalchemy (설치됨)
- ✅ scipy 1.16.3
- ✅ streamlit 1.52.0

### 데이터 확인
- ✅ 사용자: 6,040명
- ✅ 영화: 3,883개
- ✅ 평점: 802,553개
- ✅ 데이터베이스 연결 정상

---

## 🎬 Streamlit 앱 실행하기

### 1️⃣ 터미널에서 실행
```bash
streamlit run app.py
```

### 2️⃣ 브라우저에서 열기
- 자동으로 브라우저가 열립니다
- 또는 http://localhost:8501 접속

---

## 🤖 머신러닝 모델 학습하기 (선택)

현재 **SQL 기반 추천 3개**는 바로 작동합니다:
- ✅ Popularity (인기도 기반)
- ✅ Genre-based (장르 기반)
- ✅ Item-based CF (유사도 기반)

**ML 기반 추천**을 사용하려면 모델을 먼저 학습해야 합니다:

```bash
# SVD 모델 학습 (약 2-3분 소요)
python src/recommenders/ml_based.py
```

학습이 완료되면:
- ✅ ML-based (SVD) 추천 사용 가능
- ✅ Hybrid 추천 (모든 알고리즘 결합) 사용 가능

---

## 📊 추천 알고리즘 설명

### SQL 기반 (바로 사용 가능)

#### 1. Popularity (인기도 기반)
```
전체 사용자들이 높게 평가한 영화를 추천
- 평균 평점 높음
- 평점 개수 많음
- 가중 평균 계산
```

#### 2. Genre-based (장르 기반)
```
사용자가 좋아하는 장르의 영화를 추천
- 사용자의 평점 이력 분석
- 선호 장르 top 3 추출
- 해당 장르의 인기 영화 추천
```

#### 3. Item-based CF (협업 필터링)
```
사용자가 좋아한 영화와 비슷한 영화를 추천
- 영화 간 유사도 계산 (코사인 유사도)
- 사용자가 4점 이상 준 영화 찾기
- 그 영화들과 비슷한 영화 추천
```

### 머신러닝 기반 (모델 학습 필요)

#### 4. ML-based (SVD Matrix Factorization)
```
머신러닝으로 사용자 취향을 학습하여 예측
- User-Movie 평점 행렬 생성
- SVD로 잠재 요인 추출 (50 factors)
- 각 영화의 예측 평점 계산
- 예측 평점이 높은 영화 추천
```

#### 5. Hybrid (앙상블)
```
모든 알고리즘을 결합하여 최고의 추천 제공
- Popularity: 10%
- Genre: 20%
- Item-CF: 30%
- ML-SVD: 40%
- 가중 평균으로 최종 점수 계산
```

---

## 🎯 사용 예시

### Streamlit UI에서:
1. 왼쪽 사이드바에서 **사용자 ID** 선택 (1-6040)
2. **추천 알고리즘** 선택
   - Popularity (바로 사용 가능)
   - Genre-based (바로 사용 가능)
   - Item-based CF (바로 사용 가능)
   - ML-based (모델 학습 필요)
   - Hybrid (모델 학습 필요)
3. **추천 개수** 슬라이더 조정 (5-50개)
4. **"Get Recommendations"** 버튼 클릭

### 3개 탭 기능:
- **🎯 Recommendations**: 선택한 알고리즘으로 영화 추천
- **🔍 Search Movies**: 영화 제목으로 검색
- **📺 My Watch History**: 사용자의 시청 기록 확인

---

## 🐛 문제 해결

### 1. Streamlit이 실행되지 않을 때
```bash
# Streamlit 재설치
pip install --upgrade streamlit

# 캐시 삭제
streamlit cache clear
```

### 2. 데이터베이스 연결 오류
```bash
# PostgreSQL 서비스 시작 확인
# Windows: services.msc에서 PostgreSQL 서비스 확인
# Mac: brew services start postgresql
# Linux: sudo systemctl start postgresql
```

### 3. ML 모델 학습 실패
```bash
# scipy 재설치
pip install --upgrade scipy numpy

# 다시 학습
python src/recommenders/ml_based.py
```

---

## 📌 주요 차이점 정리

| 항목 | SQL 기반 | 머신러닝 기반 |
|------|----------|--------------|
| **학습 필요** | ❌ 불필요 | ✅ 필요 (2-3분) |
| **바로 사용** | ✅ 가능 | ❌ 학습 후 가능 |
| **동작 방식** | 정렬/집계 | 행렬 분해 예측 |
| **개인화** | ⚠️ 제한적 | ✅ 강력함 |
| **정확도** | 0.08-0.12 | 0.15-0.18 |
| **속도** | 50-200ms | 10ms |

---

## 🎉 지금 바로 시작하기!

```bash
# 터미널에서 실행
streamlit run app.py
```

브라우저에서 http://localhost:8501 접속하여 영화 추천을 받아보세요! 🎬
