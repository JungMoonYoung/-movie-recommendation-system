# Streamlit Cloud 배포 가이드

## 📋 사전 준비

### 1. PostgreSQL 데이터베이스 호스팅

Streamlit Cloud는 데이터베이스를 제공하지 않으므로, 별도로 PostgreSQL을 호스팅해야 합니다.

#### 추천: Neon.tech (무료)

**Neon.tech**는 무료로 PostgreSQL을 호스팅할 수 있는 서비스입니다.

1. https://neon.tech 접속
2. "Sign Up" 클릭 (GitHub 계정으로 로그인 가능)
3. 새 프로젝트 생성
   - Project name: `movielens-db` (원하는 이름)
   - Region: US East (Ohio) 또는 가까운 지역 선택
4. 연결 정보 확인
   ```
   Host: ep-xxxx-xxxx.us-east-2.aws.neon.tech
   Database: neondb
   User: your-username
   Password: your-password
   Port: 5432
   ```

#### 대안: Supabase (무료)

1. https://supabase.com 접속
2. 새 프로젝트 생성
3. Database 설정에서 연결 정보 확인

#### 대안: ElephantSQL (무료 - 20MB 제한)

1. https://www.elephantsql.com 접속
2. "Get a managed database today" 클릭
3. Tiny Turtle (무료) 플랜 선택

### 2. 데이터베이스 마이그레이션

로컬 데이터베이스의 데이터를 클라우드로 옮겨야 합니다.

#### 방법 1: pg_dump 사용 (권장)

```bash
# 1. 로컬 데이터베이스 덤프
pg_dump -U postgres -d movielens -f movielens_backup.sql

# 2. 클라우드 데이터베이스로 복원
psql -h ep-xxxx-xxxx.us-east-2.aws.neon.tech -U your-username -d neondb -f movielens_backup.sql
```

#### 방법 2: setup_db.py 수정하여 클라우드에 직접 실행

```python
# .env 파일을 클라우드 DB 정보로 수정 후
python setup_db.py
```

### 3. ML 모델 파일

`models/svd_model.pkl` 파일을 GitHub에 포함시켜야 합니다.

**주의:** 모델 파일이 100MB를 초과하면 Git LFS를 사용해야 합니다.

```bash
# 모델 파일 크기 확인
ls -lh models/svd_model.pkl

# 100MB 이하면 그대로 커밋
# 100MB 이상이면 Git LFS 사용
git lfs install
git lfs track "*.pkl"
git add .gitattributes
```

---

## 🚀 Streamlit Cloud 배포 단계

### Step 1: GitHub 저장소 생성

1. GitHub 접속 (https://github.com)
2. "New repository" 클릭
3. Repository 설정:
   - Repository name: `movie-recommendation-system`
   - Description: `MovieLens 1M 영화 추천 시스템`
   - Public 또는 Private 선택
   - **"Add a README file" 체크 해제** (이미 있음)
4. "Create repository" 클릭

### Step 2: 로컬 프로젝트를 GitHub에 푸시

현재 디렉토리에서 다음 명령어를 실행하세요:

```bash
# Git 초기화
git init

# 모든 파일 추가
git add .

# 첫 커밋
git commit -m "Initial commit: MovieLens Recommendation System"

# GitHub 원격 저장소 연결 (YOUR_USERNAME을 본인 GitHub ID로 변경)
git remote add origin https://github.com/YOUR_USERNAME/movie-recommendation-system.git

# main 브랜치로 푸시
git branch -M main
git push -u origin main
```

**중요:** `.streamlit/secrets.toml` 파일은 `.gitignore`에 포함되어 있어 GitHub에 업로드되지 않습니다.

### Step 3: Streamlit Cloud에 배포

1. https://share.streamlit.io 접속
2. "Sign in with GitHub" 클릭
3. 오른쪽 상단 "New app" 클릭
4. 배포 설정:
   - **Repository:** `YOUR_USERNAME/movie-recommendation-system`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. "Advanced settings" 클릭
6. **Python version:** `3.11` 선택
7. **Secrets** 섹션에 다음 내용 입력:

```toml
[database]
DB_HOST = "ep-xxxx-xxxx.us-east-2.aws.neon.tech"
DB_PORT = "5432"
DB_NAME = "neondb"
DB_USER = "your-username"
DB_PASSWORD = "your-password"
```

**중요:** 위 정보는 Step 1에서 생성한 Neon.tech 데이터베이스 정보를 입력하세요.

8. "Deploy!" 클릭

### Step 4: 배포 대기

- 배포 프로세스가 시작되며, 로그를 실시간으로 볼 수 있습니다.
- 첫 배포는 5~10분 정도 소요됩니다.
- 완료되면 자동으로 URL이 생성됩니다:
  - 예: `https://YOUR_APP_NAME.streamlit.app`

---

## ✅ 배포 후 확인사항

### 1. 앱 접속 테스트

생성된 URL로 접속하여 다음을 확인하세요:

- [ ] 앱이 정상적으로 로드되는지
- [ ] 사용자 목록이 표시되는지
- [ ] 각 추천 알고리즘이 작동하는지
  - [ ] 인기순 추천
  - [ ] 장르별 추천
  - [ ] 유사성 추천
  - [ ] 머신러닝 추천 (모델 파일 확인)
  - [ ] 종합 추천
- [ ] 영화 검색 기능이 작동하는지
- [ ] 시청 기록이 표시되는지

### 2. 에러 확인

만약 에러가 발생하면:

1. Streamlit Cloud 대시보드에서 "Manage app" 클릭
2. "Logs" 탭에서 에러 메시지 확인
3. 주요 에러 원인:
   - **Database connection failed:** Secrets 설정이 올바른지 확인
   - **Module not found:** `requirements.txt` 파일 확인
   - **File not found:** 모델 파일이 GitHub에 포함되었는지 확인

### 3. 성능 최적화

Streamlit Cloud 무료 플랜의 제한사항:

- **메모리:** 1GB
- **CPU:** 공유 CPU
- **Sleep 모드:** 7일 동안 접속이 없으면 자동으로 sleep 상태가 됩니다.

**최적화 팁:**
- 캐싱 적극 활용 (`@st.cache_data`)
- 추천 영화 개수를 제한 (기본값: 10개)
- 불필요한 로그 제거

---

## 🔄 업데이트 방법

코드를 수정한 후 배포된 앱을 업데이트하는 방법:

```bash
# 파일 수정 후
git add .
git commit -m "Update: 설명"
git push origin main
```

Streamlit Cloud는 GitHub의 변경사항을 자동으로 감지하고 재배포합니다.

---

## 🛠️ 문제 해결

### 문제 1: Database connection failed

**원인:** Secrets 설정이 잘못되었거나, 데이터베이스가 접속을 허용하지 않음

**해결:**
1. Streamlit Cloud의 Secrets 설정 확인
2. Neon.tech에서 IP 화이트리스트 확인 (기본적으로 모든 IP 허용)
3. 데이터베이스 연결 정보 재확인

### 문제 2: ModuleNotFoundError

**원인:** `requirements.txt`에 패키지가 누락됨

**해결:**
```bash
# requirements.txt에 누락된 패키지 추가
echo "missing-package==1.0.0" >> requirements.txt
git add requirements.txt
git commit -m "Add missing package"
git push
```

### 문제 3: ML model not found

**원인:** 모델 파일이 GitHub에 업로드되지 않음

**해결:**
```bash
# .gitignore에서 models/*.pkl 제거 (주석 처리된 상태인지 확인)
git add models/svd_model.pkl
git commit -m "Add ML model"
git push
```

### 문제 4: Out of memory

**원인:** 데이터 로딩 시 메모리 초과

**해결:**
- 추천 후보 풀 크기 줄이기 (`candidate_pool_size` 조정)
- 캐싱 최적화
- 불필요한 데이터 로딩 제거

---

## 📊 모니터링

Streamlit Cloud 대시보드에서 확인 가능한 정보:

- **App URL:** 공개 URL
- **Logs:** 실시간 로그
- **Resources:** CPU, 메모리 사용량
- **Settings:** 재배포, 삭제 등

---

## 💡 추가 기능 아이디어

배포 후 추가할 수 있는 기능:

1. **Custom Domain:** 본인의 도메인 연결
2. **Analytics:** Google Analytics 추가
3. **Authentication:** 로그인 기능 추가
4. **API:** REST API 엔드포인트 제공
5. **A/B Testing:** 알고리즘 성능 비교

---

## 🎉 배포 완료!

축하합니다! 이제 전 세계 어디서나 접속 가능한 영화 추천 시스템이 완성되었습니다.

생성된 URL을 친구들과 공유해보세요! 🎬

---

## 📞 지원

문제가 발생하면:
- Streamlit Community Forum: https://discuss.streamlit.io
- GitHub Issues: Repository의 Issues 탭
