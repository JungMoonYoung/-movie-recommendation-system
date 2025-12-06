# 🚀 빠른 배포 가이드 (5단계)

## 1️⃣ 데이터베이스 준비 (5분)

### Neon.tech 가입 및 DB 생성
1. https://neon.tech 접속 → GitHub로 로그인
2. "Create Project" 클릭
   - Name: `movielens-db`
   - Region: US East (Ohio)
3. 연결 정보 복사 (나중에 사용)
   ```
   Host: ep-xxxx-xxxx.us-east-2.aws.neon.tech
   Database: neondb
   User: username
   Password: password
   ```

### 로컬 DB 데이터 마이그레이션
```bash
# 로컬 DB 백업
pg_dump -U postgres -d movielens > movielens_backup.sql

# 클라우드로 복원
psql -h ep-xxxx-xxxx.us-east-2.aws.neon.tech -U username -d neondb < movielens_backup.sql
```

---

## 2️⃣ GitHub 저장소 생성 (3분)

1. https://github.com/new 접속
2. Repository name: `movie-recommendation-system`
3. Public 선택
4. **"Add README" 체크 해제**
5. "Create repository" 클릭

---

## 3️⃣ 코드를 GitHub에 업로드 (2분)

터미널에서 다음 명령어 실행:

```bash
# Git 초기화
git init
git add .
git commit -m "Initial commit"

# GitHub 연결 (YOUR_USERNAME을 본인 계정으로 변경)
git remote add origin https://github.com/YOUR_USERNAME/movie-recommendation-system.git

# 푸시
git branch -M main
git push -u origin main
```

---

## 4️⃣ Streamlit Cloud 배포 (3분)

1. https://share.streamlit.io 접속
2. "Sign in with GitHub" 클릭
3. "New app" 클릭
4. 설정:
   - Repository: `YOUR_USERNAME/movie-recommendation-system`
   - Branch: `main`
   - Main file: `app.py`

5. **"Advanced settings"** 클릭
6. **Secrets** 입력 (Step 1의 정보 사용):

```toml
[database]
DB_HOST = "ep-xxxx-xxxx.us-east-2.aws.neon.tech"
DB_PORT = "5432"
DB_NAME = "neondb"
DB_USER = "username"
DB_PASSWORD = "password"
```

7. "Deploy!" 클릭

---

## 5️⃣ 완료! (5~10분 대기)

배포가 완료되면 URL이 생성됩니다:
- 예: `https://your-app.streamlit.app`

접속해서 테스트하세요! 🎉

---

## ⚠️ 문제 발생 시

### 에러: Database connection failed
→ Secrets 설정이 정확한지 확인

### 에러: Module not found
→ `requirements.txt` 파일이 GitHub에 있는지 확인

### 에러: File not found (models/svd_model.pkl)
→ 모델 파일이 GitHub에 업로드되었는지 확인
```bash
git add models/svd_model.pkl
git commit -m "Add ML model"
git push
```

---

## 📚 자세한 가이드

더 자세한 설명은 `DEPLOYMENT_GUIDE.md` 파일을 참고하세요!
