"""
Streamlit Web UI for Movie Recommendation System
영화 추천 시스템 웹 UI (한국어)

Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.recommenders.popularity import get_popular_movies
from src.recommenders.genre import get_genre_based_recommendations
from src.recommenders.similarity import get_similar_movies_for_user, get_similar_movies_for_movie
from src.recommenders.ml_based import get_ml_recommendations
from src.recommenders.hybrid import get_hybrid_recommendations
from src.db_connection import get_sqlalchemy_engine
from sqlalchemy import text

# Page configuration
st.set_page_config(
    page_title="영화 추천 시스템",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .movie-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def get_user_info(user_id: int) -> dict:
    """
    Get user information from database
    사용자 정보 조회
    """
    engine = get_sqlalchemy_engine()
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT
                    u.user_id,
                    u.gender,
                    u.age,
                    u.occupation,
                    COUNT(rt.rating) as total_ratings,
                    ROUND(AVG(rt.rating)::numeric, 2) as avg_rating
                FROM users u
                LEFT JOIN ratings_train rt ON u.user_id = rt.user_id
                WHERE u.user_id = :user_id
                GROUP BY u.user_id, u.gender, u.age, u.occupation
            """)
            result = pd.read_sql(query, conn, params={'user_id': user_id})

            if result.empty:
                return None

            return result.iloc[0].to_dict()
    except Exception as e:
        st.error(f"사용자 정보 조회 오류: {e}")
        return None
    finally:
        engine.dispose()


@st.cache_data(ttl=3600)
def get_all_users() -> list:
    """모든 사용자 ID 조회"""
    engine = get_sqlalchemy_engine()
    try:
        with engine.connect() as conn:
            query = text("SELECT DISTINCT user_id FROM users ORDER BY user_id")
            result = pd.read_sql(query, conn)
            return result['user_id'].tolist()
    except Exception as e:
        st.error(f"사용자 목록 조회 오류: {e}")
        return []
    finally:
        engine.dispose()


@st.cache_data(ttl=3600)
def get_user_watch_history(user_id: int, limit: int = 50) -> pd.DataFrame:
    """
    Get user's watch history
    사용자의 시청 기록 조회
    """
    engine = get_sqlalchemy_engine()
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT
                    m.movie_id,
                    m.title,
                    STRING_AGG(DISTINCT g.genre_name, '|' ORDER BY g.genre_name) as genres,
                    rt.rating,
                    rt.timestamp
                FROM ratings_train rt
                JOIN movies m ON rt.movie_id = m.movie_id
                LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.genre_id
                WHERE rt.user_id = :user_id
                GROUP BY m.movie_id, m.title, rt.rating, rt.timestamp
                ORDER BY rt.rating DESC, rt.timestamp DESC
                LIMIT :limit
            """)
            result = pd.read_sql(query, conn, params={'user_id': user_id, 'limit': limit})
            return result
    except Exception as e:
        st.error(f"시청 기록 조회 오류: {e}")
        return pd.DataFrame()
    finally:
        engine.dispose()


@st.cache_data(ttl=3600)
def search_movies(query_str: str, limit: int = 20) -> pd.DataFrame:
    """
    Search movies by title
    제목으로 영화 검색
    """
    engine = get_sqlalchemy_engine()
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT
                    m.movie_id,
                    m.title,
                    STRING_AGG(DISTINCT g.genre_name, '|' ORDER BY g.genre_name) as genres,
                    COALESCE(AVG(rt.rating), 0) as avg_rating,
                    COUNT(rt.rating) as rating_count
                FROM movies m
                LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.genre_id
                LEFT JOIN ratings_train rt ON m.movie_id = rt.movie_id
                WHERE LOWER(m.title) LIKE LOWER(:query)
                GROUP BY m.movie_id, m.title
                ORDER BY rating_count DESC
                LIMIT :limit
            """)
            result = pd.read_sql(query, conn, params={'query': f'%{query_str}%', 'limit': limit})
            return result
    except Exception as e:
        st.error(f"영화 검색 오류: {e}")
        return pd.DataFrame()
    finally:
        engine.dispose()


def get_recommendations(user_id: int, algorithm: str, n: int) -> pd.DataFrame:
    """
    Get recommendations based on selected algorithm
    선택한 알고리즘으로 추천 생성
    """
    try:
        if algorithm == "인기순 추천":
            return get_popular_movies(n=n, min_ratings=30)
        elif algorithm == "장르별 추천":
            return get_genre_based_recommendations(user_id=user_id, n=n, top_genres=3, min_movie_ratings=30)
        elif algorithm == "유사성 추천":
            return get_similar_movies_for_user(user_id=user_id, n=n, min_rating=4.0)
        elif algorithm == "머신러닝 추천":
            # Check if model exists
            model_path = Path('models/svd_model.pkl')
            if not model_path.exists():
                st.error("❌ ML 모델을 찾을 수 없습니다!")
                st.info("먼저 모델을 학습해주세요: `python src/recommenders/ml_based.py`")
                return pd.DataFrame()
            return get_ml_recommendations(user_id=user_id, n=n)
        elif algorithm == "종합 추천":
            # Check if model exists for hybrid (which uses ML)
            model_path = Path('models/svd_model.pkl')
            if not model_path.exists():
                st.warning("⚠️ ML 모델을 찾을 수 없습니다. 하이브리드는 정확도가 낮아질 수 있습니다.")
            return get_hybrid_recommendations(user_id=user_id, n=n)
        else:
            st.error(f"알 수 없는 알고리즘: {algorithm}")
            return pd.DataFrame()
    except FileNotFoundError as e:
        st.error(f"모델 파일을 찾을 수 없습니다: {e}")
        st.info("먼저 ML 모델을 학습해주세요: `python src/recommenders/ml_based.py`")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"추천 생성 오류: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()


def display_recommendations(df: pd.DataFrame, algorithm: str):
    """
    Display recommendations in a formatted table
    추천 결과를 테이블로 표시
    """
    if df.empty:
        st.warning("추천 결과가 없습니다.")
        return

    st.markdown(f'<div class="sub-header">🎬 추천 영화 Top {len(df)} ({algorithm})</div>',
                unsafe_allow_html=True)

    # Prepare display dataframe
    display_df = df.copy()

    # Format columns based on available data
    if 'title' in display_df.columns:
        display_df = display_df.rename(columns={'title': '영화 제목'})

    if 'genres' in display_df.columns:
        display_df = display_df.rename(columns={'genres': '장르'})

    # Add score column based on algorithm
    score_col = None
    if 'weighted_rating' in df.columns:
        score_col = 'weighted_rating'
        display_df['점수'] = df['weighted_rating'].round(2)
    elif 'combined_score' in df.columns:
        score_col = 'combined_score'
        display_df['점수'] = df['combined_score'].round(2)
    elif 'recommendation_score' in df.columns:
        score_col = 'recommendation_score'
        display_df['점수'] = df['recommendation_score'].round(4)
    elif 'predicted_rating' in df.columns:
        score_col = 'predicted_rating'
        display_df['점수'] = df['predicted_rating'].round(2)
    elif 'hybrid_score' in df.columns:
        score_col = 'hybrid_score'
        display_df['점수'] = df['hybrid_score'].round(4)

    # Add rating info if available
    if 'avg_rating' in df.columns and 'rating_count' in df.columns:
        display_df['평균 평점'] = df['avg_rating'].round(2)
        display_df['평점 수'] = df['rating_count'].astype(int)

    # Select columns to display
    cols_to_show = ['영화 제목']
    if '장르' in display_df.columns:
        cols_to_show.append('장르')
    if score_col:
        cols_to_show.append('점수')
    if '평균 평점' in display_df.columns:
        cols_to_show.extend(['평균 평점', '평점 수'])

    # Filter and display
    display_df = display_df[cols_to_show]
    display_df.index = range(1, len(display_df) + 1)

    st.dataframe(display_df, use_container_width=True)


def main():
    """Main Streamlit app"""

    # Header
    st.markdown('<div class="main-header">🎬 영화 추천 시스템</div>', unsafe_allow_html=True)
    st.markdown("### 다양한 알고리즘을 활용한 개인화 영화 추천")

    # Sidebar
    st.sidebar.title("🎯 추천 설정")

    # Get all users
    all_users = get_all_users()

    if not all_users:
        st.error("데이터베이스에서 사용자를 불러올 수 없습니다.")
        return

    # User selection
    user_id = st.sidebar.selectbox(
        "사용자 ID 선택",
        options=all_users,
        index=0
    )

    # Algorithm selection
    algorithm = st.sidebar.radio(
        "추천 알고리즘 선택",
        options=[
            "인기순 추천",
            "장르별 추천",
            "유사성 추천",
            "머신러닝 추천",
            "종합 추천"
        ],
        index=4  # Default to Hybrid
    )

    # Number of recommendations
    top_n = st.sidebar.slider(
        "추천 영화 개수",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )

    # Get recommendations button
    get_recs_button = st.sidebar.button("🎬 추천 받기", type="primary")

    # Main content area
    tabs = st.tabs(["🎯 영화 추천", "🔍 영화 검색", "📺 내 시청 기록"])

    # Tab 1: Recommendations
    with tabs[0]:
        # Display user info
        user_info = get_user_info(user_id)

        if user_info:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("사용자 ID", user_info['user_id'])
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                gender = "남성" if user_info['gender'] == 'M' else "여성"
                st.metric("성별", gender)
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("나이", user_info['age'])
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("직업", user_info['occupation'])
                st.markdown('</div>', unsafe_allow_html=True)

            col5, col6 = st.columns(2)

            with col5:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("총 평점 수", int(user_info['total_ratings']))
                st.markdown('</div>', unsafe_allow_html=True)

            with col6:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("평균 평점", float(user_info['avg_rating']))
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Generate recommendations
        if get_recs_button:
            with st.spinner(f"{algorithm} 추천을 생성하고 있습니다..."):
                recommendations = get_recommendations(user_id, algorithm, top_n)

                if not recommendations.empty:
                    display_recommendations(recommendations, algorithm)
                else:
                    st.warning("이 사용자에 대한 추천 결과가 없습니다.")

    # Tab 2: Search Movies
    with tabs[1]:
        st.markdown('<div class="sub-header">🔍 영화 검색</div>', unsafe_allow_html=True)

        search_query = st.text_input("영화 제목 입력", placeholder="예: Toy Story")

        if search_query:
            with st.spinner("검색 중..."):
                search_results = search_movies(search_query)

                if not search_results.empty:
                    st.markdown(f"**{len(search_results)}개의 영화를 찾았습니다**")

                    # Display search results
                    for idx, row in search_results.iterrows():
                        with st.expander(f"🎬 {row['title']}"):
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                st.write(f"**장르:** {row['genres']}")
                                st.write(f"**평균 평점:** {row['avg_rating']:.2f} ⭐ ({int(row['rating_count'])}개 평점)")

                            with col2:
                                if st.button(f"비슷한 영화", key=f"similar_{row['movie_id']}"):
                                    with st.spinner("비슷한 영화를 찾고 있습니다..."):
                                        similar = get_similar_movies_for_movie(movie_id=row['movie_id'], n=10)

                                        if not similar.empty:
                                            st.markdown("**비슷한 영화:**")
                                            display_recommendations(similar, "유사도")
                                        else:
                                            st.warning("이 영화에 대한 유사도 데이터가 아직 계산되지 않았습니다. 인기 있는 영화들에 대해서만 유사도가 제공됩니다.")
                else:
                    st.info("영화를 찾을 수 없습니다. 다른 검색어를 시도해보세요.")

    # Tab 3: Watch History
    with tabs[2]:
        st.markdown('<div class="sub-header">📺 내 시청 기록</div>', unsafe_allow_html=True)

        with st.spinner("시청 기록을 불러오는 중..."):
            history = get_user_watch_history(user_id, limit=50)

            if not history.empty:
                st.markdown(f"**총 시청 영화 수: {len(history)}개**")

                # Filter by rating
                rating_filter = st.select_slider(
                    "평점 필터",
                    options=[1.0, 2.0, 3.0, 4.0, 5.0, "전체"],
                    value="전체"
                )

                if rating_filter != "전체":
                    filtered_history = history[history['rating'] >= rating_filter]
                else:
                    filtered_history = history

                st.markdown(f"**{len(filtered_history)}개 영화 표시 중**")

                # Display history
                display_df = filtered_history[['title', 'genres', 'rating']].copy()
                display_df.columns = ['영화 제목', '장르', '내 평점']
                display_df.index = range(1, len(display_df) + 1)

                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("시청 기록이 없습니다.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 알고리즘 설명")

    if algorithm == "인기순 추천":
        st.sidebar.info("""
        **인기순 추천**

        전체 사용자가 높게 평가한 인기 영화를 추천합니다.
        - 평균 평점이 높은 영화
        - 많은 사용자가 평가한 영화
        - 가중 평균 점수로 정렬

        ✅ 바로 사용 가능 (SQL)
        """)
    elif algorithm == "장르별 추천":
        st.sidebar.info("""
        **장르별 추천**

        사용자가 선호하는 장르의 영화를 추천합니다.
        - 사용자 평점 이력 분석
        - 상위 3개 선호 장르 추출
        - 해당 장르의 인기 영화 추천

        ✅ 바로 사용 가능 (SQL)
        """)
    elif algorithm == "유사성 추천":
        st.sidebar.info("""
        **유사성 추천**

        사용자가 좋아한 영화와 비슷한 영화를 추천합니다.
        - 영화 간 유사도 계산
        - 4점 이상 준 영화 기반
        - 코사인 유사도 사용

        ✅ 바로 사용 가능 (SQL + 유사도)
        """)
    elif algorithm == "머신러닝 추천":
        st.sidebar.info("""
        **머신러닝 추천 (SVD)**

        행렬 분해로 사용자 취향을 학습하여 예측합니다.
        - Matrix Factorization
        - SVD 알고리즘 (50 factors)
        - 예측 평점 계산

        ⚠️ 모델 학습 필요
        """)
    elif algorithm == "종합 추천":
        st.sidebar.info("""
        **종합 추천**

        - 인기도: 10%
        - 장르: 20%
        - 유사성: 30%
        - 머신러닝: 40%
        """)


if __name__ == "__main__":
    main()
