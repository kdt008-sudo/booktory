import streamlit as st
import os
import json
import pandas as pd
import pickle

# 데이터 폴더 경로
DATA_DIR = "data"

# 데이터 파일 목록 가져오기
def get_json_files(data_dir):
    return [f for f in os.listdir(data_dir) if f.endswith('.json')]

# JSON 파일을 DataFrame으로 변환
def json_to_df(filepath):
    with open(filepath, encoding='utf-8-sig') as f:
        data = json.load(f)
    # paragraphInfo를 펼쳐서 각 문단별로 행 생성
    rows = []
    for p in data.get('paragraphInfo', []):
        row = {
            'title': data.get('title'),
            'author': data.get('author'),
            'illustrator': data.get('illustrator'),
            'isbn': data.get('isbn'),
            'readAge': data.get('readAge'),
            'publishedYear': data.get('publishedYear'),
            'publisher': data.get('publisher'),
            'classification': data.get('classification'),
            'srcTextID': p.get('srcTextID'),
            'srcText': p.get('srcText'),
            'srcPage': p.get('srcPage'),
            'srcSentenceEA': p.get('srcSentenceEA'),
            'srcWordEA': p.get('srcWordEA'),
        }
        rows.append(row)
    return pd.DataFrame(rows)

# 모든 JSON 파일을 하나의 DataFrame으로 통합
def load_all_data(data_dir):
    files = get_json_files(data_dir)
    dfs = []
    for f in files:
        path = os.path.join(data_dir, f)
        dfs.append(json_to_df(path))
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()


# Streamlit 대시보드
st.set_page_config(page_title="Booktory 데이터 대시보드", layout="wide")
st.title("Booktory 데이터 대시보드")

st.write("데이터 폴더의 모든 책 정보를 시각화합니다.")

df = load_all_data(DATA_DIR)


# Sidebar 메뉴
st.sidebar.header("메뉴")
menu = st.sidebar.radio("페이지 선택", ["대시보드", "퀴즈내기"])

if menu == "대시보드":
    st.sidebar.header("필터")
    title_filter = st.sidebar.multiselect("책 제목", sorted(df['title'].unique()) if not df.empty else [], default=None)
    author_filter = st.sidebar.multiselect("저자", sorted(df['author'].unique()) if not df.empty else [], default=None)
    age_filter = st.sidebar.multiselect("연령대", sorted(df['readAge'].unique()) if not df.empty else [], default=None)
    class_filter = st.sidebar.multiselect("분류", sorted(df['classification'].unique()) if not df.empty else [], default=None)

    # 필터 적용
    filtered_df = df.copy()
    if title_filter:
        filtered_df = filtered_df[filtered_df['title'].isin(title_filter)]
    if author_filter:
        filtered_df = filtered_df[filtered_df['author'].isin(author_filter)]
    if age_filter:
        filtered_df = filtered_df[filtered_df['readAge'].isin(age_filter)]
    if class_filter:
        filtered_df = filtered_df[filtered_df['classification'].isin(class_filter)]

    if filtered_df.empty:
        st.warning("필터 결과 데이터가 없습니다.")
    else:
        st.subheader("📚 데이터 테이블")
        st.dataframe(filtered_df)
        st.subheader("📊 기본 통계")
        st.write(f"총 문단 수: {len(filtered_df)}")
        st.write(f"총 책 수: {filtered_df['isbn'].nunique()}")
        st.write("연령별 책 분포:")
        st.bar_chart(filtered_df['readAge'].value_counts())
        st.write("출판년도별 책 분포:")
        st.bar_chart(filtered_df['publishedYear'].value_counts().sort_index())
        st.write("분류별 책 분포:")
        st.bar_chart(filtered_df['classification'].value_counts())

        st.subheader("📈 추가 분석")
        # 평균 문단 수(책별)
        para_count = filtered_df.groupby('isbn')['srcTextID'].count()
        st.write(f"책별 평균 문단 수: {para_count.mean():.2f}")
        # 평균 단어 수(문단별)
        st.write(f"문단별 평균 단어 수: {filtered_df['srcWordEA'].mean():.2f}")
        # 저자별 책 수
        st.write("저자별 책 수:")
        st.bar_chart(filtered_df.groupby('author')['isbn'].nunique())
        # 출판년도별 평균 문단 수
        st.write("출판년도별 평균 문단 수:")
        st.line_chart(filtered_df.groupby('publishedYear')['srcTextID'].count())
        # paragraph 텍스트 길이 분포
        filtered_df['text_length'] = filtered_df['srcText'].str.len()
        st.write("문단 텍스트 길이 분포:")
        st.histogram(filtered_df['text_length'])

elif menu == "퀴즈내기":
    st.header("AI 동화 퀴즈 생성기")
    st.write("아래 옵션을 선택하면 AI가 퀴즈를 생성합니다.")

    # 모델 데이터 로드
    model_path = os.path.join("model", "training_docs_quiz.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            quiz_database = pickle.load(f)
        # 리스트 구조라면 dict로 변환
        if isinstance(quiz_database, list):
            quiz_dict = {}
            for item in quiz_database:
                age = item.get('readAge')
                cls = item.get('classification')
                title = item.get('title')
                qa_pairs = item.get('queAnsPairInfo', [])
                if age and cls and title:
                    quiz_dict.setdefault(age, {}).setdefault(cls, {}).setdefault(title, [])
                    for qa in qa_pairs:
                        if qa.get('question') and qa.get('ansM1'):
                            quiz_dict[age][cls][title].append({'question': qa['question'], 'answer': qa['ansM1']})
            quiz_database = quiz_dict
        elif not isinstance(quiz_database, dict):
            st.error("알 수 없는 모델 데이터 구조입니다.")
            st.stop()
    else:
        st.error("AI 퀴즈 모델 파일이 없습니다.")
        st.stop()

    # 연령, 분류, 책 제목 선택
    available_ages = list(quiz_database.keys())
    selected_age = st.selectbox("1단계: 연령 선택", available_ages)
    available_cls = list(quiz_database[selected_age].keys())
    selected_cls = st.selectbox("2단계: 분류 선택", available_cls)
    available_titles = list(quiz_database[selected_age][selected_cls].keys())
    selected_title = st.selectbox("3단계: 책 제목 선택", available_titles)

    # 퀴즈 생성 버튼
    if st.button("퀴즈 생성하기"):
        questions = quiz_database[selected_age][selected_cls][selected_title]
        if not questions:
            st.warning("해당 조건에 퀴즈가 없습니다.")
        else:
            st.subheader(f"퀴즈: {selected_title}")
            for i, q in enumerate(questions, 1):
                st.write(f"{i}. {q['question']} (정답: {q['answer']})")
