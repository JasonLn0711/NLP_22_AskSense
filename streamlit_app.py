import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

import streamlit as st
from collections import defaultdict
from search_engine import SemanticSearchEngine

# 模型快取
@st.cache_resource
def get_engine():
    return SemanticSearchEngine('data/scam_dataset_tw_54500.csv', risk_threshold=0.7)

# 讀取故事庫
@st.cache_data(ttl=600)
def load_stories():
    df = pd.read_csv('data/scam_story.csv', dtype=str)
    df.fillna('', inplace=True)
    return df

engine = get_engine()
stories_df = load_stories()

# 快取查詢結果，避免反覆計算相同 query
@st.cache_data(ttl=600)
def cached_search(query: str, top_k: int = 10):
    return engine.search(query, top_k=top_k)

@st.cache_data(ttl=600)
def cached_analysis(query: str):
    return engine.sentence_analysis(query)

@st.cache_data(ttl=600)
def cached_highlight(query: str):
    return engine.highlight_keywords(query)

# 頁面設定
st.set_page_config(
    page_title="AskSense 詐騙檢測工具",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar: Developer Info & Expertise
with st.sidebar:
    st.header("專案簡易說明")
    st.markdown(
        """
        這個工具讓你可以貼上任意文字，快速標示出可能有風險或問題的句子。  
        它會記憶你最近的檢查結果，重複檢查相同文字，速度會更快。  
        非常適合想要輕鬆找出自己文字中潛在安全或內容風險的任何人！
        """
    )
    st.markdown("---")
    st.header("開發者資訊")
    st.markdown(
        """
        **背景**
        - 國立陽明交通大學 資訊工程學系
        - AI 與資安研究

        **專長**
        - 自然語言處理 (NLP)
        - 深度學習模型開發 (SBERT)
        - 系統資安與防護策略
        """
    )
    st.markdown("© 2025 JN AskSense. All rights reserved.")

# 主區域：標題與輸入
st.title('🔍 AskSense 詐騙文字檢測 & 語意搜尋工具')
st.markdown('專為台灣用戶設計，快速且安全地揪出潛在詐騙訊息。')

# 限制最大輸入長度以防過度運算
query = st.text_area('📥 貼上要檢測的文字 (最多1000字)：', height=200, max_chars=1000)

if st.button('開始分析') and query:
    with st.spinner('分析中，請稍候…'):
        # 快取查詢與高階計算
        top10 = cached_search(query)
        analysis = cached_analysis(query)
        hits, highlighted = cached_highlight(query)
        # 彙整 Top3 詐騙類型
        type_scores = defaultdict(list)
        for r in top10:
            type_scores[r['type']].append(r['score'])
        top_types = sorted(
            ((t, max(scores)) for t, scores in type_scores.items()),
            key=lambda x: x[1], reverse=True
        )[:3]

    # 顯示最可能 Top3 類型
    st.subheader('🚩 最可能的 3 種詐騙類型')
    for t, sc in top_types:
        st.markdown(f"- **{t}** (最高相似度: {sc:.4f})")

    # 顯示關鍵詞和標示
    if hits:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('🔑 命中關鍵詞')
            st.write(', '.join(hits))
        with col2:
            st.subheader('🖍 關鍵詞標示')
            st.write(highlighted)

    # 逐句風險分析 (綠/黃/紅)
    st.subheader('🔎 逐句風險分析')
    for item in analysis:
        icon = {'綠':'🟢','黃':'🟡','紅':'🔴'}[item['level']]
        msg = f"{icon} [{item['level']}] {item['sentence']} (比對率: {item['score']:.4f})"
        if item['level'] == '紅':
            st.error(msg)
        elif item['level'] == '黃':
            st.warning(msg)
        else:
            st.success(msg)

    # 類似詐騙案例
    st.subheader('💡 類似詐騙案例')
    main_type = top_types[0][0] if top_types else None
    if main_type:
        st.markdown(f"**{main_type}** 案例示範：")
        examples = stories_df[stories_df['type'] == main_type]['Content'].tolist()
        for ex in examples[:3]:  # 前3個案例
            st.write(f"- {ex}")

    st.markdown('---')
    st.markdown('更多資源： [165 防詐達人](https://165.npa.gov.tw)')