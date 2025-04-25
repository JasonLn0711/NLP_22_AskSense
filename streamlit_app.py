import os
import zipfile
import gdown

import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

import streamlit as st
# 頁面設定
st.set_page_config(
    page_title="AskSense 詐騙檢測工具",
    layout="wide",
    initial_sidebar_state="expanded"
)
from collections import defaultdict
from search_engine import SemanticSearchEngine
import pandas as pd

MODEL_DIR = "models/paraphrase-multilingual-MiniLM-L12-v2"
ZIP_PATH  = "models/miniLM.zip"
GDRIVE_ID = st.secrets["gdrive_model_id"]
GDRIVE_URL = f"https://drive.google.com/file/d/{GDRIVE_ID}/view"

# 模型快取
@st.cache_resource
def load_model():
    # 1) download once per container
    if not os.path.exists(MODEL_DIR):
        # fetch from Drive
        success = gdown.download(
          id=GDRIVE_URL,
          output=ZIP_PATH,
          quiet=False
        )
        if success is None:
            raise RuntimeError("Model download failed: please check Drive ID and sharing settings.")
        # unzip into MODEL_DIR
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall("models/")
    # 2) load from local
    return SentenceTransformer(MODEL_DIR, local_files_only=True)

st.title("AskSense")
model = load_model()
st.success("Model loaded!")

def get_engine():
    return SemanticSearchEngine('data/scam_dataset_tw_10000.csv', risk_threshold=0.7)

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
def cached_search(query:str, top_k:int=10):
    return engine.search(query, top_k=top_k)

@st.cache_data(ttl=600)
def cached_analysis(query: str):
    return engine.sentence_analysis(query)

@st.cache_data(ttl=600)
def cached_highlight(query: str):
    return engine.highlight_keywords(query)


# Sidebar: Developer Info & Expertise
with st.sidebar:
    st.header("🔍「AskSense 詐騙檢測工具」")
    st.markdown(
        """
        這是一款能讓你可以貼上任意文字，快速標示出可能有風險或問題的句子的程式。我們的功能如下：

        1. **語意搜尋 (Semantic Search)**：透過 SBERT 模型，將輸入文字與詐騙資料庫比對。

        2. **逐句風險分析 (Sentence Risk Analysis)**：判定風險等級：紅 (高)、黃 (中)、綠 (低)。

        3. **關鍵詞擷取與標示 (Keyword Highlighting)**：自動擷取影響判斷的關鍵詞 (hits)，並在原文中標示 (highlighted)。

        4. **效能優化與安全設計**：查詢與分析皆採快取機制，避免重複計算相同輸入。

        5. **友善連結**：提供台灣官方防詐資源連結 [165 防詐達人](https://165.npa.gov.tw)，方便深入查詢。

        非常適合想要輕鬆找出自己文字中潛在安全或內容風險的任何人！
        """
    )
    st.markdown("---")
    st.header("開發者資訊")
    st.markdown(
        """
        **Jn77**
        - 國立陽明交通大學資訊學院 研究生
        """
    )
    st.markdown("© 2025 JN AskSense. 🔑 All rights reserved.")

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

    scam_yn = False

    # 顯示最可能 Top3 類型
    st.subheader('🚩 最可能的 3 種詐騙類型')
    for t, sc in top_types:
        if sc > 0.55:
            st.markdown(f"- **{t}** (最高相似度: {sc:.4f})")
            scam_yn = True
        else:
            st.markdown("聽起來...，不太像是詐騙訊息喔！")

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
    if main_type and scam_yn:
        st.markdown(f"**{main_type}** 案例示範：")
        examples = stories_df[stories_df['type'] == main_type]['content'].tolist()
        for ex in examples[:3]:  # 前3個案例
            st.write(f"- {ex}")

    st.markdown('---')
    st.markdown('更多資源： [165 防詐達人](https://165.npa.gov.tw)')
