import os
import re
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

class SemanticSearchEngine:
    def __init__(
        self,
        csv_path: str,
        risk_threshold: float = 0.85,
        keywords: list = None,
        cache_dir: str = 'cache',
        batch_size: int = 64
    ):
        # 載模型
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # 讀取 CSV
        base = os.path.dirname(__file__)
        csv_file = os.path.join(base, csv_path)
        df = pd.read_csv(csv_file, dtype=str)
        df.fillna('', inplace=True)
        self.contents = df['content'].tolist()
        self.types = df['type'].tolist()
        self.platforms = df['platform'].tolist()

        # 快取 Embeddings
        os.makedirs(os.path.join(base, cache_dir), exist_ok=True)
        emb_file = os.path.join(base, cache_dir,
                                os.path.splitext(os.path.basename(csv_path))[0] + '_embeddings.pt')
        if os.path.exists(emb_file):
            self.embeddings = torch.load(emb_file)
        else:
            self.embeddings = self.model.encode(
                self.contents,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            torch.save(self.embeddings, emb_file)

        # 其他參數
        self.risk_threshold = risk_threshold
        self.keywords = keywords or ['匯款', '限時', '官方', '165']

    def search(self, query: str, top_k: int = 3):
        q_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(q_emb, self.embeddings)[0]
        top = scores.topk(k=top_k)
        results = []
        for idx, score in zip(top.indices.tolist(), top.values.tolist()):
            results.append({
                'content': self.contents[idx],
                'type': self.types[idx],
                'platform': self.platforms[idx],
                'score': float(score)
            })
        return results

    def sentence_analysis(self, text: str):
        sents = [s.strip() for s in re.split(r'(?<=[。！？])', text) if s.strip()]
        if not sents:
            return []
        s_embs = self.model.encode(sents, convert_to_tensor=True)
        sim_mat = util.pytorch_cos_sim(s_embs, self.embeddings)
        max_sims = sim_mat.max(dim=1).values.tolist()
        analysis = []
        for sent, sim in zip(sents, max_sims):
            if sim >= self.risk_threshold:
                lvl = '紅'
            elif sim >= self.risk_threshold * 0.7:
                lvl = '黃'
            else:
                lvl = '綠'
            analysis.append({'sentence': sent, 'score': sim, 'level': lvl})
        return analysis

    def highlight_keywords(self, text: str):
        hits = [kw for kw in self.keywords if kw in text]
        hl = text
        for kw in hits:
            hl = hl.replace(kw, f"[⚠️{kw}⚠️]")
        return hits, hl