import argparse
from collections import defaultdict
from search_engine import SemanticSearchEngine

def main():
    p = argparse.ArgumentParser(description='詐騙檢測 CLI')
    p.add_argument('--csv', default='data/scam_data_tw.csv')
    p.add_argument('--threshold', type=float, default=0.85)
    args = p.parse_args()

    engine = SemanticSearchEngine(csv_path=args.csv, risk_threshold=args.threshold)
    print("✅ 已就緒，輸入文字後進行 Top10 計算並輸出 Top3 詐騙種類 (exit 離開)。")

    while True:
        q = input("\n🔍 檢測內容：")
        if q.lower() in ('exit', 'quit'):
            break

        # 取得 Top10 結果
        top10 = engine.search(q, top_k=10)
        # 按詐騙種類彙整 similarity
        type_scores = defaultdict(list)
        for r in top10:
            type_scores[r['type']].append(r['score'])
        # 計算每類最高相似度，並排序取前三
        top_types = sorted(
            ((t, max(scores)) for t, scores in type_scores.items()),
            key=lambda x: x[1], reverse=True
        )[:3]

        print("\n📊 最可能的 3 種詐騙類型：")
        for t, sc in top_types:
            print(f" - {t} (最高相似度: {sc:.4f})")

        # 關鍵詞高亮
        hits, hl = engine.highlight_keywords(q)
        if hits:
            print(f"🔑 關鍵詞：{', '.join(hits)}")
            print(f"標示後：{hl}")

        print('更多資源：https://165.npa.gov.tw')

if __name__ == '__main__':
    main()