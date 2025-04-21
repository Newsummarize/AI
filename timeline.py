from datetime import datetime
from summarize import summarize_text
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np
import json
import re
import dateparser

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def format_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').strftime('%Y.%m.%d')
    except Exception:
        return date_str

# 자연어 날짜 추출 함수
def extract_event_date(text, default_date):
    date_patterns = re.findall(r'(\d{4}[\./년\-]\s*\d{1,2}[\./월\-]?\s*\d{1,2}[일]?)|(\d{1,2}월\s*\d{1,2}일)', text)

    for full_date, short_date in date_patterns:
        raw_date = full_date or short_date
        parsed_date = dateparser.parse(
            raw_date,
            languages=['ko'],
            settings={'RELATIVE_BASE': datetime.now()}
        )
        if parsed_date and parsed_date.year > 2000:
            return parsed_date.strftime('%Y-%m-%d %H:%M:%S')

    return default_date  # 없으면 기사 발행일 사용

def summarize_timeline_by_query(data):
    if isinstance(data, str):
        data = json.loads(data)

    keyword = data['keyword']
    articles = data['articles']

    summaries = []
    event_dates = []

    for article in articles:
        content = article.get('content', '')
        published_at = article.get('published_at', '')

        if content:
            summary = summarize_text(content)
            summaries.append(summary)

            # 자연어에서 날짜 추출 (실패 시 기사 발행일 사용)
            event_date = extract_event_date(content, published_at)
            event_dates.append(event_date)

    embeddings = model.encode(summaries)

    clustering = DBSCAN(eps=0.45, min_samples=2, metric='cosine').fit(embeddings)
    labels = clustering.labels_

    clustered_data = {}
    etc_groups = []  # 기타 클러스터 저장용

    for label, date, summary in zip(labels, event_dates, summaries):
        if label == -1:
            etc_groups.append((date, summary))
        else:
            clustered_data.setdefault(label, []).append((date, summary))

    # 기타 클러스터 병합 및 정렬
    etc_groups.sort(key=lambda x: x[0])  # 날짜 순 정렬

    for i, (date, summary) in enumerate(etc_groups):
        label = f"기타_{i // 2 + 1}"  # 2개씩 묶어서 기타 1, 2 등으로 (원하면 기준 조정 가능)
        clustered_data.setdefault(label, []).append((date, summary))

    result = f"키워드: {keyword}\n\n"

    cluster_count = 1
    for label in sorted(clustered_data.keys(), key=lambda x: str(x)):
        events = sorted(clustered_data[label], key=lambda x: x[0])
        formatted_events = [f"- {format_date(date)}: {summary}" for date, summary in events]

        if not str(label).startswith("기타"):
            start_date = format_date(events[0][0])
            end_date = format_date(events[-1][0])
            event_period = f"({start_date} ~ {end_date})"
            result += f"[사건 {cluster_count}] {event_period}\n"
            cluster_count += 1
        else:
            result += f"[{label.replace('_', ' ')}]\n"

        combined_summary = " ".join([s for _, s in events])
        summarized_event = summarize_text(combined_summary, max_length=150, min_length=30)

        result += "\n".join(formatted_events)
        result += f"\n\n요약: {summarized_event}\n\n"

    return result
