from data_handler import load_data, save_data
from summarizer import summarize_text
from vectorizer import encode_text
import json

def summarize_news(json_path):
    
    data = load_data(json_path)

    for lst in data.get('lists', []):
        for article in lst.get('articles', []):
            content = article.get('content', '')
            if content:
                summary = summarize_text(content)
                article['content'] = summary

    save_data(json_path, data)
    print("요약 저장 완료")

def vectorize_news(json_path):

    data = load_data(json_path)

    for lst in data.get('lists', []):
        for article in lst.get('articles', []):
            content = article.get('content', '')
            if content:
                vector = encode_text(content)
                article['content_vector_'] = json.dumps(vector.tolist())

    save_data(json_path, data)
    print("벡터 저장 완료")
