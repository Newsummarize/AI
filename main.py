from data_handler import load_data
from search import search_articles_vectorized
from news_preprocessor import summarize_news, vectorize_news
from summarizer import summarize_text

def main():
    
    data_path = 'data/test_news.json'

    # summarize_news(data_path)
    # vectorize_news(data_path)

    articles = load_data(data_path)

    user_query = input("검색어를 입력: ")

    results = search_articles_vectorized(user_query, articles, threshold=0.40)

    if results:
        print(f"\n[검색 결과]")
        for article in results:
            summary = summarize_text(article.get('content'), 0.1)
            print(f"[{article.get('published_at')}] {summary}")
    else:
        print("\n관련 기사를 찾지 못했습니다.")

if __name__ == "__main__":
    main()
