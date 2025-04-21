import json
from timeline import summarize_timeline_by_query

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def main():
    file_path = 'data/뉴진스 어도어 분쟁_naver_news.json'  
    
    data = read_json_file(file_path)
    
    result = summarize_timeline_by_query(data)
    
    print("뉴스 요약 타임라인")
    print("=" * 60)
    print(result)

if __name__ == '__main__':
    main()