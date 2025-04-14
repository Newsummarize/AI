from summarize import summarize_text
from keyword_extract import extract_keywords
from similarity import calculate_keyword_similarity

input_text = "과수 화상병 진단·방제 속도가 크게 빨라질 전망이다. 농촌진흥청은 화상병 정밀 검사기관을 추가 지정했다고 6일 밝혔다. 정밀 진단 지침서를 4월 둘째주 배포하고 검사자 교육도 진행한다고 덧붙였다. 화상병 정밀 검사기관으로 지정한 곳은 7곳 도농업기술원(경기·강원·충북·충남·전북·경북·경남)이다. 정밀 검사기관은 농림축산식품부령으로 정하는 시설·장비·인력·검사능력을 갖춰야 한다. 종전엔 농진청 국립농업과학원 한곳만 화상병 정밀진단이 허용됐다. 그러다 보니 거리가 먼 지역은 검사가 늦어져 방제 대응이 지연된다는 지적을 받아왔다.  정부는 이를 해결하고자 2024년 7월 식물방역법을 개정해 정밀 검사기관을 추가 지정할 수 있도록 했다. 농진청은 전국 검사기관의 검사 품질을 동일하게 맞추고자 진단법 표준화 작업에 나섰다. 먼저 정밀진단 지침서를 4월 둘째주 발간·배포한다. 지침서는 화상병과 가지검은마름병의 특성, 진단법, 시료 취급법 등이 담겼다. 농진청은 이달 8~11일엔  검사 인력을 대상으로 교육도 시행한다. 이 기간 신규 정밀 검사기관의 검사 인력은 농진청 농과원에서 식물질병 진단법과 병원균 검출 방법을 배우게 된다. 이세원 농진청 농과원 식물병방제과장은 “정밀 검사기관이 전국으로 확대되면 화상병 진단·방제가 빠르고 전문적으로 이뤄질 것”이라며 “앞으로도 과수 화상병 예방과 확산 방지를 위해 지원을 확대하겠다”라고 말했다."

#임시 키워드 리스트
keyword_list_2 = ['장비', '추가', '화상병', '농림축산식품부', '농촌진흥청']

if __name__ == "__main__":
    news = input_text

    # 요약
    summary = summarize_text(news)

    # 키워드 추출
    keywords = extract_keywords(summary)
    keyword_list = [kw[0] for kw in keywords]

    similarity_score = calculate_keyword_similarity(keyword_list, keyword_list_2)

    print("\n요약: ", summary)
    print("\n키워드:", keywords)
    print(f"\n임의의 키워드와의 유사도: {similarity_score:.4f}")