from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import re

modelName = "EbanLee/kobart-summary-v3"
tokenizer = PreTrainedTokenizerFast.from_pretrained(modelName)
model = BartForConditionalGeneration.from_pretrained(modelName)

def split_sentences(text):
    # 문장 단위로 자르기
    return re.split(r'(?<=[.!?])\s+', text)

def clean_text(text):
    # 기자 이름 제거 (예: "홍길동 기자", "정지윤 기자", 등)
    text = re.sub(r'\b[\w가-힣]{2,}\s?기자\b', '', text)
    
    # 날짜 형식 제거 (예: "2019년 3월 21일", "2019.02.15", 등)
    text = re.sub(r'\d{4}[년.\- ]\d{1,2}[월.\- ]\d{1,2}[일]?', '', text)
    
    # 뉴스 출처 제거 (예: "로이터", "뉴스1", "연합뉴스", 등)
    text = re.sub(r'(로이터|뉴스1|연합뉴스|SBS|KBS|MBC|YTN|JTBC)', '', text)

    # 중복된 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def summarize_text(text, max_length=140, min_length=50, max_chars=170):
    cleaned_text = clean_text(text)
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=1024
    )

    summary_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=1.0,
        max_length=max_length,
        min_length=min_length,
        num_beams=6,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
    )

    full_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # 문장 단위로 나눈 뒤, 170자 이하가 되도록 문장을 순차적으로 붙이기
    sentences = split_sentences(full_summary)
    final_summary = ""
    for sentence in sentences:
        if len(final_summary + sentence) <= max_chars:
            final_summary += sentence.strip() + " "
        else:
            break

    return final_summary.strip()
