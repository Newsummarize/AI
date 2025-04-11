from keybert import KeyBERT
from konlpy.tag import Okt

okt = Okt()

def extract_nouns(text):
    nouns = okt.nouns(text)
    return nouns

kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

def extract_keywords(text, top_n=5):
    nouns = extract_nouns(text)
    preprocessed_text = " ".join(nouns)
    keywords = kw_model.extract_keywords(
        preprocessed_text,
        keyphrase_ngram_range=(1, 1),
        stop_words=None,
        use_maxsum=True,
        nr_candidates=20,
        top_n=top_n
    )
    return keywords