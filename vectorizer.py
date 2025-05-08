from sentence_transformers import SentenceTransformer

model = SentenceTransformer('jhgan/ko-sbert-sts')

def encode_text(text):
    return model.encode(text, convert_to_numpy=True)  # numpy 배열로 변환