from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_keyword_similarity(keywords_1, keywords_2):

    text_1 = " ".join(keywords_1)
    text_2 = " ".join(keywords_2)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text_1, text_2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return similarity[0][0]
