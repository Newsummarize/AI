import numpy as np
import torch
from vectorizer import encode_text
import json

def compute_similarity(query_vector, doc_vector):
    query_tensor = torch.tensor(query_vector, dtype=torch.float32)
    doc_tensor = torch.tensor(doc_vector, dtype=torch.float32)
    return torch.nn.functional.cosine_similarity(query_tensor, doc_tensor, dim=0).item()

def search_articles_vectorized(query, vectorized_articles, threshold):
    query_vector = encode_text(query)
    results = []

    for article in vectorized_articles.get('lists', []):
        for item in article.get('articles', []):
            if 'content_vector_' not in item:
                continue
            vector_str = item.get('content_vector_')
            if not vector_str:
                continue
            article_vector = np.array(json.loads(vector_str), dtype=np.float32)
            similarity = compute_similarity(query_vector, article_vector)

            if similarity >= threshold:
                results.append((similarity, item))

    results.sort(key=lambda x: x[0], reverse=True)

    top_results = results[:10]

    top_results.sort(key=lambda x: x[1].get('published_at', ''))
    return [item for _, item in top_results]
