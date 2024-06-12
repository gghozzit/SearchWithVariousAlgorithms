from flask import Flask, request, render_template, jsonify
import pandas as pd
from collections import Counter
import math

app = Flask(__name__)

# Load documents from CSV file
def load_documents():
    df = pd.read_csv('dataset.csv')
    return df['document'].tolist()

documents = load_documents()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    keyword = data['keyword']
    algorithm = data['algorithm']
    results = perform_search(keyword, algorithm)
    return jsonify(results)

def perform_search(keyword, algorithm):
    results = []
    keyword_terms = keyword.lower().split(' ')

    for index, doc in enumerate(documents):
        doc_terms = doc.lower().split(' ')
        score = 0

        if algorithm == 'vector-space':
            score = vector_space_model(keyword_terms, doc_terms)
        elif algorithm == 'cosine-similarity':
            score = cosine_similarity(keyword_terms, doc_terms)
        elif algorithm == 'naive-bayes':
            score = naive_bayes(keyword_terms, doc_terms)
        elif algorithm == 'extended-boolean':
            score = extended_boolean(keyword_terms, doc_terms)
        elif algorithm == 'knn':
            score = knn(keyword_terms, doc_terms)

        if score > 0:
            results.append({'doc': doc, 'score': score})

    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def vector_space_model(query_terms, doc_terms):
    return sum(1 for term in query_terms if term in doc_terms)

def cosine_similarity(query_terms, doc_terms):
    query_vector = create_term_frequency_vector(query_terms)
    doc_vector = create_term_frequency_vector(doc_terms)
    dot_product = dot_product_vectors(query_vector, doc_vector)
    magnitude_query = vector_magnitude(query_vector)
    magnitude_doc = vector_magnitude(doc_vector)
    return dot_product / (magnitude_query * magnitude_doc)

def naive_bayes(query_terms, doc_terms):
    score = 1
    for term in query_terms:
        term_frequency = doc_terms.count(term)
        score *= (term_frequency / len(doc_terms)) if len(doc_terms) > 0 else 0
    return score

def extended_boolean(query_terms, doc_terms):
    score = sum(1 for term in query_terms if term in doc_terms)
    return score ** 2

def knn(query_terms, doc_terms):
    return cosine_similarity(query_terms, doc_terms)

def create_term_frequency_vector(terms):
    return Counter(terms)

def dot_product_vectors(vector1, vector2):
    return sum(vector1[term] * vector2[term] for term in vector1 if term in vector2)

def vector_magnitude(vector):
    return math.sqrt(sum(count ** 2 for count in vector.values()))

if __name__ == '__main__':
    app.run(debug=True)
