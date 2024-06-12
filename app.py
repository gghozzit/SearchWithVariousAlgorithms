from flask import Flask, request, render_template, jsonify
from collections import Counter
import math

app = Flask(__name__)

documents = [
     'The quick brown fox jumps over the lazy dog',
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit',
        'Vector Space Model is a type of Information Retrieval Model',
        'Machine learning and natural language processing are interesting fields',
        'Artificial intelligence is transforming the world',
        'The history of natural language processing is fascinating',
        'Deep learning algorithms are very powerful',
        'Data science involves statistics, programming, and domain knowledge',
        'Python is a popular language for machine learning',
        'The field of information retrieval is crucial for search engines',
        'Big data technologies enable the processing of large datasets',
        'The Internet of Things (IoT) is connecting devices worldwide',
        'Blockchain technology provides a secure way of handling transactions',
        'Quantum computing promises to solve complex problems faster',
        'Augmented reality enhances our perception of the real world',
        'Cybersecurity is essential for protecting information systems',
        'Cloud computing offers scalable resources and storage solutions',
        'Genetic algorithms are used for optimization problems',
        "Neural networks mimic the human brain's structure and function",
        'Support vector machines are effective for classification tasks',
        'Reinforcement learning is a type of machine learning where agents learn by interacting with their environment',
        'Natural language generation creates human-like text from data',
        'Computer vision allows machines to interpret visual information',
        'Speech recognition technology converts spoken language into text',
        'Recommendation systems suggest products or content based on user preferences',
        'Edge computing processes data closer to the source of generation',
        '5G technology will significantly increase internet speeds',
        'Virtual reality creates immersive experiences through simulated environments',
        'Digital twins are virtual replicas of physical entities',
        'Autonomous vehicles are self-driving cars that use sensors and AI',
        'Robotic process automation automates repetitive tasks',

        'Hujan turun dengan deras di sore hari',
        'Anak-anak bermain di taman dengan riang',
        'Pendidikan merupakan kunci untuk masa depan yang lebih baik',
        'Kesehatan adalah aset berharga yang harus dijaga',
        'Makanan yang sehat sangat penting untuk menjaga keseimbangan tubuh',
        'Keluarga adalah tempat di mana cinta tanpa batas bermula',
        'Senyum adalah bahasa universal yang bisa dimengerti semua orang',
        'Kegiatan olahraga membantu menjaga kebugaran fisik dan mental',
        'Kebersihan adalah sebagian dari iman',
        'Pendidikan karakter merupakan pondasi bagi pembangunan bangsa',
        'Kesederhanaan adalah kunci dari kebahagiaan sejati',
        'Kebersamaan dalam keluarga menguatkan ikatan batin',
        'Alam menyediakan keindahan yang tiada tara',
        'Pariwisata berkelanjutan adalah tanggung jawab bersama',
        'Teknologi informasi membawa kemajuan yang pesat dalam berbagai bidang',
        'Kewirausahaan merupakan motor penggerak perekonomian',
        'Musik adalah bahasa yang bisa menyatukan berbagai budaya',
        'Kerja keras dan ketekunan merupakan kunci kesuksesan',
        'Kehidupan harus diisi dengan nilai-nilai positif',
        'Cinta dan kasih sayang adalah pendorong utama dalam hidup',
]

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
