from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# import spacy

app = Flask(__name__)

# # all_MiniLM-L6v2-pre-trained BERT model
# model = SentenceTransformer('all-MiniLM-L6-v2')


# @app.route('/calculate_similarity', methods=['POST'])
# def calculate_similarity():
#     data = request.get_json()

#     actual_answer = data.get("actual_answer")
#     targeted_answer = data.get("targeted_answer")

#     # Encode the texts to get the embeddings
#     embeddings = model.encode([actual_answer, targeted_answer])

#     # Calculate cosine similarity
#     similarity_score = cosine_similarity(
#         [embeddings[0]], [embeddings[1]])[0, 0]

#     similarity_score = np.float64(similarity_score)

#     return jsonify({'similarity_score': similarity_score})


# nlp = spacy.load("en_core_web_md")

# @app.route('/calculate_similarity', methods=['POST'])
# def calculate_similarity():
#     data = request.get_json()

#     actual_answer = data.get("actual_answer")
#     targeted_answer = data.get("targeted_answer")

#     actual_embedding = nlp(actual_answer).vector
#     targeted_embedding = nlp(targeted_answer).vector

#     similarity_score = cosine_similarity(
#         [actual_embedding], [targeted_embedding])[0, 0]

#     return jsonify({'similarity_score': similarity_score})


# Load DistilBERT model
model = SentenceTransformer('distilbert-base-nli-stsb')

@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    data = request.get_json()

    actual_answer = data.get("actual_answer")
    targeted_answer = data.get("targeted_answer")

    # Encode the texts to get the embeddings
    embeddings = model.encode([actual_answer, targeted_answer])

    # Calculate cosine similarity
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    similarity_score = np.float64(similarity_score)

    return jsonify({'similarity_score': similarity_score})

@app.route('/')
def home():
    return 'Server is running 🎊🔥'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
