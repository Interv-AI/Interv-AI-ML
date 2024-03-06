#install- pip install sentence-transformers
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

#all_MiniLM-L6v2-pre-trained BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    data = request.get_json() 

    actual_answer = data['actual_answer']
    targeted_answer = data['targeted_answer']

    # Encode the texts to get the embeddings
    embeddings = model.encode([actual_answer, targeted_answer])

    # Calculate cosine similarity
    similarity_score = cosine_similarity(
        [embeddings[0]], [embeddings[1]])[0, 0]

    return jsonify({'similarity_score': similarity_score})


# if __name__=='__main__':
#     app.run()
