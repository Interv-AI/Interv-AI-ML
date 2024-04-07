from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Initialize SentenceTransformer model (all-MiniLM-L6-v2)
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/analyze_answer', methods=['POST'])
def analyze_answer():
    data = request.get_json()

    actual_answer = data.get("user_answer")
    targeted_answer = data.get("targeted_answer")

    # Calculate similarity score
    similarity_score = calculate_similarity(actual_answer, targeted_answer)

    if similarity_score < 0.3:  # Set a threshold for completely different answers
        feedback = "Your answer is quite different from the reference answer."
        correct_answer = targeted_answer  # Display the targeted answer as correct
    else:
        # Provide feedback based on similarity score
        if similarity_score >= 0.75:
            feedback = "Great! Your answer is very similar to the reference answer."
        elif similarity_score >= 0.5:
            feedback = "Your answer is somewhat similar to the reference answer."
        else:
            feedback = "Your answer is different from the reference answer."

        correct_answer = targeted_answer  # No need to display correct answer when not quite different or completely different

    return jsonify({'similarity_score': similarity_score, 'feedback': feedback, 'correct_answer': correct_answer})

def calculate_similarity(actual_answer, targeted_answer):
    # Encode the texts to get the embeddings
    embeddings = model.encode([actual_answer, targeted_answer])

    # Calculate cosine similarity
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0, 0]

    similarity_score = np.float64(similarity_score)

    return similarity_score

if __name__ == '__main__':
    app.run(debug=True)
