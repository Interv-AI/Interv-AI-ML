from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from summa import summarizer
import difflib
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

    # Provide feedback based on similarity score
    if similarity_score >= 0.75:
        feedback = "Great! Your answer is very similar to the reference answer."
    elif similarity_score >= 0.5:
        feedback = "Your answer is somewhat similar to the reference answer."
    else:
        feedback = "Your answer is quite different from the reference answer."

    # Get missing points in the given answer compared to the reference answer
    missing_points = get_missing_points(actual_answer, targeted_answer)

    # Prepare feedback and tips based on missing points
    if missing_points:
        missing_points_str = "Your answer could be improved by including the following points: {}".format(', '.join(missing_points))
    else:
        missing_points_str = "Your answer covers all key points."

    return jsonify({'similarity_score': similarity_score, 'feedback': feedback, 'missing_points': missing_points_str})

def calculate_similarity(actual_answer, targeted_answer):
    # Encode the texts to get the embeddings
    embeddings = model.encode([actual_answer, targeted_answer])

    # Calculate cosine similarity
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0, 0]

    similarity_score = np.float64(similarity_score)

    return similarity_score

def get_missing_points(given_answer, actual_answer):
    # Use text summarization to extract key points from the actual answer
    actual_summary = summarizer.summarize(actual_answer, ratio=0.5)  # Adjust ratio as needed

    # Use difflib to find the difference between the given answer and the actual summary
    d = difflib.Differ()
    diff = list(d.compare(given_answer.splitlines(), actual_summary.splitlines()))

    missing_points = []
    for line in diff:
        if line.startswith('+ '):
            missing_points.append(line[2:])  # Extract the added lines (missing points)

    return missing_points

if __name__ == '__main__':
    app.run(debug=True)
