################################IMPORTANT PLEASE READ##################################################
###   Added another output to tell if the questions is right or wrong. use this as an benchmark. questions jo check kiye hain vo gpt se produced the. answers agar score below 8.5 answer wrong    #############
######### Change Threshold to 8.5 for maximum accuracy , will do further testing if time#################################################################################################

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import re

app = Flask(__name__)

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Set similarity threshold
SIMILARITY_THRESHOLD = 0.85   # Adjust as needed

@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    data = request.get_json()

    actual_answer = preprocess_text(data.get("user_answer"))
    targeted_answer = preprocess_text(data.get("targeted_answer"))
    
    # Encode the texts to get the embeddings
    actual_answer_embedding = model.encode(actual_answer, convert_to_tensor=True)
    targeted_answer_embedding = model.encode(targeted_answer, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity_score = util.pytorch_cos_sim(actual_answer_embedding, targeted_answer_embedding).item()
    
    # Check if similarity score is above the threshold
    is_correct = similarity_score >= SIMILARITY_THRESHOLD

    return jsonify({'similarity_score': similarity_score, 'is_correct': is_correct})

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def home():
    return 'Server is running 🎊🔥'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
