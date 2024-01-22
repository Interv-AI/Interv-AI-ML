from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import spacy
nlp = spacy.load("en_core_web_md")

app=Flask(__name__)
@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    actual_answer = request.form['actual_answer']
    targeted_answer = request.form['targeted_answer']

    actual_embedding = nlp(actual_answer).vector
    targeted_embedding = nlp(targeted_answer).vector

    similarity_score = cosine_similarity(
        [actual_embedding], [targeted_embedding])[0, 0]

    return jsonify({'similarity_score': similarity_score})

# if __name__=='__main__':
#     app.run()
