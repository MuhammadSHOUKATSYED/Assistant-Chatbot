from flask import Flask, request, jsonify
import spacy

app = Flask(__name__)
nlp = spacy.load("en_core_web_lg")

# Function to load QA pairs from a text file
def load_qa_pairs(filename="QA_dataset_v1.txt"):
    qa_pairs = {}
    with open(filename, "r") as file:
        for line in file:
            question, answer = line.strip().split(" | ")
            qa_pairs[question] = answer
    return qa_pairs

# Load predefined questions and answers from the text file
qa_pairs = load_qa_pairs()

@app.route('/process', methods=['POST'])
def process_message():
    data = request.json
    user_message = data.get("message")
    
    # Process the user's message using spaCy
    user_doc = nlp(user_message)

    # Find the most similar question from the predefined list
    best_match = None
    highest_similarity = 0
    
    for question, answer in qa_pairs.items():
        question_doc = nlp(question)
        similarity = user_doc.similarity(question_doc)
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = answer

    # If no similar question is found, return a default response
    if highest_similarity < 0.6:  # You can adjust the threshold based on your needs
        best_match = "Sorry, I don't understand the question."

    return jsonify({"answer": best_match})

if __name__ == '__main__':
    app.run(debug=True)
