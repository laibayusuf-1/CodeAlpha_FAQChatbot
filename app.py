from flask import Flask, render_template, request
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load FAQ data
with open("faq.json", "r") as file:
    faqs = json.load(file)

questions = [item["question"] for item in faqs]
answers = [item["answer"] for item in faqs]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)


@app.route("/", methods=["GET", "POST"])
def home():
    user_question = ""
    bot_answer = ""

    if request.method == "POST":
        user_question = request.form["question"]

        user_vector = vectorizer.transform([user_question])

        similarity = cosine_similarity(user_vector, question_vectors)

        index = similarity.argmax()

        if similarity[0][index] > 0.2:
            bot_answer = answers[index]
        else:
            bot_answer = "Sorry! I don't know the answer."

    return render_template(
        "index.html",
        user_question=user_question,
        bot_answer=bot_answer
    )


if __name__ == "__main__":
    app.run(debug=True)