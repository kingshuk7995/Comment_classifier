from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "comment" not in data:
            return jsonify({"error": "No comment provided"}), 400

        comment_text = data["comment"]
        comment_vect = vectorizer.transform([comment_text])

        prediction = model.predict(comment_vect)[0]
        prediction_proba = model.predict_proba(comment_vect).tolist()

        print(f"Received comment: {comment_text}")
        print(f"Prediction: {prediction}")
        print(f"Probabilities: {prediction_proba}")

        return jsonify({"prediction": int(prediction), "probabilities": prediction_proba})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
