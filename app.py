from flask import Flask, render_template, jsonify, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Model Setup
MODEL_PATH = "h-e-l-l-o/email-spam-classification-merged"  # Ensure model path is correct
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
class_labels = {0: "ham", 1: "spam"}

@app.route("/", methods=["GET", "POST"])
def home():
    classification_result = None  # Holds the result for classification

    if request.method == "POST":
        try:
            # Get email content from the form
            email_text = request.form.get("emailContent", "").strip()
            if not email_text:
                classification_result = {"error": "Email content cannot be empty!"}
            else:
                # Process the email text using the model
                inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()[0]

                # Get predicted class and probabilities
                predicted_class = probs.argmax()
                classification_result = {
                    "email_text": email_text,
                    "predicted_class": class_labels[predicted_class],
                    "spam_probability": f"{probs[1] * 100:.2f}%",
                    "ham_probability": f"{probs[0] * 100:.2f}%"
                }
        except Exception as e:
            classification_result = {"error": str(e)}
            print(f"Error during classification: {e}")

    return render_template("index.html", result=classification_result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
