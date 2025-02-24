import joblib
import sys

# Define paths to the model and vectorizer
model_path = "E:/ZRAP/sentiment_model.pkl"  # Path to your model
vectorizer_path = "E:/ZRAP/tfidf_vectorizer.pkl"  # Path to your vectorizer

# Load the models
classifier = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Get the review from the command line argument
review = sys.argv[1]  # The review passed as an argument

# Transform the review using the vectorizer
review_vector = vectorizer.transform([review]).toarray()

# Predict the sentiment
sentiment = classifier.predict(review_vector)

# Output the sentiment
print("Positive" if sentiment[0] == 1 else "Negative")
