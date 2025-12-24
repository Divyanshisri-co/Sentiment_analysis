# Sentiment_analysis
This is a **Flask web application** that predicts the sentiment of IMDB movie reviews.   The app classifies reviews as **Positive** or **Negative** using a trained **Logistic Regression** model with **TF-IDF features**.
## Features

- Predicts sentiment: Positive / Negative  
- Text preprocessing includes lowercasing, stopwords removal, and lemmatization  
- User-friendly web interface using Flask  

---

## Project Structure

IMDB_Sentiment_Project/
├─ app.py # Flask application
├─ models/ # Saved model files
│ ├─ sentiment_model.pkl
│ ├─ tfidf_vectorizer.pkl
│ └─ label_encoder.pkl
├─ templates/ # HTML template
│ └─ index.html

yaml
Copy code

---

## How to Run

1. Open terminal in project folder  
2. Run the Flask app:

```bash
python app.py
Open browser → http://127.0.0.1:5000/

Enter a movie review → see sentiment prediction

Author
GitHub: Divyanshisri-co
