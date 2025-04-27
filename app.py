from flask import Flask, render_template, request
import pandas as pd 
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import nltk
import joblib
from pandarallel import pandarallel


#nltk.download('punkt')
#nltk.download('stopwords')

# Preprocess text
def preprocess_text(text):
    import string
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def initialize_app():
    # Initialize Flask app
    app = Flask(__name__)

    
    parquet_dir = r"C:\Users\MOHAMMED NOMAAN\OneDrive\ドキュメント\dataset"
    all_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith('.parquet')]

    df_list = [pd.read_parquet(file) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # Initialize pandarallel
    pandarallel.initialize(progress_bar=True)

    df['Cleaned_Review'] = df['text'].parallel_apply(preprocess_text)
    

    # Splitting data
    X = df['Cleaned_Review']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_file = 'logistic_regression_model.pkl'
    vectorizer_file = 'tfidf_vectorizer.pkl'

    # Define absolute paths for model and vectorizer
    model_file = os.path.join(os.getcwd(), 'logistic_regression_model.pkl')
    vectorizer_file = os.path.join(os.getcwd(), 'tfidf_vectorizer.pkl')


    if os.path.exists(model_file) and os.path.exists(vectorizer_file):
        model = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)
    else:
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_tfidf, y_train)

    
        joblib.dump(model, model_file)
        joblib.dump(vectorizer, vectorizer_file)

    
    X_test_tfidf = vectorizer.transform(X_test)
    test_accuracy = accuracy_score(y_test, model.predict(X_test_tfidf))

    # Routes
    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        review = request.form['review']
        cleaned_review = preprocess_text(review)
        review_tfidf = vectorizer.transform([cleaned_review])
        prediction = model.predict(review_tfidf)
        if prediction[0] == 2:
            result = "Genuine"
        elif prediction[0] == 1:
            result = "Neutral"
        else:
            result = "Fake"

        
        accuracy = round(test_accuracy * 100, 2)
        return render_template('index.html', review=review, result=result, accuracy=accuracy)

    return app

if __name__ == '__main__':
    app = initialize_app()
    app.run(debug=False, use_reloader = False)
