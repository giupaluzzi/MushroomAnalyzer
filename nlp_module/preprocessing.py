import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_resources():
    nltk.download('stopwords')

def clean_text(text):
    # Obtain all the stopwords
    stop_words = set(stopwords.words('english'))
    
    # Divide the text in words
    words = nltk.word_tokenize(text.lower())

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    # Remove stopwords and punctuation
    result = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and w not in punctuation]

    return ' '.join(result)
