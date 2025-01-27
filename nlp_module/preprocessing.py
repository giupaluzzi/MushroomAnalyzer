import nltk
from string import punctuation
from nltk.corpus import stopwords

def download_resources():
    nltk.download('stopwords')

def clean_text(text):
    # Obtain all the stopwords
    stop_words = set(stopwords.words('english'))
    
    # Divide the text in words
    words = nltk.word_tokenize(text)

    # Remove stopwords and punctuation
    result = [w for w in words if w not in stop_words and w not in punctuation]

    return ' '.join(result)
