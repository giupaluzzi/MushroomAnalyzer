from string import punctuation
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

# Function to clean the input text by removing stopwords and punctuation, and applying lemmatization
def clean_text(text):

    # Obtain a set of stopwords for the English language
    stop_words = set(stopwords.words('english'))
    
    # Tokenize the text by converting it into individual words (in lowercase)
    words = word_tokenize(text.lower())

    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Remove stopwords and punctuation, and apply lemmatization to each word
    result = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and w not in punctuation]

    # Return the cleaned text as a string
    return ' '.join(result)
