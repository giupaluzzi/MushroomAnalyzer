import nltk
# Function to download the necessary NLTK resources
def download_resources():
    # Download required resources from NLTK
    nltk.download('stopwords')  # For stop words
    nltk.download('wordnet')    # For WordNet lexical database
    nltk.download('omw-1.4')    # For multilingual WordNet (useful for synonym generation)
    nltk.download('punkt')      # For sentence and word tokenization