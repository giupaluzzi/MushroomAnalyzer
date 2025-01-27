import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def download_stopwords():
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Load data from csv
def load_data(path):
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        print(f"Errore: il file {path} non è stato trovato.")
        return None

# Clean the text: removing stopwords and punctuation
def clean_text(text):    
    
    # Divide the sentence in words
    words = word_tokenize(text)

    # Remove stopwords
    res = [w for w in words if w.casefold() not in stop_words and w not in punctuation]
    return ''.join(res)

# Preprocess data -> clean all the characteristics
def preprocess_data(data):
    data['clean_characteristics'] = data['Characteristics'].apply(clean_text)
    return data

# Feature Extraction
def feature_extraction(data):
    tfidf = TfidfVectorizer(max_features=100)
    X = tfidf.fit_transform(data['clean_characteristics'])
    return X, tfidf

# Prepare target
def prepare_target(data):
    return data['Toxicity'].map({'Edible': 0, 'Inedible': 1, 'Poisonous': 2})

# Divide dataset in training and test
def split_dataset(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    return X_train, X_test, Y_train, Y_test

# Train the model (Naive Bayes)
def train_model(X_train, Y_train):
    model = MultinomialNB()
    model.fit(X_train, Y_train)
    return model

# Test the model and get the predictions
def evaluate_model(model, X_test, Y_test):
    pred = model.predict(X_test)
    print(classification_report(Y_test, pred))
    return pred

def plot_confusion_matrix(test, pred):
    cm = confusion_matrix(test, pred)
    sns.heatmap(cm, square=True, annot=True, cbar=False, 
            xticklabels=['Edible','Inedible','Poisonous'], yticklabels=['Edible','Inedible','Poisonous']) 

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()



#Test
def main():
    download_stopwords()

    data = load_data("./Dataset/fungi_info.csv")
    print(data.head())

    data = preprocess_data(data)

    X, tfidf = feature_extraction(data)
    Y = prepare_target(data)

    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    model = train_model(X_train, Y_train)

    Y_pred = evaluate_model(model, X_test, Y_test)

    plot_confusion_matrix(Y_test, Y_pred)

#    example = ["Red cap with white spots, grows in forest"]
#   pred = model.predict([example])
#    print(f"Prediction for new example: {pred}")

# Esegui il main
if __name__ == "__main__":
    main()