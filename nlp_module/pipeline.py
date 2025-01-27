from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from plot_cm import plot_confusion_matrix

def train_and_evaluate_model(X,y):
    data_train, data_test, target_train, target_test = train_test_split(X, y)

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    model.fit(data_train, target_train)

    prediction = model.predict(data_test)
    print(classification_report(target_test, prediction))

    plot_confusion_matrix(target_test, prediction)
    
    return model
