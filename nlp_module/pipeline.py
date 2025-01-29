from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from plot_cm import plot_confusion_matrix

# Test RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate_model(X,y):
    data_train, data_test, target_train, target_test = train_test_split(X, y, test_size=0.3)

#    model = make_pipeline(TfidfVectorizer(max_features=1000, ngram_range=(1, 2)), MultinomialNB(class_prior=[0.33, 0.33, 0.34]))

    model = make_pipeline(TfidfVectorizer(max_features=1000, ngram_range=(1, 2)), RandomForestClassifier(n_estimators=100, class_weight='balanced'))
    model.fit(data_train, target_train)

    prediction = model.predict(data_test)
    print(classification_report(target_test, prediction, zero_division=0))

    plot_confusion_matrix(target_test, prediction)
    
    return model
