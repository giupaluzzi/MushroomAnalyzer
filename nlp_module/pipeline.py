from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Naive Bayes
# from sklearn.naive_bayes import MultinomialNB

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate_model(X,y):
    # Split the data into training and test sets. 70% of data will be used for training and 30% for testing.
    data_train, data_test, target_train, target_test = train_test_split(X, y, test_size=0.3)

    #   First Idea: Naive Bayes -> MultinomialNB is too sensitive to class imbalance; in fact, it didn't perform well in the tests
    #   model = make_pipeline(TfidfVectorizer(max_features=1000, ngram_range=(1, 2)), MultinomialNB(class_prior=[0.33, 0.33, 0.34]))

    #   class_prior is used for balancing different class occurrences

    # Define the model pipeline
    # It's divided in:
    # 1. TfidVectorizer -> converts raw text in numerical features using TF-IDF method
    #                      max_features -> limits the number of features to the top 1000 most frequent terms
    #                      ngram_range -> select the "grams" to use (in this case unigrams and bigrams) 
    # 2. RandomForestClassifier -> model to perform classification using decision trees
    #                              n_estimators -> specify how many decision trees are in the forest
    #                              class_weight -> adjust weights inversely propotional to class frequencies (useful for imbalanced datasets)
    model = make_pipeline(TfidfVectorizer(max_features=1000, ngram_range=(1, 2)), RandomForestClassifier(n_estimators=100, class_weight='balanced'))
    
    # Fit the model to the training data
    model.fit(data_train, target_train)

    # Predict the labels for the test data
    predictions = model.predict(data_test)

    # Print the classification report to evaluate the performance of the model
    print(classification_report(target_test, predictions, zero_division=0))
    
    return model