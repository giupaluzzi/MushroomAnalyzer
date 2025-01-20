from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(model, test_generator):
    # Predizioni sul dataset di test
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Classi predette
    y_true = test_generator.classes # Etichette vere

    # Calcolo della matrice di confusione
    cm = confusion_matrix(y_true, y_pred_classes)
    class_labels = list(test_generator.class_indices.keys())

    # Visualizzazione della matrice
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Report di classificazione
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels))
