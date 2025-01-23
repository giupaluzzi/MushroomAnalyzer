from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(model, test_generator, top_n=None):
    # Predizioni sul dataset di test
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Classi predette
    y_true = test_generator.classes # Etichette vere

    # Calcolo della matrice di confusione
    cm = confusion_matrix(y_true, y_pred_classes)
    class_labels = list(test_generator.class_indices.keys())

    # Normalizzazione della matrice
    cm_normalized = cm.astype('float') / cm.sum(axis=1) [:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    # Filtraggio classi più problematiche
    if top_n is not None:
        error_counts = np.sum(cm, axis=1) - np.diag(cm)
        top_classes_idx = np.argsort(error_counts)[-top_n:]
        cm_normalized = cm_normalized[np.ix_(top_classes_idx, top_classes_idx)]
        class_labels = [class_labels[i] for i in top_classes_idx]

    # Visualizzazione della matrice
    plt.figure(figsize=(20, 15))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('Confusion_matrix_normalized.png', dpi=300)
    plt.show()

    # Report di classificazione
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels))
