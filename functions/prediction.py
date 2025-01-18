import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def test_model(model, train_set, input_size, test_image_path):
    """
    Testa un'immagine singola utilizzando il modello addestrato.
    
    Args:
        model: Il modello addestrato o il percorso al file del modello salvato.
        train_set: Il set di training per recuperare le classi.
        input_size: Dimensione dell'input del modello (es. (224, 224)).
        test_image_path: Percorso dell'immagine da testare.

    Returns:
        Classe predetta come stringa.
    """
    # Mappa numeri alle classi (es. indice numerico → nome della specie)
    class_mapping = {v: k for k, v in train_set.class_indices.items()}
    
    # Legge e preprocessa l'immagine di test
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        raise FileNotFoundError(f"Immagine non trovata: {test_image_path}")

    test_image = cv2.resize(test_image, input_size)  # Ridimensiona l'immagine
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)  # Converti da BGR a RGB
    plt.imshow(test_image)  # Mostra l'immagine originale
    plt.axis('off')
    plt.show()

    # Prepara l'immagine per il modello
    test_image = np.expand_dims(test_image, axis=0)  # Aggiunge una dimensione per il batch
    test_image = test_image / 255.0  # Normalizza i valori dei pixel

    # Carica il modello se necessario
    if isinstance(model, str):
        print(f"Caricamento modello da: {model}")
        model = load_model(model)

    # Predice la classe dell'immagine
    prediction_prob = model.predict(test_image)
    predicted_class_index = np.argmax(prediction_prob, axis=1)[0]
    predicted_class = class_mapping[predicted_class_index]

    # Mostra il risultato
    plt.title(f"Predizione: {predicted_class}")
    plt.show()

    return predicted_class
