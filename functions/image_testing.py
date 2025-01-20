import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Funzione per testare un'immagine singola
def image_testing(model_path, image_path, image_size):
    # Carica il modello salvato
    model = load_model(model_path)

    # Ricompila il modello per evitare l'avviso
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Carica il dizionario delle classi
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)

    # Inverti il dizionario per ottenere un mapping indice -> nome classe
    index_to_class = {v: k for k, v in class_indices.items()}

    # Preprocessa l'immagine
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizzazione

    # Predici la classe
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)
    predicted_class_name = index_to_class[predicted_class_idx]

    # Visualizza l'immagine con la classe predetta
    plt.figure(figsize=(6, 6))
    plt.imshow(load_img(image_path))
    plt.title(f"Classe Predetta: {predicted_class_name}", fontsize=14)
    plt.axis('off')
    plt.show()

    return predicted_class_name

# Percorsi
model_path = "Mushrooms_model_1.h5"  # Modifica con il percorso corretto del tuo modello
image_path = "MushroomAnalyzer/test/stropharia_rugosaannulata.jpg"
image_size = (224, 224)

# Test e visualizzazione dell'immagine
predicted_class = image_testing(model_path, image_path, image_size)
print(f"Classe predetta: {predicted_class}")
