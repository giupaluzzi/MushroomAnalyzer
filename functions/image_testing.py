import json
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

def image_testing(model_path, image_path, image_size):
    # Tests an image
    
    # Load the saved model
    model = load_model(model_path)

    # Ricompila il modello per evitare l'avviso
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Load the dictionary with the classes
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)

    # Invert the dictionary to obtain a mapping of index -> class name
    index_to_class = {v: k for k, v in class_indices.items()}

    # Preprocess the image
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)
    predicted_class_name = index_to_class[predicted_class_idx]

    # Display the image with the predicted class
    plt.figure(figsize=(6, 6))
    plt.imshow(load_img(image_path))
    plt.title(f"Predicted Class: {predicted_class_name}", fontsize=14)
    plt.axis('off')
    plt.show()

    return predicted_class_name

# Paths
model_path = "Mushrooms_model_3.h5"  
image_path = "MushroomAnalyzer/test/marasmius_griseoradiatus.jpg"
image_size = (224, 224)

# Test and display of the image
predicted_class = image_testing(model_path, image_path, image_size)
print(f"Predicted Class: {predicted_class}")
