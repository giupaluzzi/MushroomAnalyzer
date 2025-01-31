import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

def test_model(model, train_set, input_size, test_image_path):
    # Tests an image using a trained model

    # Map numerical indices to class labels (numeric index -> species name)
    class_mapping = {v: k for k, v in train_set.class_indices.items()}
    
    # Read and preprocess the test image
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        raise FileNotFoundError(f"Image not found: {test_image_path}")

    test_image = cv2.resize(test_image, input_size)  # Resize the image
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    plt.imshow(test_image)  # Display the original image
    plt.axis('off')
    plt.show()

    # Prepare the image for the model
    test_image = np.expand_dims(test_image, axis=0)  # Add a batch dimension
    test_image = test_image / 255.0  # Normalize pixel values

    # Load the model if a file path is provided
    if isinstance(model, str):
        print(f"Loading model from: {model}")
        model = load_model(model)

    # Predict the class of the image
    prediction_prob = model.predict(test_image)
    predicted_class_index = np.argmax(prediction_prob, axis=1)[0]
    predicted_class = class_mapping[predicted_class_index]

    # Display the result
    #plt.title(f"Prediction: {predicted_class}")
    #plt.show()

    return predicted_class
