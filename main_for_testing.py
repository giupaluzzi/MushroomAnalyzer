import os
import json 
import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import MobileNetV2, ResNet50


import matplotlib.pyplot as plt
from keras.utils import img_to_array, load_img
import numpy as np


from functions.datasetting import split_dataset
from functions.dataloader import load_dataset
from functions.accuracy_and_loss import plot_loss, plot_accuracy, plot_training_history
from functions.training import train_mushrooms_model, evaluate_mushrooms_model
from functions.prediction import test_model
from functions.modelCNN import model_1


def main():
    # Main directories
    source_dir = "Dataset/MIND.Funga_Dataset"                               # Source dataset directory
    train_dir = "Dataset/Mushroom_Dataset_Training"        # Training set directory
    val_dir = "Dataset/Mushroom_Dataset_Validation"        # Validation set directory

    # Split the dataset
    #split_dataset(source_dir, train_dir, val_dir, test_ratio=0.2)

  
    # Load training and validation datasets
    image_size = (400, 400)  # Required image dimensions for the model
    batch_size = 32          # Batch size for training
    print("Loading training and validation datasets")
    train_set, val_set = load_dataset(train_dir, val_dir, image_size, batch_size)

    # Save class indices to a JSON file
    with open('class_indices.json', 'w') as f:
        json.dump(train_set.class_indices, f)


    # Inizialize CNN model
    #print("Creating the model")
    #model = model_1(image_size=(224, 224, 3), num_classes = len(train_set.class_indices))


    # Inizialize ImageNet model
    print("Creating the model")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(400, 400, 3))
    for layer in base_model.layers:
        layer.trainable = False  # Freeze pre-trained layers

    
    # Add custom layers
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(len(train_set.class_indices), activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

    optimizer = Adam(learning_rate=0.001)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Training configuration
    num_epochs = 30
    patience = 5
    model_num = 6

    # Configure the callbacks
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", 
        patience=patience, 
        restore_best_weights=True,
        verbose=1
)

    model_checkpoint_callback = ModelCheckpoint(
        filepath=f"best_model_{model_num}.h5",  # Save the best model based on validation loss
        monitor="val_loss",
        save_best_only=True,
        verbose=1
)

    # Train the model
    print("Training the model")
    early_stopping = Truepatience = 5

    history, model_path, early_stop = train_mushrooms_model(
        model=model,
        train_data=train_set,
        validation_data=val_set,
        use_early_stopping=early_stopping,
        patience=patience,
        model_id=model_num,
        epochs=num_epochs,
)


    # Evaluate the model
    print("Evaluating the model")
    evaluate_mushrooms_model(model, train_set, val_set)


    # Plot training history
    print("Plotting training history")
    plot_training_history(history)


    # Test an image
    print("Testing an image")
    test_image_path = "test/tulostoma_exasperatum.jpg"  # Path of the test image
    predicted_class = test_model(model_path, train_set, image_size, test_image_path)
    print(f"Prediction: {predicted_class}")

    

    # Load the image for display
    img = load_img(test_image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0  # Normalize the image

    # Model prediction
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    predicted_class_idx = np.argmax(prediction)  # Index of predicted class

    # Get the class name associated with the predicted index
    predicted_class_name = list(train_set.class_indices.keys())[predicted_class_idx]

    # Display the image with the prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class_name}")
    plt.axis('off')  
    plt.show()

if __name__ == "__main__":
    main()
