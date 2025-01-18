import os
import json 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2

# Visualizza il risultato con un grafico
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

# Importa le funzioni implementate
from functions.datasetting import split_dataset
from functions.dataloader import load_dataset
from functions.accuracy_and_loss import plot_loss, plot_accuracy, plot_training_history
from functions.training import train_mushrooms_model, evaluate_mushrooms_model
from functions.prediction import test_model
#from functions.move_images_back import move_images_back_to_source
def main():
    # Percorsi principali
    source_dir = "/Users/stefanomorici/Desktop/AiLab project testing/MushroomAnalyzer/Dataset/MIND.Funga_Dataset"
    train_dir = "Mushroom_Dataset_Training"
    val_dir = "Mushroom_Dataset_Validation"

    # Suddivisione 80% training e 20% validation
    split_dataset(source_dir, train_dir, val_dir, test_ratio=0.2)

  
    # Carica i dati
    image_size = (224, 224)  # Dimensioni immagine richieste dal modello
    batch_size = 32          # Dimensione batch
    print("Caricamento dei dati di training e validazione...")
    train_set, val_set = load_dataset(train_dir, val_dir, image_size, batch_size)

    # Salva le classi in un file JSON
    with open('class_indices.json', 'w') as f:
        json.dump(train_set.class_indices, f)

    # Inizializza il modello
    print("Creazione del modello...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False  # Congela i layer pre-addestrati
    
    # Aggiungi strati personalizzati
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(len(train_set.class_indices), activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

    # Compila il modello
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    num_epochs = 20
    patience = 3
    model_num = 1

    # Configura le callback
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", 
        patience=patience, 
        restore_best_weights=True,
        verbose=1
)

    model_checkpoint_callback = ModelCheckpoint(
        filepath=f"best_model_{model_num}.h5",  # Salva il miglior modello in base alla validation loss
        monitor="val_loss",
        save_best_only=True,
        verbose=1
)

    # Addestra il modello
    print("Addestramento del modello...")
    num_epochs = 20
    early_stopping = Truepatience = 3
    model_num = 1

    history, model_path, early_stop = train_mushrooms_model(
        model=model,
        train_data=train_set,
        validation_data=val_set,
        use_early_stopping=early_stopping,
        patience=patience,
        model_id=model_num,
        epochs=num_epochs,
)


    # Valuta il modello
    print("Valutazione del modello...")
    evaluate_mushrooms_model(model, train_set, val_set)

    # Visualizza i risultati dell'addestramento
    print("Tracciamento della cronologia...")
    plot_training_history(history)

    # Restituisci le immagini alla cartella principale
    #move_images_back_to_source(source_dir, train_dir, val_dir)


    # Testa un'immagine singola
    print("Test di un'immagine...")
    test_image_path = "MushroomAnalyzer/test/hygrocybe_hololeuca.jpg"  # percorso immagine di test
    predicted_class = test_model(model_path, train_set, image_size, test_image_path)
    print(f"Predizione: {predicted_class}")

    

    # Carica l'immagine per visualizzarla
    img = image.load_img(test_image_path, target_size=image_size)
    img_array = image.img_to_array(img) / 255.0  # Normalizza l'immagine

    # Previsione del modello
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    predicted_class_idx = np.argmax(prediction)  # Indice della classe predetta

    # Ottieni la classe associata all'indice
    predicted_class_name = list(train_set.class_indices.keys())[predicted_class_idx]

    # Visualizza il grafico dell'immagine e la predizione
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predizione: {predicted_class_name}")
    plt.axis('off')  # Non mostrare gli assi
    plt.show()

if __name__ == "__main__":
    main()
