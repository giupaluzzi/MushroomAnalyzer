from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(training_path, validation_path, target_size, batch_size):
    """
    Carica i dati di training e validation applicando data augmentation.
    """
    # Creazione del generatore di immagini con data augmentation
    augmentation = ImageDataGenerator(
        rescale=1./255,        # Normalizzazione dei valori dei pixel
        shear_range=0.2,       # Distorsione delle immagini
        zoom_range=0.2,        # Applicazione di zoom casuale
        horizontal_flip=True   # Inversione orizzontale casuale
    )

    #Generatore di immagini per il validation set senza augmentation
    val_preprocessing = ImageDataGenerator(rescale=1./255)

    print('\n--- Preparazione del TRAINING SET ---')
    train_data = augmentation.flow_from_directory(
        training_path,
        target_size=(target_size[0], target_size[1]),  # Ridimensiona immagini
        batch_size=batch_size,                         # Batch size per training
        class_mode='categorical',                      # Classificazione multiclasse
        color_mode='rgb'                               # Immagini a colori
    )

    print('\n--- Preparazione del VALIDATION SET ---')
    val_data = val_preprocessing.flow_from_directory(
        validation_path,
        target_size=(target_size[0], target_size[1]),  # Ridimensiona immagini
        batch_size=batch_size,                         # Batch size per validation
        class_mode='categorical',                      # Classificazione multiclasse
        color_mode='rgb'                               # Immagini a colori
    )

    
    return train_data, val_data


