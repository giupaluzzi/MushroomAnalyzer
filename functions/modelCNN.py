from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def model_1(image_size, num_classes):
    """
    Crea un modello di rete neurale convoluzionale (CNN) per la classificazione delle immagini.

    - image_size: dimensione delle immagini di input (es. (224, 224, 3))
    - num_classes: numero di classi da predire (output)
    """
    # Modello sequenziale
    model = Sequential()

    # Primo layer convoluzionale
    model.add(Conv2D(32, (3, 3), input_shape=image_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Secondo layer convoluzionale
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Terzo layer convoluzionale
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten per trasformare le mappe di caratteristiche in un vettore
    model.add(Flatten())

    # Strato denso completamente connesso
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Aggiungi Dropout per prevenire overfitting

    # Strato di output
    model.add(Dense(num_classes, activation='softmax'))  # Funzione softmax per la classificazione multiclasse

    # Compilazione del modello
    #model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
