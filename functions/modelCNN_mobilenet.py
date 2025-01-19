from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

def model_with_mobilenet(image_size, num_classes):
        """
        Utilizza MobileNetV2 come base per il trasferimento di apprendimento e aggiunge
        un layer finale per la classificazione del fungo.

        - image_size: la dimensione delle immagini di input
        - num_classes: numero di classi da predire 
        """

        # Carica MobileNetV2 pre-addestrato senza la parte finale (head)
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Congela i pesi del modello pre-addestrato
        base_model.trainable = False

        # Crea il modello
        model = Sequential()

        # Aggiungi il modello MobileNetV2 come base 
        model.add(base_model)

        # Aggiungi un layer di pooling globale per ridurre la dimensionalità
        model.add(GlobalAveragePooling2D())

        # Aggiungi un layer denso per la classificazione finale
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))     # Dropout per evitare overfitting

        # Aggiungi il layer di output (softmax per la classificazione multiclasse)
        model.add(Dense(num_classes, activation='softmax'))

        # Compila il modello
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        return model