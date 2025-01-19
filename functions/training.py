from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_mushrooms_model(model, train_data, validation_data, use_early_stopping, patience, model_id, epochs):
    """
    Addestra il modello utilizzando i set di training e validation.
    Supporta l'early stopping per evitare overfitting.

    -model: il modello Keras da addestrare
    - train_data: dataset di training
    - validation_data: dataset di validation
    - use_early_stopping: bool, se True abilita EarlyStopping
    - patience: numero di epoche senza miglioramento prima di fermare il training
    - model_id: identificativo numerico per il salvataggio del modello
    - epochs: numero massimo di epoche per il training 
    """

    # Calcola i pesi delle classi in base ai dati di training
    class_labels = train_data.classes       # etichette per ogni immagine
    class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)   
    class_weights = dict(enumerate(class_weights))  # conversione in dizionario

    # Configurazione Early Stopping
    early_stop = None
    if use_early_stopping:
        early_stop = EarlyStopping(
            monitor='val_loss',            # Monitora la perdita sulla validation
            mode='min',                    # Minimizziamo la perdita
            patience=patience,             # Numero di epoche senza miglioramenti
            restore_best_weights=True,     # Ripristina i pesi migliori
            verbose=1                      # Mostra messaggi durante il training
        )

    # Configurazione del checkpoint per salvare il modello
    model_filename = f"Mushrooms_model_{model_id}.h5"
    checkpoint = ModelCheckpoint(
        filepath=model_filename,           # Salva il file del modello
        monitor='val_loss',                    # Monitora la perdita sul training
        verbose=1,                         # Mostra messaggi durante il salvataggio
        save_best_only=True,               # Salva solo il miglior modello
        mode='min'                         # Minimizziamo la perdita
    )

    # Lista delle callback
    callbacks = [checkpoint]
    if early_stop:
        callbacks.append(early_stop)

    # Avvio del training del modello
    history = model.fit(
        train_data,                        # Dati di training
        validation_data=validation_data,  # Dati di validation
        callbacks=callbacks,               # Callback configurati
        epochs=epochs,                      # Numero di epoche
        class_weight=class_weights         # Pesi delle classi
    )

    return history, model_filename, early_stop






def evaluate_mushrooms_model(model, training_data, validation_data):
    """
    Valuta il modello sui set di training e validation.
    Stampa l'accuratezza e la perdita per entrambi i set.
    """
    print("--- Valutazione del modello ---")
    
    # Valutazione sul training set
    train_metrics = model.evaluate(training_data, verbose=1)
    train_loss, train_accuracy = train_metrics[0], train_metrics[1]
    print(f"Training Set - Accuratezza: {train_accuracy:.4f}, Perdita: {train_loss:.4f}")
    
    # Valutazione sul validation set
    val_metrics = model.evaluate(validation_data, verbose=1)
    val_loss, val_accuracy = val_metrics[0], val_metrics[1]
    print(f"Validation Set - Accuratezza: {val_accuracy:.4f}, Perdita: {val_loss:.4f}")
    
    return {
        "training": {"accuracy": train_accuracy, "loss": train_loss},
        "validation": {"accuracy": val_accuracy, "loss": val_loss}
    }
