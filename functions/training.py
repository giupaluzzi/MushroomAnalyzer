from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_mushrooms_model(model, train_data, validation_data, use_early_stopping, patience, model_id, epochs):
    # Trains the model using the training and validation datasets.
    # Uses the early stopping function to prevent overfitting
    # Uses ModelCheckpoint to save the best model

    # Retrieve class labels from training data
    class_labels = train_data.classes

    # Compute class weights based on training data to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)   
    class_weights = dict(enumerate(class_weights))  # convert to a dictionary

    # Configure the EarlyStopping callback
    early_stop = None
    if use_early_stopping:
        early_stop = EarlyStopping(
            monitor='val_loss',            # Monitor validation loss
            mode='min',                    # Minimize the loss
            patience=patience,             # Number of epochs with no improvements
            restore_best_weights=True,     # Restore the best weights after stopping
            verbose=1                      # Display messages during training
        )

    # Configure the ModelCheckpoint callback
    model_filename = f"Mushrooms_model_{model_id}.h5"
    checkpoint = ModelCheckpoint(
        filepath=model_filename,           # Path to save the model file
        monitor='val_loss',                # Monitor validation loss
        verbose=1,                         # Dispaly messages during saving
        save_best_only=True,               # Save only the best model
        mode='min'                         # Minimize the loss
    )

    # Combine the callbacks into a list
    callbacks = [checkpoint]
    if early_stop:
        callbacks.append(early_stop)

    # Start model training
    history = model.fit(
        train_data,                        # Training data
        validation_data=validation_data,   # Validation Data
        callbacks=callbacks,               # Callbacks
        epochs=epochs,                     # Epochs
        class_weight=class_weights         # Class weights
    )

    return history, model_filename, early_stop






def evaluate_mushrooms_model(model, training_data, validation_data):
    # Evaluates the model on training and validation datasets

    print("--- Evaluating the model ---")
    
    # Evaluate on the training set
    train_metrics = model.evaluate(training_data, verbose=1)
    train_loss, train_accuracy = train_metrics[0], train_metrics[1]
    print(f"Training Set - Accuracy: {train_accuracy:.4f}, Loss: {train_loss:.4f}")
    
    # Evaluate on the validation set
    val_metrics = model.evaluate(validation_data, verbose=1)
    val_loss, val_accuracy = val_metrics[0], val_metrics[1]
    print(f"Validation Set - Accuracy: {val_accuracy:.4f}, Loss: {val_loss:.4f}")
    
    return {
        "training": {"accuracy": train_accuracy, "loss": train_loss},
        "validation": {"accuracy": val_accuracy, "loss": val_loss}
    }
