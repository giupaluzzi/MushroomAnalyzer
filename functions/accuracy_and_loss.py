import matplotlib.pyplot as plt

def plot_loss(history):
    # Plots the loss curve for training and validation datasets

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.title('Loss Trend', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_accuracy(history):
    # Plots the accuracy curve for training and validation datasets

    plt.figure(figsize=(10, 6))
    plt.plot(history.history.get('accuracy', history.history.get('acc')), label='Training Accuracy', color='green', linewidth=2)
    plt.plot(history.history.get('val_accuracy', history.history.get('val_acc')), label='Validation Accuracy', color='orange', linewidth=2)
    plt.title('Accuracy Trend', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_training_history(history):
    # Plots both the loss and accuracy trends in two separate graphs

    print("Generating loss and accuracy plots.")
    plot_loss(history)
    plot_accuracy(history)
