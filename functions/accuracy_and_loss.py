import matplotlib.pyplot as plt

def plot_loss(history):
    """
    Traccia la curva della perdita (loss) per il training e la validation.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.title('Andamento della Perdita', fontsize=16)
    plt.xlabel('Epoche', fontsize=14)
    plt.ylabel('Perdita', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_accuracy(history):
    """
    Traccia la curva dell'accuratezza (accuracy) per il training e la validation.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history.get('accuracy', history.history.get('acc')), label='Training Accuracy', color='green', linewidth=2)
    plt.plot(history.history.get('val_accuracy', history.history.get('val_acc')), label='Validation Accuracy', color='orange', linewidth=2)
    plt.title('Andamento dell\'Accuratezza', fontsize=16)
    plt.xlabel('Epoche', fontsize=14)
    plt.ylabel('Accuratezza', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_training_history(history):
    """
    Traccia sia la perdita che l'accuratezza in due grafici separati.
    """
    print("Generazione dei grafici di perdita e accuratezza...")
    plot_loss(history)
    plot_accuracy(history)
