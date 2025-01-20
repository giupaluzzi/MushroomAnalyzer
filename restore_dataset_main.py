from functions.restore_dataset import restore_dataset  # Importa la funzione

def main():
    # Specifica i percorsi delle directory
    train_dir = "Mushroom_Dataset_Training"
    val_dir = "Mushroom_Dataset_Validation"
    source_dir = "/Users/stefanomorici/Desktop/AiLab project/MushroomAnalyzer/Dataset/MIND.Funga_Dataset"

    # Esegui la funzione per ripristinare il dataset
    restore_dataset(train_dir, val_dir, source_dir)

if __name__ == "__main__":
    main()
