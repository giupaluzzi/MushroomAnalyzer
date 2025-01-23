import os
import random
import shutil

def split_dataset(source_dir, train_dir, val_dir, test_ratio=0.2):
    """
    Suddivide un dataset in training set e validation set.
    
    Args:
        source_dir (str): Path alla directory sorgente contenente le sottocartelle di classi.
        train_dir (str): Path della directory per il training set.
        val_dir (str): Path della directory per il validation set.
        split_ratio (float): Proporzione di immagini da includere nel training set (default: 0.8).
    
    Returns:
        None
    """
    # Crea le directory per training e validation set, test set se non esistono
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Itera su tutte le sottocartelle nella directory sorgente
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for class_name in class_dirs:
        source_class_path = os.path.join(source_dir, class_name)
        
        # Percorsi delle sottodirectory per la classe
        train_class_path = os.path.join(train_dir, class_name)
        val_class_path = os.path.join(val_dir, class_name)

        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(val_class_path, exist_ok=True)

        # Elenco di tutti i file immagine nella sottodirectory della classe
        all_images = [f for f in os.listdir(source_class_path) if os.path.isfile(os.path.join(source_class_path, f))]
        
        # Mescola casualmente le immagini
        random.shuffle(all_images)

        # Determina il numero di immagini per ciascun set
        val_split_index = int(len(all_images) * test_ratio)
        train_images = all_images[val_split_index:]
        val_images = all_images[:val_split_index]
        

        # Sposta le immagini nel rispoettivo set
        for image in train_images:
            shutil.move(
                os.path.join(source_class_path, image),
                os.path.join(train_class_path, image)
            )
        
        for image in val_images:
            shutil.move(
                os.path.join(source_class_path, image),
                os.path.join(val_class_path, image)
            )
        
        print(f"Classe '{class_name}': {len(train_images)} immagini nel training set, {len(val_images)} nel validation set.")

    print("Suddivisione del dataset completata.")

        
        
        