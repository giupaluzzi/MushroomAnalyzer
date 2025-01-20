import os
import shutil

def restore_dataset(train_dir, val_dir, source_dir):
    """
    Ripristina il dataset iniziale spostando le immagini dalle cartelle di training e validation
    nelle cartelle originali.
    
    Args:
        train_dir (str): Directory contenente le immagini di training.
        val_dir (str): Directory contenente le immagini di validation.
        source_dir (str): Directory principale del dataset iniziale, in cui verranno ripristinate le immagini.
    """
    # Controlla le cartelle di training e validation
    class_dirs_train = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    class_dirs_val = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]

    # Crea le cartelle originali, se non esistono
    os.makedirs(source_dir, exist_ok=True)
    
    # Restituisci le immagini dalla cartella di training alla cartella principale
    for class_name in class_dirs_train:
        source_class_path = os.path.join(train_dir, class_name)
        target_class_path = os.path.join(source_dir, class_name)
        os.makedirs(target_class_path, exist_ok=True)
        
        # Sposta le immagini da training a source
        for image in os.listdir(source_class_path):
            image_path = os.path.join(source_class_path, image)
            shutil.move(image_path, os.path.join(target_class_path, image))
        print(f"Immagini della classe '{class_name}' ripristinate dal training set.")

    # Restituisci le immagini dalla cartella di validation alla cartella principale
    for class_name in class_dirs_val:
        source_class_path = os.path.join(val_dir, class_name)
        target_class_path = os.path.join(source_dir, class_name)
        os.makedirs(target_class_path, exist_ok=True)
        
        # Sposta le immagini da validation a source
        for image in os.listdir(source_class_path):
            image_path = os.path.join(source_class_path, image)
            shutil.move(image_path, os.path.join(target_class_path, image))
        print(f"Immagini della classe '{class_name}' ripristinate dal validation set.")
    
    print("Ripristino del dataset completato.")


