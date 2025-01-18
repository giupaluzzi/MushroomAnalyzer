import os
import shutil

def move_images_back_to_source(source_dir, train_dir, val_dir):
    """
    Sposta le immagini dalle cartelle di training e validation alla cartella principale del dataset.
    """
    # Aggiungi il percorso completo per le cartelle di training e validation
    train_path = os.path.join(source_dir, train_dir)
    val_path = os.path.join(source_dir, val_dir)

    # Sposta le immagini dalla cartella di training alla cartella principale
    for class_name in os.listdir(train_path):
        class_train_path = os.path.join(train_path, class_name)
        if os.path.isdir(class_train_path):
            for image in os.listdir(class_train_path):
                image_path = os.path.join(class_train_path, image)
                destination_path = os.path.join(source_dir, class_name, image)
                shutil.move(image_path, destination_path)

    # Sposta le immagini dalla cartella di validation alla cartella principale
    for class_name in os.listdir(val_path):
        class_val_path = os.path.join(val_path, class_name)
        if os.path.isdir(class_val_path):
            for image in os.listdir(class_val_path):
                image_path = os.path.join(class_val_path, image)
                destination_path = os.path.join(source_dir, class_name, image)
                shutil.move(image_path, destination_path)

    # Elimina le cartelle di training e validation (ora vuote)
    shutil.rmtree(train_path)
    shutil.rmtree(val_path)
    
    print("Immagini restituite alla cartella principale.")
