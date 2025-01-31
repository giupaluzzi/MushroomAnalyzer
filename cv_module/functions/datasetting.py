import os
import random
import shutil

def split_dataset(source_dir, train_dir, val_dir, test_ratio=0.2):
    # Split the dataset into training and validation set 

    # Create the directories for train and validation set, if they don't already exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Iterate on all the folders(classes) in the main directory
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    # Loop through each class folder
    for class_name in class_dirs:
        source_class_path = os.path.join(source_dir, class_name)
        
        # Paths for validation and training folders for the current class
        train_class_path = os.path.join(train_dir, class_name)
        val_class_path = os.path.join(val_dir, class_name)

        # Create folders for the current class if they don't already exist
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(val_class_path, exist_ok=True)

        # List of all the images files in the currrent class folder
        all_images = [f for f in os.listdir(source_class_path) if os.path.isfile(os.path.join(source_class_path, f))]
        
        # Shuffle the list of all the images randomly
        random.shuffle(all_images)

        # Determine the number of images for each set 
        val_split_index = int(len(all_images) * test_ratio)
        train_images = all_images[val_split_index:]     # Remaining images go to the training set
        val_images = all_images[:val_split_index]       # First part of the images go to the validation set 
        

        # Move the images to their respective training and validation folders 
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
        
        # Print the number of images in each set 
        print(f"Class '{class_name}': {len(train_images)} images in the training set, {len(val_images)} in the validation set.")

    print("The Dataset is split.")

        
        
        