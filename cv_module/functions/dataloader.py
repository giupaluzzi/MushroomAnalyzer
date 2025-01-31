from keras.preprocessing.image import ImageDataGenerator

def load_dataset(training_path, validation_path, target_size, batch_size):
    # Loads training and validation datasets with data augmentation applied

    # Create an ImageDataGenerator for training with data augmentation
    augmentation = ImageDataGenerator(
        rescale=1./255,        # Normalize pixels values
        shear_range=0.2,       # Apply shearing transformations
        zoom_range=0.2,        # Apply random zoom transformations
        horizontal_flip=True   # Randomly flip images horizontally
    )

    # Create an ImageDataGenerator for validation without data augmentation
    val_preprocessing = ImageDataGenerator(rescale=1./255)

    # Load and preprocess the training set
    print('\n--- Creating TRAINING SET ---')
    train_data = augmentation.flow_from_directory(
        training_path,
        target_size=(target_size[0], target_size[1]),  # Resize images to target size
        batch_size=batch_size,                         # Batch size for the set 
        class_mode='categorical',                      # Multi-class classification
        color_mode='rgb'                               # rgb images
    )

    # Load and preprocess the validation set 
    print('\n--- Creating VALIDATION SET ---')
    val_data = val_preprocessing.flow_from_directory(
        validation_path,
        target_size=(target_size[0], target_size[1]),  # Resize images to target size
        batch_size=batch_size,                         # Batch size for the set 
        class_mode='categorical',                      # Multi-class classification
        color_mode='rgb'                               # rgb images
    )

    
    return train_data, val_data


