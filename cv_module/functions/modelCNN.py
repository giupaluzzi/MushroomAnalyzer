from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

def model_1(image_size, num_classes):
    # Creates a CNN model for image classification

    # Sequential model
    model = Sequential()


    # First convolutional layer
    model.add(Conv2D(64, (5, 5), input_shape=image_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional layer
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third convolutional layer
    #model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten
    model.add(Flatten())

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))  # softmax function for multi-class classification


    return model
