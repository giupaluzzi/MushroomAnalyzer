from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

def model_1(image_size, num_classes):
    # Creates a CNN model for image classification

    # Sequential model
    model = Sequential()

    # First convolutional layer
    model.add(Conv2D(32, (3, 3), input_shape=image_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third convolutional layer
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten
    model.add(Flatten())

    # Dense layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Add dropout to prevent overfitting

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))  # softmax function for multi-class classification

    # Compile the model
    #model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
