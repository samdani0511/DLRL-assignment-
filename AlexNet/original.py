from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class AlexNet(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        # First Convolutional Layer
        self.add(Conv2D(96, kernel_size=(11, 11), strides=4, padding='valid', activation='relu', input_shape=input_shape))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

        # Second Convolutional Layer
        self.add(Conv2D(256, kernel_size=(5, 5), strides=1, padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

        # Third, Fourth, and Fifth Convolutional Layers
        self.add(Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.add(Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

        # Flatten Layer
        self.add(Flatten())

        # Fully Connected Layers
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5)) # Dropout for regularization
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(num_classes, activation='softmax')) # Output layer for classification

# Example usage:
input_shape = (224, 224, 3) # ImageNet input size
num_classes = 1000 # Number of classes in ImageNet
model = AlexNet(input_shape, num_classes)
model.summary()
