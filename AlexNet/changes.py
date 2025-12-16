from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.initializers import HeNormal

class AlexNetImproved(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        # -------- Feature Extraction --------
        # Conv Block 1
        self.add(Conv2D(
            96, kernel_size=(11, 11), strides=4, padding='same',
            activation='relu', kernel_initializer=HeNormal(),
            input_shape=input_shape
        ))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        # Conv Block 2
        self.add(Conv2D(
            256, kernel_size=(5, 5), padding='same',
            activation='relu', kernel_initializer=HeNormal()
        ))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        # Conv Block 3
        self.add(Conv2D(384, (3, 3), padding='same', activation='relu', kernel_initializer=HeNormal()))
        self.add(Conv2D(384, (3, 3), padding='same', activation='relu', kernel_initializer=HeNormal()))
        self.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=HeNormal()))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        # -------- Classifier --------
        # Replace Flatten with GAP to reduce parameters
        self.add(GlobalAveragePooling2D())

        self.add(Dense(1024, activation='relu', kernel_initializer=HeNormal()))
        self.add(Dropout(0.5))
        self.add(Dense(num_classes, activation='softmax'))


# Example usage
input_shape = (224, 224, 3)
num_classes = 1000
model = AlexNetImproved(input_shape, num_classes)
model.summary()
