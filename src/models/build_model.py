from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from ..config import IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE, DROPOUT_RATE


def build_cnn () -> Model:
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)

    # Softmax output layer for multi-class classification
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    compile_model(model)
    return model


def compile_model (model : Model) -> Model:
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
