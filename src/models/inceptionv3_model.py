from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model

from ..config import IMAGE_SIZE, NUM_CLASSES
from .build_model import compile_model


def build_inceptionv3() -> Model:
    """Build a classification model based on pretrained InceptionV3.

    The convolutional base is frozen to reuse ImageNet features. A global average
    pooling layer is added, followed by dropout and a softmax classifier matching
    the number of target classes.
    """
    # Input tensor
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # Pre‑trained InceptionV3 base without the top classification head
    base = InceptionV3(weights="imagenet", include_top=False, input_tensor=inputs)
    # Freeze the base layers for transfer learning
    base.trainable = False

    # Global average pooling reduces the spatial dimensions
    x = GlobalAveragePooling2D()(base.output)
    # Regularisation
    x = Dropout(0.5)(x)
    # Final dense layer for our classes
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    compile_model(model)
    return model
