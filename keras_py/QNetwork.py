from tensorflow.keras.layers import Dense, Activation, LeakyReLU, Input
from tensorflow.keras import initializers
from tensorflow.keras.models import Model


def q_network(input_shape, action_size):
    inputs = Input(shape=input_shape)
    x = Dense(1024, activation="relu")(inputs)
    x = Dense(512, activation="relu")(x)
    outputs = Dense(action_size)(x)
    return Model(inputs=inputs, outputs=outputs)
