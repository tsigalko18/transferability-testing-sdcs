import numpy as np
from donkeycar.parts.keras import Convolution2D, Lambda, Flatten, Dense, Dropout, Input, MaxPooling2D
from donkeycar.parts.keras import KerasPilot, Model

# definitions
DAVE2_MC = 'dave2-mc-dropout'
CHAUFFEUR_MC = 'chauffeur-mc-dropout'
EPOCH_MC = 'epoch-mc-dropout'


def get_own_model_with_dropout(model_name, input_shape=None):
    model = None

    if model_name == DAVE2_MC:
        model = get_dave2_mc_dropout_model(input_shape)
    elif model_name == CHAUFFEUR_MC:
        model = get_chauffeur_mc_dropout_model(input_shape)
    elif model_name == EPOCH_MC:
        model = get_epoch_mc_dropout_model(input_shape)
    else:
        raise Exception("unknown model type: %s" % model_name)

    class OwnModelWithDropout(KerasPilot):
        def __init__(self, *args, **kwargs):
            super(OwnModelWithDropout, self).__init__(*args, **kwargs)
            self.model = model
            self.compile()

        def compile(self):
            self.model.compile(optimizer=self.optimizer, loss='mse')

        def run(self, img_arr):
            img_arr = img_arr.reshape((1,) + img_arr.shape)

            img_arr = img_arr[0, :, :, :]

            # take a batch of data of the same image
            x = np.array([img_arr for idx in range(16)])  # TODO: use cfg.BATCH_SIZE

            # get predictions from a sample pass
            outputs = self.model.predict(x)

            # average over all passes is the final steering angle
            steering_angle = np.mean(outputs[0])

            # variance of predictions gives the uncertainty
            uncertainty = np.var(outputs[0])
            # print(uncertainty)

            steering = steering_angle
            throttle = np.mean(outputs[1])

            return steering, throttle

    return OwnModelWithDropout


def get_output_layers(x, activation_steer='linear', activation_throttle='linear'):
    return [
        Dense(1, activation=activation_steer, name='n_outputs0')(x),
        Dense(1, activation=activation_throttle, name='n_outputs1')(x)
    ]


def get_dave2_mc_dropout_model(input_shape=(140, 320, 3)):
    if not input_shape:
        input_shape = (140, 320, 3)

    img_in = Input(shape=input_shape, name='img_in')
    x = img_in

    # Normalization is carried out in donkey car train data generator
    x = Lambda(lambda x: x * 255 / 127.5 - 1., input_shape=input_shape, name='lambda_norm')(x)
    # 5x5 Convolutional layers with stride of 2x2

    x = Convolution2D(24, (5, 5), strides=(2, 2), name='conv1', activation='elu')(x)
    x = Dropout(rate=0.05)(x, training=True)

    x = Convolution2D(36, (5, 5), strides=(2, 2), name='conv2', activation='elu')(x)
    x = Dropout(rate=0.05)(x, training=True)

    x = Convolution2D(48, (5, 5), strides=(2, 2), name='conv3', activation='elu')(x)
    x = Dropout(rate=0.05)(x, training=True)

    # 3x3 Convolutional layers with stride of 1x1
    x = Convolution2D(64, (3, 3), name='conv4', activation='elu')(x)
    x = Dropout(rate=0.05)(x, training=True)

    x = Convolution2D(64, (3, 3), name='conv5', activation='elu')(x)
    x = Dropout(rate=0.05)(x, training=True)

    # Flatten before passing to Fully Connected layers
    x = Flatten()(x)

    # Three fully connected layers
    x = Dense(100, name='fc1', activation='elu')(x)
    x = Dropout(rate=0.05)(x, training=True)

    x = Dense(50, name='fc2', activation='elu')(x)
    x = Dropout(rate=0.05)(x, training=True)

    x = Dense(10, name='fc3', activation='elu')(x)
    x = Dropout(rate=0.05)(x, training=True)

    # Output layer
    outputs = get_output_layers(x)

    return Model(inputs=[img_in], outputs=outputs)


def get_chauffeur_mc_dropout_model(input_shape=(140, 320, 3)):
    # from donkeycar.parts.keras import SpatialDropout2D

    if not input_shape:
        input_shape = (140, 320, 3)

    def get_convolution_kernels(n, kernel_size):
        return Convolution2D(n,
                             kernel_size,
                             kernel_initializer="he_normal",
                             bias_initializer="he_normal",
                             activation='relu',
                             padding='same')

    img_in = Input(shape=input_shape, name='img_in')
    x = img_in

    # (Convolution -> Spatial Dropout -> Max Pooling) x5
    x = Convolution2D(16, (5, 5), input_shape=input_shape, kernel_initializer="he_normal", bias_initializer="he_normal",
                      activation='relu', padding='same')(x)
    x = Dropout(0.05)(x, training=True)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = get_convolution_kernels(20, (5, 5))(x)
    x = Dropout(0.05)(x, training=True)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = get_convolution_kernels(40, (3, 3))(x)
    x = Dropout(0.05)(x, training=True)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = get_convolution_kernels(60, (3, 3))(x)
    x = Dropout(0.05)(x, training=True)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = get_convolution_kernels(80, (2, 2))(x)
    x = Dropout(0.05)(x, training=True)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = get_convolution_kernels(128, (2, 2))(x)
    x = Dropout(0.05)(x, training=True)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flattening and dropping-out
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    # Classification
    outputs = get_output_layers(x)

    return Model(inputs=[img_in], outputs=outputs)


# half
def get_epoch_mc_dropout_model(input_shape=(140, 320, 3)):
    if not input_shape:
        input_shape = (140, 320, 3)

    img_in = Input(shape=input_shape)
    x = img_in

    x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.1)(x, training=True)

    x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.1)(x, training=True)

    x = Convolution2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.1)(x, training=True)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x, training=True)

    outputs = get_output_layers(x)

    return Model(inputs=[img_in], outputs=outputs)


if __name__ == '__main__':
    in_size = (240 - 100, 320, 3)
    model_names = [DAVE2_MC, CHAUFFEUR_MC, EPOCH_MC]
    models = [get_own_model_with_dropout(model_name, in_size)().model for model_name in model_names]

    print(f"Number of model parameters for input of size of {in_size}:\n")
    for model, name in zip(models, model_names):
        print(f"{name} has {model.count_params()} parameters")
