'''
Adapted from https://github.com/wuhanstudio/adversarial-driving/blob/master/model/adversarial_driving.py
Credits: Wu Han
@misc{wu2021adversarial,
      title={Adversarial Driving: Attacking End-to-End Autonomous Driving Systems},
      author={Han Wu, Syed Yunas and Wenjie Ruan},
      year={2021},
      eprint={2103.09151},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
'''

import argparse
import os
import pickle

import cv2
import donkeycar as dk
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from visualodometry.utils import normalize_and_crop

tf.compat.v1.disable_eager_execution()
import tensorflow.python.keras.backend as K

# Definitions
NO_LEFT = 'noleft'
MODEL_PATH = 'model'
TRAINING_SET_PATH = 'data'


class AdversarialDriving:

    def __init__(self, model):

        self.model = model

        # Initialize FGSM Attack
        # Get the loss and gradient of the loss wrt the inputs
        self.attack_type = "fgsmr_left"
        self.activate = False
        self.epsilon = 1

        # MINOR MODIFICATION: Our models predict both steering and throttle commands, but we attack on the steering only
        # self.loss = K.mean(-self.model.output, axis=-1)
        self.loss = K.mean(-self.model.output[0], axis=-1)

        self.grads = K.gradients(self.loss, self.model.input)

        # Get the sign of the gradient
        self.delta = K.sign(self.grads[0])

        self.sess = tf.compat.v1.keras.backend.get_session()

        self.perturb = 0
        self.perturbs = []
        self.perturb_percent = 0
        self.perturb_percents = []
        self.n_attack = 1

        # self.unir_no_left = np.zeros((1, 160, 320, 3))
        # self.unir_no_right = np.zeros((1, 160, 320, 3))

        self.unir_no_left = np.zeros((1, 66, 200, 3))
        self.unir_no_right = np.zeros((1, 66, 200, 3))

        self.result = {}

    def init(self, attack_type, activate):

        # Reset Training Process
        if self.attack_type != attack_type:
            self.perturb = 0
            self.perturbs = []
            self.perturb_percent = 0
            self.perturb_percents = []
            self.n_attack = 1

        self.attack_type = attack_type

        if activate == 1:
            self.activate = True
            print("Attacker:", attack_type)
        else:
            self.activate = False
            print("No Attack")

            # Initialize FGSM Attack
            # Get the loss and gradient of the loss wrt the inputs
            if attack_type == "fgsmr_left" or attack_type == "unir_no_right" or attack_type == "unir_no_right_train":
                # self.loss = -self.model.output
                self.loss = -self.model.output[0]
            if attack_type == "fgsmr_right" or attack_type == "unir_no_left" or attack_type == "unir_no_left_train":
                # self.loss = self.model.output
                self.loss = self.model.output[0]

            self.grads = K.gradients(self.loss, self.model.input)
            # Get the sign of the gradient
            self.delta = K.sign(self.grads[0])

            # Save Universal Adversarial Perturbation
            if attack_type == "unir_no_left_train":
                pickle.dump(self.unir_no_left, open("unir_no_left.pickle", "wb"))
            if attack_type == "unir_no_right_train":
                pickle.dump(self.unir_no_right, open("unir_no_right.pickle", "wb"))

            print("Initialized", attack_type)

    def set_unir_no_left(self, unir_no_left):
        self.unir_no_left = unir_no_left

    def set_unir_no_right(self, unir_no_right):
        self.unir_no_right = unir_no_right

    # Deep Fool: Project on the lp ball centered at 0 and of radius xi
    def proj_lp(self, v, xi=10, p=2):

        # SUPPORTS only p = 2 and p = Inf for now
        if p == 2:
            v = v * min(1, xi / np.linalg.norm(v.flatten('C')))
            # v = v / np.linalg.norm(v.flatten(1)) * xi
        elif p == np.inf:
            v = np.sign(v) * np.minimum(abs(v), xi)
        else:
            raise ValueError('Values of p different from 2 and Inf are currently not supported...')

        return v

    def attack(self, input):

        # Random Noise
        if self.attack_type == "random":
            # Random Noise [-1, 1]
            # noise = np.random.randint(2, size=(160, 320, 3)) - 1
            noise = np.random.randint(2, size=(66, 200, 3)) - 1
            return noise

        # Apply FGSM Attack
        if self.attack_type.startswith("fgsmr_"):
            # Perturb the image
            noise = self.epsilon * self.sess.run(self.delta, feed_dict={self.model.input: np.array([input])})
            # return noise.reshape(160, 320, 3)
            return noise.reshape(66, 200, 3)

        # Apply Universal Adversarial Perturbation
        if self.attack_type == "unir_no_left":
            # return self.unir_no_left.reshape(160, 320, 3)
            return self.unir_no_left.reshape(66, 200, 3)

        if self.attack_type == "unir_no_right":
            # return self.unir_no_right.reshape(160, 320, 3)
            return self.unir_no_right.reshape(66, 200, 3)

        # Train Universal Perturbation
        if self.attack_type == "unir_no_left_train" or self.attack_type == "unir_no_right_train":

            image = np.array([input])

            # MINOR MODIFICATION: Our model predicts both steer and throttle, but we are only interested in the steer
            # y_true = float(self.model.predict(image, batch_size=1))
            y_true = float(self.model.predict(image, batch_size=1)[0])

            target_sign = 0

            if self.attack_type == "unir_no_left_train":
                target_sign = 1
            if self.attack_type == "unir_no_right_train":
                target_sign = -1
            if np.sign(y_true) != target_sign:
                x_adv = image
                y_h = y_true

                while (np.sign(y_h) != target_sign):
                    # print("Attack: ", y_h)
                    grads_array = self.sess.run(self.grads, feed_dict={self.model.input: np.array(x_adv)})
                    grads_array = np.array(grads_array[0])

                    # Fix
                    if np.isclose(np.linalg.norm(grads_array.flatten()), 0):
                        break

                    grads_array = 50 * grads_array / np.linalg.norm(grads_array.flatten())

                    x_adv = x_adv + grads_array

                    # y_h = self.model.predict(x_adv, batch_size=1)
                    y_h = self.model.predict(x_adv, batch_size=1)[0]

                    # print("After DeepFool: ", y_true, " --> ", y_h)

                if self.attack_type == "unir_no_left_train":
                    self.unir_no_left = self.unir_no_left + (x_adv - image)
                    # Project on l_p ball
                    self.unir_no_left = self.proj_lp(self.unir_no_left)

                    # y_uni = self.model.predict(image + self.unir_no_left, batch_size=1)
                    y_uni = self.model.predict(image + self.unir_no_left, batch_size=1)[0]

                if self.attack_type == "unir_no_right_train":
                    self.unir_no_right = self.unir_no_right + (x_adv - image)
                    # Project on l_p ball
                    self.unir_no_right = self.proj_lp(self.unir_no_right)

                    # y_uni = self.model.predict(image + self.unir_no_right, batch_size=1)
                    y_uni = self.model.predict(image + self.unir_no_right, batch_size=1)[0]

                # print("After Universal: ", y_true, " --> ", y_uni)
                self.perturb = self.perturb + (y_uni - y_true)
                self.perturbs.append(float(self.perturb / self.n_attack))
                self.perturb_percent = self.perturb_percent + (y_uni - y_true) / (np.abs(y_true))
                self.perturb_percents.append(float(self.perturb_percent * 100 / self.n_attack))
                self.n_attack = self.n_attack + 1

            return 0


def parse_arguments():
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument(f'--{NO_LEFT}',
                                 action="store_true",
                                 help="Whether to create a universal image that will make predictions deviate towards left (true) or right (false)")

    argument_parser.add_argument(f"--{MODEL_PATH}",
                                 type=str,
                                 help="Path to the model to perturbate")

    argument_parser.add_argument(f'--{TRAINING_SET_PATH}',
                                 type=str,
                                 help="Path to the training set used for training the model"
                                 )

    return vars(argument_parser.parse_args())


def main():
    # Getting program arguments
    args = parse_arguments()
    print(args)

    # Loading the model
    model = keras.models.load_model(args[MODEL_PATH])

    # Defining the type of the attack
    attack_type = 'unir_no_left' if args[NO_LEFT] else 'unir_no_right'

    # Creating the Adversarial Driving class
    adversarial_driving = AdversarialDriving(model)
    adversarial_driving.init(attack_type=attack_type + "_train", activate=False)

    cfg = dk.load_config()

    # Iterating over the training images to get the universal attack
    for filename in tqdm([fn for fn in os.listdir(args[TRAINING_SET_PATH]) if '.jpg' in fn]):
        # Reading the image and converting it to np.array
        path = os.path.join(args[TRAINING_SET_PATH], filename)
        image = np.array(cv2.imread(path))

        # Cropping image
        image = normalize_and_crop(image, cfg)
        image = cv2.resize(image, (200, 66))  # for DAVE2
        # image = image[100:, :]
        # image = image / 255

        # Using the image to update the universal attack image
        adversarial_driving.attack(image)

    # Dumping the obtained perturbation
    adversarial_driving.init(attack_type=attack_type + "_train", activate=False)

    # Displaying the perturbation image
    file = open(attack_type + '.pickle', 'rb')
    perturbation = pickle.load(file)[0]
    file.close()

    # Storing the image as a .png file
    cv2.imwrite(f"perturbation_no{'left' if args[NO_LEFT] else 'right'}.png", perturbation)

    # Displaying the perturbation (exaggerating noise)
    plt.imshow(perturbation * 255 / np.max(perturbation))
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
