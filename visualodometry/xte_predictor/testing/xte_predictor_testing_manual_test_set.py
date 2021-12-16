import warnings

warnings.filterwarnings('ignore')

import os
import cv2

import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras

PREDICTOR_SIM = 'predictor_sim'
SHOW_PREDICTIONS = 'show_predictions'
STORE_PREDICTIONS = 'store_predictions'


def get_arguments():
    import argparse
    parser = argparse.ArgumentParser()

    # parser.add_argument('--' + PREDICTOR_SIM, action='store_true',
    #                     help="Whether the predictor takes real images_paths or pseudo-sim images_paths")
    parser.add_argument('--' + SHOW_PREDICTIONS, action='store_true', help='Path to the model')
    # parser.add_argument('--' + STORE_PREDICTIONS, action='store_true', help="Whether to store XTE predictions or not")

    args = vars(parser.parse_args())

    return args


def load_data(predictor_real, crop=100, size=(256, 256)):
    # Getting paths depending on the type of dataset (either real images_real or pseudo-sim)
    xte_csv_path = 'xte_predictor_testset/xtes.csv'
    images_path = 'xte_predictor_testset/images_real' if predictor_real else 'xte_predictor_testset/images_pseudosim'

    # Getting the labels (XTEs)
    frames = pd.read_csv(xte_csv_path, sep=';')
    labels = list(frames['xte'])

    # Getting the images_real
    images = []
    images_names = sorted(
        [file_name for file_name in os.listdir(images_path) if '.jpg' in file_name or 'png' in file_name],
        key=lambda name: int(name.split('_')[0]))

    for name in images_names:
        path = os.path.join(images_path, name)
        image = cv2.imread(path)

        # Real images_paths aren't cropped in folder
        if predictor_real:
            image = image[crop:, :]

        image = cv2.resize(image, size)
        # plt.imshow(image)
        # plt.show()
        images.append(image)

    return np.array(images), np.array(labels)


def store_predictions(predictions, xte_predictor_real=True):
    file = open("Predictions_Real.csv" if xte_predictor_real else "Predictions_Sim.csv", 'w')
    for i in range(len(predictions)):
        p = predictions[i]
        file.write(str(p[0]) + "\n")
    file.close()


def show_predictions(X, Y, y_hat):
    for image, xte, xte_hat in zip(X, Y, y_hat):
        img = image
        cv2.putText(img, "~XTE: " + str(xte_hat),
                    (64, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2)
        cv2.imshow("XTE: " + str(xte), img)
        cv2.waitKey(100)
    cv2.waitKey()


def plot_absolute_errors_barchart(errors, title='XTE Predictor: Absolute error distribution on test set'):
    bin_size = 0.05
    bins = np.zeros(int(np.ceil(np.max(np.abs(errors)) / bin_size)))

    for error in np.sort(np.abs(errors)):
        idx = int(error / bin_size)
        bins[idx] += 1

    fig, ax = plt.subplots()
    x = np.arange(len(bins)) * bin_size
    ax.bar(x, bins, 0.03, label='Absolute Error')

    ax.set_ylabel('Cardinality')
    ax.set_xlabel("|Error|")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    plt.show()


def print_mean_errors_for_classes(errors, Y):
    error_classes = [[], [], [], [], []]
    for err, Y in zip(errors, Y):
        label = int(Y)
        error_classes[label].append(np.abs(err))

    for i in [0, 1, 2, -1, -2]:
        err_class = error_classes[i]
        print("Mean absolute error for 'class' {} is: {:.4f} (class has {} images_real)".format(i, np.mean(err_class),
                                                                                                len(err_class)))


def print_mean_squared_errors_for_classes(errors, Y):
    error_classes = [[], [], [], [], []]
    for err, Y in zip(errors, Y):
        label = int(Y)
        error_classes[label].append(err ** 2)

    for i in [0, 1, 2, -1, -2]:
        err_class = error_classes[i]
        print("Mean squared error for 'class' {} is: {:.4f} (class has {} images_real)".format(i, np.mean(err_class),
                                                                                               len(err_class)))


def main(prediction_real=True):
    # Getting program args
    args = get_arguments()
    print(args)

    # Loading model
    model_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                              'xte_predictor_b.h5' if not prediction_real else 'xte_predictor_a.h5'
                              )
    model = keras.models.load_model(model_path)

    # Loading the data
    X, Y = load_data(prediction_real)

    # Getting predictions
    y_hat = model.predict(X)

    if args[SHOW_PREDICTIONS]:
        # Showing predictions
        show_predictions(X, Y, y_hat)

    errors = []
    for i in range(len(y_hat)):
        error = y_hat[i] - Y[i]
        error = abs(error)

        # if error < 0.20:  # tolerance level
        #     error = [0]

        errors.append(error)

    errors = np.array(errors)

    # if args[STORE_PREDICTIONS]:
    store_predictions(errors, prediction_real)

    print("MSE: ", np.mean(errors ** 2))
    print("MAE: ", np.mean(np.abs(errors)))
    print("Absolute Error mode: {}\n".format(np.percentile(np.abs(errors), 50)))

    # Plotting the barchart about the distribution of errors
    title = 'Visual Odometry Real:' if prediction_real else 'Visual Odometry Sim:'
    title += ' Absolute error distribution on test set'
    plot_absolute_errors_barchart(errors, title)

    # Mean errors for "classes" (-2, -1, 0, 1, 2)
    print_mean_errors_for_classes(errors, Y)

    # Mean squared errors for "classes" (-2, -1, 0, 1, 2)
    print_mean_squared_errors_for_classes(errors, Y)


if __name__ == '__main__':
    main(prediction_real=True)
    main(prediction_real=False)
