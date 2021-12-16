import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from cyclegan import CycleGan, gen_G, gen_F, disc_X, disc_Y, generator_loss_fn, discriminator_loss_fn


def load_data(actual_dir=None):
    list_files = sorted(os.listdir(actual_dir))
    print("Found %d images in %s" % (len(list_files), actual_dir))

    images = []
    for image in tqdm(list_files):
        path = os.path.join(actual_dir, image)
        if path.endswith(".jpg"):
            img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[100:, :]
            img = cv2.resize(img, (120, 160))
            # img = tf.cast(img, dtype=tf.float32)
            # img = (img / 127.5) - 1.0
            # plt.imshow(img)
            # plt.show()
            # print()
        else:
            continue

        images.append([img, image])

    return images


def translate(images, DIR_CYCLEGAN_MODEL, IS_SIM):
    # load cyclegan
    global DIR_PSEUDO_REAL, DIR_PSEUDO_SIM
    cyclegan_model = CycleGan(generator_G=gen_G,
                              generator_F=gen_F,
                              discriminator_X=disc_X,
                              discriminator_Y=disc_Y
                              )
    latest = tf.train.latest_checkpoint(DIR_CYCLEGAN_MODEL)
    cyclegan_model.load_weights(latest).expect_partial()
    print('Latest checkpoint restored:', latest)
    model_epoch = latest[-3:]

    # compile the model
    cyclegan_model.compile(
        gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
    )

    if IS_SIM:
        DIR_PSEUDO_REAL = os.path.join(os.getcwd(),
                                       "data", "training_xte_predictor",
                                       "sim2pseudoreal" + "-cg" + model_epoch)

        if not os.path.exists(DIR_PSEUDO_REAL):
            print("Created output directory %s" % DIR_PSEUDO_REAL)
            os.makedirs(DIR_PSEUDO_REAL)
    else:
        DIR_PSEUDO_SIM = os.path.join(os.getcwd(),
                                      "data", "training_xte_predictor",
                                      "real2pseudosim" + "-cg" + model_epoch)

        if not os.path.exists(DIR_PSEUDO_SIM):
            print("Created output directory %s" % DIR_PSEUDO_SIM)
            os.makedirs(DIR_PSEUDO_SIM)

    # _, ax = plt.subplots(4, 2, figsize=(10, 15))
    for i, img in enumerate(images):
        if i % 100 == 0 and i > 0:
            print("Translated %d/%d images (%d%%)" % (i, len(images), round(i / len(images) * 100)))

        test_image = img[0]
        name = img[1]

        img = test_image
        img = np.array([img])

        if IS_SIM:
            prediction = cyclegan_model.gen_G(img, training=False)[0].numpy()
        else:
            prediction = cyclegan_model.gen_F(img, training=False)[0].numpy()

        prediction = ((prediction * 127.5) + 127.5).astype(np.uint16)

        # img = (img * 127.5 + 127.5).astype(np.uint16)
        # img = np.squeeze(img, axis=0)

        # ax[i, 0].imshow(img)
        # ax[i, 1].imshow(prediction)
        # ax[i, 0].set_title("Input image %s" % name)
        # ax[i, 1].set_title("Translated image")
        # ax[i, 0].axis("off")
        # ax[i, 1].axis("off")

        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
        if IS_SIM:
            cv2.imwrite(os.path.join(DIR_PSEUDO_REAL, name), prediction)
        else:
            cv2.imwrite(os.path.join(DIR_PSEUDO_SIM, name), prediction)

        # plt.tight_layout()
        # plt.show()


def main():
    os.chdir('..')
    DIR_SIM = os.path.join(os.getcwd(), "data", "training_xte_predictor", "sim")
    DIR_REAL = os.path.join(os.getcwd(), "data", "training_xte_predictor", "real")
    DIR_CYCLEGAN_MODEL = os.path.join(os.getcwd(), "models", "cyclegan")

    IS_SIM = False

    actual_dir = DIR_SIM if IS_SIM else DIR_REAL
    images = load_data(actual_dir=actual_dir)
    translate(images, DIR_CYCLEGAN_MODEL, IS_SIM)


if __name__ == '__main__':
    main()
