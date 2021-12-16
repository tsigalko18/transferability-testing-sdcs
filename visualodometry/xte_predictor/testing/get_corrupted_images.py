from visualodometry.corruptions.hendrycks import *

warnings.filterwarnings('ignore')

from PIL import Image

import os

import numpy as np

DIR = ''  # directory with images here


def img_crop(img_arr, top, bottom):
    if bottom is 0:
        end = img_arr.shape[0]
    else:
        end = -bottom
    return img_arr[top:end, ...]


def normalize_and_crop(img_arr):
    img_arr = img_arr.astype(np.float32) * 1.0 / 255.0

    img_arr = img_crop(img_arr, 100, 0)
    if len(img_arr.shape) == 2:
        img_arrH = img_arr.shape[0]
        img_arrW = img_arr.shape[1]
        img_arr = img_arr.reshape(img_arrH, img_arrW, 1)
    return img_arr


def load_PIL_img(filename):
    try:
        img = Image.open(filename)
        img = img.convert('RGB')
    except Exception as e:
        print(e)
        print('failed to load PIL image:', filename)
        img = None
    return img


def load_scaled_image_arr(filename):
    '''
    load an image from the filename, and use the cfg to resize if needed
    also apply cropping and normalize
    '''
    try:
        img = Image.open(filename)
        img_arr = np.array(img)
        img_arr = normalize_and_crop(img_arr)
    except Exception as e:
        print(e)
        print('failed to load image:', filename)
        img_arr = None
    return img_arr


def get_inputs_by_path(path, data):
    print(path)
    inputs = []

    print("\nCollecting {} images_paths ...".format(len(data)))
    for img_name in data:
        # Reading the image
        img = load_PIL_img(os.path.join(path, img_name))
        img = img.resize((224, 224))
        # plt.imshow(img)
        # plt.show()

        inputs.append(img)
    print("{} images_paths collected.".format(len(inputs)))
    return inputs


def corrupt(image, corruption, severity):
    """
    Corruptions
    1. gaussian_noise
    2. shot_noise
    3. impulse_noise
    4. speckle_noise
    5. gaussian_blur
    6. glass_blur
    7. defocus_blur
    8. motion_blur
    9. zoom_blur
    10. fog
    11. frost
    12. snow
    13. spatter only [1-3] look like rain drops [4-5] mud
    14. contrast
    15. brightness
    16. saturate
    17. jpeg_compression
    18. pixelate
    19. elastic_transform
    """

    im = None

    if severity not in np.arange(1, 6):
        print("Severity must range between 1 and 5, value %s is invalid" % str(severity))
        exit()

    if corruption.lower() == "gaussian_noise":
        im = gaussian_noise(image, severity=severity)
    elif corruption.lower() == "shot_noise":
        im = shot_noise(image, severity=severity)
    elif corruption.lower() == "impulse_noise":
        im = impulse_noise(image, severity=severity)
    elif corruption.lower() == "speckle_noise":
        im = speckle_noise(image, severity=severity)
    elif corruption.lower() == "gaussian_blur":
        im = gaussian_blur(image, severity=severity)
    elif corruption.lower() == "glass_blur":
        im = glass_blur(image, severity=severity)
    elif corruption.lower() == "defocus_blur":
        im = defocus_blur(image, severity=severity)
    elif corruption.lower() == "motion_blur":
        im = motion_blur(image, severity=severity)
    elif corruption.lower() == "zoom_blur":
        im = zoom_blur(image, severity=severity)
    elif corruption.lower() == "fog":
        im = fog(image, severity=severity)
    elif corruption.lower() == "frost":
        im = frost(image, severity=severity)
    elif corruption.lower() == "snow":
        im = snow(image, severity=severity)
    elif corruption.lower() == "spatter":
        im = spatter(image, severity=severity)
    elif corruption.lower() == "contrast":
        im = contrast(image, severity=severity)
    elif corruption.lower() == "brightness":
        im = brightness(image, severity=severity)
    elif corruption.lower() == "saturate":
        im = saturate(image, severity=severity)
    elif corruption.lower() == "jpeg_compression":
        im = jpeg_compression(image, severity=severity)
    elif corruption.lower() == "pixelate":
        im = pixelate(image, severity=severity)
    elif corruption.lower() == "elastic_transform":
        im = elastic_transform(image, severity=severity)
    else:
        print("Corruption %s is invalid" % str(corruption))
        exit()

    if im is None:
        print("Corruption %s not applied" % corruption)
        exit()

    if isinstance(im, np.ndarray):  # some corruptions return a numpy array
        im = Image.fromarray(np.uint8(im)).convert('RGB')  # convert them back to PILImage

    return im


def corrupt_images(name, images, corruption, severity):
    inputs = get_inputs_by_path(os.path.join(os.getcwd(),
                                             DIR,
                                             name.split("-Run")[0],
                                             name.replace(".csv", "")), images)

    for image in inputs:
        corrupt(image, corruption, severity)
