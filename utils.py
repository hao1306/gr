from pyquaternion import Quaternion
from typing import Dict, Tuple, Any, List, Callable, Union
import cv2
import os
import numpy as np
import matplotlib.image as mpimg
import math

# declaration of image parameters
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 224, 224, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def angle_of_rotation(yaw: float) -> float:
    """
    Given a yaw angle (measured from x axis), find the angle needed to rotate by so that
    the yaw is aligned with the y axis (pi / 2).
    :param yaw: Radians. Output of quaternion_yaw function.
    :return: Angle in radians.
    """
    return (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)


def make_2d_rotation_matrix(angle_in_radians: float) -> np.ndarray:
    """
    Makes rotation matrix to rotate point in x-y plane counterclockwise
    by angle_in_radians.
    """

    return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                     [np.sin(angle_in_radians), np.cos(angle_in_radians)]])


def convert_global_coords_to_local(coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   rotation: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector. The rotation is meant to be a z-axis rotation.
    :param coordinates: x,y locations. array of shape [n_steps, 2].
    :param translation: Tuple of (x, y, z) location that is the center of the new frame.
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations in frame stored in array of share [n_times, 2].
    """
    yaw = angle_of_rotation(quaternion_yaw(Quaternion(rotation)))

    transform = make_2d_rotation_matrix(angle_in_radians=yaw)

    coords = (coordinates - np.atleast_2d(np.array(translation)[:2])).T

    return np.dot(transform, coords).T[:, :2]


def convert_local_coords_to_global(coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   rotation: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Converts local coordinates to global coordinates.
    :param coordinates: x,y locations. array of shape [n_steps, 2]
    :param translation: Tuple of (x, y, z) location that is the center of the new frame
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations stored in array of share [n_times, 2].
    """
    yaw = angle_of_rotation(quaternion_yaw(Quaternion(rotation)))

    transform = make_2d_rotation_matrix(angle_in_radians=-yaw)

    return np.dot(transform, coordinates.T).T[:, :2] + np.atleast_2d(np.array(translation)[:2])

class LetterBox:
    # YOLOv5 LetterBox class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, size, auto=False, stride=32):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top:top + h, left:left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


"""
* @brief Function to crop the imeages to the required shape
* @param The image to crop
* @return The cropped image
"""


def crop(image):
    # Crop the image (removing the sky at the top and the car front at the bottom)

    return image[20:-20, 90:-90, :]  # remove the sky and the car front


"""
* @brief Resize the image to the input shape used by the network model
* @param Image file to Resize
* @return The Resized image for the network
"""


def resize(image):
    letterbox = LetterBox(size=(1600, 900))
    
    output = letterbox(image)
    #print(output.shape)
    return output


"""
* @brief Fuinction to convert the color space of the image from RGB to YUV.
* @param The image to change the colorspace.
* @return The converted image. 
"""


def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


"""
* @brief Function to run the preprocess of the images.
* @param The images from the dataset
* @return The preprocessed images.
"""


def preprocess(image):
    # image = crop(image)
    # image = resize(image)
    image = cv2.resize(image, (450, 800))
    image = random_brightness(image)
    image = random_shadow(image)
    image = rgb2yuv(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def preprocess_all(image):
    # image = crop(image)
    # image = cv2.resize(image(450, 800))
    image = random_brightness(image)
    image = random_shadow(image)
    image = rgb2yuv(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image



def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle



"""
* @brief Randomly generates shadows in the image.
* @param The image to add shadow on.
* @return The image with added shadows.
"""


def random_shadow(image):
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    mask = np.zeros_like(image[:, :, 1])
    mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


"""
* @brief Function to adjust the brightness of the images.
* @param The image to process
* @return The brightness equalized image.
"""


def random_brightness(image):
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
