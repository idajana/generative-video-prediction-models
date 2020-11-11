#!/usr/bin/python3

import cv2
import numpy as np


def depthFromValue(value, depth_min, depth_max):
    return depth_max + value.astype(float) / 255 * (depth_min - depth_max)


def valueFromDepth(d, depth_min, depth_max):
    return np.maximum(np.minimum(np.round(255 * (d - depth_max) / (depth_min - depth_max)), 255), 0)


def translate(image, pixel_size, depth_min, depth_max, translation):
    M = np.float32([[1, 0, pixel_size * translation[0]], [0, 1, pixel_size * translation[1]]])
    result = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    translation_brightness = int(round(255.0 / (depth_max - depth_min) * translation[2]))

    asdf = result[:, :].astype(float) + translation_brightness
    result[:, :] = np.maximum(np.minimum(asdf[:, :], 255), 0)
    return result


def rotateX(image, pixel_size, depth_min, depth_max, angle, vector):
    result = np.zeros(image.shape, np.uint8)

    j = np.arange(image.shape[1])
    for i in range(image.shape[0]):
        y = (i - image.shape[0] / 2) / pixel_size + vector[0]

        d = depthFromValue(image[i], depth_min, depth_max) - vector[1]

        y_new = y * np.cos(angle) - d * np.sin(angle)
        d_new = y * np.sin(angle) + d * np.cos(angle)
        i_new = np.maximum(np.minimum(np.round((y_new - vector[0]) * pixel_size + image.shape[0] / 2), image.shape[0] - 1), 0).astype(int)
        value_new = np.maximum(valueFromDepth(d_new + vector[1], depth_min, depth_max), result[i_new, j]).astype(np.uint8)

        mask = np.zeros(image.shape, np.bool)
        mask[i_new, j] = (image[i] != 0)  # Don't change value if image is black and depth unknown
        np.putmask(result, mask, value_new)

    return result


# Not tested
def rotateY(image, pixel_size, depth_min, depth_max, angle, vector):
    result = np.zeros(image.shape, np.uint8)

    i = np.arange(image.shape[0])
    for j in range(image.shape[1]):
        x = (j - image.shape[1] / 2) / pixel_size + vector[0]

        d = depthFromValue(image[:, j], depth_min, depth_max) - vector[1]

        x_new = x * np.cos(angle) - d * np.sin(angle)
        d_new = x * np.sin(angle) + d * np.cos(angle)
        j_new = np.maximum(np.minimum(np.round((x_new - vector[0]) * pixel_size + image.shape[1] / 2), image.shape[1] - 1), 0).astype(int)
        value_new = np.maximum(valueFromDepth(d_new + vector[1], depth_min, depth_max), result[i, j_new]).astype(np.uint8)

        mask = np.zeros(image.shape, np.bool)
        mask[i, j_new] = (image[i, j_new] != 0)  # Don't change value if image is black and depth unknown
        np.putmask(result, mask, value_new)

        # j_new = max(min(int(round((x_new - vector[0]) * pixel_size + image.shape[1] / 2)), image.shape[1] - 1), 0)
        # result[i][j_new] = max(valueFromDepth(d_new + vector[1], depth_min, depth_max), result[i][j_new])

    return result


if __name__ == '__main__':
    file_directory = 'images/'
    image = cv2.imread(file_directory + 'cylinder_r_193_h_647.urdf_angle_-1.5707963267948966.png', 0)
    image = rotateX(image, 20.0, 15.0, 40.0, -0.5, (0.0, 25.0))
    cv2.imwrite(file_directory + 'side-calculated.png', image)
