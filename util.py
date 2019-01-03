import os

import matplotlib.pyplot as plt

from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.filters import sobel

from scipy import ndimage

import numpy as np
import cv2


RESULTS_DIR = 'results'
SUB_DIRS = ['cell_detection', 'white_detection', 'red_detection', 'processed_data']
CELL_DIRS = ['edge', 'color']

DEV_MODE = False


def build_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_dirs(sub_name):
    root_path = os.getcwd()
    results_path = os.path.join(root_path, RESULTS_DIR)
    subroot_path = os.path.join(results_path, sub_name)

    for sub_dir in SUB_DIRS:
        sub_path = os.path.join(subroot_path, sub_dir)
        if sub_dir == 'cell_detection':
            for cell_dir in CELL_DIRS:
                cell_path = os.path.join(sub_path, cell_dir)
                build_dir(cell_path)
        else:
            build_dir(sub_path)


def show_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def show_ind_channels(img):
    figure, plots = plt.subplots(ncols=3, nrows=1)
    for i, subplot in zip(range(3), plots):
        temp = img[:, :, i]
        subplot.imshow(temp, cmap='gray')
        subplot.set_axis_off()
    plt.show()


def grayscale_img(alg, img):

    if alg == 'avg':
        return (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3

    if alg == 'luma':
        return 0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 2]

    if alg == 'desat':
        temp = np.maximum(img[:, :, 0], img[:, :, 1])
        a = np.maximum(temp, img[:, :, 2])
        temp = np.minimum(img[:, :, 0], img[:, :, 1])
        b = np.minimum(temp, img[:, :, 2])
        return (a + b) / 2

    if alg == 'dcmin':
        temp = np.minimum(img[:, :, 0], img[:, :, 1])
        return np.minimum(temp, img[:, :, 2])

    if alg == 'dcmax':
        temp = np.maximum(img[:, :, 0], img[:, :, 1])
        return np.maximum(temp, img[:, :, 2])

    if alg == 'red':
        return img[:, :, 0]

    if alg == 'green':
        return img[:, :, 1]

    if alg == 'blue':
        return img[:, :, 2]


def invert_img(img):
    return 255 - img


def impose_img(lines, pic):
    out = pic.copy()
    out[lines > 0.5] = [255,255,255,0]
    return out


def apply_threshold(img, thresh):
    t_value = thresh/100 * 255
    img[img < t_value] = 0
    img[img >= t_value] = 255
    return img


def apply_smoothing(img):
    return median(img, disk(5))


def apply_filler(img):
    return ndimage.binary_fill_holes(img, structure=np.ones((15,15)))


def normalize(img):
    return img/255


def conv2uint8(img):
    return np.array(img, dtype="uint8") * 250


class ImgProcessor:

    def __init__(self, sub_name, gray_scale_mode='dcmax', save_mode=True):
        root_path = os.getcwd()
        results_path = os.path.join(root_path, RESULTS_DIR)
        self.subroot_path = os.path.join(results_path, sub_name)
        self.gray_mode = gray_scale_mode
        self.save = save_mode

    def save_img(self, sub_dir, file_name, img):
        output_file = os.path.join(self.subroot_path, *sub_dir)
        output_file = os.path.join(output_file, file_name)

        cv2.imwrite(output_file, img)

    def detect_red_vessels(self, org_img, name, bgr_base, threshold):

        lower = np.array([max(0, x - threshold) for x in bgr_base], dtype="uint8")
        upper = np.array([min(255, x + threshold) for x in bgr_base], dtype="uint8")

        mask = cv2.inRange(org_img, lower, upper)
        img_masked = cv2.bitwise_and(org_img, org_img, mask=mask)

        img_gray = grayscale_img(self.gray_mode, img_masked)
        img_smooth = apply_smoothing(img_gray)

        if self.save:
            file_name, file_type = os.path.splitext(name)
            output_name = '{}_{}{}'.format(file_name, post_fix, file_type)
            self.save_img(sub_dir, output_name, img_smooth*255)

        return normalize(img_smooth)

    def detect_white_matter(self, org_img, name, red_scale, green_scale, blue_scale, threshold=100):
        t_value = threshold/100*255

        f_img = org_img
        org_img[red_scale < t_value] = [0, 0, 0]
        org_img[green_scale < t_value] = [0, 0, 0]
        org_img[blue_scale < t_value] = [0, 0, 0]

        org_img[red_scale >= t_value] = [255, 255, 255]
        org_img[green_scale >= t_value] = [255, 255, 255]
        org_img[blue_scale >= t_value] = [255, 255, 255]

        img_gray = grayscale_img(self.gray_mode, f_img)
        img_inv = invert_img(img_gray)
        img_filled = apply_filler(img_inv)
        img_smooth = apply_smoothing(img_filled)
        img_fill = apply_filler(img_smooth)
        img_proc = invert_img(img_fill*255)

        if DEV_MODE:
            show_img(img_fill)

        if self.save:
            sub_dir = ['white_detection']
            file_name, file_type = os.path.splitext(name)
            post_fix = 'white_th-{}'.format(threshold)
            output_name = '{}_{}{}'.format(file_name, post_fix, file_type)
            self.save_img(sub_dir, output_name, img_proc)

        return normalize(img_proc)

    def cell_detection(self, org_img, name, mode=0, threshold=100, boundaries=None):

        sub_dir = ['cell_detection']
        post_fix = 'cell_'
        if mode == 1:
            lower = boundaries[0]
            upper = boundaries[1]
            mask = cv2.inRange(org_img, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8"))
            img = cv2.bitwise_and(org_img, org_img, mask=mask)
            sub_dir.append('color')
            post_fix += 'color_(lb-{}_ub{}-)'.format(','.join(str(x) for x in lower),
                                                     ','.join(str(x) for x in upper))
        else:
            img = org_img
            sub_dir.append('edge')
            post_fix += 'edge_th-{}'.format(threshold)

        img_gray = grayscale_img(self.gray_mode, img)

        img_inv = invert_img(img_gray)

        img_thresh = apply_threshold(img_inv, threshold)
        img_smooth = apply_smoothing(img_thresh)
        img_filled = apply_filler(img_smooth)
        img_proc = conv2uint8(img_filled)
        img_edges = sobel(img_filled)

        if mode == 1:
            img_proc = invert_img(img_proc)

        _, contours, _ = cv2.findContours(img_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if DEV_MODE:
            show_img(img_gray)
            show_img(img_proc)
            show_img(img_edges)

        if self.save:
            file_name, file_type = os.path.splitext(name)
            output_name = '{}_{}{}'.format(file_name, post_fix, file_type)
            self.save_img(sub_dir, output_name, img_edges*255)

        return normalize(img_proc), len(contours)


def test_util():
    create_dirs('test')


if __name__ == "__main__":
    test_util()