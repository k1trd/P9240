from statistics import median

import util
import os
import cv2
import csv
import codecs
import json

import xml.etree.ElementTree as et
import numpy as np

DEV_MODE = False

DATA_PATH = ['data', 'breastpathq', 'validation']
ANOT_PATH = ['data', 'breastpathq', 'cells', 'Sedeen']
LABEL_PATH = ['data', 'breastpathq', 'val_labels.csv']

OUTPUT_DIR = 'validation'
OUTPUT_DATA_PATH = 'raw_data'
OUTPUT_DATA_PREFIX = 'raw_dataset'

CELL_EDGE_THRESHOLD = range(45, 46)
WHITE_THRESHOLD = range(79, 80)
COLOR_BOUNDARIES = [([30, 0, 40], [135, 255, 145])]
VESSEL_BASE = [80, 50, 200]
VESSEL_THRESHOLD = range(74, 75)

ANOT_POSTFIX = '.session'
ANOT_TYPE = '.xml'

EVALUATE_CELL_DETECTION = False

COUNT_LIMIT = -1

ADD_LABEL = True
SAVE_PROCESSED_DATA = True

APPEND_RAW = True
APPEND_CELL_EDGE_DETECT = False
APPEND_CELL_COLOR_DETECT = False
APPEND_WHITE_DETECT = False
APPEND_VESSEL_DETECT = False

# Classification Labels and BINS
# 0 = 0(Inclusive)- 0.1 (Exclusive)
# 1 = 0.1 (Inclusive) - 0.45 (Exclusive)
# 2 = 0.45 (Inclusive) - 1.0 (Inclusive)
CLASSIFY_LABELS = False
BIN_DATASET = False

# BIN_RANGE = [lower_bound(inclusive), upper_bound(exclusive)]
BIN_RANGE = [0.00, 0.10]
# BIN_RANGE = [0.10, 0.45]
# BIN_RANGE = [0.45, 1.1]

# Normalize Data
NORMALIZE_LABEL = True
NORM_VERSION = 2
NORMALIZE_MAP_FILE = ['data', 'normalize_label_mapping_v2.json']


def get_cell_count(path, name):
    file_name, file_type = os.path.splitext(name)
    file_name = '{}{}{}'.format(file_name, ANOT_POSTFIX, ANOT_TYPE)
    file_path = os.path.join(path, file_name)
    anot = et.parse(file_path).getroot()
    cells = sum(1 for x in anot.iter('point'))
    return cells


def evaluate_cell_detection(true_count, anot_size, over_pred, under_pred):
    if true_count < anot_size:
        over_pred.append((anot_size-true_count))
    elif true_count > anot_size:
        under_pred.append(true_count-anot_size)
    return over_pred, under_pred


def main():
    root_path = os.getcwd()
    util.create_dirs(OUTPUT_DIR)

    output_path = os.path.join(root_path, 'results', OUTPUT_DIR)
    input_path = os.path.join(root_path, *DATA_PATH)
    anot_path = os.path.join(root_path, *ANOT_PATH)
    label_path = os.path.join(root_path, *LABEL_PATH)

    img_util = util.ImgProcessor(OUTPUT_DIR, save_mode=False)

    labels = {}

    if NORMALIZE_LABEL:
        with open(os.path.join(root_path, *NORMALIZE_MAP_FILE)) as json_file:
            norm_label_map = json.load(json_file)

    if ADD_LABEL:
        print('Preparing Labels')
        with codecs.open(label_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            for row in reader:
                id = '{}_{}'.format(row[0], row[1])
                p = row[2]

                if NORMALIZE_LABEL:
                    labels[id] = norm_label_map[p]
                else:
                    labels[id] = float(p)


    print('Start Pre-processing: {}'.format(input_path))

    count = 0
    edge_over_pred = []
    edge_under_pred = []
    color_over_pred = []
    color_under_pred = []

    full_labels = []
    full_dataset = []

    for file_name in os.listdir(input_path):
        if COUNT_LIMIT != -1 and count >= COUNT_LIMIT:
            break

        file_path = os.path.join(input_path, file_name)
        if os.path.isfile(file_path):
            print("\tProcessing: {}".format(file_name))
            img = cv2.imread(file_path)
            processed_img = []

            name, type = os.path.splitext(file_name)
            info = name.split('_')
            slide = int(info[0])
            rid = int(info[1])

            if BIN_DATASET:
                p = labels.get(name, -1)
                if not (p == -1 or BIN_RANGE[0] <= p < BIN_RANGE[1]):
                    continue

            if APPEND_RAW:
                for i in range(3):
                    processed_img.append(img[:, :, i])

            if APPEND_CELL_EDGE_DETECT:
                if EVALUATE_CELL_DETECTION:
                    true_cell_count = get_cell_count(anot_path, file_name)

                for thr in CELL_EDGE_THRESHOLD:
                    new_img, anot_size = img_util.cell_detection(img, file_name, threshold=thr)
                    processed_img.append(new_img)
                    if EVALUATE_CELL_DETECTION:
                        edge_over_pred, edge_under_pred = evaluate_cell_detection(true_cell_count,
                                                                              anot_size, edge_over_pred, edge_under_pred)
                    if DEV_MODE:
                        break

            if APPEND_CELL_COLOR_DETECT:
                for boundary in COLOR_BOUNDARIES:
                    new_img, anot_size = img_util.cell_detection(img, file_name, mode=1, boundaries=boundary)
                    processed_img.append(new_img)
                    if EVALUATE_CELL_DETECTION:
                        color_over_pred, color_under_pred = evaluate_cell_detection(true_cell_count,
                                                                              anot_size, color_over_pred, color_under_pred)
                    if DEV_MODE:
                        break

            if APPEND_VESSEL_DETECT:
                for thr in VESSEL_THRESHOLD:
                    new_img = img_util.detect_red_vessels(img, file_name, bgr_base=VESSEL_BASE, threshold=thr)
                    processed_img.append(new_img)

                    if DEV_MODE:
                        break

            if APPEND_WHITE_DETECT:
                red_scale = util.grayscale_img('red', img)
                green_scale = util.grayscale_img('green', img)
                blue_scale = util.grayscale_img('blue', img)

                for thr in WHITE_THRESHOLD:
                    temp_img = img.copy()

                    new_img = img_util.detect_white_matter(temp_img, file_name, red_scale,
                                                           green_scale, blue_scale, threshold=thr)
                    processed_img.append(new_img)
                    if DEV_MODE:
                        break

            full_dataset.append(processed_img)

            if ADD_LABEL:
                if CLASSIFY_LABELS:
                    p = labels[name]
                    label = [0, 0, 0]
                    if 0 <= p < 0.1:
                        label[0] = 1
                    elif 0.1 <= p < 0.45:
                        label[1] = 1
                    else:
                        label[2] = 1
                else:
                    label = labels[name]
            else:
                label = 0
            data_row = [slide, rid, label]
            full_labels.append(data_row)

            count += 1

    print('Saving Dataset')

    if SAVE_PROCESSED_DATA:
        nb_batchs = int(len(full_dataset)/1000)
        prefix = OUTPUT_DATA_PREFIX

        if CLASSIFY_LABELS:
            prefix += '_class'

        if BIN_DATASET:
            prefix += '_bin({}-{})'.format(BIN_RANGE[0], BIN_RANGE[1])

        if NORMALIZE_LABEL:
            prefix = '{}_norm{}'.format(prefix, NORM_VERSION)

        for i in range(nb_batchs):
            output_file = '{}_{}'.format(prefix, str(i))
            lb = i*1000
            ub = lb+1000

            np.savez_compressed(os.path.join(output_path, OUTPUT_DATA_PATH, output_file), labels=full_labels[lb:ub],
                                dataset=full_dataset[lb:ub])

        remainder = len(full_dataset) % 1000

        output_file = '{}_{}'.format(prefix, str(nb_batchs))
        if remainder > 0:
            lb = nb_batchs * 1000
            ub = lb + remainder

            np.savez_compressed(os.path.join(output_path, OUTPUT_DATA_PATH, output_file), labels=full_labels[lb:ub],
                                    dataset=full_dataset[lb:ub])

    # Calculate Cell Detection Performance
    if EVALUATE_CELL_DETECTION:
        edge_avg_over_pred = 0
        edge_avg_under_pred = 0

        if len(edge_over_pred) > 0:
            edge_avg_over_pred = median(edge_over_pred)
        if len(edge_under_pred) > 0:
            edge_avg_under_pred = median(edge_under_pred)

        color_avg_over_pred = 0
        color_avg_under_pred = 0

        if len(color_over_pred) > 0:
            color_avg_over_pred = median(color_over_pred)
        if len(color_under_pred) > 0:
            color_avg_under_pred = median(color_under_pred)

        print('Performance:')
        print('\tEdge Avg Under Pred = {}'.format(edge_avg_under_pred))
        print('\tEdge Avg Over Pred = {}'.format(edge_avg_over_pred))
        print('\tColor Avg Under Pred = {}'.format(color_avg_under_pred))
        print('\tColor Avg Over Pred = {}'.format(color_avg_over_pred))

    print('Done Pre-processing')


if __name__ == "__main__":
    main()