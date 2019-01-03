import os
import csv
import codecs
import numpy as np

PATIENT_INFO_FILE = ['data', 'datasets_patient_ids.csv']


def train_val_split(features, labels, val_ratio=0.2):
    root_path = os.getcwd()
    patient_file = os.path.join(root_path, *PATIENT_INFO_FILE)
    slide2patient = {}

    with codecs.open(patient_file, "r", encoding="utf-8") as temp_file:
        reader = csv.reader(temp_file)
        next(reader, None)
        for row in reader:
            slide2patient[row[1]] = row[0]

    train_labels = []
    train_features = []
    val_labels = []
    val_features = []

    unique_slides = set(labels[:, 0])
    unique_patients = set()
    for slide_id in unique_slides:
        unique_patients.add(slide2patient[str(int(slide_id))])

    max_train_patient = int(len(unique_patients) * (1-val_ratio))
    nb_patients_added = 0
    patients_added = set()

    for i in range(len(labels)):
        slide = int(labels[i, 0])
        pat_id = slide2patient[str(slide)]

        if pat_id in patients_added:
            train_features.append(features[i])
            train_labels.append(labels[i])
        else:
            if nb_patients_added < max_train_patient:
                train_features.append(features[i])
                train_labels.append(labels[i])
                patients_added.add(pat_id)
                nb_patients_added += 1
            else:
                val_features.append(features[i])
                val_labels.append(labels[i])

    train_labels = np.asarray(train_labels)
    train_features = np.asarray(train_features)
    val_labels = np.asarray(val_labels)
    val_features = np.asarray(val_features)

    return train_features, val_features, train_labels, val_labels
