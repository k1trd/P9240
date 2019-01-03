from sklearn.metrics import mean_squared_error
from math import sqrt

from keras.models import Input, Model, load_model
from keras.optimizers import Adam, SGD, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping

import data_manager
import inception
import prediction_probability

import os
import csv
import codecs
import time
import numpy as np
import keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Modes: 0 = training | 1 = prediction
MODE = 0

# ------------------
# Training Params
# ------------------

# Half Dataset
TRAINING_SPLIT_DATA_PATH = ['results', 'train_split2', 'raw_data']
VALIDATION_SPLIT_DATA_PATH = ['results', 'validation_split2', 'raw_data']

# Full Data
# TRAINING_SPLIT_DATA_PATH = ['results', 'train', 'raw_data']
# VALIDATION_SPLIT_DATA_PATH = []

TEST_SPLIT_DATA_PATH = ['results', 'validation', 'raw_data']

SUMMARY_TRAINING_LOG = ['results', 'training_summary_log.csv']
HISTORY_TRAINING_PATH = ['results']
HISTORY_PREFIX = 'training_history_log'
PREDICTION_LOG = 'training_prediction_log'

MODELS_PATH = ['models']
# DATA_FILE = 'processed_dataset'
DATA_FILE = 'raw_dataset_norm3'

# INPUT_SHAPE = (4, 512, 512)
INPUT_SHAPE = (3, 512, 512)
ENABLE_MODEL_REDUCTION = True
PRE_TRAINED_MODEL = ['models', 'model_average_e(100)_bs(10)_mr(True)_i(1)_test-rmse(0.099404)_test-pk(0.909051).hdf5']
# PRE_TRAINED_MODEL = ['models','models_average_e().hdf5']
LOAD_WEIGHTS_NEW_MODEL = []

# LOSS_FUNCTION = 'mean_squared_error'
LOSS_FUNCTION = 'root_mean_squared_error'
LEARNING_OPTIMIZER = Adam(decay=1e-10, clipnorm=1.0, clipvalue=0.5)  # adam | sgd | adagrad

NUM_EPOCHS = 10
BATCH_SIZE = 10
VALIDATION_RATIO = 0.2

# ROTATIONS: 1 = 0 Deg, 2 = 90 Deg, 3 = 180 Deg, 4 = 270 Deg
NB_ROTATION = 1
ENABLE_VERTICAL_MIRROR = True
ENABLE_HORIZONTAL_MIRROR = True

# Model Selection Params
TOP_K_MODEL_SAVE = 5
TEMP_MODEL_DIR = 'top_k_temp'
# ------------------
# Prediction Params
# ------------------

OUTPUT_TEMPLATE = ['data', 'datasets_Sample_Submission_Updated.csv']
OUTPUT_FILE = ['results', 'CI2_RU_Results.csv']
# OUTPUT_TEMPLATE = ['data', 'breastpathq', 'val_labels.csv']
# OUTPUT_FILE = ['results', 'CI2_RU_Results_val.csv']

TEST_DATA_PATH = ['results', 'test', 'processed_data']
# TEST_DATA_PATH = ['results', 'validation', 'processed_data']
PREDICT_MODEL = 'model_e50_bs20_pk0.863.hdf5'


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def train_model():
    print('Start: Training Model')
    root_path = os.getcwd()

    train_path = os.path.join(root_path, *TRAINING_SPLIT_DATA_PATH)
    test_path = os.path.join(root_path, *TEST_SPLIT_DATA_PATH)

    if len(VALIDATION_SPLIT_DATA_PATH) > 0:
        val_path = os.path.join(root_path, *VALIDATION_SPLIT_DATA_PATH)

    history_name = '{}_ep{}_bs{}.csv'.format(HISTORY_PREFIX, NUM_EPOCHS, BATCH_SIZE)

    history_file = os.path.join(root_path, *HISTORY_TRAINING_PATH)
    history_file = os.path.join(history_file, history_name)

    print('Preparing Data')

    # Get Training Data
    print('Retrieving Data for Training')
    train_features = None
    train_label = None
    data_retrieved = False
    for file_name in os.listdir(train_path):
        file_path = os.path.join(train_path, file_name)
        if os.path.isfile(file_path):
            if DATA_FILE == file_name[:-6]:
                data_retrieved = True
                np_file = np.load(file_path)

                if train_features is None:
                    train_features = np_file['dataset']
                    train_label = np_file['labels']
                else:
                    train_features = np.concatenate((train_features, np_file['dataset']), axis=0)
                    train_label = np.concatenate((train_label, np_file['labels']), axis=0)

    if not data_retrieved:
        print('Retrieved No Training Data')

    if len(VALIDATION_SPLIT_DATA_PATH) == 0:
        print('Splitting Training Dataset Into Training and Validation')
        x_train, x_val, y_train, y_val = data_manager.train_val_split(train_features, train_label,
                                                                      val_ratio=VALIDATION_RATIO)
    else:
        print('Retrieving Data for Validation')
        x_train = train_features
        y_train = train_label

        x_val = None
        y_val = None
        data_retrieved = False
        for file_name in os.listdir(val_path):
            file_path = os.path.join(val_path, file_name)
            if os.path.isfile(file_path):
                if DATA_FILE == file_name[:-6]:
                    data_retrieved = True
                    np_file = np.load(file_path)

                    if x_val is None:
                        x_val = np_file['dataset']
                        y_val = np_file['labels']
                    else:
                        x_val = np.concatenate((x_val, np_file['dataset']), axis=0)
                        y_val = np.concatenate((y_val, np_file['labels']), axis=0)

        if not data_retrieved:
            print('Retrieved No Validation Data')

    # Get Testing Dataset
    print('Retrieving Data for Testing')
    test_features = None
    test_label = None
    data_retrieved = False
    for file_name in os.listdir(test_path):
        file_path = os.path.join(test_path, file_name)
        if os.path.isfile(file_path):
            if DATA_FILE == file_name[:-6]:
                data_retrieved = True
                np_file = np.load(file_path)
                if test_features is None:
                    test_features = np_file['dataset']
                    test_label = np_file['labels']
                else:
                    test_features = np.concatenate((test_features, np_file['dataset']), axis=0)
                    test_label = np.concatenate((test_label, np_file['labels']), axis=0)

    if not data_retrieved:
        print('Retrieved No Testing Data')

    # Use validation as test data
    x_test = test_features
    y_test = test_label

    print('Preparing_Model')

    model_path = os.path.join(root_path, *MODELS_PATH)
    model_temp_path = os.path.join(model_path, TEMP_MODEL_DIR)

    if not os.path.exists(model_temp_path):
        os.makedirs(model_temp_path)

    if len(PRE_TRAINED_MODEL) == 0:
        print('Creating New Model')
        model_input = Input(INPUT_SHAPE)
        x = inception.build_inception_v4(model_input, enable_reduction=ENABLE_MODEL_REDUCTION)
        model = Model(model_input, x, name='inception_v4')
        if len(LOAD_WEIGHTS_NEW_MODEL) > 0:
            weights2load = os.path.join(root_path, *LOAD_WEIGHTS_NEW_MODEL)
            model.load_weights(weights2load)
    else:
        print('Loading Existing Model')
        pre_model_path = os.path.join(root_path, *PRE_TRAINED_MODEL)
        cus_obj = None
        if LOSS_FUNCTION == 'root_mean_squared_error':
            cus_obj = {'root_mean_squared_error': root_mean_squared_error}
        model = load_model(pre_model_path, custom_objects=cus_obj)

    if LOSS_FUNCTION == 'root_mean_squared_error':
        model.compile(loss=root_mean_squared_error, optimizer=LEARNING_OPTIMIZER, metrics=['mae'])
    else:
        model.compile(loss=LOSS_FUNCTION, optimizer=LEARNING_OPTIMIZER, metrics=['mae'])

    print(model.summary())

    print('Training Model')

    best_val_scores = []
    best_avrg_scores = []

    best_val_dic = {}
    best_avrg_dic = {}

    nb_mirror_aug = 1

    if ENABLE_VERTICAL_MIRROR:
        nb_mirror_aug += 1
    if ENABLE_HORIZONTAL_MIRROR:
        nb_mirror_aug += 1

    # Generate Fulls Labels for all rotations and mirrors
    y_val_full = []
    y_train_full = []
    for j in range(nb_mirror_aug):
        for i in range(NB_ROTATION):
            y_val_full = np.concatenate((y_val_full, y_val[:, 2]))

        for i in range(NB_ROTATION):
            y_train_full = np.concatenate((y_train_full, y_train[:, 2]))

    start = time.time()

    print('Train on {} samples, validate on {} samples'.format(len(y_train_full), len(y_val_full)))

    # Create History Log File
    with codecs.open(history_file, "w", encoding="utf-8") as history_csv:
        history_csv.write('epoch,time,train_rmse,train_pk,val_rmse,val_pk, avg_pk\n')

    for i in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(i, NUM_EPOCHS))

        # Train on mirror
        train_preds = []
        val_preds = []
        train_rmse = 0
        for j in range(nb_mirror_aug):
            # Mirror Image if enabled
            # First = Original
            # Second = Vertical (If Vertical not enable then horizontal)
            # Third = Horizontal (If both enabled)
            if j == 0:
                x_mirror_train = x_train
                x_mirror_val = x_val
            elif j == 1:
                if ENABLE_VERTICAL_MIRROR:
                    x_mirror_train = np.flip(x_train, 2)
                    x_mirror_val = np.flip(x_val, 2)
                elif ENABLE_HORIZONTAL_MIRROR:
                    x_mirror_train = np.flip(x_train, 3)
                    x_mirror_val = np.flip(x_val, 3)
            elif j == 2:
                x_mirror_train = np.flip(x_train, 3)
                x_mirror_val = np.flip(x_val, 3)

            # Rotate Image 90 Degrees
            for k in range(NB_ROTATION):
                history = model.fit(np.rot90(x_mirror_train, k, axes=(2, 3)), y_train[:, 2], verbose=0,
                                    epochs=1, batch_size=BATCH_SIZE)
                train_rmse += history.history['loss'][0]
                temp_preds = model.predict(np.rot90(x_mirror_train, k, axes=(2, 3)))
                temp_preds = temp_preds.reshape(temp_preds.shape[0])
                # temp_preds = np.nan_to_num(temp_preds)
                train_preds = np.concatenate((train_preds, temp_preds))

            for k in range(NB_ROTATION):
                temp_preds = model.predict(np.rot90(x_mirror_val, k, axes=(2, 3)))
                temp_preds = temp_preds.reshape(temp_preds.shape[0])
                # temp_preds = np.nan_to_num(temp_preds)
                val_preds = np.concatenate((val_preds, temp_preds))

        train_rmse = train_rmse / (nb_mirror_aug * NB_ROTATION)
        train_pk = prediction_probability.predprob(y_train_full, train_preds)
        val_pk = prediction_probability.predprob(y_val_full, val_preds)

        val_rmse = sqrt(mean_squared_error(y_val_full, val_preds))
        epoch_end = time.time()

        if isinstance(train_pk, tuple):
            train_pk = 0
        if isinstance(val_pk, tuple):
            val_pk = 0

        avrg_pk = (train_pk + val_pk) / 2
        print(
            ' - {}s - train_rmse: {:.6f} - train_pk: {:.6f} - val_rmse: {:.6f} - val_pk: {:.6f} - average_pk: {:.6f}'.format(
                int(epoch_end - epoch_start), train_rmse, train_pk, val_rmse, val_pk, avrg_pk))

        with codecs.open(history_file, "a", encoding="utf-8") as history_csv:
            history_csv.write('{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(i,
                                                                                  int(epoch_end - epoch_start),
                                                                                  train_rmse, train_pk, val_rmse,
                                                                                  val_pk, avrg_pk))

        # Save best validation rmse
        if len(best_val_scores) < TOP_K_MODEL_SAVE:
            # Save first K models
            # [rmse, val_pk, train_pk, train_rmse, index_id]
            val_info = [None, 0, 0, 0, 0]

            val_info[0] = val_rmse
            val_info[1] = val_pk
            val_info[2] = train_pk
            val_info[3] = train_rmse

            index = len(best_val_scores)
            val_info[4] = index

            best_val_scores.append(val_rmse)
            best_val_dic[str(val_rmse)] = val_info
            temp_best_val_weights = os.path.join(model_temp_path, 'temp_best_rmse_weights{}.hdf5'.format(index))
            model.save_weights(temp_best_val_weights)
        else:
            best_val_scores.append(val_rmse)
            best_val_scores = sorted(best_val_scores)

            if best_val_scores[TOP_K_MODEL_SAVE] != val_rmse:
                info2remove = best_val_dic.pop(str(best_val_scores[TOP_K_MODEL_SAVE]), None)
                # Save if new top K model

                # [rmse, val_pk, train_pk, train_rmse, index_id]
                val_info = [None, 0, 0, 0, 0]

                val_info[0] = val_rmse
                val_info[1] = val_pk
                val_info[2] = train_pk
                val_info[3] = train_rmse

                if info2remove is not None:
                    index = info2remove[4]
                    val_info[4] = index

                    best_val_dic[str(val_rmse)] = val_info
                    temp_best_val_weights = os.path.join(model_temp_path, 'temp_best_rmse_weights{}.hdf5'.format(index))
                    model.save_weights(temp_best_val_weights)
                else:
                    temp_best_avrg_weights = os.path.join(model_temp_path, 'temp_best_rmse_weights_train_rmse({:.6f})_val-pk({:.6f})_train-pk({:.6f}).hdf5'.format(train_rmse,
                                                                                                                                                                   val_pk,
                                                                                                                                                                   train_pk))
                    model.save_weights(temp_best_avrg_weights)
            best_val_scores = best_val_scores[:-1]

        # Save if best average PK
        if len(best_avrg_scores) < TOP_K_MODEL_SAVE:
            # Save first K models
            # [avrg_pk, val_pk, train_pk, train_rmse, index_id]
            val_info = [None, 0, 0, 0, 0]

            val_info[0] = avrg_pk
            val_info[1] = val_pk
            val_info[2] = train_pk
            val_info[3] = train_rmse

            index = len(best_avrg_scores)
            val_info[4] = index

            best_avrg_scores.append(avrg_pk)
            best_avrg_dic[str(avrg_pk)] = val_info
            temp_best_avrg_weights = os.path.join(model_temp_path, 'temp_best_avrg_weights{}.hdf5'.format(index))
            model.save_weights(temp_best_avrg_weights)
        else:
            best_avrg_scores.append(avrg_pk)
            best_avrg_scores = sorted(best_avrg_scores, reverse=True)

            if best_avrg_scores[TOP_K_MODEL_SAVE] != avrg_pk:
                # Save if new top K model
                info2remove = best_avrg_dic.pop(str(best_avrg_scores[TOP_K_MODEL_SAVE]), None)

                # [rmse, val_pk, train_pk, train_rmse, index_id]
                val_info = [None, 0, 0, 0, 0]

                val_info[0] = avrg_pk
                val_info[1] = val_pk
                val_info[2] = train_pk
                val_info[3] = train_rmse

                if info2remove is not None:
                    index = info2remove[4]
                    val_info[4] = index

                    best_avrg_dic[str(avrg_pk)] = val_info
                    temp_best_avrg_weights = os.path.join(model_temp_path,
                                                          'temp_best_avrg_weights{}.hdf5'.format(index))
                    model.save_weights(temp_best_avrg_weights)
                else:
                    temp_best_avrg_weights = os.path.join(model_temp_path, 'temp_best_avrg_weights_train_rmse({:.6f})_val-pk({:.6f})_train-pk({:.6f}).hdf5'.format(train_rmse,
                                                                                                                                                                   val_pk,
                                                                                                                                                                   train_pk))
                    model.save_weights(temp_best_avrg_weights)

            best_avrg_scores = best_avrg_scores[:-1]

    end = time.time()
    print('Training Time: {}s'.format(int(end - start)))

    '''
    print('Saving Last Model:')
    final_model_name = 'model_final_e({})_bs({})_mr({})_train_rmse({:.6f})_train-pk({:.6f})_val-rmse({:.6f})_val-pk({:.6f}).hdf5'.format(
        NUM_EPOCHS,
        BATCH_SIZE,
        ENABLE_MODEL_REDUCTION,
        train_rmse,
        train_pk,
        val_rmse,
        val_pk)

    model.save(os.path.join(model_path, final_model_name))
    '''

    print('Evaluating Model')

    # Write Prediction Log
    def log_preds(y, y_, file):
        with codecs.open(file, "w", encoding="utf-8") as pred_csv:
            pred_csv.write('y_true,y_pred\n')
            for l in range(len(y)):
                pred_csv.write('{},{}\n'.format(y[l], y_[l]))

    # Write Summary Log
    log_file = os.path.join(root_path, *SUMMARY_TRAINING_LOG)
    if not os.path.exists(log_file):
        with codecs.open(log_file, "a", encoding="utf-8") as log_csv:
            log_csv.write(
                'model_version,i,nb_epochs,batch_size,nb_rotations,mirror_vertical,mirror_horizontal,model_reduction,test_rmse,test_pk,val_pk,train_pk,train_rmse\n')

    # Create Directory to save all top k models
    save_model_path = os.path.join(model_path, 'top_models-ep({})-bs({})-r({})-rm({})-rd({})'.format(NUM_EPOCHS,
                                                                                                     BATCH_SIZE,
                                                                                                     NB_ROTATION,
                                                                                                     ENABLE_HORIZONTAL_MIRROR or ENABLE_VERTICAL_MIRROR,
                                                                                                     ENABLE_MODEL_REDUCTION))
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    # Create Directory to save all prediction logs

    pred_path = os.path.join(root_path, *HISTORY_TRAINING_PATH)
    pred_path = os.path.join(pred_path,
                             '{}_ep{}_bs{}_r({})_rm({})_mr({})'.format(PREDICTION_LOG, NUM_EPOCHS, BATCH_SIZE,
                                                                       NB_ROTATION,
                                                                       ENABLE_HORIZONTAL_MIRROR or ENABLE_VERTICAL_MIRROR,
                                                                       ENABLE_MODEL_REDUCTION))
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    # Evaluate Best Validation Models
    for best_val_score in best_val_dic.values():
        itr = best_val_score[4]
        temp_best_val_weights = os.path.join(model_temp_path, 'temp_best_rmse_weights{}.hdf5'.format(best_val_score[4]))
        model.load_weights(temp_best_val_weights)
        val_preds = model.predict(x_test)
        val_preds = val_preds.reshape(val_preds.shape[0])
        val_preds = np.nan_to_num(val_preds)
        p_k = prediction_probability.predprob(y_test[:, 2], val_preds)
        test_rmse_val = sqrt(mean_squared_error(y_test[:, 2], val_preds))
        print('Test P_K Score (Val Model {}):'.format(itr))
        val_str_pk = '{:.6f}'.format(p_k)
        print(val_str_pk)

        best_model_name = 'model_rmse_e({})_bs({})_mr({})_i({})_test-rmse({:.6f})_train_rmse({:.6f})_test-pk({})_val-pk({:.6f})_train-pk({:.6f}).hdf5'.format(
            NUM_EPOCHS,
            BATCH_SIZE,
            ENABLE_MODEL_REDUCTION,
            itr,
            test_rmse_val,
            best_val_score[3],
            val_str_pk,
            best_val_score[1],
            best_val_score[2])

        model.save(os.path.join(save_model_path, best_model_name))

        pred_name_file = '{}_val_ep{}_bs{}_mr({})_i({}).csv'.format(PREDICTION_LOG, NUM_EPOCHS, BATCH_SIZE,
                                                                    ENABLE_MODEL_REDUCTION, itr)
        pred_file = os.path.join(pred_path, pred_name_file)
        log_preds(y_test[:, 2], val_preds, pred_file)

        with codecs.open(log_file, "a", encoding="utf-8") as log_csv:
            log_csv.write(
                'rmse,{},{},{},{},{},{},{},{:.6f},{},{:.6f},{:.6f},{:.6f}\n'.format(NUM_EPOCHS, itr, BATCH_SIZE,
                                                                                    NB_ROTATION,
                                                                                    ENABLE_VERTICAL_MIRROR,
                                                                                    ENABLE_HORIZONTAL_MIRROR,
                                                                                    ENABLE_MODEL_REDUCTION,
                                                                                    test_rmse_val, val_str_pk,
                                                                                    best_val_score[1],
                                                                                    best_val_score[2],
                                                                                    best_val_score[3]))

    # Evaluate Best Average Models
    for best_avrg_score in best_val_dic.values():
        itr = best_avrg_score[4]
        temp_best_avg_weights = os.path.join(model_temp_path,
                                             'temp_best_avrg_weights{}.hdf5'.format(best_avrg_score[4]))
        model.load_weights(temp_best_avg_weights)
        avg_preds = model.predict(x_test)
        avg_preds = avg_preds.reshape(avg_preds.shape[0])
        avg_preds = np.nan_to_num(avg_preds)
        p_k = prediction_probability.predprob(y_test[:, 2], avg_preds)
        test_rmse_avg = sqrt(mean_squared_error(y_test[:, 2], avg_preds))
        print('Test P_K Score (Avg Model {}):'.format(itr))
        avg_str_pk = '{:.6f}'.format(p_k)
        print(avg_str_pk)

        best_model_name = 'model_average_e({})_bs({})_mr({})_i({})_test-rmse({:.6f})_train_rmse({:.6f})_test-pk({})_val-pk({:.6f})_train-pk({:.6f}).hdf5'.format(
            NUM_EPOCHS,
            BATCH_SIZE,
            ENABLE_MODEL_REDUCTION,
            itr,
            test_rmse_avg,
            best_val_score[3],
            avg_str_pk,
            best_avrg_score[1],
            best_avrg_score[2])

        model.save(os.path.join(save_model_path, best_model_name))

        pred_name_file = '{}_avg_ep{}_bs{}_mr({})_i({}).csv'.format(PREDICTION_LOG, NUM_EPOCHS, BATCH_SIZE,
                                                                    ENABLE_MODEL_REDUCTION, itr)
        pred_file = os.path.join(pred_path, pred_name_file)
        log_preds(y_test[:, 2], avg_preds, pred_file)

        with codecs.open(log_file, "a", encoding="utf-8") as log_csv:
            log_csv.write(
                'average,{},{},{},{},{},{},{},{:.6f},{},{:.6f},{:.6f},{:.6f}\n'.format(NUM_EPOCHS, itr, BATCH_SIZE,
                                                                                       NB_ROTATION,
                                                                                       ENABLE_VERTICAL_MIRROR,
                                                                                       ENABLE_HORIZONTAL_MIRROR,
                                                                                       ENABLE_MODEL_REDUCTION,
                                                                                       test_rmse_avg, avg_str_pk,
                                                                                       best_avrg_score[1],
                                                                                       best_avrg_score[2],
                                                                                       best_avrg_score[3]))

    print('Done: Training Model')


def predict_test_data():
    print('Start: Predicting Test Data')

    print('Preparing Data')
    root_path = os.getcwd()
    test_path = os.path.join(root_path, *TEST_DATA_PATH)
    test_features = None
    ids = None
    for file_name in os.listdir(test_path):
        file_path = os.path.join(test_path, file_name)
        if os.path.isfile(file_path):
            if DATA_FILE in file_name:
                np_file = np.load(file_path)

                if test_features is None:
                    test_features = np_file['dataset']
                    ids = np_file['labels']
                else:
                    test_features = np.concatenate((test_features, np_file['dataset']), axis=0)
                    ids = np.concatenate((ids, np_file['labels']), axis=0)

    print('Load Model')
    model_path = os.path.join(root_path, *MODELS_PATH)
    pred_model_name = os.path.join(model_path, PREDICT_MODEL)
    predict_model = load_model(pred_model_name)

    print('Preparing Predictions')
    preds = predict_model.predict(test_features)
    preds = preds.reshape(preds.shape[0])
    pred_dic = {}
    for i in range(len(ids)):
        slide = str(int(ids[i, 0]))
        rid = str(int(ids[i, 1]))
        p = preds[i]

        info = '{}_{}'.format(slide, rid)
        pred_dic[info] = p

    print('Calculating P_K Score')
    p_k = prediction_probability.predprob(ids[:, 2], preds)
    print('P_K Score:')
    if not isinstance(p_k, tuple):
        str_pk = '{:.6f}'.format(p_k)
    else:
        str_pk = 'nan'
    print(str_pk)

    print('Generating Submission File')

    data_rows = []
    template = os.path.join(root_path, *OUTPUT_TEMPLATE)
    with codecs.open(template, "r", encoding="utf-8") as temp_file:
        reader = csv.reader(temp_file)
        next(reader, None)
        for row in reader:
            slide = row[0]
            rid = row[1]
            info = '{}_{}'.format(slide, rid)
            p = pred_dic.get(info, '0')
            data = '{},{},{}\n'.format(slide, rid, p)
            data_rows.append(data)

    output_file = os.path.join(root_path, *OUTPUT_FILE)
    with codecs.open(output_file, "w", encoding="utf-8") as submission_file:
        submission_file.write('slide,rid,p\n')
        for row in data_rows:
            submission_file.write(row)
    submission_file.close()

    print('Done: Predicting Test Data')


if __name__ == "__main__":
    if MODE == 0:
        train_model()
    else:
        predict_test_data()
