#!/usr/bin/env python

from PIL import Image
import scipy
import logging
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import os
import sys
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt
from tensorflow.keras import backend as K
from keras.models import load_model, save_model

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import os

from aio_set_reader import PlantImageContainer
import datetime
import random as r

import argparse


def load_images_from_folder(folder: str) -> np.asarray:
    """
    Функция подгрузки необходимого набора искусственных изображений из передаваемого каталога.

    :param folder: папка с изображениями, сохраненными в формате .png

    :return: список формата Numpy, содержащие AIO в объектах класса Image из Pillow
    """

    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(np.asarray(img).astype(np.float32))
    return np.asarray(images)


def build_regr_model(hp):
    """
    Архитектура нейронной сети, которая будет выделять некоторые признаки из AIO,
    которые в дальнейшем могут быть идентифицированы как важные

    KN: Чтобы сдампить признаки нам надо functional API
    (https://stackoverflow.com/questions/61010584/how-can-i-extract-flatten-layer-output-for-each-epoch)
    (https://www.machinelearningnuggets.com/tensorflow-keras-functional-api/)

    Также надо уходить от бинов

    :param hp: (вот что это такое, я пока так и не понял. Работает, и ладно)
    :return: (Модель, очевидно, тоже не знаю, как пояснить пока что)
    """
    global n_col, n_row

    regression_model = Sequential()
    regression_model.add(Conv2D(n_col / 4, kernel_size=(3, 3), padding='same', strides=(1, 1),
                                input_shape=(n_row, n_col, 3),
                                activation=hp.Choice('first_conv2d_activation', ['relu', 'tanh'],)))

    if hp.Boolean("need_batch_norm_after_first_conv2d"):
        regression_model.add(BatchNormalization())
    regression_model.add(MaxPooling2D(pool_size=(2, 2)))

    regression_model.add(Conv2D(hp.Int(
        'second_conv2d_out_channels', min_value=n_col / 4, max_value=2 * n_col, step=n_col / 4, ),
        kernel_size=(3, 3), padding='same', strides=(1, 1),
        activation=hp.Choice('second_conv2d_activation', ['relu', 'tanh'], )))
    if hp.Boolean("need_batch_norm_after_second_conv2d"):
        regression_model.add(BatchNormalization())
    regression_model.add(MaxPooling2D(pool_size=(2, 2)))

    regression_model.add(Conv2D(hp.Int(
        'third_conv2d_out_channels', min_value=n_col / 4, max_value=2 * n_col, step=n_col / 4, ),
        kernel_size=(3, 3), padding='same', strides=(1, 1),
        activation=hp.Choice('second_conv2d_activation', ['relu', 'tanh'], )))
    if hp.Boolean("need_batch_norm_after_second_conv2d"):
        regression_model.add(BatchNormalization())
    regression_model.add(MaxPooling2D(pool_size=(2, 2)))

    regression_model.add(Flatten(name='flatten'))
    regression_model.add(Dense(128, activation='relu'))  # с этого слоя снимаем
    regression_model.add(Dense(1, activation='linear'))

    regression_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    return regression_model


if __name__ == '__main__':
    # парсер аргументов командной строки
    cnn_arg_parser = argparse.ArgumentParser(description='Processing CV input/output folders')

    # параметризуем его
    cnn_arg_parser.add_argument('path_tuner_log', type=str, help='Path to folder with cnn tuner logs')
    cnn_arg_parser.add_argument('path_learn_log', type=str, help='Path to folder with cnn learn logs')
    cnn_arg_parser.add_argument('path_df', type=str, help='Path to folder with plant dataframe table')
    cnn_arg_parser.add_argument('path_images', type=str, help='Path to folder with AIO for current plant dataset')
    cnn_arg_parser.add_argument('model_save_folder', type=str, help='Path to folder for saving models')
    cnn_arg_parser.add_argument('idx_save_folder', type=str, help='Path to folder for saving train-test split indices')
    cnn_arg_parser.add_argument('accuracy_save_folder', type=str,
                                help='Path to folder for saving accuracies on each learn iteration')

    args = cnn_arg_parser.parse_args()

    # импорт данных
    plant_tab = pd.read_csv(args.path_df)

    # построение интервальных классов, на которых обучаем модель

    plant_dataset = PlantImageContainer()

    # загружаем дни и изображения, по дням строим метки классов для датасета
    aio_days = np.asarray([np.uint8(resp) for resp in plant_tab["resp"]])

    aio_images = PlantImageContainer.load_images_from_folder(args.path_images)

    aio_labels = np.asarray(list(plant_tab["resp"]))

    # параметры для нейросети
    n_row = 128
    n_col = 128
    n_data = len(aio_labels)
    n_tuner = int(0.5 * n_data)
    n_trials_max = 10
    n_seedrng = 12345678
    print(n_data, n_tuner, n_trials_max, n_seedrng)

    # тюнинг-подбор гиперпараметров
    idxsplit = r.sample(range(n_data), k=n_data)
    tuner_idx = idxsplit[:n_tuner]

    train_images, train_labels = aio_images[tuner_idx], aio_labels[tuner_idx]

    train_images = train_images / 255.0

    tuner = kt.RandomSearch(build_regr_model,
                            objective='val_loss',
                            max_trials=n_trials_max,
                            seed=n_seedrng,
                            overwrite=True,
                            project_name=args.path_tuner_log)

    tuner.search(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)
    tuner.results_summary()
    best_model = tuner.get_best_models()[0]

    # сохраним модель перед обучением, чтобы потом посмотреть
    best_model.save(f"{args.model_save_folder}/{len(best_model.layers)}_regr_model_" +
                    str(datetime.datetime.now().date()) + ".h5", save_format="h5")

    # а теперь обучение модели при кросс-валидации
    num_folds = 10
    batch_size = 64
    verbosity = 1

    mse_per_fold_tr, mse_per_fold_vd = [], []
    mae_per_fold_tr, mae_per_fold_vd = [], []

    models = []
    tr_corr = []
    te_corr = []

    data_images = aio_images / 255.0
    idx_split = r.sample(range(n_data), k=n_data)
    n_split = int(0.90 * n_data)
    train_idx = idxsplit[:n_split]
    check_idx = idxsplit[n_split:]
    train_images = data_images[train_idx]
    train_labels = aio_labels[train_idx]

    # сохраним индексы образцов для обучения, чтобы потом на локальной машине предсказания построить
    with open(f"{args.idx_save_folder}/{len(best_model.layers)}_train_idx_" + str(datetime.datetime.now().date()) + ".txt",
              "w") as fl_save:
        fl_save.write(str(train_idx))

    with open(f"{args.idx_save_folder}/{len(best_model.layers)}_test_idx_" + str(datetime.datetime.now().date()) + ".txt",
              "w") as fl_save:
        fl_save.write(str(check_idx))

    for j, fold_no in enumerate(range(0, 100, num_folds)):
        kfold = KFold(n_splits=num_folds, shuffle=True)
        for k, (tr, valid) in enumerate(kfold.split(train_images, train_labels)):
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
            history = best_model.fit(train_images[tr], train_labels[tr],
                                     batch_size=batch_size,
                                     epochs=100,
                                     verbose=verbosity,
                                     validation_split=0.2)
            # plot_accuracy(history, fold_no)
            scores = best_model.evaluate(train_images[tr], train_labels[tr], verbose=0)
            mae_per_fold_tr.append(scores[1])
            mse_per_fold_tr.append(scores[0])
            tr_corr.append(np.median(np.abs(best_model.predict(train_images[tr]) - aio_days[tr])))
            # Generate generalization metrics
            scores = best_model.evaluate(train_images[valid], train_labels[valid], verbose=0)
            print(f'Score for fold {j * num_folds + k}: {best_model.metrics_names[0]} of {scores[0]};'
                  f' {best_model.metrics_names[1]} of {scores[1]}')
            mae_per_fold_vd.append(scores[1])
            mse_per_fold_vd.append(scores[0])
            models.append(best_model)
            te_corr.append(np.median(np.abs(best_model.predict(train_images[valid]) - aio_days[valid])))

    print("Means: ")
    print(np.mean(mse_per_fold_tr), np.mean(mse_per_fold_vd))

    with open(f"{args.accuracy_save_folder}/{len(best_model.layers)}_MSE_per_fold_vd" + str(datetime.datetime.now().date())
              + ".txt", "w") as fl_save:
        fl_save.write(str(mse_per_fold_vd))

    with open(f"{args.accuracy_save_folder}/{len(best_model.layers)}_MAE_per_fold_vd" + str(datetime.datetime.now().date())
              + ".txt", "w") as fl_save:
        fl_save.write(str(mae_per_fold_vd))

    with open(f"{args.accuracy_save_folder}/{len(best_model.layers)}_MSE_per_fold_tr" + str(datetime.datetime.now().date())
              + ".txt", "w") as fl_save:
        fl_save.write(str(mse_per_fold_tr))

    with open(f"{args.accuracy_save_folder}/{len(best_model.layers)}_MAE_per_fold_tr" + str(datetime.datetime.now().date())
              + ".txt", "w") as fl_save:
        fl_save.write(str(mae_per_fold_tr))

    # сохраним лучшую по точности модель
    best_model_id = np.argmax(mae_per_fold_vd)
    best_model_cross = models[best_model_id]
    best_model_id, np.argmax(te_corr)

    best_model_cross.save(f"{args.model_save_folder}/{len(best_model_cross.layers)}_regr_model_trained_cross_"
                          + str(datetime.datetime.now().date()) + ".h5", save_format="h5")
