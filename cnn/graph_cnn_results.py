import json
from keras.models import load_model
import os
import ast
import numpy as np
import pandas as pd
from custom_visualization import draw_train_valid_error
from image_processing.image_creation import snp_weather_colnames
from image_processing.data_processing import delete_redundant_columns
from image_processing.aio_set_reader import PlantImageContainer

import argparse

# загрузим модель по заданному пути к файлу
model = load_model("tmp_scripts_for_models/models_for_demonstration/10_regr_model_trained_cross_2023-07-02.h5")
print(model.summary())

# задаем директорию с файлами-checkpoint тюнера (пока тоже вручную)
tuner_folders = ["tmp_scripts_for_models/models_for_demonstration/tuner_2layers",
                 "tmp_scripts_for_models/models_for_demonstration/tuner_3layers"]
cnn_names = ["7 layers", "10 layers"]
# ищем рекурсивно все json-файлы в папке логов тюнера
for i, tuner_folder in enumerate(tuner_folders):
    json_files = []
    for root, dirs, files in os.walk(tuner_folder):
        for tmp_file in files:
            if tmp_file.endswith(".json"):
                json_files.append(os.path.join(root, tmp_file))

    # собираем данные об ошибке прогноза за все попытки подбора гиперпараметров модели при помощи keras_tuner
    # храним показатели в виде списков, потому что модуль os отсортировал в лексико-графическом порядке самостоятельно
    # по названиям папок 'trial_%%', которые записаны в путях к файлам json
    tuner_mae_train, tuner_mse_train = [], []
    tuner_mae_valid, tuner_mse_valid = [], []

    # этот кусок кода сильно не нравится, надо будет переделать
    for file in json_files:
        with open(file, "r") as fl:
            data = json.load(fl)
            if "metrics" in data.keys():  # там есть 2 лишних json'а, их этим условием отбрасываем
                # print(data['metrics'])
                metrics = data['metrics']['metrics']

                mse_train = metrics['loss']
                mae_train = metrics['mean_absolute_error']

                mse_valid = metrics['val_loss']
                mae_valid = metrics['val_mean_absolute_error']

                # mae train
                val = mae_train['observations'][0]['value'][0]
                tuner_mae_train.append(val)

                # mse train
                val = mse_train['observations'][0]['value'][0]
                tuner_mse_train.append(val)

                # mae valid
                val = mae_valid['observations'][0]['value'][0]
                tuner_mae_valid.append(val)

                # mse valid
                val = mse_valid['observations'][0]['value'][0]
                tuner_mse_valid.append(val)

    # отрисовка графиков
    metric_names = ["MSE", "MAE"]

    draw_train_valid_error(tuner_mae_train, tuner_mae_valid, metric_names[1],
                           f"Comparison of MAE for training_validation in hyperparameters tuning, {cnn_names[i]} CNN",
                           18,
                           f"Comparison of MAE for training_validation in hyperparameters tuning, {cnn_names[i]} CNN".
                           replace(" ", "_"))
    draw_train_valid_error(tuner_mse_train, tuner_mse_valid, metric_names[0],
                           f"Comparison of MSE for training_validation in hyperparameters tuning, {cnn_names[i]} CNN",
                           18,
                           f"Comparison of MAE for training_validation in hyperparameters tuning, {cnn_names[i]} CNN".
                           replace(" ", "_"))

# также построим график тренировочной/валидационной ошибки для модели

# задаем директорию
learn_folder = "tmp_scripts_for_models/models_for_demonstration/accuracies"

# зададим вручную названия графиков и метрик (потому что фактически по-другому никак)
p = "Comparison of MAE for training/validation samples per fold, 10 layers CNN"
graph_names = ["Comparison of MAE for training_validation samples per fold, 10 layers CNN",
               "Comparison of MSE for training_validation samples per fold, 10 layers CNN",
               "Comparison of MAE for training_validation samples per fold, 7 layers CNN",
               "Comparison of MSE for training_validation samples per fold, 7 layers CNN"]

# пройдем по всем файлам и заполним список массивов оишбок каждой нейросети на каждой метрике по фолдам
error_lists = []
for file in os.listdir(learn_folder):
    filepath = os.path.join(learn_folder, file)
    if os.path.isfile(filepath) and filepath.endswith(".txt"):
        with open(filepath, "r") as fl:
            error_lists.append(ast.literal_eval(fl.read()))

# рисуем графики для каждой метрики для каждой нейросети
for k, i in enumerate(range(0, len(error_lists) - 1, 2)):
    train = error_lists[i]
    valid = error_lists[i + 1]
    draw_train_valid_error(train, valid, metric_names[1] if k % 2 == 0 else metric_names[0], graph_names[k], 18,
                           graph_names[k].replace(" ", "_"))

# также посчитаем ошибку на тестовых данных
total_df = pd.read_csv("../tables_for_AIO/total_df_for_aio_vigna.csv")
with open("tmp_scripts_for_models/train_idx_2023-07-01.txt", "r") as fl:
    total_idx = range(total_df.shape[0])
    train_idx = ast.literal_eval(fl.read())
    test_idx = list(set(total_idx) - set(train_idx))

test_df = total_df.iloc[test_idx]
true_resp = test_df["resp"]
test_df = delete_redundant_columns(test_df, ["resp", "doy", "year", "seqid", "geo_id"])

plant_reader = PlantImageContainer()
aio_images = plant_reader.load_images_from_folder("../AIO_set/for_model")

prediction = model.predict(aio_images[test_idx]).flatten()

mae = np.sum(np.fabs(prediction - true_resp)) / test_df.shape[0]
mse = np.sum((prediction - true_resp)**2) / test_df.shape[0]

print(f"MAE on test sample for fitted model with 3 conv layers = {mae}")
print(f"MSE on test sample for fitted model with 3 conv layers = {mse}")
