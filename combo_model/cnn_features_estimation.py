import pandas as pd
import numpy as np

import tensorflow as tf

import ast
from tensorflow.keras.models import Model
from keras.models import load_model
from image_processing.aio_set_reader import PlantImageContainer

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore, Score

import matplotlib.pyplot as plt
from matplotlib import cm
from collections import Counter

from cnn.custom_visualization import draw_images_table, draw_activity_maps

import argparse
from no_df_model import ComboModelTuner
from pca_tsne_umap import pca_features, t_sne_features
# from rams import score_ram_combo, grad_ram_combo


def process_estimation_metric(metric: callable, scorer: callable, _model: Model, images: list, features_: np.ndarray,
                              labels, _batch_size: int = 318, ind: int = 0) -> np.asarray:
    """
    Функция обработки метрик выделения областей "важности" изображения нейронной сетью

    :param metric: класс метрики, которой производится оценка
    :param scorer: класс скорэра, который является "оберткой" меток образцов
    :param _model: объект модели, которую будем оценивать
    :param images: список из искусственных изображений
    :param labels: список меток изображений
    :param _batch_size: размер куска датасета изображений, который передается в метрику оценки за один вызов

    :return: список карт, подсвечивающих области "важности" каждого образца, переведенный в формат Numpy
    """

    processed_images = []
    _score = None
    replace_2_linear = ReplaceToLinear()
    if ind == 0:
        if type(scorer) == CategoricalScore:
            tmp_metric = metric(_model, model_modifier=replace_2_linear, clone=True)
        else:
            tmp_metric = metric(_model, model_modifier=replace_2_linear)
    if ind == 0:
        for i in range(0, len(images), _batch_size):
            if type(scorer) == CategoricalScore:
                _score = scorer(list(labels[i: i + _batch_size]))
            batch_maps = tmp_metric(_score if _score else scorer, [images[:, :, :, None][i: i + _batch_size], features_[i: i + _batch_size]])
            processed_images += list(batch_maps[0])
    else:
        for i in range(len(images)):
            batch_maps = metric(_model, "max_pool_map", features_[i], images[i])
            processed_images += list(batch_maps)

    return np.array(processed_images)


def get_top_features(plant_df: pd.DataFrame, model_estimation_pictures: np.asarray, n_top: int, alpha: float) -> list:
    """
    Функция получения наиболее "важных" для предсказания модели генетических (и не только)
    маркеров на основании показателей интенсивности соответствующих пикселей.

    :param model_estimation_pictures: список всех изображений, полученных на основе AIO в качестве метрик оценки
    качества модели
    :param n_top: количество наиболее важных признаков, которое мы выделяем из каждого образца
    :param alpha: процент, которые отбираем уже среди лучших выбранных маркеров на основании частоты встречаемости
    :return: список из трех списков: 1 - список всех уникальных столбцов, которые попали в
    топ хоть в одном из рассматриваемых изображений; 2 - топ 'alpha'% параметров, которые встречаются
    """

    meaningful_features = []
    for img in model_estimation_pictures:
        pixel_dict = {(s, k): img[s, k] for k in range(img.shape[1]) for s in range(img.shape[0])}
        sorted_pixel_dict = dict(sorted(pixel_dict.items(), key=lambda item: item[1], reverse=True))  # по убыванию
        meaningful_features += [(s, k) for s, k in list(sorted_pixel_dict.keys())[:n_top]]

    aio_shape = model_estimation_pictures[0].shape
    # получаем названия признаков по их координатам (общие по всем классам)
    meaningful_col_names = []
    for i, j in meaningful_features:
        col_name = plant_df.columns[(i * aio_shape[1] + j) % (plant_df.shape[1])]
        meaningful_col_names.append(col_name)

    # оставим отдельно только уникальные
    meaningful_col_names_uniq = set(meaningful_col_names)

    # посмотрим на топ alpha%, которые встречаются в наибольшем числе классов
    columns_frequency = sorted(Counter(meaningful_col_names).items(), key=lambda item: item[1], reverse=True)
    alpha_quantile = [i for i, j in columns_frequency[: int(len(meaningful_col_names) * alpha)]]
    return [meaningful_col_names_uniq, alpha_quantile]


def regression_score_func(output):
    return tf.math.abs(1.0 / (output[:, 0] + tf.keras.backend.epsilon()))


def save_comparison(save_path: str, metric_tables: list, metric_names: list,
                    comparison_tables: list, table_names: list):
    # пишем в консоль резуьтаты сравнения простым хардкодом
    with open(save_path, "w") as file:
        # saliency maps + gradCAMs + scoreCAMs
        for i, metric_nm in enumerate(metric_names):
            file.write(f"{metric_nm}s \n")
            file.write("######################### \n")
            for name, table in zip(table_names, comparison_tables):
                file.write(f"Intersection between model and {name}:{list(set(table) & set(metric_tables[i]))} \n")
                file.write("######################### \n")


def process_classification_cnn(score_func: callable, model_folder: str, df_folder: str, labels: pd.DataFrame, images_folder: str,
                               _class_bins: list, results_folder: str):
    """
    Функция обработки результатов работы модели на имеющихся данных при решении задачи КЛАССИФИКАЦИИ, а также
    визуализации при помощи тепловых карт разного типа - ScoreCAM, GradCAM, Saliency Maps

    :param score_func: функция скорера, которая работает в паре с построением функций
    :param model_folder: каталог, содержащий обученную версию модели
    :param df_folder: каталог, содержащий датасет исследуемого типа растений
    :param images_folder: каталог, содержащий обучающий и тестовый наборы данных, соответствующих табличному датасету
    :param _class_bins: список меток классов, на которые модель
    :param results_folder: каталог, в который сохраняются текстовые файлы с результатами
    :return: None
    """
    # загружаем датасет
    plant_df = pd.read_csv(df_folder)

    # загружаем модель
    classification_model = load_model(model_folder)

    # создаем все необходимое для задачи классификации
    plant_dataset = PlantImageContainer(_class_bins)
    aio_images = PlantImageContainer.load_images_from_folder(images_folder)  # подгружаем изображения
    # интервалы зацветания описываются серединами отрезков
    aio_days = np.asarray([np.uint8(resp) for resp in plant_df["resp"]])
    aio_labels = plant_dataset.get_image_labels(aio_days)

    metric_names = ["Saliency Map", "ScoreCAM", "GradCAM"]
    metric_functions = [Saliency, score_ram_combo, grad_ram_combo]
    # metric_functions = [Saliency, Scorecam, GradcamPlusPlus]
    metric_batch_sizes = [318, 78, 78]
    top_metric_features = [20, 20, 20]
    alphas = [0.05, 0.05, 0.05]

    n_row, n_col = 5, 3
    top_feature_results = []
    for i, nm in enumerate(metric_names):
        metric_maps = process_estimation_metric(metric_functions[i], score_func, classification_model,
                                                aio_images[:metric_batch_sizes[i]], aio_labels[:metric_batch_sizes[i]],
                                                metric_batch_sizes[i], i)
        draw_images_table(list(metric_maps[:15]), n_row, n_col, nm, 22, False)
        metric_unique_features, top_metric_features = get_top_features(plant_df, metric_maps[:15],
                                                                       top_metric_features[i], alphas[i])
        top_feature_results.append(top_metric_features)

    save_comparison(results_folder, top_feature_results, metric_names, [BORUTA_SNP, BOOTSTRAP_SNP, TOTAL_SNP],
                    ["Boruta", "Bootstrap", "Total"])


def process_regression_cnn(score_func: callable, model_folder: str, pheno_folder: str, gen_folder: str,
                           labels: np.ndarray, images_folder: str, results_folder: str):
    """
    Функция обработки результатов работы модели на имеющихся данных при решении задачи РЕГРЕССИИ, а также
    визуализации при помощи тепловых карт разного типа - ScoreCAM, GradCAM, Saliency Maps

    :param score_func: функция скорера, которая работает в паре с построением функций
    :param model_folder: каталог, содержащий обученную версию модели
    :param df_folder: каталог, содержащий датасет исследуемого типа растений
    :param images_folder: каталог, содержащий обучающий и тестовый наборы данных, соответствующих табличному датасету
    :param results_folder: каталог, в который сохраняются текстовые файлы с результатами
    :return: None
    """
    # загружаем датасет
    pheno_df = pd.read_csv(pheno_folder)
    gen_df = pd.read_csv(gen_folder)
    gen_df = gen_df.drop(["Unnamed: 0"], axis=1)

    # загружаем модель
    regression_model = load_model(model_folder, custom_objects={'custom_loss_mae': ComboModelTuner.custom_loss_mae,
                                                                'custom_loss_mse': ComboModelTuner.custom_loss_mse})
    # regression_model.complile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
    #                           loss=ComboModelTuner.custom_loss_mae,
    #                           metrics=[ComboModelTuner.custom_loss_mse])

    # загружаем набор AIO для построения оценок, метки фенотипа
    plant_images = PlantImageContainer.load_images_from_folder(images_folder)
    pca_features_ = pca_features(plant_images, n_components=5)
    t_sne_features_ = t_sne_features(plant_images, n_components=2)
    total_features = np.concatenate((pca_features_, t_sne_features_), axis=1)

    metric_names = ["Saliency Map", "ScoreCAM", "GradCAM"]
    metric_functions = [Saliency, score_ram_combo, grad_ram_combo]
    # metric_functions = [Saliency, Scorecam, GradcamPlusPlus]
    metric_batch_sizes = [320, 96, 96]
    top_features_sizes = [20, 20, 20]
    alphas = [0.05, 0.05, 0.05]

    n_row, n_col = 5, 3
    top_feature_results = []
    for i, nm in enumerate(metric_names):
        if i + 2 == 2:
            tf.compat.v1.disable_eager_execution()
        metric_maps = process_estimation_metric(metric_functions[i+2], score_func, regression_model,
                                                plant_images, total_features, labels[:metric_batch_sizes[i+2]],
                                                metric_batch_sizes[i+2], i+2)
        draw_images_table(list(metric_maps[:15]), n_row, n_col, nm, 22, False)
        metric_unique_features, top_metric_features = get_top_features(gen_df, metric_maps[:15],
                                                                       top_features_sizes[i], alphas[i])
        top_feature_results.append(top_metric_features)
    print(top_feature_results)
    # save_comparison(results_folder, top_feature_results, metric_names, [BORUTA_SNP, BOOTSTRAP_SNP, TOTAL_SNP],
    #                 ["Boruta", "Bootstrap", "Total"])


# pheno_folder = "../datasets/wheat/wheat_pheno_num_sync.csv"
# gen_folder = "../datasets/wheat/markers_poly_filtered_sync.csv"
# model_crop_height = "checkpoints/model_checkpoints/model_saves/height_crop/grid_cv_trained_model_iter2.h5"
# model_crop_brown = "checkpoints/model_checkpoints/model_saves/crop_brown/grid_cv_trained_model_iter2.h5"
# model_crop_yellow = "checkpoints/model_checkpoints/model_saves/crop_yellow/crop_yellow4/grid_cv_trained_model_iter4.5"
#
# # model_classification_folder = "../models/vigna/saved_models/best_model_tuner_cross_1306.h5"
#
# aio_folder = "../AIO_set_wheat/for_model"
#
#
# regression_results_folder7 = "data/regression7_comparison.txt"
# regression_results_folder9 = "data/regression9_comparison.txt"
#
# # classification_results_folder = "data/classification_comparison.txt"
# df = pd.read_csv(pheno_folder)
# process_regression_cnn(regression_score_func, model_crop_height, pheno_folder, gen_folder, df[["Урожайность.зерна..г.", "Высота.растений..см"]].to_numpy(), aio_folder, regression_results_folder7)
# process_regression_cnn(regression_score_func, model_crop_brown, pheno_folder, gen_folder, df[["Урожайность.зерна..г.", "Бурая.ржавчина..."]].to_numpy(), aio_folder, regression_results_folder9)
# process_regression_cnn(regression_score_func, model_crop_yellow, pheno_folder, gen_folder, df[["Урожайность.зерна..г.", "Желтая.ржавчина..."]].to_numpy(), aio_folder, regression_results_folder9)
#
