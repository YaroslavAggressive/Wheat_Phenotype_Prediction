import logging
import os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import Sequential, Model
from keras.saving import load_model

import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np

from image_processing.aio_set_reader import PlantImageContainer
from collections import Counter
import multiprocessing
import copy

from pca_tsne_umap import pca_features, t_sne_features
from no_df_model import ComboModelTuner

from typing import Callable

# need for gradients evaluation in GradCAM
tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()


def get_top_features(plant_df: pd.DataFrame, model_estimation_pictures: np.asarray, n_top: int, alpha: float,
                     draw: bool = False,
                     color: str = "", title_: str = "", x_label: str = "", y_label: str = "",
                     save_name: str = "") -> list:
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

    # отрисовка
    if draw:
        dct = Counter(meaningful_col_names)
        for elem in dct.most_common():
            print(elem)
        keys = [i + 1 for i, elem in enumerate(dct.most_common())]
        values = [elem[1] for elem in dct.most_common()]
        plt.bar(keys, values)
        plt.axvline(x=int(0.05 * len(values)), color="r", linestyle="--")
        # plt.hist(keys, values, color=color)
        # plt.gca().set_xlim(min(Counter(meaningful_col_names).values()), max(Counter(meaningful_col_names).values()))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title_)
        plt.grid()
        plt.savefig(save_name)
        plt.cla()
    # посмотрим на топ alpha%, которые встречаются в наибольшем числе классов
    columns_frequency = sorted(Counter(meaningful_col_names).items(), key=lambda item: item[1], reverse=True)
    alpha_quantile = [i for i, j in columns_frequency[: int(len(meaningful_col_names) * alpha)]]
    return [meaningful_col_names_uniq, alpha_quantile]


def grad_ram(model: Model, layer_name: str, image: np.array) -> np.array:

    tf.compat.v1.disable_eager_execution()

    regr = model.output[0]
    layer_output = model.get_layer(layer_name).output
    grads = tf.gradients(regr, layer_output)[0]
    gradient_function = K.function([model.input], [layer_output, grads])
    output, grads_val = gradient_function(np.array([image]))
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    weights = np.mean(grads_val, axis=(0, 1))
    ram = np.dot(output, weights)
    ram = tf.keras.activations.linear(ram)  # passing through linear activation (regression task)
    ram /= np.max(ram)  # scale 0 to 1.0

    tf.compat.v1.enable_eager_execution()

    return ram


def grad_ram_combo(model: Model, layer_name: str, dict_features: np.array, image: np.array) -> np.array:

    regr = model.output[0]
    layer_output = model.get_layer(layer_name).output
    grads = tf.gradients(regr, layer_output)[0]
    gradient_function = K.function([model.input], [layer_output, grads])
    output, grads_val = gradient_function([np.array([image[:, :, None]]), np.array([dict_features])])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    weights = np.mean(grads_val, axis=(0, 1))
    ram = np.dot(output, weights)
    ram = tf.keras.activations.linear(ram)  # passing through linear activation (regression task)
    ram /= np.max(ram)  # scale 0 to 1.0

    return ram


def grad_ram_pp(model: Model, layer_name: str, image: np.array) -> np.array:
    regr = model.output[0]
    layer_output = model.get_layer(layer_name).output
    grads = tf.gradients(regr, layer_output)[0]
    # grads = normalize(grads)

    first = K.exp(regr) * grads
    second = K.exp(regr) * grads * grads
    third = K.exp(regr) * grads * grads * grads

    gradient_function = K.function([model.input], [regr, first, second, third, layer_output, grads])
    regr, conv_first_grad, conv_second_grad, conv_third_grad, conv_output, grads_val = \
        gradient_function(np.array([image]))
    global_sum = np.sum(conv_output[0].reshape((-1, conv_first_grad[0].shape[2])), axis=0)

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * \
                  global_sum.reshape((1, 1, conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num / alpha_denom

    weights = np.maximum(conv_first_grad[0], 0.0)
    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0), axis=0)
    alphas /= alpha_normalization_constant.reshape((1, 1, conv_first_grad[0].shape[2]))
    deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad[0].shape[2])), axis=0)

    ram = np.sum(deep_linearization_weights * conv_output[0], axis=2)
    ram = tf.keras.activations.linear(ram)  # passing through linear activation (regression task)
    ram /= np.max(ram)  # scale 0 to 1.0
    return ram


def grad_ram_pp_combo(model: Model, layer_name: str, dict_features: np.array, image: np.array) -> np.array:
    regr = model.output[0]
    layer_output = model.get_layer(layer_name).output
    grads = tf.gradients(regr, layer_output)[0]
    # grads = normalize(grads)

    first = K.exp(regr) * grads
    second = K.exp(regr) * grads * grads
    third = K.exp(regr) * grads * grads * grads

    gradient_function = K.function([model.input], [regr, first, second, third, layer_output, grads])
    regr, conv_first_grad, conv_second_grad, conv_third_grad, conv_output, grads_val = \
        gradient_function([np.array([dict_features]), np.array([image])])
    global_sum = np.sum(conv_output[0].reshape((-1, conv_first_grad[0].shape[2])), axis=0)

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * \
                  global_sum.reshape((1, 1, conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num / alpha_denom

    weights = np.maximum(conv_first_grad[0], 0.0)
    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0), axis=0)
    alphas /= alpha_normalization_constant.reshape((1, 1, conv_first_grad[0].shape[2]))
    deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad[0].shape[2])), axis=0)

    ram = np.sum(deep_linearization_weights * conv_output[0], axis=2)
    ram = tf.keras.activations.linear(ram)  # passing through linear activation (regression task)
    ram /= np.max(ram)  # scale 0 to 1.0
    return ram


def score_ram(model: Model, layer_name: str, image: np.array, max_n=-1) -> np.array:
    act_map_array = Model(inputs=model.input,
                          outputs=model.get_layer(layer_name).output).predict(np.array([image]))

    # extract effective maps
    if max_n != -1:
        act_map_std_list = [np.std(act_map_array[0, :, :, k]) for k in range(act_map_array.shape[3])]
        unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_n)[:max_n]
        max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
        act_map_array = act_map_array[:, :, :, max_N_indices]

    input_shape = model.layers[0].output_shape[1:]  # get input shape
    # 1. upsampled to original input size
    act_map_resized_list = [cv2.resize(act_map_array[0, :, :, k], input_shape[:2], interpolation=cv2.INTER_LINEAR)
                            for k in range(act_map_array.shape[3])]

    # 2. normalize the raw activation value in each activation map into [0, 1]
    act_map_normalized_list = []
    for act_map_resized in act_map_resized_list:
        if np.max(act_map_resized) - np.min(act_map_resized) != 0:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)

    # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(np.array([image]))
        for k in range(3):
            masked_input[0, :, :, k] *= act_map_normalized
        masked_input_list.append(masked_input)

    # 6. get final class discriminative localization map as linear combination of all activation maps
    ram = np.sum(act_map_array[0, :, :, :], axis=2)
    ram = tf.keras.activations.linear(ram)  # passing through linear activation (regression task)
    ram /= np.max(ram)  # scale 0 to 1.0
    return ram


def score_ram_combo(model: Model, layer_name: str, dict_features: np.array, image: np.array, max_n=-1) -> np.array:
    act_map_array = Model(inputs=model.input,
                          outputs=model.get_layer(layer_name).output).predict([np.array([image[:, :, None]]), np.array([dict_features])])

    # extract effective maps
    if max_n != -1:
        act_map_std_list = [np.std(act_map_array[0, :, :, k]) for k in range(act_map_array.shape[3])]
        unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_n)[:max_n]
        max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
        act_map_array = act_map_array[:, :, :, max_N_indices]

    input_shape = model.layers[0].output_shape[0][1:]  # get input shape
    # 1. upsampled to original input size
    act_map_resized_list = [cv2.resize(act_map_array[0, :, :, k], input_shape[:2], interpolation=cv2.INTER_LINEAR)
                            for k in range(act_map_array.shape[3])]

    # 2. normalize the raw activation value in each activation map into [0, 1]
    act_map_normalized_list = []
    for act_map_resized in act_map_resized_list:
        if np.max(act_map_resized) - np.min(act_map_resized) != 0:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)

    # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(np.array([image]))
        masked_input[0, :, :] = masked_input[0, :, :] * act_map_normalized
        masked_input_list.append(masked_input)

    # 6. get final class discriminative localization map as linear combination of all activation maps
    ram = np.sum(act_map_array[0, :, :, :], axis=2)
    ram = tf.keras.activations.linear(ram)  # passing through linear activation (regression task)
    ram /= np.max(ram)  # scale 0 to 1.0
    return ram


def process_model_rams(model_for_rams: tf.keras.Model, df_for_images: pd.DataFrame, ram_metric: Callable,
                       images_: np.array, features_: np.array, layer_name: str, top_snp_per_image: int = 40,
                       top_snp_overall: float = 0.01, draw_stat: bool = False, color_bar: str = "", title_: str = "",
                       xlabel_: str = "", ylabel_: str = "", save_: bool = False, save_name: str = "") -> list:

    """
    Функция выделения карт важности определенной RAM-метрикой (Regression Activation Map) моделью сверточной нейросети

    :param model_for_rams: сама модель
    :param ram_metric: метрика построения карт важности
    :param images_: набор изображений, на которых модель обучалась/проверялась/тестировалась
    :param features_: набор сторонних признаках, также подающихся на вход модели
    :param layer_name: номер слоя, с которого снимаются карты активации
    :param top_snp_per_image: количество пикселей с наибольшей интенсивностью, которые выбираются с каждого изображения
    :param top_snp_overall: процент пикселей с наибольшей интенсивностью, которые выбираются как самые важные
    :param draw_stat: индикатор необходимости построения частотной гистограммы признаков
    :param color_bar: цвет гистограммы частот
    :param title_: подпись графика
    :param xlabel_: подпись оси абсцисс
    :param ylabel_: подпись оси ординат
    :param save_: индикатор необходимости сохранения статистической гистограммы после отрисовки
    :param save_name: путь до файла, в который сохраняется частотная гистограмма (если необходимо)

    :return: Список важных снипов с наибольшей частотой по всем изображениям + список уникальных признаков, которым
     соответствуют выбранные пиксели
    """

    processed_rams = []
    tf.compat.v1.disable_eager_execution()
    for i in range(len(images_)):
        batch_map = ram_metric(model_for_rams, layer_name, features_[i], images_[i])
        processed_rams += list([batch_map])
        print(f"RAM {i} from metric {ram_metric.__name__} for crop height finished")
    tf.compat.v1.enable_eager_execution()
    if draw_stat:
        result = get_top_features(df_for_images, np.array(processed_images), top_snp_per_image, top_snp_overall,
                                  draw=draw_stat, color=color_bar, title_=title_,
                                  x_label=xlabel_, y_label=ylabel_,
                                  save_name=save_name if save_ else "")
    else:
        result = get_top_features(df_for_images, np.array(processed_images), top_snp_per_image, top_snp_overall)
    print("Top features for GradRAM metric, model crop height")
    print(result[1])
    return result


model_crop_height = "checkpoints/model_checkpoints/model_saves/height_crop/grid_cv_trained_model_iter4.h5"
model_crop_brown = "checkpoints/model_checkpoints/model_saves/crop_brown/grid_cv_trained_model_iter4.h5"
model_crop_yellow = "checkpoints/model_checkpoints/model_saves/crop_yellow/crop_yellow4/grid_cv_trained_model_iter4.h5"

model_height = tf.keras.models.load_model(model_crop_height,
                                          custom_objects={'custom_loss_mae': ComboModelTuner.custom_loss_mae},
                                          compile=False)
model_brown = tf.keras.models.load_model(model_crop_brown,
                                         custom_objects={'custom_loss_mae': ComboModelTuner.custom_loss_mae},
                                         compile=False)
model_yellow = tf.keras.models.load_model(model_crop_yellow,
                                          custom_objects={'custom_loss_mae': ComboModelTuner.custom_loss_mae},
                                          compile=False)
model_height.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
                     loss=ComboModelTuner.custom_loss_mae,
                     metrics=[ComboModelTuner.custom_loss_mse])

model_brown.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
                    loss=ComboModelTuner.custom_loss_mae,
                    metrics=[ComboModelTuner.custom_loss_mse])

model_height.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
                     loss=ComboModelTuner.custom_loss_mae,
                     metrics=[ComboModelTuner.custom_loss_mse])

# загружаем дни и изображения, по дням строим метки классов для датасета
n = 960
folder_images = "../AIO_set_wheat/for_model"
images = PlantImageContainer.load_images_from_folder(folder_images)[:n]
pca_features_ = pca_features(images, n_components=5)  # По совету КН взять 5
tsne = t_sne_features(images, n_components=2)  # по совету КН взять 2

total_features = np.concatenate((pca_features_, tsne), axis=1)

df_wheat = pd.read_csv("../datasets/wheat/wheat_pheno_num_sync.csv")
df_gen = pd.read_csv("../datasets/wheat/markers_poly_filtered_sync.csv")

data_images = images / 255.0
# урожайность высота
metric_unique_features, top_metric_features = process_model_rams(model_for_rams=model_height, df_for_images=df_gen,
                                                                 ram_metric=grad_ram_combo, images_=data_images,
                                                                 features_=total_features, layer_name="conv_deconv_1",
                                                                 top_snp_per_image=30, top_snp_overall=0.01,
                                                                 draw_stat=True, color_bar="r",
                                                                 title_="Частотная гистограмма снипов модели прогнозирования высоты и урожайности пшеницы метрикой GradRAM",
                                                                 xlabel_="Номер SNP", ylabel_="Частота",
                                                                 save_=True,
                                                                 save_name="../plots/feature_hist_crop_height_grad.png")
# урожайность бурая ржавчина
metric_unique_features, top_metric_features = process_model_rams(model_for_rams=model_brown, df_for_images=df_gen,
                                                                 ram_metric=grad_ram_combo, images_=data_images,
                                                                 features_=total_features, layer_name="conv_deconv_1",
                                                                 top_snp_per_image=30, top_snp_overall=0.01,
                                                                 draw_stat=True, color_bar="r",
                                                                 title_="Частотная гистограмма снипов модели прогнозирования высоты и бурой ржавчины пшеницы метрикой GradRAM",
                                                                 xlabel_="Номер SNP", ylabel_="Частота",
                                                                 save_=True,
                                                                 save_name="../plots/feature_hist_crop_brown_grad.png")
# урожайность желтая ржавчина
metric_unique_features, top_metric_features = process_model_rams(model_for_rams=model_yellow, df_for_images=df_gen,
                                                                 ram_metric=grad_ram_combo, images_=data_images,
                                                                 features_=total_features, layer_name="conv_deconv_1",
                                                                 top_snp_per_image=30, top_snp_overall=0.01,
                                                                 draw_stat=True, color_bar="r",
                                                                 title_="Частотная гистограмма снипов модели прогнозирования высоты и желтой ржавчины пшеницы метрикой GradRAM",
                                                                 xlabel_="Номер SNP", ylabel_="Частота",
                                                                 save_=True,
                                                                 save_name="../plots/feature_hist_crop_yellow_grad.png")

# дальше посчитаем те же таблицы, но при помощи ScoreRAM
# урожайность высота
metric_unique_features, top_metric_features = process_model_rams(model_for_rams=model_height, df_for_images=df_gen,
                                                                 ram_metric=score_ram_combo, images_=data_images,
                                                                 features_=total_features, layer_name="conv_deconv_1",
                                                                 top_snp_per_image=30, top_snp_overall=0.01,
                                                                 draw_stat=True, color_bar="r",
                                                                 title_="Частотная гистограмма снипов модели прогнозирования высоты и урожайности пшеницы метрикой ScoreRAM",
                                                                 xlabel_="Номер SNP", ylabel_="Частота",
                                                                 save_=True,
                                                                 save_name="../plots/feature_hist_crop_height_score.png")

# урожайность бурая ржавчина
metric_unique_features, top_metric_features = process_model_rams(model_for_rams=model_brown, df_for_images=df_gen,
                                                                 ram_metric=score_ram_combo, images_=data_images,
                                                                 features_=total_features, layer_name="conv_deconv_1",
                                                                 top_snp_per_image=30, top_snp_overall=0.01,
                                                                 draw_stat=True, color_bar="r",
                                                                 title_="Частотная гистограмма снипов модели прогнозирования высоты и бурой ржавчины пшеницы метрикой ScoreRAM",
                                                                 xlabel_="Номер SNP", ylabel_="Частота",
                                                                 save_=True,
                                                                 save_name="../plots/feature_hist_crop_brown_score.png")

# урожайность желтая ржавчина
metric_unique_features, top_metric_features = process_model_rams(model_for_rams=model_yellow, df_for_images=df_gen,
                                                                 ram_metric=score_ram_combo, images_=data_images,
                                                                 features_=total_features, layer_name="conv_deconv_1",
                                                                 top_snp_per_image=30, top_snp_overall=0.01,
                                                                 draw_stat=True, color_bar="r",
                                                                 title_="Частотная гистограмма снипов модели прогнозирования высоты и желтой ржавчины пшеницы метрикой ScoreRAM",
                                                                 xlabel_="Номер SNP", ylabel_="Частота",
                                                                 save_=True,
                                                                 save_name="../plots/feature_hist_crop_yellow_score.png")
