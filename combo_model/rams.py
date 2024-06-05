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

import multiprocessing
import copy

# need for gradients evaluation in GradCAM
tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()


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


from pca_tsne_umap import pca_features, t_sne_features
from no_df_model import ComboModelTuner

loaded_model = tf.keras.models.load_model("checkpoints/grid_cv_trained_model_iter0.h5",
                                          custom_objects={'custom_loss_mae': ComboModelTuner.custom_loss_mse},
                                          compile=False)
loaded_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
                            loss=ComboModelTuner.custom_loss_mae,
                            metrics=[ComboModelTuner.custom_loss_mse])
print(loaded_model.summary())
# model_combo = load_model("logs/model_name_v3_x_0_trainable_model3.h5",
#                          custom_objects={"custom_loss_simple": custom_loss_simple,
#                                          "custom_loss_simple_2": custom_loss_simple_2,
#                                          "custom_loss_last": custom_loss_last,
#                                          "custom_loss_last_2": custom_loss_last_2})
# # print(model_combo.summary())
#
# # убираем ненужные слои ввода и вывода
# model_processed = Model(inputs=model_combo.inputs[1:], outputs=model_combo.outputs[1])
# print(model_processed.summary())

# загружаем дни и изображения, по дням строим метки классов для датасета
folder_images = "../AIO_set_wheat/for_model"
images = PlantImageContainer.load_images_from_folder(folder_images)
pca_features_ = pca_features(images, n_components=5)  # По совету КН взять 5
tsne = t_sne_features(images, n_components=2)  # по совету КН взять 2
total_features = np.concatenate((pca_features_, tsne), axis=1)
df_wheat = pd.read_csv("../datasets/wheat/wheat_pheno_num_sync.csv")
labels = df_wheat[["Урожайность.зерна..г.", "Высота.растений..см"]].to_numpy()


data_images = images / 255.0
data_vector = labels

# gradram_combo = grad_ram_combo(loaded_model, "max_pool_map", total_features[20], data_images[20])
gradram_combo = score_ram_combo(loaded_model, "max_pool_map", total_features[0], data_images[0])
plt.imshow(gradram_combo)
plt.imsave('gradram_combo.jpg', gradram_combo)

# gradram_pp_combo = grad_ram_pp_combo(model_processed, "max_pool_map", aio_dict.to_numpy()[0], data_images[0])
# plt.imshow(gradram_pp_combo)
# plt.imsave('gradram_pp_combo.jpg', gradram_pp_combo)

# scoreram_combo = score_ram_combo(model_processed, "max_pool_map", aio_dict.to_numpy()[0], data_images[0])
# plt.imshow(scoreram_combo)
# plt.imsave('scoreram_combo.jpg', scoreram_combo)
