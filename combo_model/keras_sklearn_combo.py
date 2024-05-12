from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense)
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import GlobalAveragePooling2D
from keras.layers import Conv2DTranspose
from keras.layers import Resizing
from keras.losses import MeanAbsoluteError, MeanSquaredError
from keras.metrics import Accuracy

from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer

import itertools

import pandas as pd

import keras_tuner
import tensorflow as tf
import keras
import numpy as np
from dataclasses import dataclass

from image_processing.aio_set_reader import PlantImageContainer

import copy


@dataclass
class KerasSKLearnComboModel:

    data: np.ndarray
    labels: np.ndarray
    test_size: float = 0.1
    rand_state: int = 12345
    n_epochs: int = 20
    n_row: int = 200
    n_col: int = 200
    input_channels: int = 1

    def __post_init__(self):
        """
        После инициализации разбиваем данные на обучающую и тестовую (для проверки после обучения) выборку
        :return: None
        """

        self.train_data, self.test_data, self.train_labels, self.test_labels = \
            train_test_split(self.data, self.labels, test_size=self.test_size, random_state=self.rand_state)

    def grad(self, loss_func_: keras.losses.Loss, inputs_: np.ndarray, targets_: np.ndarray):
        with tf.GradientTape() as tape:
            features = self.feature_model(inputs_, training=True)
            self.regression_model.fit(features, targets_)
            predicts = self.regression_model.predict(features)
            loss_value_1 = loss_func_(y_pred=predicts[:, 0], y_true=targets_[:, 0])
            loss_value_2 = loss_func_(y_pred=predicts[:, 1], y_true=targets_[:, 1])
        # return tape.gradient([loss_value_1, loss_value_2], self.feature_model.trainable_variables)
        return tape.gradient(loss_value_1, loss_value_2, self.feature_model.trainable_variables)

    def build_features_model(self, hp: dict) -> None:
        """
                Функция построения модели нейросети с функциональным интерфейсом keras

                :param hp: набор гиперпараметров, отвечающих за конфигурация нейросети
                :return: граф-представление нейросети
                """

        inp_node = Input((self.n_row, self.n_col, self.input_channels), name="img_input")

        conv_node_1 = Conv2D(hp['first_conv2d_out_channels'],
                             kernel_size=(hp['first_conv2d_kernel_size'], hp['first_conv2d_kernel_size']),
                             padding='same',
                             strides=(1, 1),
                             activation=hp['first_conv2d_activation'], name="conv_map_1")(inp_node)
        if hp['need_extra_conv2d']:
            conv_node_1 = Conv2D(hp['extra_conv2d_out_channels'],
                                 kernel_size=(hp['extra_conv2d_kernel_size'], hp['extra_conv2d_kernel_size']),
                                 padding='same',
                                 strides=(1, 1),
                                 activation=hp['extra_conv2d_activation'], name="conv_map_extra")(conv_node_1)

        if hp['need_batch_norm_after_first_conv2d']:
            batch_node_1 = BatchNormalization()(conv_node_1)
            mp_node_1 = MaxPooling2D(pool_size=(2, 2))(batch_node_1)
        else:
            mp_node_1 = MaxPooling2D(pool_size=(2, 2))(conv_node_1)

        conv_node_2 = Conv2D(hp['second_conv2d_out_channels'],
                             kernel_size=(hp['second_conv2d_kernel_size'], hp['second_conv2d_kernel_size']),
                             padding='same',
                             strides=(1, 1),
                             activation=hp['second_conv2d_activation'], name="conv_map_2")(mp_node_1)

        if hp['need_batch_norm_after_second_conv2d']:
            batch_node_2 = BatchNormalization()(conv_node_2)
            mp_node_2 = MaxPooling2D(pool_size=(2, 2), name="max_pool_map")(batch_node_2)
        else:
            mp_node_2 = MaxPooling2D(pool_size=(2, 2), name="max_pool_map")(conv_node_2)

        if hp['need_deconv_block']:
            deconv_node_2 = Conv2DTranspose(
                hp['second_conv2d_out_channels'],
                kernel_size=(hp['second_conv2d_kernel_size'], hp['second_conv2d_kernel_size']),
                padding='same',
                strides=(2, 2),
                activation=hp['second_conv2d_activation'],
                name="deconv_2"
            )(mp_node_2)
            concat_node_2 = Concatenate(name="concat_2", axis=3)([deconv_node_2, conv_node_2])
            conv_node_deconv_2 = Conv2D(
                hp['second_conv2d_out_channels'],
                kernel_size=(hp['second_conv2d_kernel_size'], hp['second_conv2d_kernel_size']),
                padding='same',
                strides=(1, 1),
                activation=hp['second_conv2d_activation'],
                name="conv_deconv_2"
            )(concat_node_2)
            deconv_node_1 = Conv2DTranspose(
                hp['first_conv2d_out_channels'],
                kernel_size=(hp['first_conv2d_kernel_size'], hp['first_conv2d_kernel_size']),
                padding='same',
                strides=(2, 2),
                activation=hp['first_conv2d_activation'],
                name="deconv_1"
            )(conv_node_deconv_2)
            concat_node_1 = Concatenate(name="concat_1", axis=3)([deconv_node_1, conv_node_1])
            mp_node_2 = Conv2D(
                hp['first_conv2d_out_channels'],
                kernel_size=(hp['first_conv2d_kernel_size'], hp['first_conv2d_kernel_size']),
                padding='same',
                strides=(1, 1),
                activation=hp['first_conv2d_activation'],
                name="conv_deconv_1"
            )(concat_node_1)

        if hp['use_gap_1_or_flatten_0'] == 0:
            flatten_node = Flatten(name='flatten')(mp_node_2)
            dense_node = Dense(hp['num_feature_output'], activation=hp['dense_output_activation'],
                               name="img_feature_output")(flatten_node)
        elif hp['use_gap_1_or_flatten_0'] == 1:
            dense_node = GlobalAveragePooling2D(name="img_feature_output")(mp_node_2)

        # concatenate_features = Concatenate(name="concat_features")([dense_node])

        self.feature_model = Model(inputs=inp_node, outputs=dense_node, name="feature_model")

    def build_regression_model(self, hps: dict) -> None:
        self.regression_model = MultiOutputRegressor(RandomForestRegressor(**hps))
        # self.regression_model = RandomForestRegressor(**hps)

    def predict(self, data_img: np.ndarray, data_dict: np.ndarray = []) -> np.ndarray:
        # в дальнешем при добавлении статистических методов анализа структуры популяции data_dict будет передаваться и
        # для обучения модели, и для получения прогноза (сюда)
        features = self.feature_model.predict(data_img)
        return self.regression_model.predict(features)

    def fit(self, hps_cnn: dict, hps_reg: dict):
        train_mae_1, train_mae_2 = [], []
        train_mse_1, train_mse_2 = [], []
        valid_mae_1, valid_mae_2 = [], []
        valid_mse_1, valid_mse_2 = [], []

        train_accuracy_1, train_accuracy_2 = [], []
        valid_accuracy_1, valid_accuracy_2 = [], []

        self.build_regression_model(hps_reg)
        self.build_features_model(hps_cnn)

        skf = KFold(n_splits=10)
        optimizer = keras.optimizers.SGD(learning_rate=0.01)
        for epoch in range(self.n_epochs):
            epoch_loss_mae_1, epoch_loss_mae_2 = MeanAbsoluteError(), MeanAbsoluteError()
            epoch_loss_mse_1, epoch_loss_mse_2 = MeanSquaredError(), MeanSquaredError()
            epoch_accuracy_1, epoch_accuracy_2 = Accuracy(), Accuracy()

            # Training loop - using batches of 32
            for i, (train_idx, valid_idx) in enumerate(skf.split(self.train_data, self.train_labels)):
                train_data, train_labels = self.train_data[train_idx], self.train_labels[train_idx]
                valid_data, valid_labels = self.train_data[valid_idx], self.train_labels[valid_idx]

                # Optimize the model
                grads = self.grad(epoch_accuracy_1, valid_data, valid_labels)
                optimizer.apply_gradients(zip(grads, self.feature_model.trainable_variables))
                train_features = self.feature_model.predict(train_data)

                self.regression_model.fit(train_features, valid_labels)

                # считаем ошибку на обучающих/тестовых данных
                train_predict = self.regression_model.predict(train_features)
                valid_features = self.feature_model.predict(valid_data)
                valid_prediction = self.regression_model.predict(valid_features)

                # оценка качества модели на обучающей и тестовой выборке

                # Compare predicted label to actual label
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                epoch_accuracy_1.update_state(train_labels[0], train_predict[0])
                epoch_accuracy_2.update_state(train_labels[1], train_predict[1])

                epoch_loss_mae_1.update_state()
                epoch_loss_mae_2.update_state()

                epoch_loss_mse_2.update_state()
                epoch_loss_mse_2.update_state()

            # End epoch
            train_mse_1.append(epoch_loss_mse_1.result())
            train_mse_2.append(epoch_loss_mse_2.result())

            train_mae_1.append(epoch_loss_mae_1.result())
            train_mae_2.append(epoch_loss_mae_2.result())

            valid_mse_1.append(epoch_loss_mse_1.result())
            valid_mse_2.append(epoch_loss_mse_2.result())

            valid_mae_1.append(epoch_loss_mae_1.result())
            valid_mae_2.append(epoch_loss_mae_2.result())

            if epoch % 50 == 0:
                print("Epoch {:03d}: MAE Label 1: {:.3f}, Accuracy Label 1: {:.3%}, Accuracy Label 2: {:.3%}".
                      format(epoch, epoch_loss_mae_1.result(), epoch_accuracy_1.result(), epoch_accuracy_2.result()))


class ComboModelTuner:

    @staticmethod
    @tf.function
    def custom_loss_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        error = y_true - y_pred
        squared_error = tf.abs(error)
        result = tf.reduce_mean(squared_error)
        return result

    @staticmethod
    @tf.function
    def custom_loss_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        error = y_true - y_pred
        squared_error = tf.square(error)
        result = tf.reduce_mean(squared_error)
        return result

    @staticmethod
    def random_hyper_tuning(model: KerasSKLearnComboModel, iters_num: int, hps_cnn: dict, hps_reg: dict):

        valid_mae_label_1, valid_mae_label_2 = [], []
        valid_mse_label_1, valid_mse_label_2 = [], []
        valid_accuracy_label_1, valid_accuracy_label_2 = [], []

        for iter_ in range(iters_num):
            # Сборка комбинации случайных гиперпараметров в заданын границах
            cnn_hp_comb, reg_hp_comb = {}, {}

            for param in hps_cnn:
                if len(hps_cnn[param]) > 1:
                    if any(isinstance(x, bool) for x in hps_cnn[param]) or any(isinstance(x, str) for x in hps_cnn[param]):
                        cnn_hp_comb[param] = hps_cnn[param][np.random.randint(len(hps_cnn[param]))]
                    else:
                        cnn_hp_comb[param] = np.random.randint(low=min(hps_cnn[param]), high=max(hps_cnn[param]))
                else:
                    cnn_hp_comb[param] = hps_cnn[param][0]

            for param in hps_reg:
                if len(hps_reg[param]) > 1:
                    if any(isinstance(x, bool) for x in hps_reg[param]) or any(isinstance(x, str) for x in hps_reg[param]):
                        reg_hp_comb[param] = hps_reg[param][np.random.randint(len(hps_reg[param]))]
                    else:
                        reg_hp_comb[param] = np.random.randint(low=min(hps_reg[param]), high=max(hps_reg[param]))
                else:
                    reg_hp_comb[param] = hps_reg[param][0]

            model.fit(cnn_hp_comb, reg_hp_comb)

            # считаем ошибку модели на тестовой выборке
            print(f"Random Tuning iter #{iter_} finished successfully")

    @staticmethod
    def grid_hyper_tuning(model: KerasSKLearnComboModel, hps_cnn: dict, hps_reg: dict):
        cnn_hp_combos = itertools.product(*hps_cnn)
        reg_hp_combos = itertools.product(*hps_reg)

        valid_mae_label_1, valid_mae_label_2 = [], []
        valid_mse_label_1, valid_mse_label_2 = [], []
        valid_accuracy_label_1, valid_accuracy_label_2 = [], []

        for i, tmp_hps_cnn in enumerate(cnn_hp_combos):
            for j, tmp_hps_reg in enumerate(reg_hp_combos):
                model.fit(tmp_hps_cnn, tmp_hps_reg)

                # считаем ошибку модели на тестовой выборке
                valid_predict = model.predict
                print(f"Grid Tuning iter #{i * len(reg_hp_combos) + j} finished successfully")


# гиперпараметры модели features со сверточной нейросети
cnn_hps = {'first_conv2d_out_channels': [32, 64],
           'first_conv2d_kernel_size': [3, 5, 7],
           'first_conv2d_activation': ['tanh', 'relu'],
           'need_extra_conv2d': [False, True],
           'extra_conv2d_out_channels': [32, 64],
           'extra_conv2d_kernel_size': [3, 5, 7],
           'extra_conv2d_activation': ['tanh', 'relu'],
           'need_batch_norm_after_first_conv2d': [True, False],
           'second_conv2d_kernel_size': [3, 5],
           'second_conv2d_out_channels': [64, 128],
           'second_conv2d_activation': ['tanh', 'relu'],
           'need_batch_norm_after_second_conv2d': [True, False],
           'dense_output_activation': ['sigmoid', 'linear'],
           'use_gap_1_or_flatten_0': [0, 1],
           'need_deconv_block': [False, True],
           'num_feature_output': [64, 128, 256]
           }

# гиперпараметры модели регрессии по фичам
reg_hp = {'n_estimators': [5, 20, 50, 100],
          'max_features': ['log2', 'sqrt'],
          'max_depth': [(i + 1) * 5 for i in range(7)],
          'min_samples_split': [2, 6, 10],
          'bootstrap': [True, False],
          # "n_outputs_": [2]
          }


# загрузка регрессионных меток
wheat_data = pd.read_csv("../datasets/wheat/wheat_pheno_num_sync.csv")
wheat_data = wheat_data.drop(["Unnamed: 0", "number"], axis=1)
labels = wheat_data[["Урожайность.зерна..г.", "Высота.растений..см"]]

# загрузка изображений
images_path = "../AIO_set_wheat/for_model"
images = PlantImageContainer.load_images_from_folder(images_path)

# разделение на train/validation
images_train, images_val, labels_train, labels_val = train_test_split(images,
                                                                      labels,
                                                                      test_size=0.2,
                                                                      random_state=123)

train_dataset = tf.data.Dataset.from_tensor_slices((images_train, labels_train.values))
val_dataset = tf.data.Dataset.from_tensor_slices((images_val, labels_val.values))

# (Пока просто средними значениями) импутируем данные, поскольку присутствуют пропуски
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = KNNImputer(n_neighbors=2, weights='uniform')
labels = imp.fit_transform(labels.to_numpy().reshape(-1, 2))
combo_model = KerasSKLearnComboModel(data=images,
                                     labels=labels)
ComboModelTuner.random_hyper_tuning(combo_model, 20, cnn_hps, reg_hp)
# ComboModelTuner.grid_hyper_tuning(combo_model, cnn_hps, reg_hp)
