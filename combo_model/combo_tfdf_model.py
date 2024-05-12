from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense)
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import GlobalAveragePooling2D
from keras.layers import Conv2DTranspose
from keras.layers import Resizing

import keras_tuner
import tensorflow as tf
import keras
import numpy as np
from dataclasses import dataclass

import tensorflow_decision_forests as tfdf


x_train = np.random.rand(1000, 28, 28, 1)
y_train = np.random.randint(0, 10, (1000, 1))
x_val = np.random.rand(1000, 28, 28, 1)
y_val = np.random.randint(0, 10, (1000, 1))


@dataclass
class ComboModel(keras_tuner.HyperModel):

    n_epochs: int = 100
    n_row: int = 200
    n_col: int = 200
    input_channels: int = 1
    n_data: int = 100  # len(aio_labels)
    random_seed: int = 1234567890
    n_dict_features: int = 30
    n_trait: int = 1

    optimizer: keras.optimizers.Optimizer = None
    model: keras.models.Model = None

    def combo_model_functional(self, hp):
        """
        Функция построения модели нейросети с функциональным интерфейсом keras

        :param hp: набор гиперпараметров, отвечающих за конфигурация нейросети
        :return: граф-представление нейросети
        """

        inp_node = Input((self.n_row, self.n_col, self.input_channels), name="img_input")
        inp_dict_model = Input(self.n_dict_features, name="pop_struct_input")

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

        concatenate_features = Concatenate(name="concat_features")(inp_dict_model, dense_node)

        reg_forest_1 = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION,
                                                    num_trees=hp['num_estimators'],
                                                    max_depth=hp['max_depth'],
                                                    bootstrap_training_dataset=hp['bootstrap'])
        forest_1_pred = reg_forest_1(concatenate_features)

        reg_forest_2 = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION,
                                                    num_trees=hp['num_estimators'],
                                                    max_depth=hp['max_depth'],
                                                    bootstrap_training_dataset=hp['bootstrap'])
        forest_2_pred = reg_forest_2(concatenate_features)

        combo_model = Model(inputs=[inp_node, inp_dict_model], outputs=[forest_1_pred, forest_2_pred], name="feature_model")

        return combo_model

    def build(self, hp):
        """
        Builds a convolutional model.
        """
        # Гиперпараметры сверточной части модели
        model_hp = {
            # сначала идут параметры сверточной части модели
            'first_conv2d_out_channels': [32, 64],
            'first_conv2d_kernel_size': [3, 5, 7],
            'first_conv2d_activation': ['tanh', 'relu'],
            'need_extra_conv2d': [False, True],
            'extra_conv2d_out_channels': [32, 64],
            'extra_conv2d_kernel_size': [3, 5, 7],
            'extra_conv2d_activation': ['tanh', 'relu'],
            'need_batch_norm_after_first_conv2d': [True, False],
            'second_conv2d_kernel_size': [3, 5],
            'second_conv2d_out_channels': [128, 64],
            'second_conv2d_activation': ['tanh', 'relu'],
            'need_batch_norm_after_second_conv2d': [True, False],
            'dense_output_activation': ['sigmoid', 'linear'],
            'use_gap_1_or_flatten_0': [1, 0],
            'need_deconv_block': [False, True],
            'num_feature_output': [128, 64, 256],
            # а дальше идут параметры регрессионного случайного леса
            'n_estimators': [5, 20, 50, 100],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [(i + 1) * 5 for i in range(7)],
            'min_samples_split': [2, 6, 10],
            'bootstrap': [True, False]
        }

        # возвращаем собранную модель
        return self.feature_model_functional(model_hp)

    @staticmethod
    @tf.function
    def custom_loss_mse(y_true, y_pred):
        error = y_true - y_pred
        squared_error = tf.square(error)
        result = tf.reduce_mean(squared_error)

        return result

    @staticmethod
    @tf.function
    def custom_loss_mae(y_true, y_pred):
        error = y_true - y_pred
        abs_error = tf.abs(error)
        result = tf.reduce_mean(abs_error)

        return result

    # Function to run the train step.
    # здесь надо подумать как исправить эту функцию
    @tf.function
    def run_train_step(self, images_, pop_comps_, labels_1_, labels_2_):
        with tf.GradientTape() as tape:
            logits_1, logits_2 = self.model(images_, pop_comps_)
            loss_1 = loss_fn(labels, logits_1)
            loss_2 = loss_fn(labels, logits_2)
            # Add any regularization losses.
            if self.model.losses:
                loss_1 += tf.math.add_n(self.model.losses)
                loss_2 += tf.math.add_n(self.model.losses)
        gradients = tape.gradient(loss_1, self.model.trainable_variables)
        gradients = tape.gradient(loss_2, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    # Function to run the validation step.
    @tf.function
    def run_val_step(self, images_, pop_comps_, labels_1_, labels_2_):
        logits = self.model(images)
        loss = loss_fn(labels, logits)
        # Update the metric.
        epoch_loss_metric.update_state(loss)

    def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):
        # Convert the datasets to tf.data.Dataset.
        batch_size = hp.Int("batch_size", 32, 128, step=32, default=64)
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        validation_data = tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)

        # Define the optimizer.
        self.optimizer = keras.optimizers.Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3))
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # The metric to track validation loss.
        epoch_loss_metric = keras.metrics.Mean()

        # Assign the model to the callbacks.
        for callback in callbacks:
            callback.set_model(model)

        # Record the best validation loss value
        best_epoch_loss = float("inf")

        # The custom training loop.
        for epoch in range(self.n_epochs):
            print(f"Epoch: {epoch}")

            # Iterate the training data to run the training step.
            for images, labels_1, labels_2 in train_ds:
                self.run_train_step(images, labels_1, labels_2)

            # Iterate the validation data to run the validation step.
            for images, labels in validation_data:
                self.run_val_step(images, labels)

            # Calling the callbacks after epoch.
            epoch_loss = float(epoch_loss_metric.result().numpy())
            for callback in callbacks:
                # The "my_metric" is the objective passed to the tuner.
                callback.on_epoch_end(epoch, logs={"my_metric": epoch_loss})
            epoch_loss_metric.reset_state()

            print(f"Epoch loss: {epoch_loss}")
            best_epoch_loss = min(best_epoch_loss, epoch_loss)

        # Return the evaluation metric value.
        return best_epoch_loss


tuner = keras_tuner.RandomSearch(
    objective=keras_tuner.Objective("my_metric", "min"),
    max_trials=2,
    hypermodel=ComboModel(),
    directory="results",
    project_name="custom_training",
    overwrite=True,
)

tuner.search(x=x_train, y=y_train, validation_data=(x_val, y_val))

best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps.values)

best_model = tuner.get_best_models()[0]
best_model.summary()