from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense)
from tensorflow.keras.layers import BatchNormalization


def build_classification_model(hp):
    """
    Архитектура сверточной нейронной сети, которая будет выделять некоторые признаки из AIO,
    которые в дальнейшем могут быть идентифицированы как важные.

    Эта версия состоит из 2 слоев и решает задачу классификации (использовалась как первый прототип на предзащите)

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

    regression_model.add(Flatten(name='flatten'))
    regression_model.add(Dense(128, activation='relu'))  # с этого слоя снимаем
    regression_model.add(Dense(1, activation='linear'))

    regression_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    return regression_model


def build_regression_model_v1(hp):
    """
    Архитектура сверточной нейронной сети, которая выделяет важные признаки из AIO,
    которые в дальнейшем могут быть идентифицированы как важные.

    Эта версия состоит из 2 слоев и решает задачу регрессии (данный вариант используется для демонстрации
    непосредственно на защите третьего этапа конкурса)

    :param hp: (вот что это такое, я пока так и не понял. Работает, и ладно)
    :return: (Модель, очевидно, тоже не знаю, как пояснить пока что)
    """
    global n_col, n_row

    regression_model = Sequential()
    regression_model.add(Conv2D(n_col / 4, kernel_size=(3, 3), padding='same', strides=(1, 1),
                                input_shape=(n_row, n_col, 3),
                                activation=hp.Choice('first_conv2d_activation', ['relu', 'tanh'], )))

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

    regression_model.add(Flatten(name='flatten'))
    regression_model.add(Dense(128, activation='relu'))  # с этого слоя снимаем
    regression_model.add(Dense(1, activation='linear'))

    regression_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    return regression_model


def build_regression_model_v2(hp):
    """
    Архитектура сверточной нейронной сети, которая выделяет важные признаки из AIO,
    которые в дальнейшем могут быть идентифицированы как важные.

    Эта версия состоит из 3 слоев и решает задачу регрессии (вариант получен случайно,
    но также обучена для представления результатов на защите 3 этапа конкурса BSR)

    :param hp: (вот что это такое, я пока так и не понял. Работает, и ладно)
    :return: (Модель, очевидно, тоже не знаю, как пояснить пока что)
    """
    global n_col, n_row

    regression_model = Sequential()
    regression_model.add(Conv2D(n_col / 4, kernel_size=(3, 3), padding='same', strides=(1, 1),
                                input_shape=(n_row, n_col, 3),
                                activation=hp.Choice('first_conv2d_activation', ['relu', 'tanh'], )))

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
