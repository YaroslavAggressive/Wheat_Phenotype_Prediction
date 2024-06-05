import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib import cm


def draw_images_table(list_of_aio: list[Image], n_row: int, n_col: int, prefix: str, fontsize: int, show=False):
    """
    Демонстрационная отрисовка переданного в функцию списка AIO для иллюстрации результатов на защите

    :param list_of_aio: список изображений, которые планируем отрисовать
    :param n_row: число строк в 'таблице' изображений
    :param n_col: число столбцов в 'таблице' изображений
    :param prefix: префикс к названию отображаемой метрики
    :param fontsize: размер шрифта подписей к изображениям
    :param show: бинарный индикатор необходимости вывода графика в отдельном окне figure модуля pyplot
    :return: None
    """
    if len(list_of_aio) != n_col * n_row:
        raise Exception(f"Invalid dimensions of rows/columns to draw {len(list_of_aio)} images from 'list_of_aio'")
    f, ax = plt.subplots(nrows=n_row, ncols=n_col, figsize=(48, 16))
    if n_row == 1 or n_col == 1:
        for i in range(max([n_col, n_row])):
            ax[i].set_title(f'{prefix} for image #{i + 1}', fontsize=fontsize)
            ax[i].imshow(list_of_aio[i], cmap='jet')
            ax[i].axis('off')
    else:
        for i in range(n_row):
            for j in range(n_col):
                ax[i, j].set_title(f'{prefix} for image #{i * n_col + j + 1}', fontsize=fontsize)
                ax[i, j].imshow(list_of_aio[i * n_col + j], cmap='jet')
                ax[i, j].axis('off')
    if show:
        plt.show()
    # plt.cla()


def draw_activity_maps(list_of_maps: np.asarray, n_row: int, n_col: int, prefix: str, fontsize: int = 18, show = False):
    """
    Отрисовка метрик качества оценки полученной модели

    :param list_of_maps: список изображений, содержащих области "важности" определенных признаков для
    прогноза сети
    :param n_row: число строк в 'таблице' изображений
    :param n_col: число столбцов в 'таблице' изображений
    :param prefix: префикс к названию отображаемой метрики
    :param fontsize: размер шрифта подписей к изображениям
    :param show: бинарный индикатор необходимости вывода графика в отдельном окне figure модуля pyplot
    :return: None
    """

    if len(list_of_maps) != n_col * n_row:
        raise Exception(f"Invalid dimensions of rows/columns to draw {len(list_of_maps)} images from 'list_of_aio'")

    f, ax = plt.subplots(nrows=n_row, ncols=n_col, figsize=(48, 16))
    if n_row == 1 or n_col == 1:
        for i in range(max([n_col, n_row])):
            ax[i].set_title(f'{prefix} for image #{i + 1}', fontsize=fontsize)
            # heatmap = np.uint8(cm.jet(list_of_maps[i])[..., :3] * 255)
            ax[i].imshow(list_of_maps[i], cmap='jet', alpha=0.5)  # overlay
            ax[i].axis('off')
    else:
        for i in range(n_row):
            for j in range(n_col):
                ax[i, j].set_title(f'{prefix} for image #{i * n_col + j + 1}', fontsize=fontsize)
                # heatmap = np.uint8(cm.jet(list_of_maps[i * n_col + j])[..., :3] * 255)
                ax[i, j].imshow(list_of_maps[i * n_col + j], cmap='jet', alpha=0.5)  # overlay
                ax[i, j].axis('off')
    if show:
        plt.show()
    # plt.cla()


def draw_train_valid_error(error_train: list, error_valid: list, metric_name: str, title: str, fontsize: int,
                           savename: str = "", show: bool = False):
    """
    Функция построения сравнения графиков метрики качества модели на тренировочном/валидационном наборах

    :param error_train: качество модели на обучаещем наборе данных
    :param error_valid: качество модели на валидационном наборе данных
    :param metric_name: название метрики, по которой модель оценивается
    :param title: подпись к графику
    :param fontsize: размер шрифта подписи к графику
    :param savename: название файла, в который будет сохранен график
    :param show: бинарный индикатор необходимости отрисовки графика в отдельном окне figure модуляpyplot
    :return: None
    """

    if len(error_valid) != len(error_train):
        raise Exception("Incorrect dimensions of cnn quality metrics for train and validation parts")
    fold_num = range(len(error_train))
    plt.plot(fold_num, error_valid, color='r', label=f'{metric_name} for validation')
    plt.plot(fold_num, error_train, color='g', label=f'{metric_name} for train')
    plt.xlabel("Fold Number")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid()
    if savename:
        plt.savefig(savename)
    if show:
        plt.show()
    plt.cla()
