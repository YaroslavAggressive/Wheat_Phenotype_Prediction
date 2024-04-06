import os
from PIL import Image
import numpy as np
import pandas as pd
import multiprocessing as mp

THREADS_NUMBER = 20  # пока такое значение, чтобы машина не померла


def get_column_interval(column_values: list) -> list:
    return [min(column_values), max(column_values)]


def scale_value(value: float, old_borders: list, new_borders: list) -> float:
    return (new_borders[1] - new_borders[0]) / (old_borders[1] - old_borders[0]) * (value - old_borders[0])


def weather_to_pixel(value: float) -> dict:
    """
    Принцип перевода погодных факторов в пиксели результирующего искусственного изображения по 3 цветовым каналам

    :param value: реальное значения погодного фактора

    :return: [r, g, b] - массив значений интенсивностей цветовых каналов
    """

    value = int(100 * value)
    r = value % 256
    b = value // 256
    g = 0 if value >= 0 else 255
    return {"r": r, "g": g, "b": b}


def snp_to_pixel_rgb(value: float, interval: list) -> dict:  # в интервале должно быть 3 значения: [0, 1, 2] или [-1, 0, 1]
    """
    Принцип перевода в интенсивность пикселя: есть 2 односторонних интервала - [ interval[0], interval[1] ) и
    [ interval[1], interval[2] ). В случае если признк принимает значение на границах интервалов (значений SNP),
    то при этом задается максимальная интенсивность одного из трех каналов, отвечающего за интенсивность цветов в
    пикселе. Если значение промежуточное (только для канал red и green), то значение шкалируется с интервала исходных
    значений на интервал значений пикселя [0, 255].

    :param value: начальное значения признака у образца
    :param interval: интервал значений признака среди всех имеющихся образцов

    :return: [r, g, b] - массив значений интенсивностей цветовых каналов
    """

    r, g, b = 0, 0, 0
    if interval[0] <= value < interval[1]:  # -1 <= val < 0 или 0 <= val < 1
        if value == interval[0]:
            r = 255
        else:
            r = scale_value(value, [interval[0], interval[2]], [0, 255])
    elif value == interval[1]:  # 0 <= val < 1 или 1 <= val < 2
        if value == interval[0]:
            g = 255
        else:
            g = scale_value(value, [interval[0], interval[2]], [0, 255])
    else:  # val == 2 или val == 1
        b = 255
    return {"r": r, "g": g, "b": b}


def snp_to_pixel_bw(value: int) -> dict:
    """
    Функция перевода значения снипа в интенсивность пикселя. В данном случае пиксели должны быть черно-белыми, поэтому
    интервал
    :param value:
    :return:
    """
    pass


def snp_weather_colnames(all_colnames: list, meteo_names: list) -> list:
    """
    Дополнительная функция разделения всех признаков растений, находящихся в таблице
    на две семантически различные части - генетические и погодные.

    :param all_colnames: полный список признаков, описывающих один образец растения
    :param meteo_names: список названий поголных параметров
    :return: список из двух списков, содеражщих отдельно генетические и погодные образцы соответственно
    """

    snp_columns, weather_columns = [], []
    weather_columns = []
    for name in all_colnames:
        flag = False
        for meteo_name in meteo_names:
            if name.startswith(meteo_name):
                weather_columns.append(name)
                flag = True
                break
        if not flag:
            snp_columns.append(name)
    return [snp_columns, weather_columns]


def create_image_dataset(df: pd.DataFrame, n_rows: int, n_columns: int, snp_columns: list, weather_columns: list,
                         original_image_path: str = "", view_image_path: str = "", multithread: bool=False) -> list:
    """
    Основная функция формирования датасета искусственных изображений для обучения и тестирования модели.

    :param df: Полный датасет
    :param n_rows: Число столбцов пикселей в результирующем изображении (которое используется именно в модели)
    :param n_columns: Число строк пикселей в результирующем изображении (которое используется именно в модели)
    :param snp_columns: Список имен генетических столбцов-признаков, которые есть в датасете
    :param weather_columns: Список имен погодных столбцов-признаков, которые есть в датасете
    :param original_image_path: Путь для сохранения датасета изображений в удобном для использования в модели формате
    :param view_image_path: Путь для сохранения датасета изображений в удобном для визуального анализа формате
    :param multithread: Логический флаг, определяющий пользователем необходимость распараллелить вычисления

    :return: Список объектов синтетических изображений в классе Image библиотеки Pillow
    """

    images = []
    if multithread:
        with mp.Pool(mp.cpu_count() - 3) as aio_process_pool:  # оставляем три ядра на фоновые процессы машины
            threads_data = []
            for i in df.index:
                plant = df.iloc[i]
                threads_data.append((plant, n_rows, n_columns, snp_columns, weather_columns,
                                     original_image_path + "/" + "AIO_" + str(i + 1) + ".png",
                                     view_image_path + "/" + "AIO_view_" + str(i + 1) + ".png", i))

            images = aio_process_pool.starmap(process_single_image, threads_data)
    else:
        for i in df.index:
            plant = df.loc[i]
            plant_img = process_single_image(plant, n_rows, n_columns, snp_columns, weather_columns,
                                             original_image_path + "/" + "AIO_" + str(i + 1) + ".png"
                                             if original_image_path else "",
                                             view_image_path + "/" + "AIO_view_" + str(i + 1) + ".png"
                                             if view_image_path else "", i)
            images.append(plant_img)
    return images


def create_aio_for_plant(row: pd.DataFrame, rows: int, cols: int, snp_columns: list,
                         weather_columns: list, ind: int=0) -> Image:
    """
    Функция непосредственно для создания AIO для заданного образца

    :param row: ряд данных образца в полном датасете
    :param rows: число столбцов пикселей в результирующем изображении (которое используется именно в модели)
    :param cols: число строк пикселей в результирующем изображении (которое используется именно в модели)
    :param snp_columns: список имен генетических факторов, которые имеются в наборе данных
    :param weather_columns: список имен погодных факторов, которые имеются в наборе данных

    :return: объект класса Image пакета Pillow, содержащий искусственное изображение
    """
    # if rows * cols != row.shape[0]:
    #     raise Exception("Incorrect dimensions of AIO_set and input dataset cell")
    array_red, array_green, array_blue = np.zeros((rows, cols)), np.zeros((rows, cols)), np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            idx = (i * cols + j) % row.shape[0]  # сдвиг индексов, если записали уже все признаки по разу, начинаем проход заново, заносим сколько влезет
            if idx < len(snp_columns):
                # interval = get_column_interval(total_df[snp_columns[i * cols + j]])
                val = row[snp_columns[idx]]
                rgb = snp_to_pixel(val, [0, 1, 2])
            else:
                val = row[weather_columns[idx - len(snp_columns)]]
                rgb = weather_to_pixel(val)
            array_red[i, j] = rgb["r"]
            array_green[i, j] = rgb["g"]
            array_blue[i, j] = rgb["b"]

    red_img = Image.fromarray(array_red).convert("L")
    green_img = Image.fromarray(array_green).convert("L")
    blue_img = Image.fromarray(array_blue).convert("L")
    img = Image.merge("RGB", (red_img, green_img, blue_img))
    return img


def process_single_image(row: pd.DataFrame, rows: int, cols: int, snp_columns: list, weather_columns: list,
                         original_image_path: str = "", view_image_path: str = "", idx: int = 0) -> Image:
    """
    Метод, выполняющий создание/сохранение синтетического изображения в двух форматах -
    непосредственно для модели и для визуального просмотра исследователем. В основном написан для
    распараллеливания создания AIO (экономия времени).

    :param row: растение-образец из общей таблицы данных, который требуется обработать
    :param rows: число пикселей в столбце результирующего синтетического изображения
    :param cols: число пикселей в ряду результирующего синтетического изображения
    :param snp_columns: названия колонок генетических факторов
    :param weather_columns: названия колонок погодных факторов
    :param original_image_path: каталог, в который требуется сохранять изображения для использования моделью
    :param view_image_path: каталог, в который требуется сохранять изображения для использования моделью
    :param idx: индекс для отладочной информации в командную строку - номер образца в общем датасете

    :return: объект искусственного изображения для данного образца в виде класса Image
    """

    plant_img = create_aio_for_plant(row, rows, cols, snp_columns, weather_columns)
    base_width, base_height = 600, 600  # required width and height of input images (for visual estimation)
    view_img = plant_img.resize((base_width, base_height), Image.Resampling.LANCZOS)
    if view_image_path:  # сохранение изображений для визуальной оценки
        view_img.save(view_image_path)
    if original_image_path:  # сохранение изображений непосредственно для модели
        plant_img.save(original_image_path)
    print(f"Image #{idx + 1} processed")
    return plant_img


