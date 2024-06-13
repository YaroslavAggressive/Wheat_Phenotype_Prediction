import os
from PIL import Image
import numpy as np
import pandas as pd
import multiprocessing as mp

THREADS_NUMBER = 20  # пока такое значение, чтобы машина не померла


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


def snp_to_pixel_bw(value: int, max_allels: float) -> float:
    """
    Функция перевода значения снипа в интенсивность черно-белого пикселя. Черным пиксель становится, если
    соответствующее поле имеет пропуск данных (NA), наиболее светлый - при максимальном номере аллели среди
    всех имеющихся.

    :param value: числовая метка аллели для генетического маркера
    :param max_allels: максимальное число аллелей по всем генетическим маркерам датасета
    :return: Значение соответствующего пикселя
    """
    if np.isnan(value):
        return 0.0
    else:
        return 255.0 / (max_allels + 1) * (value + 1)


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
                         for_model_path: str = "", for_user_path: str = "", multithread: bool=False) -> list:
    """
    Основная функция формирования датасета искусственных изображений для обучения и тестирования модели.

    :param df: Полный датасет
    :param n_rows: Число столбцов пикселей в результирующем изображении (которое используется именно в модели)
    :param n_columns: Число строк пикселей в результирующем изображении (которое используется именно в модели)
    :param snp_columns: Список имен генетических столбцов-признаков, которые есть в датасете
    :param weather_columns: Список имен погодных столбцов-признаков, которые есть в датасете
    :param for_model_path: Путь для сохранения датасета изображений в удобном для использования в модели формате
    :param for_user_path: Путь для сохранения датасета изображений в удобном для визуального анализа формате
    :param multithread: Логический флаг, определяющий пользователем необходимость распараллелить вычисления

    :return: Список объектов синтетических изображений в классе Image библиотеки Pillow
    """

    max_allel = max_allels_number(df)

    images = []
    if multithread:
        with mp.Pool(mp.cpu_count() - 3) as aio_process_pool:  # оставляем три ядра на фоновые процессы машины
            threads_data = []
            for i in df.index:
                plant = df.iloc[i]
                threads_data.append((plant, n_rows, n_columns, snp_columns, weather_columns,
                                     for_model_path + "/" + "AIO_" + str(i + 1) + ".png",
                                     for_user_path + "/" + "AIO_view_" + str(i + 1) + ".png", max_allel, i))

            images = aio_process_pool.starmap(process_single_image, threads_data)
    else:
        for i in df.index:
            plant = df.loc[i]
            plant_img = process_single_image(plant, n_rows, n_columns, snp_columns, weather_columns,
                                             for_model_path + "/" + "AIO_" + str(i + 1) + ".png"
                                             if for_model_path else "",
                                             for_user_path + "/" + "AIO_view_" + str(i + 1) + ".png"
                                             if for_user_path else "", max_allel, i)
            images.append(plant_img)
    return images


def create_aio_for_plant(row: pd.DataFrame, rows: int, cols: int, snp_columns: list,
                         weather_columns: list, max_allel: float) -> Image:
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
    arr = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            idx = (i * cols + j) % row.shape[0]  # сдвиг индексов, если записали уже все признаки по разу, начинаем проход заново, заносим сколько влезет
            if idx < len(snp_columns):
                # interval = get_column_interval(total_df[snp_columns[i * cols + j]])
                val = row[snp_columns[idx]]
                bw = snp_to_pixel_bw(val, max_allel)
            else:
                val = row[weather_columns[idx - len(snp_columns)]]
                bw = weather_to_pixel(val)
            arr[i, j] = bw

    img = Image.fromarray(np.uint8(arr), 'L')
    return img


def process_single_image(row: pd.DataFrame, width: int, height: int, snp_columns: list, weather_columns: list,
                         for_model_path: str = "", for_user_path: str = "", max_allel: float = 1.0,
                         idx: int = 0) -> Image:
    """
    Метод, выполняющий создание/сохранение синтетического изображения в двух форматах -
    непосредственно для модели и для визуального просмотра исследователем. В основном написан для
    распараллеливания создания AIO (экономия времени).

    :param row: растение-образец из общей таблицы данных, который требуется обработать
    :param width: число пикселей в столбце результирующего синтетического изображения
    :param height: число пикселей в ряду результирующего синтетического изображения
    :param snp_columns: названия колонок генетических факторов
    :param weather_columns: названия колонок погодных факторов
    :param for_model_path: каталог, в который требуется сохранять изображения для использования моделью
    :param for_user_path: каталог, в который требуется сохранять изображения для использования моделью
    :param max_allel: максимальное число аллелей среди всех генетических маркером имеющихся
    :param idx: индекс для отладочной информации в командную строку - номер образца в общем датасете

    :return: объект искусственного изображения для данного образца в виде класса Image
    """

    plant_img = create_aio_for_plant(row, width, height, snp_columns, weather_columns, max_allel)
    base_width, base_height = 600, 600  # required width and height of input images (for visual estimation)
    view_img = plant_img.resize((base_width, base_height), Image.Resampling.LANCZOS)
    if for_user_path:  # сохранение изображений для визуальной оценки
        view_img.save(for_user_path)
    if for_model_path:  # сохранение изображений непосредственно для модели
        plant_img.save(for_model_path)
    print(f"Image #{idx + 1} processed")
    return plant_img


def max_allels_number(df: pd.DataFrame) -> float:
    """
    Функция поиска максимального количества аллелей среди всех имеющихся генетических маркеров для дальнейшего
    преобразования признаков аллелей в числовые метки, а следом и в значения пикселя в градациях серого

    :param df: датасет с генетическими данным, который необходимо исследовать
    :return: Максимум аллелей для одного генетического маркера среди всех имеющихся снипов
    """

    total_max = 1.0
    for col in df:
        temp_max = max(df[col])
        if temp_max > total_max:
            total_max = temp_max
    return total_max


print("DF#1")
df_1 = pd.read_csv("../datasets/wheat/wheat_pheno_num_sync.csv")
df_1 = df_1.drop(["Unnamed: 0", "number"], axis=1)

print("DF#2")
df_2 = pd.read_csv("../datasets/wheat/markers_poly_filtered_sync.csv")
df_2 = df_2.drop(["Unnamed: 0"], axis=1)

create_image_dataset(df_2, 200, 200, df_2.columns, [], "../AIO_set_wheat/for_model", "../AIO_set_wheat/for_user")
