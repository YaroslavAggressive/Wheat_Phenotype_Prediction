import numpy as np
import os
import cv2 as cv


class PlantImageContainer:
    """
    Формарование датасета AIO растительной культуры для последующего решения
    задачи классификации.
    """
    def __init__(self, class_bins=[]):
        self.class_bins = class_bins
        self.n_labels = len(class_bins) - 1 if class_bins else 0
        self.class_mids = [(0.5 * (class_bins[i] + class_bins[i+1])) for i in range(len(class_bins) - 1)]

    @staticmethod
    def load_images_from_folder(folder: str) -> np.asarray:
        """
        Функция подгрузки необходимого набора искусственных изображений из передаваемого каталога.

        :param folder: папка с изображениями, сохраненными в формате .png

        :return: список формата Numpy, содержащие AIO в объектах класса Image из Pillow
        """

        images = []
        for filename in os.listdir(folder):
            img = cv.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(np.asarray(img).astype(np.float32))
        return np.asarray(images)

    def get_image_labels(self, resp: np.asarray) -> np.asarray:
        """
        Функция по созданию классовых меток AIO (в данном случае они формируются отдельно
        от считанных данных, поэтому важно передавать отклики в том же порядке, в котором
        хранятся искусственные изображения)

        :param resp: вектор значений целевой переменной - фенотипа, по кторой разбиваем на классы AIO.
        :return: список классовых меток для всех изображений в датасете
        """
        labels = []
        for resp_i in resp:
            for i in range(self.n_labels):
                if self.class_bins[i] <= resp_i < self.class_bins[i + 1]:
                    labels.append(np.uint8(i))
        return np.asarray(labels)
