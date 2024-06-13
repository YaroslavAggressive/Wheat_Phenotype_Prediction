import numpy as np
import pandas as pd
from custom_visualization import draw_train_valid_error
import matplotlib.pyplot as plt
import pickle
import os


def draw_research_results(folder_name: str) -> list:
    metrics = []
    for i, fl_name in enumerate(os.listdir(folder_name)):
        if os.path.isfile(folder_name + "/" + fl_name):
            with open(folder_name + "/" + fl_name, "rb") as fl:
                metric = pickle.load(fl)
                metrics.append(metric)
    return metrics


height_crop = "height_crop"
crop_brown_rust = "crop_brown"
crop_yellow_rust = "crop_yellow"

dir_ = f"C:/Users/tyryk/PycharmProjects/vkr_2_0/combo_model/checkpoints/"
metrics_dr = f"/metrics/metrics/"

# сначала урожайность и высота

# train mae
add_dir = "/mae/train"
trains = draw_research_results(dir_ + metrics_dr + height_crop + add_dir)
# valid mae
add_dir = "/mae/valid"
valids = draw_research_results(dir_ + metrics_dr + height_crop + add_dir)
# результаты по эпохам
draw_train_valid_error(trains[0], valids[0], "MAE",
                       "График MAE прогноза на выборках train/valid выборках между фолдами", 8,
                       f"../plots/mae_folds_{height_crop}_tr_val.jpg")
# результаты по фолдам
draw_train_valid_error(trains[1], valids[1], "MAE",
                       "График MAE прогноза на выборках train/valid выборках по всем эпохам", 8,
                       f"../plots/mae_epochs_{height_crop}_tr_val.jpg")

# train mse
add_dir = "/mse/train"
trains = draw_research_results(dir_ + metrics_dr + height_crop + add_dir)
# valid mse
add_dir = "/mse/valid"
valids = draw_research_results(dir_ + metrics_dr + height_crop + add_dir)
# результаты по эпохам
draw_train_valid_error(trains[0], valids[0], "MSE",
                       "График MSE прогноза на выборках train/valid выборках между фолдами", 8,
                       f"../plots/mse_folds_{height_crop}_tr_val.jpg")
# результаты по фолдам
draw_train_valid_error(trains[1], valids[1], "MSE",
                       "График MSE прогноза на выборках train/valid выборках по всем эпохам", 8,
                       f"../plots/mse_epochs_{height_crop}_tr_val.jpg")

# затем урожайность и бурая ржавчина
# train mae
add_dir = "/mae/train"
trains = draw_research_results(dir_ + metrics_dr + crop_brown_rust + add_dir)
# valid mae
add_dir = "/mae/valid"
valids = draw_research_results(dir_ + metrics_dr + crop_brown_rust + add_dir)
# результаты по эпохам
draw_train_valid_error(trains[0], valids[0], "MAE",
                       "График MAE прогноза на выборках train/valid выборках между фолдами", 8,
                       f"../plots/mae_folds_{crop_brown_rust}_tr_val.jpg")
# результаты по фолдам
draw_train_valid_error(trains[1], valids[1], "MAE",
                       "График MAE прогноза на выборках train/valid выборках по всем эпохам", 8,
                       f"../plots/mae_epochs_{crop_brown_rust}_tr_val.jpg")

# train mse
add_dir = "/mse/train"
trains = draw_research_results(dir_ + metrics_dr + crop_brown_rust + add_dir)
# valid mse
add_dir = "/mse/valid"
valids = draw_research_results(dir_ + metrics_dr + crop_brown_rust + add_dir)
# результаты по эпохам
draw_train_valid_error(trains[0], valids[0], "MSE",
                       "График MSE прогноза на выборках train/valid выборках между фолдами", 8,
                       f"../plots/mse_folds_{crop_brown_rust}_tr_val.jpg")
# результаты по фолдам
draw_train_valid_error(trains[1], valids[1], "MSE",
                       "График MSE прогноза на выборках train/valid выборках по всем эпохам", 8,
                       f"../plots/mse_epochs_{crop_brown_rust}_tr_val.jpg")

# и урожайность + желтая ржавчина
# train mae
# for i, dr in enumerate(os.listdir(dir_ + metrics_dr + crop_yellow_rust)):
#     os.mkdir(f"../plots/{crop_yellow_rust}_data_{i}")
#     add_dir = "/mae/train"
#     trains = draw_research_results(dir_ + metrics_dr + crop_yellow_rust + "/" + dr + add_dir)
#     # valid mae
#     add_dir = "/mae/valid"
#     valids = draw_research_results(dir_ + metrics_dr + crop_yellow_rust + "/" + dr + add_dir)
#     # результаты по эпохам
#     draw_train_valid_error(trains[0], valids[0], "MAE",
#                            "График MAE прогноза на выборках train/valid выборках между фолдами", 8,
#                            f"../plots/{crop_yellow_rust}_data_{i}/mae_folds_{crop_yellow_rust}_tr_val_{i}.jpg")
#     # результаты по фолдам
#     draw_train_valid_error(trains[1], valids[1], "MAE",
#                            "График MAE прогноза на выборках train/valid выборках по всем эпохам", 8,
#                            f"../plots/{crop_yellow_rust}_data_{i}/mae_epochs_{crop_yellow_rust}_tr_val_{i}.jpg")
#
#     # train mse
#     add_dir = "/mse/train"
#     trains = draw_research_results(dir_ + metrics_dr + crop_yellow_rust + "/" + dr + add_dir)
#     # valid mse
#     add_dir = "/mse/valid"
#     valids = draw_research_results(dir_ + metrics_dr + crop_yellow_rust + "/" + dr + add_dir)
#     # результаты по эпохам
#     draw_train_valid_error(trains[0], valids[0], "MSE",
#                            "График MSE прогноза на выборках train/valid выборках между фолдами", 8,
#                            f"../plots/{crop_yellow_rust}_data_{i}/mse_folds_{crop_yellow_rust}_tr_val_{i}.jpg")
#     # результаты по фолдам
#     draw_train_valid_error(trains[1], valids[1], "MSE",
#                            "График MSE прогноза на выборках train/valid выборках по всем эпохам", 8,
#                            f"../plots/{crop_yellow_rust}_data_{i}/mse_epochs_{crop_yellow_rust}_tr_val_{i}.jpg")

# отрисуем тестовые ошибки
# урожайность + высота растений
add_dir = "/mae"
tests_mae = draw_research_results(dir_ + metrics_dr + height_crop + add_dir)

# plt.plot(list(range(len(tests_mae[0]))), tests_mae[0], color="r")
plt.figure(figsize=(10, 6))
plt.bar(list(range(len(tests_mae[0]))), tests_mae[0], color="red")
plt.xlabel("Номер итерации подбора параметров")
plt.ylabel("MAE")
plt.title("MAE прогнозирования урожайности и высоты растений на тестовой выборке")
plt.grid()
plt.savefig("../plots/tests_mae_mse/test_crop_height_mae.jpg")
plt.cla()

# урожайность + бурая ржавчина
add_dir = "/mae"
tests_mae = draw_research_results(dir_ + metrics_dr + crop_brown_rust + add_dir)

# plt.plot(list(range(len(tests_mae[0]))), tests_mae[0], color="r")
plt.figure(figsize=(10, 6))
plt.bar(list(range(len(tests_mae[0]))), tests_mae[0], color="green")
plt.xlabel("Номер итерации подбора параметров")
plt.ylabel("MAE")
plt.title("MAE прогнозирования урожайности и бурой ржавчины на тестовой выборке")
plt.grid()
plt.savefig("../plots/tests_mae_mse/test_crop_brown_mae.jpg")
plt.cla()

# урожайность + желтая ржавчина
tests_mae = draw_research_results(dir_ + metrics_dr + crop_yellow_rust)

# plt.plot(list(range(len(tests_mae[0]))), tests_mae[0], color="r")
plt.figure(figsize=(10, 6))
plt.bar(list(range(len(tests_mae[0]))), tests_mae[0], color="blue")
plt.xlabel("Номер итерации подбора параметров")
plt.ylabel("MAE")
plt.title("MAE прогнозирования урожайности и желтой ржавчины на тестовой выборке")
plt.grid()
plt.savefig("../plots/tests_mae_mse/test_crop_yellow_mae.jpg")
plt.cla()
