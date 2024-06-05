import numpy as np
import pandas as pd
from custom_visualization import draw_train_valid_error
import pickle
import os


def draw_research_results(folder_name: str) -> list:
    metrics = []
    for i, fl_name in enumerate(os.listdir(folder_name)):
        with open(folder_name + "/" + fl_name, "rb") as fl:
            metric = pickle.load(fl)
            metrics.append(metric)
    return metrics


height_crop = "height_crop"
crop_brown_rust = "crop_brown_rust"
crop_yellow_rust = "crop_yellow_rust"

dir_ = f"C:/Users/tyryk/PycharmProjects/vkr_2_0/combo_model/checkpoints/"
metrics_dr = f"/metrics/metrics/"

# сначала урожайность и высота

# train mae
add_dir = "mae/train"
trains = draw_research_results(dir_ + height_crop + metrics_dr + height_crop + "_test/" + add_dir)
# valid mae
add_dir = "mae/valid"
valids = draw_research_results(dir_ + height_crop + metrics_dr + height_crop + "_test/" + add_dir)
# результаты по эпохам
draw_train_valid_error(trains[0], valids[0], "MAE",
                       "График MAE прогноза на выборках train/valid выборках между фолдами", 8,
                       f"../plots/mae_folds_{height_crop}_tr_val.jpg")
# результаты по фолдам
draw_train_valid_error(trains[1], valids[1], "MAE",
                       "График MAE прогноза на выборках train/valid выборках по всем эпохам", 8,
                       f"../plots/mae_epochs_{height_crop}_tr_val.jpg")

# train mse
add_dir = "mse/train"
trains = draw_research_results(dir_ + height_crop + metrics_dr + height_crop + "_test/" + add_dir)
# valid mse
add_dir = "mse/valid"
valids = draw_research_results(dir_ + height_crop + metrics_dr + height_crop + "_test/" + add_dir)
# результаты по эпохам
draw_train_valid_error(trains[0], valids[0], "MSE",
                       "График MSE прогноза на выборках train/valid выборках между фолдами", 8,
                       f"../plots/mse_folds_{height_crop}_tr_val.jpg")
# результаты по фолдам
draw_train_valid_error(trains[1], valids[1], "MSE",
                       "График MSE прогноза на выборках train/valid выборках по всем эпохам", 8,
                       f"../plots/mse_epochs_{height_crop}_tr_val.jpg")
exit(0)
# затем урожайность и бурая ржавчина
# train mae
add_dir = "mae/train"
trains = draw_research_results(dir_ + height_crop + metrics_dr + height_crop + "_test/" + add_dir)
# valid mae
add_dir = "mae/valid"
valids = draw_research_results(dir_ + height_crop + metrics_dr + height_crop + "_test/" + add_dir)
# результаты по эпохам
draw_train_valid_error(trains[0], valids[0], "MAE",
                       "График MAE прогноза на выборках train/valid выборках между фолдами", 8,
                       f"../plots/mae_folds_{height_crop}_tr_val.jpg")
# результаты по фолдам
draw_train_valid_error(trains[1], valids[1], "MAE",
                       "График MAE прогноза на выборках train/valid выборках по всем эпохам", 8,
                       f"../plots/mae_epochs_{height_crop}_tr_val.jpg")

# train mse
add_dir = "mse/train"
trains = draw_research_results(dir_ + height_crop + metrics_dr + height_crop + "_test/" + add_dir)
# valid mse
add_dir = "mse/valid"
valids = draw_research_results(dir_ + height_crop + metrics_dr + height_crop + "_test/" + add_dir)
# результаты по эпохам
draw_train_valid_error(trains[0], valids[0], "MSE",
                       "График MSE прогноза на выборках train/valid выборках между фолдами", 8,
                       f"plots/mse_folds_{height_crop}_tr_val.jpg")
# результаты по фолдам
draw_train_valid_error(trains[1], valids[1], "MSE",
                       "График MSE прогноза на выборках train/valid выборках по всем эпохам", 8,
                       f"plots/mse_epochs_{height_crop}_tr_val.jpg")

# и урожайность + желтая ржавчина
# train mae
add_dir = "mae/train"
trains = draw_research_results(dir_ + height_crop + metrics_dr + height_crop + "_test/" + add_dir)
# valid mae
add_dir = "mae/valid"
valids = draw_research_results(dir_ + height_crop + metrics_dr + height_crop + "_test/" + add_dir)
# результаты по эпохам
draw_train_valid_error(trains[0], valids[0], "MAE",
                       "График MAE прогноза на выборках train/valid выборках между фолдами", 8,
                       f"plots/mae_folds_{height_crop}_tr_val.jpg")
# результаты по фолдам
draw_train_valid_error(trains[1], valids[1], "MAE",
                       "График MAE прогноза на выборках train/valid выборках по всем эпохам", 8,
                       f"plots/mae_epochs_{height_crop}_tr_val.jpg")

# train mse
add_dir = "mse/train"
trains = draw_research_results(dir_ + height_crop + metrics_dr + height_crop + "_test/" + add_dir)
# valid mse
add_dir = "mse/valid"
valids = draw_research_results(dir_ + height_crop + metrics_dr + height_crop + "_test/" + add_dir)
# результаты по эпохам
draw_train_valid_error(trains[0], valids[0], "MSE",
                       "График MSE прогноза на выборках train/valid выборках между фолдами", 8,
                       f"plots/mse_folds_{height_crop}_tr_val.jpg")
# результаты по фолдам
draw_train_valid_error(trains[1], valids[1], "MSE",
                       "График MSE прогноза на выборках train/valid выборках по всем эпохам", 8,
                       f"plots/mse_epochs_{height_crop}_tr_val.jpg")
