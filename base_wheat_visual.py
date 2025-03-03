import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from support.phenotype_normalization import rank_based_transform, data_standardization


def draw_hist(data: list, fig_width: int, fig_height: int, col_type: str, xlabel_name: str, ylabel_name: str,
              title_name: str, save_path: str, show: bool = False, bins: int = 20):
    plt.figure(figsize=(fig_width, fig_height))
    plt.hist(data, bins=bins, color=col_type, alpha=0.7, rwidth=0.85)
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    plt.title(title_name)
    plt.grid()
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.cla()


def draw_base_plot(x_data: list, y_data: list, fig_width: int, fig_height: int, l_type: str,
                   xlabel_name: str, ylabel_name: str, title_name: str, save_path: str,
                   show: bool = False):
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(x_data, y_data, l_type)
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    plt.title(title_name)
    plt.grid()
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.cla()


def draw_base_data_visual(df: pd.DataFrame):
    # посмотрим визуально на распределение фенотипов имеющихся данных -
    # урожайность зерна (в граммах) и высоту растений (в см)

    # урожайность
    crop_yield = df["Урожайность.зерна..г."]
    draw_base_plot(x_data=list(range(crop_yield.shape[0])), y_data=list(crop_yield), fig_width=10, fig_height=6,
                   l_type='.',
                   xlabel_name="Номер образца", ylabel_name="Урожайность зерна (г)",
                   title_name="Показатель урожайности зерна исследованных образцов",
                   save_path="plots/crop_yield_wheat.jpg")

    # высота растений
    height = df["Высота.растений..см"]
    draw_base_plot(x_data=list(range(height.shape[0])), y_data=list(height), fig_width=10, fig_height=6, l_type='.',
                   xlabel_name="Номер образца", ylabel_name="Высота растений (см)",
                   title_name="Высота исследованных образцов пшеницы", save_path="plots/height_wheat.jpg")

    # бурая ржавчина
    brown_rust = df["Бурая.ржавчина..."]
    draw_base_plot(x_data=list(range(brown_rust.shape[0])), y_data=list(brown_rust), fig_width=10, fig_height=6,
                   l_type='.',
                   xlabel_name="Номер образца", ylabel_name="Бурая ржавчина",
                   title_name="Показатели бурой ржавчины у исследованных образцов пшеницы",
                   save_path="plots/brown_rust_wheat.jpg")

    # желтая ржавчина
    yellow_rust = df["Желтая.ржавчина..."]
    draw_base_plot(x_data=list(range(yellow_rust.shape[0])), y_data=list(yellow_rust), fig_width=10, fig_height=6,
                   l_type='.',
                   xlabel_name="Номер образца", ylabel_name="Желтая ржавчина",
                   title_name="Показатели желтой ржавчины у исследованных образцов пшеницы",
                   save_path="plots/yellow_rust_wheat.jpg")


def draw_base_hists(df: pd.DataFrame):
    # отфильтруем nan-ы
    crop_yield_filtered = df[df["Урожайность.зерна..г."].notnull()]["Урожайность.зерна..г."]
    height_filtered = df[df["Высота.растений..см"].notnull()]["Высота.растений..см"]
    brown_rust_filtered = df[df["Бурая.ржавчина..."].notnull()]["Бурая.ржавчина..."]
    yellow_rust_filtered = df[df["Желтая.ржавчина..."].notnull()]["Желтая.ржавчина..."]

    # распределение показателей урожайности
    draw_hist(data=list(crop_yield_filtered), fig_width=10, fig_height=6, col_type='r', xlabel_name="Урожайность (г)",
              ylabel_name="Частоты показателей",
              title_name="Распределение фильтрованных частот урожайности зерна с образцов пшеницы",
              save_path="plots/crop_yield_wheat_hist.jpg",
              bins=7)

    # распределение частот урожайности
    draw_hist(data=list(height_filtered), fig_width=10, fig_height=6, col_type='g', xlabel_name="Высота растений (см)",
              ylabel_name="Частоты показателей",
              title_name="Распределение фильтрованных высот образцов пшеницы",
              save_path="plots/height_wheat_hist.jpg",
              bins=7)

    # распределение показателей бурой ржавчины
    draw_hist(data=list(brown_rust_filtered), fig_width=10, fig_height=6, col_type='b', xlabel_name="Бурая ржавчина",
              ylabel_name="Частоты показателей",
              title_name="Распределение показателей бурой ржавчины у исследованных образцов пшеницы",
              save_path="plots/brown_rust_wheat_hist.jpg",
              bins=5)

    # распределение показателей желтой ржавчины
    draw_hist(data=list(yellow_rust_filtered), fig_width=10, fig_height=6, col_type='k', xlabel_name="Желтая ржавчина",
              ylabel_name="Частоты показателей",
              title_name="Распределение показателей желтой ржавчины у исследованных образцов пшеницы",
              save_path="plots/yellow_rust_wheat_hist.jpg",
              bins=7)


def draw_standard_hists(df: pd.DataFrame):
    # далее визуализируем данные с учетом нормализации

    # отфильтруем nan-ы
    crop_yield_filtered = df[df["Урожайность.зерна..г."].notnull()]["Урожайность.зерна..г."]
    height_filtered = df[df["Высота.растений..см"].notnull()]["Высота.растений..см"]
    brown_rust_filtered = df[df["Бурая.ржавчина..."].notnull()]["Бурая.ржавчина..."]
    yellow_rust_filtered = df[df["Желтая.ржавчина..."].notnull()]["Желтая.ржавчина..."]

    # первый метод нормализации ('фенотип' минус 'среднее' делить на 'дисперсию')
    # распределение частот высот
    draw_hist(data=list(data_standardization(crop_yield_filtered)), fig_width=10, fig_height=6, col_type='r',
              xlabel_name="Урожайность (г)",
              ylabel_name="Частоты нормализованных показателей",
              title_name="Распределение нормализованных стандартно частот урожайности зерна с образцов пшеницы",
              save_path="plots/filter_stand_norm_crop_yield_wheat_hist.jpg")

    # распределение частот урожайности
    draw_hist(data=list(data_standardization(height_filtered)), fig_width=10, fig_height=6, col_type='g',
              xlabel_name="Высота растений (см)",
              ylabel_name="Частоты нормализованных показателей",
              title_name="Распределение нормализованных стандартно высот образцов пшеницы",
              save_path="plots/filter_stand_norm_height_wheat_hist.jpg")

    # распределение частот бурой ржавчины
    draw_hist(data=list(data_standardization(brown_rust_filtered)), fig_width=10, fig_height=6, col_type='b',
              xlabel_name="Бурая ржавчина",
              ylabel_name="Частоты нормализованных показателей",
              title_name="Распределение нормализованных стандартно частот показателей бурой ржавчины",
              save_path="plots/filter_stand_norm_brown_rust_wheat_hist.jpg")

    # распределение частот желтой ржавчины
    draw_hist(data=list(data_standardization(yellow_rust_filtered)), fig_width=10, fig_height=6, col_type='k',
              xlabel_name="Желтая ржавчина",
              ylabel_name="Частоты нормализованных показателей",
              title_name="Распределение нормализованных стандартно частот показателей желтой ржавчины",
              save_path="plots/filter_stand_norm_yellow_rust_wheat_hist.jpg")

    # второй вариант нормализации (через rankdata из scipy.stats и логистрическую функцию)
    # здесь еще нужно отфильтровать nan-ы

    # распределение частот высот
    draw_hist(data=list(rank_based_transform(crop_yield_filtered)), fig_width=10, fig_height=6, col_type='r',
              xlabel_name="Урожайность (г)",
              ylabel_name="Частоты нормализованных показателей",
              title_name="Распределение нормализованных логистически частот урожайности зерна с образцов пшеницы",
              save_path="plots/filter_log_norm_crop_yield_wheat_hist.jpg",
              bins=7)

    # распределение частот урожайности
    draw_hist(data=list(rank_based_transform(height_filtered)), fig_width=10, fig_height=6, col_type='g',
              xlabel_name="Высота растений (см)",
              ylabel_name="Частоты нормализованных показателей",
              title_name="Распределение нормализованных логистически высот образцов пшеницы",
              save_path="plots/filter_log_norm_height_wheat_hist.jpg",
              bins=7)

    # распределение частот бурой ржавчины
    draw_hist(data=list(rank_based_transform(brown_rust_filtered)), fig_width=10, fig_height=6, col_type='b',
              xlabel_name="Бурая ржавчина",
              ylabel_name="Частоты нормализованных показателей",
              title_name="Распределение нормализованных логистически частот бурой ржавчины образцов пшеницы",
              save_path="plots/filter_log_norm_brown_rust_wheat_hist.jpg",
              bins=7)

    # распределение частот желтой ржавчина
    draw_hist(data=list(rank_based_transform(yellow_rust_filtered)), fig_width=10, fig_height=6, col_type='k',
              xlabel_name="Желтая ржавчина",
              ylabel_name="Частоты нормализованных показателей",
              title_name="Распределение нормализованных логистически частот желтой ржавчины образцов пшеницы",
              save_path="plots/filter_log_norm_yellow_rust_wheat_hist.jpg",
              bins=7)


def plot_corr(df: pd.DataFrame, title: str, show: bool = False, save_path: str = ""):
    f = plt.figure(figsize=(20, 10))
    mask = np.triu(np.ones_like(df_my_pheno.corr()))
    dataplot = sns.heatmap(df_my_pheno.corr(), cmap="YlGnBu", annot=True, mask=mask, annot_kws={"fontsize": 32})
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns,
               fontsize=12, rotation=15)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns,
               fontsize=12, rotation=-15)
    plt.title(title, fontsize=22)
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)


def draw_bars(df: pd.DataFrame, fig_width: int, fig_height: int, xlabel_name: str, ylabel_name: str,
              title_name: str, save_path: str, split: int = 30, show: bool = False):
    tmp_df = (df.isna().sum() * 100.0 / len(df)).sort_values(ascending=False)[:split]
    plt.figure(figsize=(fig_width, fig_height))
    plt.bar(list(tmp_df.index), list(tmp_df), color='maroon')
    plt.xticks(range(df[list(tmp_df.index)].select_dtypes(['number']).shape[1]),
               df[list(tmp_df.index)].select_dtypes(['number']).columns,
               fontsize=7, rotation=30)
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    plt.title(title_name)
    plt.grid()
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.cla()


# загрузка исходных данных
name_1 = "datasets/wheat/markers_poly_filtered_sync.csv"
name_2 = "datasets/wheat/wheat_pheno_num_sync.csv"

df_1 = pd.read_csv(name_1)
# print(df_1['NC_057794.1_1230844_T_C'])
df_2 = pd.read_csv(name_2)
# print(df_2.columns)

# вся отрисовка
# draw_base_data_visual(df_2)
# draw_base_hists(df_2)
# draw_standard_hists(df_2)

# корреляционная матрица 4 использованных фенотипов и в целом

# 1. По всем фенотипам
df_2 = df_2.drop(["Unnamed: 0", "number", "name", "section"], axis=1)
df_1 = df_1.drop(["Unnamed: 0"], axis=1)
# plot_corr(df_2, 'Корреляционная матрица фенотипических показателей пшеницы', False, "plots/corr_all_phenotypes.jpg")
# plt.cla()

# 2. Только 'Высота растений', 'Урожайность', 'Бурая ржавчина' и 'Желтая ржавчина'
import numpy as np
df_my_pheno = df_2[["Урожайность.зерна..г.", "Высота.растений..см", "Бурая.ржавчина...", "Желтая.ржавчина..."]]
# mask = np.triu(np.ones_like(df_my_pheno.corr()))
# dataplot = sns.heatmap(df_my_pheno.corr(), cmap="YlGnBu", annot=True, mask=mask, annot_kws={"fontsize": 16})
plot_corr(df_my_pheno, 'Корреляционная матрица 4 выбранных фенотипических показателей', False, "plots/corr_4_phenotypes.jpg")
exit(0)

# графики доли пропусков по снипам и по фенотипам
# 1. ОНП (генетические маркеры)
draw_bars(df_1, 10, 6, "Генетические маркеры", "Процент пропусков (%)",
          "Процент отсутствующих данных по каждому SNP", "plots/nans_snps.jpg")

# 2. Фенотипы
draw_bars(df_my_pheno, 10, 6, "Фенотипические показатели", "Процент пропусков (%)",
          "Процент отсутствующих данных по фенотипам", "plots/nans_pheno.jpg")