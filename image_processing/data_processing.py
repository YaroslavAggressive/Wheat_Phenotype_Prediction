import pandas as pd
import numpy as np
import os

import argparse

YEAR_DAYS = 365


def create_complete_df(df_total: pd.DataFrame, df_meteo: pd.DataFrame, col_names: list,
                       days_before: int, days_after: int) -> pd.DataFrame:
    """
    Функция итоговой сборки датасета растений с погодными/генетическими для работы моделей

    :param df_total: полный датасет с генетичесикми маркерами по образцам
    :param df_meteo: датасет из погодных параметров по дням, одна строка - один день с погодными факторами
    :param col_names: список параметров, которые требуется собрать по дням для одного образца растения
    :param days_before: дней до дня посадки образцов, по которым собираем параметры
    :param days_after: дней после дня посадки образцов, по которым собираем параметры
    :return: итоговый датафрейм для работы
    """

    basic_names = list(df_meteo.columns)

    # убираем doy, year, geoid, они не нужны фактически для обучения моделей
    for name in ["geo_id", "year", "doy"]:
        basic_names.pop(basic_names.index(name))

    days_indexes = [i for i in range(-days_before, days_after + 1) for j in [i] * len(basic_names)]

    for i, colname in zip(days_indexes, col_names):
        for j in range(df_total.shape[0]):
            year = df_total.loc[j, "year"]
            geoid = df_total.loc[j, "geo_id"]

            sowing_doy = df_total.loc[j, "doy"]
            doy = sowing_doy + days_indexes[i]
            if doy <= 0:
                year -= 1
                doy += YEAR_DAYS
            elif doy > YEAR_DAYS:
                year += 1
                doy -= YEAR_DAYS

            row = df_meteo[(df_meteo["year"] == year) & (df_meteo["doy"] == doy) & (df_meteo["geo_id"] == geoid)]

            for name in basic_names:
                if colname.startswith(name):
                    df_total.loc[j, colname] = row[name].values[0]

    return df_total


def create_gene_df(df_total: pd.DataFrame, df_genes: pd.DataFrame) -> pd.DataFrame:

    """
    Функция сборки итогового набора данных по генетическим/погодным параметрам, описывающим образцы

    :param df_total: датасет с генетическими маркерами, описывающими каждый образец
    :param df_genes: датасет с генетическими факторами, влияющими на развитие образцов
    :return: Итоговый набор данных, содержащий все изучаемые показатели
    """

    for j, name in enumerate(df_genes.columns):
        for i in range(df_total.shape[0]):
            seqid = df_total.loc[i, "seqid"]
            row = df_genes[df_genes["seqid"] == seqid]
            df_total.loc[i, name] = row[name].values[0]

    return df_total


def vigna_to_table(df_genes: pd.DataFrame, df_meteo: pd.DataFrame, df_resp: pd.DataFrame,
                   days_before: int, days_after: int, colnames: list) -> pd.DataFrame:

    """
    Второй вариант создания итоговой таблицы данных для дальнейшего построения AIO. Здесь к погодным данным
    приписываются генетические. Фактически, единственный правильный подход к обработке имеющихся данных.

    :param df_genes: датасет с генетическими параметрами для каждого образца
    :param df_meteo: датасет с погодными данными за days_before до и days_after после посадки образцов
    :param df_resp: результирующий датасет, который под конец останется со всеми параметрами
    :param days_before: число дней до посадки образцов, за которые учитываем погодные данные
    :param days_after: число дней после посадки образцов, за которые учитываем погодные данные
    :param colnames: список параметров окружающей среды, которые учитываем до и после посадки
    :return: Итоговый датасет с климатическими и генетическими параметрами, по которым формируем уже искусственные
    изображения
    """

    weather_col_names = [name + str(i + 1) for i in range(days_after + days_before + 1) for name in colnames]

    df_resp = df_resp.reindex(columns=list(df_resp.columns) + list(df_genes.columns)[1:] + weather_col_names,
                              fill_value=0)

    df_with_genes = create_gene_df(df_resp, df_genes)
    df_total = create_complete_df(df_with_genes, df_meteo, weather_col_names, days_before, days_after)

    return df_total


# creating vigna dataset for

df_gen = pd.read_csv("../datasets/vigna/full_tab_gwas.csv")
df_meteo = pd.read_csv("../datasets/vigna/meteo_vigna.csv")
df_date = pd.read_csv("../datasets/vigna/pheno_tab_muc.csv")

# df_gen_with_date = merge_columns_to_df(df_gen, df_date, ["year", "geo_id", "doy"])
# df_gen_with_date.to_csv("../tables_for_AIO/full_tab_gwas_with_date.csv", index=False)

# creating total dataframe
df_gen = df_gen.drop(["resp"])
total_df = vigna_to_table(df_genes=df_gen, df_meteo=df_meteo, df_resp=df_date, days_before=5, days_after=20,
                          colnames=["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "GWETTOP", "ALLSKY_SFC_LW_DWN", "dl"])
total_df.to_csv("../tables_for_AIO/total_df_for_aio_vigna.csv", index=False)

# creating chickpea dataset
# import allel

# ffl = allel.read_vcf("../datasets/chickpea/merged.005.085.recode.vcf.gz")
#
# gt = allel.GenotypeArray(ffl['calldata/GT'])
# print(len(gt.count_het(axis=1)))
# exit(0)
# print(ffl['samples'])

# creating total tables for chickpea data

# load and watch over the weather data tables for Kuban/Astrakhan
aos_2022_df = pd.read_csv("../datasets/chickpea/pogoda_aos_28_04_2022.csv")
print(aos_2022_df.shape)

# так как таблицы будем делать по годам, все имена файлов с погодой в один список впихнем
chickpea_df = pd.read_csv("../tables_for_AIO/total_df_for_aio_chickpea_28042016.csv")
for colname in chickpea_df.columns:
    print(f"column = {colname}, value = {chickpea_df[colname][0]}")
vigna_df = pd.read_csv("../tables_for_AIO/total_df_for_aio_vigna.csv")
for colname in vigna_df.columns:
    print(f"column = {colname}, value = {vigna_df[colname][0]}")
weather_filenames = ["pogoda_aos_28_04_2022.csv", "pogoda_kos_02_05_2017.csv", "pogoda_kos_18_05_2016.csv",
                     "pogoda_kos_28_04_2016.csv", "pogoda_kos_13_05_2022.csv"]

df_folder = "../datasets/chickpea/"
weather_filenames = ["pogoda_kos_28_04_2016.csv"]
genotype_filenames = ["chickpea_raw_df_col_row_names.csv"]
days_before = 5
days_after = 20

for fl in weather_filenames:
    print(fl.split("_"))

weather_parameters = ["ALLSKY_SFC_LW_DWN", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR"]

for filename_meteo, filename_gen in zip(weather_filenames, genotype_filenames):
    # сначала обработаем погоду
    filepath = df_folder + filename_meteo

    # создаем новый объект датасета, в котором переформатируем данные
    tmp_meteo_df = pd.read_csv(filepath, sep=";")

    # поскольку pandas считает первую строку параметров столбцами фрейма данных, немного подшаманить приедтся
    first_row = pd.DataFrame([tmp_meteo_df.columns])
    tmp_meteo_df.columns = range(len(tmp_meteo_df.columns))
    tmp_meteo_df = pd.concat([first_row, tmp_meteo_df], axis=0)

    # разворачиваем датафрейм в один ряд
    res_df = tmp_meteo_df.unstack().to_frame().T
    res_df.columns = [param + str(i + 1) for i in range(days_before + days_after) for param in weather_parameters]

    # необходимо датафрейм согласовать с генетическим по колиечству записей и "склеить" по строкам

    # скачаем датафрейм, к которому будем добавлят погоду
    tmp_geno_df = pd.read_csv(df_folder + filename_gen)
    n_rows = tmp_geno_df.shape[0]

    # все образцы посажены в один день (вроде как, хотя тут не совсем уверен), поэтому "размножим" просто погоду
    res_df = res_df.loc[res_df.index.repeat(n_rows)].reset_index(drop=True)
    total_df = pd.concat([tmp_geno_df, res_df], axis=1)

    # сохранение в файл для добавления длины светового дня
    total_df.to_csv("../tables_for_AIO/chickpea_geo_meteo_noDL_28042016.csv", index=False)

# здесь теперь перепишем порядок столбцов дата фрейма, чтобы сохранить порядок в создании AIO
# df_chickpea_for_aio = pd.read_csv("../tables_for_AIO/total_df_for_aio_chickpea_28042016.csv")
# basic_meteo_parameters = ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "GWETTOP", "ALLSKY_SFC_LW_DWN", "DL"]
# days_before = 5
# days_after = 20
# weather_parameters = [param + str(i + 1)
#                       for i in range(days_before + days_after)
#                       for param in basic_meteo_parameters]
# for i in range(len(df_chickpea_for_aio.columns)):
#     print(df_chickpea_for_aio.columns[i])
# df_chickpea_for_aio.columns = list(df_chickpea_for_aio.columns[:len(df_chickpea_for_aio.columns) -
#                                                                 len(weather_parameters)]) + weather_parameters
# for name in df_chickpea_for_aio.columns:
#     print(name)
# # сохраняем
# df_chickpea_for_aio.to_csv("../tables_for_AIO/total_df_for_aio_chickpea_28042016.csv", index=False)




