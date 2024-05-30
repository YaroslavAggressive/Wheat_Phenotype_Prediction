import pandas as pd
import matplotlib.pyplot as plt

from support.phenotype_normalization import rank_based_transform, data_standardization

# загрузка исходных данныъ
name_1 = "datasets/wheat/markers_poly_filtered_sync.csv"
name_2 = "datasets/wheat/wheat_pheno_num_sync.csv"

df_1 = pd.read_csv(name_1)
print(df_1['NC_057794.1_1230844_T_C'])

df_2 = pd.read_csv(name_2)
print(df_2.columns)

# посмотрим визуально на распределение фенотипов имеющихся данных -
# урожайность зерна (в граммах) и высоту растений (в см)

# урожайность
crop_yield = df_2["Урожайность.зерна..г."]
fig, ax = plt.subplots()
plt.plot(list(range(crop_yield.shape[0])), list(crop_yield), '.')
plt.xlabel("Номер образца")
plt.ylabel("Урожайность зерна (г)")
plt.title("Показатель урожайности зерна исследованных образцов")
plt.grid()
plt.savefig("crop_yield_wheat.jpg")
# plt.show()
plt.cla()

# высота растений
height = df_2["Высота.растений..см"]
plt.plot(list(range(height.shape[0])), list(height), '.')
plt.xlabel("Номер образца")
plt.ylabel("Высота растений (см)")
plt.title("Высота исследованных образцов пшеницы")
plt.grid()
plt.savefig("height_wheat.jpg")
# plt.show()
plt.cla()

# еще попробуем отфильтровать nan-ы
no_nan_df = df_2[df_2["Урожайность.зерна..г."].notnull() & df_2["Высота.растений..см"].notnull()]
crop_yield_filtered = no_nan_df["Урожайность.зерна..г."]
height_filtered = no_nan_df["Высота.растений..см"]

# распределение показателей урожайности
plt.hist(list(crop_yield_filtered), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.xlabel("Номер образца")
plt.ylabel("Частоты показателей урожайности (г)")
plt.title("Распределение фильтрованных частот урожайности зерна с образцов пшеницы")
plt.grid()
plt.savefig("plots/height_wheat_hist.jpg")
# plt.show()
plt.cla()

# распределение частот урожайности
plt.hist(list(height_filtered))
plt.xlabel("Номер образца")
plt.ylabel("Частота высот растений (см)")
plt.title("Распределение фильтрованных высот образцов пшеницы")
plt.grid()
plt.savefig("plots/crop_yield_wheat_hist.jpg")
# plt.show()
plt.cla()

# далее визуализируем данные с учетом нормализации

# первый метод нормализации ('фенотип' минус 'среднее' делить на 'дисперсию')
# распределение частот высот
plt.hist(list(data_standardization(crop_yield)), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.xlabel("Номер образца")
plt.ylabel("Частоты нормализованных показателей урожайности (г)")
plt.title("Распределение нормализованных стандартно частот урожайности зерна с образцов пшеницы")
plt.grid()
plt.savefig("plots/filter_stand_norm_height_wheat_hist.jpg")
# plt.show()
plt.cla()

# распределение частот урожайности
plt.hist(list(data_standardization(height)))
plt.xlabel("Номер образца")
plt.ylabel("Частота нормализованных высот растений (см)")
plt.title("Распределение нормализованных стандартно высот образцов пшеницы")
plt.grid()
plt.savefig("plots/filter_stand_norm_crop_yield_wheat_hist.jpg")
# plt.show()
plt.cla()

# второй вариант нормализации (через rankdata из scipy.stats и логистрическую функцию)
# здесь еще нужно отфильтровать nan-ы

# распределение частот высот
plt.hist(list(rank_based_transform(crop_yield_filtered)), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.xlabel("Номер образца")
plt.ylabel("Частоты нормализованных показателей урожайности (г)")
plt.title("Распределение нормализованных логистически частот урожайности зерна с образцов пшеницы")
plt.grid()
plt.savefig("plots/filter_log_norm_height_wheat_hist.jpg")
# plt.show()
plt.cla()

# распределение частот урожайности
plt.hist(list(rank_based_transform(height_filtered)))
plt.xlabel("Номер образца")
plt.ylabel("Частота нормализованных высот растений (см)")
plt.title("Распределение нормализованных логистически высот образцов пшеницы")
plt.grid()
plt.savefig("plots/filter_log_norm_crop_yield_wheat_hist.jpg")
# plt.show()
plt.cla()

