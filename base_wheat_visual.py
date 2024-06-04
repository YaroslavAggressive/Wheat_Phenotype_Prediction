import pandas as pd
import matplotlib.pyplot as plt

from support.phenotype_normalization import rank_based_transform, data_standardization

# загрузка исходных данных
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
plt.figure(figsize=(10, 6))
fig, ax = plt.subplots()
plt.plot(list(range(crop_yield.shape[0])), list(crop_yield), '.')
plt.xlabel("Номер образца")
plt.ylabel("Урожайность зерна (г)")
plt.title("Показатель урожайности зерна исследованных образцов")
plt.grid()
plt.savefig("plots/crop_yield_wheat.jpg")
# plt.show()
plt.cla()

# высота растений
height = df_2["Высота.растений..см"]
plt.figure(figsize=(10, 6))
plt.plot(list(range(height.shape[0])), list(height), '.')
plt.xlabel("Номер образца")
plt.ylabel("Высота растений (см)")
plt.title("Высота исследованных образцов пшеницы")
plt.grid()
plt.savefig("plots/height_wheat.jpg")
# plt.show()
plt.cla()

# бурая ржавчина
brown_rust = df_2["Бурая.ржавчина..."]
plt.figure(figsize=(10, 6))
plt.plot(list(range(brown_rust.shape[0])), list(brown_rust), '.')
plt.xlabel("Номер образца")
plt.ylabel("Бурая ржавчина")
plt.title("Показатели бурой ржавчины у исследованных образцов пшеницы")
plt.grid()
plt.savefig("plots/brown_rust_wheat.jpg")
# plt.show()
plt.cla()

# желтая ржавчина
yellow_rust = df_2["Желтая.ржавчина..."]
plt.figure(figsize=(10, 6))
plt.plot(list(range(yellow_rust.shape[0])), list(yellow_rust), '.')
plt.xlabel("Номер образца")
plt.ylabel("Желтая ржавчина")
plt.title("Показатели желтой ржавчины у исследованных образцов пшеницы")
plt.grid()
plt.savefig("plots/yellow_rust_wheat.jpg")
# plt.show()
plt.cla()

# еще попробуем отфильтровать nan-ы
crop_yield_filtered = df_2[df_2["Урожайность.зерна..г."].notnull()]["Урожайность.зерна..г."]
height_filtered = df_2[df_2["Высота.растений..см"].notnull()]["Высота.растений..см"]
brown_rust_filtered = df_2[df_2["Бурая.ржавчина..."].notnull()]["Бурая.ржавчина..."]
yellow_rust_filtered = df_2[df_2["Желтая.ржавчина..."].notnull()]["Желтая.ржавчина..."]

# распределение показателей урожайности
plt.figure(figsize=(10, 6))
plt.hist(list(crop_yield_filtered), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.xlabel("Урожайность (г)")
plt.ylabel("Частоты показателей")
plt.title("Распределение фильтрованных частот урожайности зерна с образцов пшеницы")
plt.grid()
plt.savefig("plots/height_wheat_hist.jpg")
# plt.show()
plt.cla()

# распределение частот урожайности
plt.figure(figsize=(10, 6))
plt.hist(list(height_filtered))
plt.xlabel("Высота растений (см)")
plt.ylabel("Частота показателей")
plt.title("Распределение фильтрованных высот образцов пшеницы")
plt.grid()
plt.savefig("plots/crop_yield_wheat_hist.jpg")
# plt.show()
plt.cla()

# распределение показателей бурой ржавчины
plt.figure(figsize=(10, 6))
plt.hist(list(brown_rust_filtered))
plt.xlabel("Бурая ржавчина")
plt.ylabel("Частота показателей")
plt.title("Распределение показателей бурой ржавчины у исследованных образцов пшеницы")
plt.grid()
plt.savefig("plots/brown_rust_wheat_hist.jpg")
plt.show()
plt.cla()

# распределение показателей желтой ржавчины
plt.figure(figsize=(10, 6))
plt.hist(list(yellow_rust_filtered), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.xlabel("Желтая ржавчина")
plt.ylabel("Частота показателей")
plt.title("Распределение показателей желтой ржавчины у исследованных образцов пшеницы")
plt.grid()
plt.savefig("plots/yellow_rust_wheat_hist.jpg")
plt.show()
plt.cla()

# далее визуализируем данные с учетом нормализации

# первый метод нормализации ('фенотип' минус 'среднее' делить на 'дисперсию')
# распределение частот высот
plt.figure(figsize=(10, 6))
plt.hist(list(data_standardization(crop_yield)), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.xlabel("Урожайность (г)")
plt.ylabel("Частоты нормализованных показателей")
plt.title("Распределение нормализованных стандартно частот урожайности зерна с образцов пшеницы")
plt.grid()
plt.savefig("plots/filter_stand_norm_height_wheat_hist.jpg")
# plt.show()
plt.cla()

# распределение частот урожайности
plt.figure(figsize=(10, 6))
plt.hist(list(data_standardization(height)))
plt.xlabel("Высота растений (см)")
plt.ylabel("Частота нормализованных показателей")
plt.title("Распределение нормализованных стандартно высот образцов пшеницы")
plt.grid()
plt.savefig("plots/filter_stand_norm_crop_yield_wheat_hist.jpg")
# plt.show()
plt.cla()

# распределение частот бурой ржавчины
plt.figure(figsize=(10, 6))
plt.hist(list(data_standardization(brown_rust_filtered)), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.xlabel("Бурая ржавчина")
plt.ylabel("Частоты нормализованных показателей")
plt.title("Распределение нормализованных стандартно частот показателей бурой ржавчины")
plt.grid()
plt.savefig("plots/filter_stand_norm_brown_rust_wheat_hist.jpg")
# plt.show()
plt.cla()

# распределение частот желтой ржавчины
plt.figure(figsize=(10, 6))
plt.hist(list(data_standardization(yellow_rust_filtered)))
plt.xlabel("Желтая ржавчина")
plt.ylabel("Частота нормализованных показателей")
plt.title("Распределение нормализованных стандартно частот показателей желтой ржавчины")
plt.grid()
plt.savefig("plots/filter_stand_norm_yellow_rust_wheat_hist.jpg")
# plt.show()
plt.cla()

# второй вариант нормализации (через rankdata из scipy.stats и логистрическую функцию)
# здесь еще нужно отфильтровать nan-ы

# распределение частот высот
plt.figure(figsize=(10, 6))
plt.hist(list(rank_based_transform(crop_yield_filtered)), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.xlabel("Урожайность (г)")
plt.ylabel("Частоты нормализованных показателей")
plt.title("Распределение нормализованных логистически частот урожайности зерна с образцов пшеницы")
plt.grid()
plt.savefig("plots/filter_log_norm_height_wheat_hist.jpg")
# plt.show()
plt.cla()

# распределение частот урожайности
plt.figure(figsize=(10, 6))
plt.hist(list(rank_based_transform(height_filtered)))
plt.xlabel("Высота растений (см)")
plt.ylabel("Частота нормализованных показателей")
plt.title("Распределение нормализованных логистически высот образцов пшеницы")
plt.grid()
plt.savefig("plots/filter_log_norm_crop_yield_wheat_hist.jpg")
# plt.show()
plt.cla()

# распределение частот бурой ржавчины
plt.figure(figsize=(10, 6))
plt.hist(list(rank_based_transform(brown_rust_filtered)), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.xlabel("Бурая ржавчина")
plt.ylabel("Частоты нормализованных показателей")
plt.title("Распределение нормализованных логистически частот бурой ржавчины образцов пшеницы")
plt.grid()
plt.savefig("plots/filter_log_norm_brown_rust_wheat_hist.jpg")
# plt.show()
plt.cla()

# распределение частот желтой ржавчина
plt.figure(figsize=(10, 6))
plt.hist(list(rank_based_transform(yellow_rust_filtered)))
plt.xlabel("Желтая ржавчина")
plt.ylabel("Частота нормализованных показателей")
plt.title("Распределение нормализованных логистически частот желтой ржавчины образцов пшеницы")
plt.grid()
plt.savefig("plots/filter_log_norm_yellow_rust_wheat_hist.jpg")
# plt.show()
plt.cla()

