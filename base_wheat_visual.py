import pandas as pd
import matplotlib.pyplot as plt

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
crop_yeild = df_2["Урожайность.зерна..г."]
fig, ax = plt.subplots()
plt.plot(list(range(crop_yeild.shape[0])), list(crop_yeild), '.')
plt.xlabel("Номер образца")
plt.ylabel("Урожайность зерна (г)")
plt.title("Показатель урожайности зерна исследованных образцов")
plt.grid()
plt.savefig("crop_yield_wheat.jpg")
plt.show()
plt.cla()

# высота растений
height = df_2["Высота.растений..см"]
plt.plot(list(range(height.shape[0])), list(height), '.')
plt.xlabel("Номер образца")
plt.ylabel("Высота растений (см)")
plt.title("Высота исследованных образцов пшеницы")
plt.grid()
plt.savefig("height_wheat.jpg")
plt.show()
plt.cla()

# распределение частот высот
plt.hist(list(crop_yeild), bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.xlabel("Номер образца")
plt.ylabel("Частоты показателей урожайности (г)")
plt.title("Распределение частот урожайности зерна с образцов пшеницы")
plt.grid()
plt.savefig("height_wheat_hist.jpg")
plt.show()
plt.cla()

# распределение частот урожайности
plt.hist(list(height))
plt.xlabel("Номер образца")
plt.ylabel("Частота высот растений (см)")
plt.title("Распределение высот образцов пшеницы")
plt.grid()
plt.savefig("crop_yield_wheat_hist.jpg")
plt.show()
plt.cla()

