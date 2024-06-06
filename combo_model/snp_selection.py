import pickle

from combo_model.pca_tsne_umap import pca_features, t_sne_features
from combo_model.no_df_model import ComboModelTuner
from image_processing.aio_set_reader import PlantImageContainer

import  numpy as np
import pandas as pd
import tensorflow as tf

from support.phenotype_normalization import rank_based_transform, data_standardization

# поиск пересечений снипов и сохранение в табличный мастаб latex
crop_height = ['NC_057803.1_4754622_C_T', 'NC_057797.1_534615859_A_G', 'NC_057798.1_604438876_A_G',
               'NC_057796.1_10207963_A_G', 'NC_057794.1_149436352_G_C', 'NC_057796.1_10196988_G_T',
               'NC_057794.1_39102905_T_C', 'NC_057797.1_531991977_A_G', 'NC_057800.1_732137392_A_G',
               'NC_057796.1_10209215_G_T', 'NC_057795.1_382123600_C_T', 'NC_057795.1_594638876_C_T',
               'NC_057798.1_722770323_T_A', 'NC_057798.1_757041516_A_G', 'NC_057795.1_383327892_A_G',
               'NC_057803.1_4755814_A_G', 'NC_057801.1_677461618_C_G', 'NC_057799.1_19376950_C_T',
               'NC_057801.1_834376177_G_T', 'NC_057797.1_739918017_G_A', 'NC_057799.1_75635911_G_C',
               'NC_057799.1_75744012_G_C', 'NC_057797.1_537751215_T_G', 'NC_057799.1_43102981_G_A',
               'NC_057795.1_600072180_G_A', 'NC_057798.1_430792946_T_A', 'NC_057798.1_429693834_T_C',
               'NC_057795.1_10570763_C_G', 'NC_057796.1_28507212_A_G', 'NC_057797.1_739918057_A_G',
               'NC_057801.1_367640535_C_A', 'NC_057797.1_776842743_A_G', 'NC_057795.1_383970956_G_A',
               'NC_057797.1_725456771_T_C', 'NC_057794.1_245870738_T_C', 'NC_057796.1_22406463_G_C',
               'NC_057798.1_142433913_C_T', 'NC_057795.1_594638836_T_G', 'NC_057794.1_59894735_C_T',
               'NC_057797.1_29704897_T_G', 'NC_057795.1_636669724_A_C', 'NC_057798.1_7426102_C_T',
               'NC_057795.1_599743536_C_T', 'NC_057795.1_599461234_T_C', 'NC_057802.1_563064506_G_C',
               'NC_057794.1_46694769_C_T', 'NC_057800.1_732137368_T_C', 'NC_057800.1_10043422_A_T',
               'NC_057801.1_165366658_A_G', 'NC_057798.1_752474511_A_G']

crop_brown = ['NC_057797.1_13234957_A_C', 'NC_057796.1_304241564_T_C', 'NC_057797.1_9171771_G_A',
              'NC_057798.1_753176654_C_T', 'NC_057802.1_553182224_G_C', 'NC_057798.1_811195497_A_G',
              'NC_057797.1_694523621_C_A', 'NC_057796.1_304241506_G_A', 'NC_057796.1_305900442_T_G',
              'NC_057797.1_9158310_A_G', 'NC_057801.1_756450913_T_A', 'NC_057801.1_12477398_C_G',
              'NC_057796.1_27639689_A_G', 'NC_057801.1_214091004_A_G', 'NC_057796.1_27639647_T_C',
              'NC_057796.1_255085454_T_C', 'NC_057798.1_702328067_G_A', 'NC_057798.1_98236710_T_G',
              'NC_057796.1_307965282_G_T', 'NC_057797.1_694697835_G_T', 'NC_057796.1_398837919_C_G',
              'NC_057801.1_824083368_G_A', 'NC_057795.1_588246527_G_C', 'NC_057795.1_686682819_G_C',
              'NC_057797.1_694698079_T_C', 'NC_057801.1_794615048_A_G', 'NC_057799.1_615951454_T_C',
              'NC_057801.1_34427962_C_T', 'NC_057801.1_372328967_G_A', 'NC_057799.1_8321833_A_T',
              'NC_057802.1_553182237_G_A', 'NC_057797.1_671555569_G_C', 'NC_057801.1_614024000_G_C',
              'NC_057797.1_16787287_C_G', 'NC_057796.1_255085393_A_T', 'NC_057803.1_38700849_G_A',
              'NC_057802.1_624104_A_G', 'NC_057801.1_217007657_A_C', 'NC_057801.1_83381054_A_T',
              'NC_057795.1_589139981_T_C', 'NC_057800.1_8368421_G_A', 'NC_057802.1_413392816_G_A',
              'NC_057797.1_730298253_C_A', 'NC_057801.1_250750415_T_C', 'NC_057796.1_412732720_T_C',
              'NC_057795.1_686682777_C_T', 'NC_057796.1_307965317_G_C', 'NC_057794.1_485230890_C_T',
              'NC_057798.1_98236735_G_A', 'NC_057801.1_419053524_G_A']

crop_yellow = ['NC_057798.1_608949069_T_C', 'NC_057794.1_65502599_G_T', 'NC_057801.1_620652034_C_T',
               'NC_057798.1_609871637_A_G', 'NC_057799.1_608700361_G_A', 'NC_057796.1_295026318_C_T',
               'NC_057796.1_297803003_C_T', 'NC_057798.1_541165534_T_C', 'NC_057795.1_600072180_G_A',
               'NC_057796.1_295026324_C_G', 'NC_057802.1_413392110_C_T', 'NC_057801.1_620652074_G_A',
               'NC_057796.1_232848824_T_C', 'NC_057801.1_12397731_T_C', 'NC_057801.1_17996663_G_A',
               'NC_057802.1_382214610_G_A', 'NC_057794.1_61327479_T_C', 'NC_057801.1_82259596_T_G',
               'NC_057798.1_736653096_G_A', 'NC_057799.1_75744012_G_C', 'NC_057801.1_12476113_G_A',
               'NC_057798.1_752474363_G_A', 'NC_057794.1_48914576_T_A', 'NC_057795.1_71295734_T_C',
               'NC_057795.1_71295653_C_T', 'NC_057795.1_670070685_G_A', 'NC_057799.1_2959129_G_A',
               'NC_057795.1_51477020_A_C', 'NC_057801.1_164669803_A_G', 'NC_057798.1_540671709_G_A',
               'NC_057798.1_735606871_C_T', 'NC_057794.1_57389185_C_T', 'NC_057802.1_382214449_G_A',
               'NC_057799.1_615944126_G_T', 'NC_057796.1_50601227_C_T', 'NC_057801.1_82128337_A_G',
               'NC_057798.1_674451531_A_G', 'NC_057796.1_146297383_C_T', 'NC_057798.1_736582333_A_G',
               'NC_057795.1_191227009_T_C', 'NC_057799.1_48021727_C_T', 'NC_057797.1_786122359_C_T',
               'NC_057797.1_5358333_G_A', 'NC_057795.1_599461234_T_C', 'NC_057801.1_822973489_A_G',
               'NC_057799.1_557582543_G_T', 'NC_057794.1_41905540_A_G', 'NC_057799.1_635746095_T_G',
               'NC_057801.1_165205171_A_C', 'NC_057797.1_759445639_T_C']

# print(list(set(crop_yellow) & set(crop_brown)))
# print(list(set(crop_height) & set(crop_yellow)))
# print(list(set(crop_height) & set(crop_brown)))
# for i in range(0, 30):
#     crop_yellow[i] = crop_yellow[i].replace("_", "\_")
#     crop_brown[i] = crop_brown[i].replace("_", "\_")
#     crop_height[i] = crop_height[i].replace("_", "\_")
#
# for i in range(0, 30, 3):
#     print(f"${crop_height[i]}$ & ${crop_height[i + 1]}$ & ${crop_height[i + 2]}$ \\\ \hline")
# print("#################")
# for i in range(0, 30, 3):
#     print(f"${crop_brown[i]}$ & ${crop_brown[i + 1]}$ & ${crop_brown[i + 2]}$ \\\ \hline")
# print("#################")
# for i in range(0, 30, 3):
#     print(f"${crop_yellow[i]}$ & ${crop_yellow[i + 1]}$ & ${crop_yellow[i + 2]}$ \\\ \hline")
#

# brown_idx = pd.read_pickle("../combo_model/checkpoints/train_test_indices/indices/crop_brown/train_test_split.txt")
# crop_idx = pd.read_pickle("../combo_model/checkpoints/train_test_indices/indices/height_crop/train_test_split.txt")
# print(len(crop_idx))
# print(len(brown_idx))

with open("../combo_model/checkpoints/train_test_indices/indices/crop_brown/train_test_split.txt", "rb") as fl:
    brown_idx = pickle.load(fl)
with open("../combo_model/checkpoints/train_test_indices/indices/height_crop/train_test_split.txt", "rb") as fl:
    crop_idx = pickle.load(fl)
crop_idx = np.array(list(range(400)))[~crop_idx]
brown_idx = np.array(list(range(400)))[~brown_idx]
print(crop_idx)
print(brown_idx)

model_crop_height = "../combo_model/checkpoints/model_checkpoints/model_saves/height_crop/grid_cv_trained_model_iter4.h5"
model_crop_brown = "../combo_model/checkpoints/model_checkpoints/model_saves/crop_brown/grid_cv_trained_model_iter4.h5"
# model_crop_yellow = "../combo_model/checkpoints/model_checkpoints/model_saves/crop_yellow/crop_yellow4/grid_cv_trained_model_iter4.h5"

model_height = tf.keras.models.load_model(model_crop_height,
                                          custom_objects={'custom_loss_mae': ComboModelTuner.custom_loss_mae},
                                          compile=False)
model_brown = tf.keras.models.load_model(model_crop_brown,
                                         custom_objects={'custom_loss_mae': ComboModelTuner.custom_loss_mae},
                                         compile=False)
# model_yellow = tf.keras.models.load_model(model_crop_yellow,
#                                           custom_objects={'custom_loss_mae': ComboModelTuner.custom_loss_mae},
#                                           compile=False)
model_height.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
                     loss=ComboModelTuner.custom_loss_mae,
                     metrics=[ComboModelTuner.custom_loss_mse])

model_brown.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
                    loss=ComboModelTuner.custom_loss_mae,
                    metrics=[ComboModelTuner.custom_loss_mse])

# model_height.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
#                      loss=ComboModelTuner.custom_loss_mae,
#                      metrics=[ComboModelTuner.custom_loss_mse])

# загружаем дни и изображения, по дням строим метки классов для датасета
folder_images = "../AIO_set_wheat/for_model"
images = PlantImageContainer.load_images_from_folder(folder_images)
pca_features_ = pca_features(images, n_components=5)  # По совету КН взять 5
tsne = t_sne_features(images, n_components=2)  # по совету КН взять 2
total_features = np.concatenate((pca_features_, tsne), axis=1)
df_wheat = pd.read_csv("../datasets/wheat/wheat_pheno_num_sync.csv")
print(df_wheat.head(5))
df_gen = pd.read_csv("../datasets/wheat/markers_poly_filtered_sync.csv")

# labels_height = df_wheat[["Урожайность.зерна..г.", "Высота.растений..см"]].to_numpy()
# labels_yellow = df_wheat[["Урожайность.зерна..г.", "Желтая.ржавчина..."]].to_numpy()
# labels_brown = df_wheat[["Урожайность.зерна..г.", "Бурая.ржавчина..."]].to_numpy()

labels_height = df_wheat[["Высота.растений..см"]].to_numpy()
labels_crop = df_wheat[["Урожайность.зерна..г."]].to_numpy()
labels_brown = df_wheat[["Бурая.ржавчина..."]].to_numpy()

print(labels_crop.flatten()[~np.isnan(labels_crop.flatten())].mean())
print(labels_height.flatten()[~np.isnan(labels_height.flatten())].mean())
print(labels_brown.flatten()[~np.isnan(labels_brown.flatten())].mean())

# нормализация данных
labels_height = rank_based_transform(labels_height.flatten()[~np.isnan(labels_height.flatten())])
labels_crop = rank_based_transform(labels_crop.flatten()[~np.isnan(labels_crop.flatten())])
labels_brown = rank_based_transform(labels_brown.flatten()[~np.isnan(labels_brown.flatten())])

# ошибка на тестовых урожайность высота
pred_crop_height = model_height.predict([images[crop_idx], total_features[crop_idx]])
err_crop = np.abs(labels_crop[crop_idx].flatten() - pred_crop_height[:, 0].flatten())
err_height = np.abs(labels_height[crop_idx].flatten() - pred_crop_height[:, 1].flatten())
print(err_crop.mean())
print(err_height.mean())
print("###########")

# ошибка на тестовых урожайность бурая ржавчина
pred_crop_brown = model_height.predict([images[brown_idx], total_features[brown_idx]])
err_crop = np.abs(labels_crop[brown_idx].flatten() - pred_crop_brown[:, 0].flatten())
err_brown = np.abs(labels_brown[brown_idx].flatten() - pred_crop_brown[:, 1].flatten())
print(err_crop.mean())
print(err_brown.mean())


