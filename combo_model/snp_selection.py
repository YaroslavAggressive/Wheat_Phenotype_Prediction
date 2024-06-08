import pickle

from combo_model.pca_tsne_umap import pca_features, t_sne_features
from combo_model.no_df_model import ComboModelTuner
from image_processing.aio_set_reader import PlantImageContainer

import  numpy as np
import pandas as pd
import tensorflow as tf

from support.phenotype_normalization import rank_based_transform, data_standardization

# поиск пересечений снипов и сохранение в табличный мастаб latex
# сначала GradRAM
# crop_height_grad = ['NC_057803.1_4754622_C_T', 'NC_057797.1_534615859_A_G', 'NC_057798.1_604438876_A_G',
#                     'NC_057796.1_10207963_A_G', 'NC_057794.1_149436352_G_C', 'NC_057796.1_10196988_G_T',
#                     'NC_057794.1_39102905_T_C', 'NC_057797.1_531991977_A_G', 'NC_057800.1_732137392_A_G',
#                     'NC_057796.1_10209215_G_T', 'NC_057795.1_382123600_C_T', 'NC_057795.1_594638876_C_T',
#                     'NC_057798.1_722770323_T_A', 'NC_057798.1_757041516_A_G', 'NC_057795.1_383327892_A_G',
#                     'NC_057803.1_4755814_A_G', 'NC_057801.1_677461618_C_G', 'NC_057799.1_19376950_C_T',
#                     'NC_057801.1_834376177_G_T', 'NC_057797.1_739918017_G_A', 'NC_057799.1_75635911_G_C',
#                     'NC_057799.1_75744012_G_C', 'NC_057797.1_537751215_T_G', 'NC_057799.1_43102981_G_A',
#                     'NC_057795.1_600072180_G_A', 'NC_057798.1_430792946_T_A', 'NC_057798.1_429693834_T_C',
#                     'NC_057795.1_10570763_C_G', 'NC_057796.1_28507212_A_G', 'NC_057797.1_739918057_A_G',
#                     'NC_057801.1_367640535_C_A', 'NC_057797.1_776842743_A_G', 'NC_057795.1_383970956_G_A',
#                     'NC_057797.1_725456771_T_C', 'NC_057794.1_245870738_T_C', 'NC_057796.1_22406463_G_C',
#                     'NC_057798.1_142433913_C_T', 'NC_057795.1_594638836_T_G', 'NC_057794.1_59894735_C_T',
#                     'NC_057797.1_29704897_T_G', 'NC_057795.1_636669724_A_C', 'NC_057798.1_7426102_C_T',
#                     'NC_057795.1_599743536_C_T', 'NC_057795.1_599461234_T_C', 'NC_057802.1_563064506_G_C',
#                     'NC_057794.1_46694769_C_T', 'NC_057800.1_732137368_T_C', 'NC_057800.1_10043422_A_T',
#                     'NC_057801.1_165366658_A_G', 'NC_057798.1_752474511_A_G']
#
# crop_brown_grad = ['NC_057797.1_13234957_A_C', 'NC_057796.1_304241564_T_C', 'NC_057797.1_9171771_G_A',
#                    'NC_057798.1_753176654_C_T', 'NC_057802.1_553182224_G_C', 'NC_057798.1_811195497_A_G',
#                    'NC_057797.1_694523621_C_A', 'NC_057796.1_304241506_G_A', 'NC_057796.1_305900442_T_G',
#                    'NC_057797.1_9158310_A_G', 'NC_057801.1_756450913_T_A', 'NC_057801.1_12477398_C_G',
#                    'NC_057796.1_27639689_A_G', 'NC_057801.1_214091004_A_G', 'NC_057796.1_27639647_T_C',
#                    'NC_057796.1_255085454_T_C', 'NC_057798.1_702328067_G_A', 'NC_057798.1_98236710_T_G',
#                    'NC_057796.1_307965282_G_T', 'NC_057797.1_694697835_G_T', 'NC_057796.1_398837919_C_G',
#                    'NC_057801.1_824083368_G_A', 'NC_057795.1_588246527_G_C', 'NC_057795.1_686682819_G_C',
#                    'NC_057797.1_694698079_T_C', 'NC_057801.1_794615048_A_G', 'NC_057799.1_615951454_T_C',
#                    'NC_057801.1_34427962_C_T', 'NC_057801.1_372328967_G_A', 'NC_057799.1_8321833_A_T',
#                    'NC_057802.1_553182237_G_A', 'NC_057797.1_671555569_G_C', 'NC_057801.1_614024000_G_C',
#                    'NC_057797.1_16787287_C_G', 'NC_057796.1_255085393_A_T', 'NC_057803.1_38700849_G_A',
#                    'NC_057802.1_624104_A_G', 'NC_057801.1_217007657_A_C', 'NC_057801.1_83381054_A_T',
#                    'NC_057795.1_589139981_T_C', 'NC_057800.1_8368421_G_A', 'NC_057802.1_413392816_G_A',
#                    'NC_057797.1_730298253_C_A', 'NC_057801.1_250750415_T_C', 'NC_057796.1_412732720_T_C',
#                    'NC_057795.1_686682777_C_T', 'NC_057796.1_307965317_G_C', 'NC_057794.1_485230890_C_T',
#                    'NC_057798.1_98236735_G_A', 'NC_057801.1_419053524_G_A']
#
# crop_yellow_grad = ['NC_057798.1_608949069_T_C', 'NC_057794.1_65502599_G_T', 'NC_057801.1_620652034_C_T',
#                     'NC_057798.1_609871637_A_G', 'NC_057799.1_608700361_G_A', 'NC_057796.1_295026318_C_T',
#                     'NC_057796.1_297803003_C_T', 'NC_057798.1_541165534_T_C', 'NC_057795.1_600072180_G_A',
#                     'NC_057796.1_295026324_C_G', 'NC_057802.1_413392110_C_T', 'NC_057801.1_620652074_G_A',
#                     'NC_057796.1_232848824_T_C', 'NC_057801.1_12397731_T_C', 'NC_057801.1_17996663_G_A',
#                     'NC_057802.1_382214610_G_A', 'NC_057794.1_61327479_T_C', 'NC_057801.1_82259596_T_G',
#                     'NC_057798.1_736653096_G_A', 'NC_057799.1_75744012_G_C', 'NC_057801.1_12476113_G_A',
#                     'NC_057798.1_752474363_G_A', 'NC_057794.1_48914576_T_A', 'NC_057795.1_71295734_T_C',
#                     'NC_057795.1_71295653_C_T', 'NC_057795.1_670070685_G_A', 'NC_057799.1_2959129_G_A',
#                     'NC_057795.1_51477020_A_C', 'NC_057801.1_164669803_A_G', 'NC_057798.1_540671709_G_A',
#                     'NC_057798.1_735606871_C_T', 'NC_057794.1_57389185_C_T', 'NC_057802.1_382214449_G_A',
#                     'NC_057799.1_615944126_G_T', 'NC_057796.1_50601227_C_T', 'NC_057801.1_82128337_A_G',
#                     'NC_057798.1_674451531_A_G', 'NC_057796.1_146297383_C_T', 'NC_057798.1_736582333_A_G',
#                     'NC_057795.1_191227009_T_C', 'NC_057799.1_48021727_C_T', 'NC_057797.1_786122359_C_T',
#                     'NC_057797.1_5358333_G_A', 'NC_057795.1_599461234_T_C', 'NC_057801.1_822973489_A_G',
#                     'NC_057799.1_557582543_G_T', 'NC_057794.1_41905540_A_G', 'NC_057799.1_635746095_T_G',
#                     'NC_057801.1_165205171_A_C', 'NC_057797.1_759445639_T_C']

# print(list(set(crop_yellow_grad) & set(crop_brown_grad)))
# print(list(set(crop_height_grad) & set(crop_yellow_grad)))
# print(list(set(crop_height_grad) & set(crop_brown_grad)))
# for i in range(0, 30):
#     crop_yellow_grad[i] = crop_yellow_grad[i].replace("_", "\_")
#     crop_brown_grad[i] = crop_brown_grad[i].replace("_", "\_")
#     crop_height_grad[i] = crop_height_grad[i].replace("_", "\_")
#
# for i in range(0, 30, 3):
#     print(f"${crop_height_grad[i]}$ & ${crop_height_grad[i + 1]}$ & ${crop_height_grad[i + 2]}$ \\\ \hline")
# print("#################")
# for i in range(0, 30, 3):
#     print(f"${crop_brown_grad[i]}$ & ${crop_brown_grad[i + 1]}$ & ${crop_brown_grad[i + 2]}$ \\\ \hline")
# print("#################")
# for i in range(0, 30, 3):
#     print(f"${crop_yellow_grad[i]}$ & ${crop_yellow_grad[i + 1]}$ & ${crop_yellow_grad[i + 2]}$ \\\ \hline")

# потом ScoreRAM
crop_height_score = ['NC_057797.1_771854938_A_G', 'NC_057797.1_716038008_C_T', 'NC_057796.1_459141313_G_C',
                     'NC_057797.1_713867972_C_T', 'NC_057801.1_250750415_T_C', 'NC_057797.1_714115096_G_A',
                     'NC_057799.1_79508903_A_T', 'NC_057799.1_79386833_A_G', 'NC_057797.1_706881785_G_A',
                     'NC_057795.1_678962006_C_G', 'NC_057803.1_58787211_A_G', 'NC_057801.1_378790448_A_G',
                     'NC_057797.1_713868991_G_T', 'NC_057796.1_9228188_T_G', 'NC_057795.1_380089360_A_G',
                     'NC_057794.1_17980126_G_A', 'NC_057798.1_217331887_G_A', 'NC_057797.1_749534485_T_C',
                     'NC_057796.1_81402317_G_A', 'NC_057794.1_91392131_G_A', 'NC_057795.1_6493872_G_T',
                     'NC_057794.1_17980242_T_A', 'NC_057801.1_2470409_A_T', 'NC_057801.1_372328967_G_A',
                     'NC_057797.1_764116145_A_C', 'NC_057798.1_216687375_C_G', 'NC_057797.1_771766207_C_T',
                     'NC_057798.1_782878661_T_A', 'NC_057797.1_713859498_G_A', 'NC_057797.1_716038298_C_G',
                     'NC_057798.1_782869557_G_A', 'NC_057799.1_78151584_A_G', 'NC_057796.1_82273528_G_C',
                     'NC_057801.1_791664763_T_C', 'NC_057797.1_716038013_C_T', 'NC_057801.1_845341651_C_G',
                     'NC_057794.1_588695922_T_G', 'NC_057797.1_773425898_T_C', 'NC_057802.1_501806884_A_C',
                     'NC_057797.1_717835426_C_A', 'NC_057799.1_31116815_T_A', 'NC_057795.1_505298970_A_C',
                     'NC_057794.1_23176267_G_C', 'NC_057798.1_100647286_A_G', 'NC_057796.1_255085454_T_C',
                     'NC_057796.1_458769505_T_C', 'NC_057796.1_81402261_T_G', 'NC_057795.1_626648000_G_A',
                     'NC_057794.1_588695950_G_A', 'NC_057794.1_588695835_C_T', 'NC_057796.1_260893724_G_T',
                     'NC_057798.1_571688650_T_C', 'NC_057794.1_7957538_A_G', 'NC_057798.1_103087760_T_C',
                     'NC_057801.1_10237260_C_G', 'NC_057795.1_337753627_C_G', 'NC_057801.1_252513491_T_G',
                     'NC_057797.1_428289939_C_T', 'NC_057798.1_19526385_T_C', 'NC_057795.1_18633033_G_C',
                     'NC_057799.1_125624319_G_C', 'NC_057797.1_615300_G_A', 'NC_057800.1_407330877_A_G',
                     'NC_057798.1_740429813_C_T', 'NC_057798.1_548647157_G_A', 'NC_057799.1_648812562_G_A',
                     'NC_057795.1_183247004_A_G', 'NC_057799.1_598617613_G_A', 'NC_057795.1_34374333_T_C',
                     'NC_057801.1_83381054_A_T', 'NC_057799.1_608700334_T_C', 'NC_057795.1_10008143_C_T',
                     'NC_057795.1_10107068_G_A', 'NC_057797.1_2920918_C_A', 'NC_057799.1_652761735_G_A',
                     'NC_057801.1_367264922_G_A', 'NC_057796.1_21404114_C_T', 'NC_057796.1_21206559_T_C',
                     'NC_057796.1_413066310_T_C', 'NC_057798.1_696464605_A_G', 'NC_057797.1_714538462_T_C',
                     'NC_057797.1_714801206_C_T', 'NC_057796.1_28507212_A_G', 'NC_057798.1_181301905_C_A',
                     'NC_057801.1_433151740_C_G', 'NC_057797.1_779130662_G_A', 'NC_057797.1_2433252_G_A',
                     'NC_057797.1_2433176_T_C', 'NC_057794.1_17980141_G_A', 'NC_057794.1_18288043_G_T',
                     'NC_057797.1_2818429_C_T', 'NC_057794.1_136099413_T_C', 'NC_057797.1_428289846_G_A',
                     'NC_057801.1_677731887_G_C', 'NC_057797.1_709529341_G_A', 'NC_057798.1_33314446_G_A']

crop_brown_score = ['NC_057801.1_758239019_C_T', 'NC_057795.1_34388352_A_G', 'NC_057801.1_792603733_G_A',
                    'NC_057796.1_482775353_A_G', 'NC_057796.1_482775184_C_G', 'NC_057796.1_17077819_A_C',
                    'NC_057802.1_579563396_C_T', 'NC_057798.1_757040101_A_T', 'NC_057799.1_645798885_T_C',
                    'NC_057798.1_757039626_C_G', 'NC_057798.1_695648153_A_C', 'NC_057795.1_598095552_G_A',
                    'NC_057800.1_616149343_C_G', 'NC_057795.1_254248020_T_C', 'NC_057798.1_695648085_G_A',
                    'NC_057802.1_552247369_T_C', 'NC_057799.1_147376611_A_G', 'NC_057795.1_46548657_G_A',
                    'NC_057802.1_579563423_T_C', 'NC_057801.1_113543415_T_G', 'NC_057798.1_714755056_G_T',
                    'NC_057802.1_589669423_T_C', 'NC_057796.1_482775343_G_A', 'NC_057799.1_594901214_C_T',
                    'NC_057794.1_481681952_A_T', 'NC_057798.1_691231172_G_A', 'NC_057801.1_705250417_C_T',
                    'NC_057799.1_2959267_C_A', 'NC_057795.1_315224571_T_C', 'NC_057798.1_15170502_A_G',
                    'NC_057803.1_140565032_C_T', 'NC_057799.1_595044022_A_G', 'NC_057795.1_40923945_G_A',
                    'NC_057800.1_35490312_C_T', 'NC_057800.1_643183809_A_G', 'NC_057796.1_458757_T_C',
                    'NC_057798.1_142428314_C_T', 'NC_057797.1_385389613_T_C', 'NC_057802.1_2592517_T_C',
                    'NC_057798.1_768737165_C_T', 'NC_057797.1_23432228_G_A', 'NC_057795.1_539706178_G_A',
                    'NC_057799.1_2959240_G_A', 'NC_057796.1_384808_C_T', 'NC_057795.1_684737982_C_A',
                    'NC_057799.1_85529322_G_A', 'NC_057797.1_385389005_T_A', 'NC_057800.1_719068972_A_G',
                    'NC_057798.1_741260422_A_G', 'NC_057795.1_23097244_A_G', 'NC_057796.1_475991746_C_G',
                    'NC_057796.1_476652559_C_T', 'NC_057798.1_52332178_T_C', 'NC_057798.1_740269205_A_T',
                    'NC_057800.1_29909097_C_A', 'NC_057799.1_594901257_G_A', 'NC_057797.1_582943_T_C',
                    'NC_057795.1_628644285_A_G', 'NC_057795.1_314901142_C_T', 'NC_057794.1_562004651_C_T',
                    'NC_057795.1_40923845_T_A', 'NC_057798.1_707527775_T_A', 'NC_057796.1_410860500_A_C',
                    'NC_057798.1_798036828_C_T', 'NC_057798.1_769237141_T_A', 'NC_057797.1_643940422_G_A',
                    'NC_057796.1_26795552_T_G', 'NC_057795.1_596115909_G_A', 'NC_057798.1_109139885_T_G',
                    'NC_057799.1_16963560_T_G', 'NC_057796.1_497161988_T_C', 'NC_057798.1_808531647_T_C',
                    'NC_057795.1_1675285_C_T', 'NC_057795.1_315224629_T_G', 'NC_057798.1_54187123_C_A',
                    'NC_057798.1_741232332_G_C', 'NC_057802.1_169622549_T_A', 'NC_057801.1_822113027_T_C',
                    'NC_057798.1_19112565_T_C', 'NC_057795.1_459108230_C_T', 'NC_057796.1_26802685_C_T',
                    'NC_057802.1_552275390_C_G', 'NC_057799.1_580130073_C_A', 'NC_057801.1_748228569_C_T',
                    'NC_057798.1_691231151_G_A', 'NC_057795.1_267900146_C_T', 'NC_057797.1_747396166_A_G',
                    'NC_057796.1_21206559_T_C', 'NC_057797.1_106572194_G_C', 'NC_057798.1_122460900_C_A',
                    'NC_057799.1_2833545_A_G', 'NC_057798.1_62027727_A_G', 'NC_057795.1_459397497_T_G',
                    'NC_057797.1_767010779_T_C', 'NC_057795.1_260557144_G_C', 'NC_057795.1_259130818_A_T']

crop_yellow_score = ['NC_057800.1_643183742_T_C', 'NC_057800.1_643183809_A_G', 'NC_057799.1_650297795_C_T',
                     'NC_057800.1_721490579_G_C', 'NC_057799.1_650274749_A_G', 'NC_057799.1_651429960_C_G',
                     'NC_057795.1_63940731_T_C', 'NC_057795.1_63993951_G_A', 'NC_057795.1_63940872_G_A',
                     'NC_057798.1_745116782_T_C', 'NC_057800.1_10043337_A_G', 'NC_057802.1_613722105_T_A',
                     'NC_057799.1_577315886_G_A', 'NC_057798.1_46408407_T_C', 'NC_057800.1_752937907_G_A',
                     'NC_057796.1_384808_C_T', 'NC_057798.1_796943577_C_T', 'NC_057802.1_613713317_T_C',
                     'NC_057801.1_843649062_C_T', 'NC_057803.1_79986504_G_A', 'NC_057800.1_721490573_C_T',
                     'NC_057802.1_20575062_C_T', 'NC_057796.1_22814569_A_G', 'NC_057800.1_29087454_C_T',
                     'NC_057796.1_22814526_T_C', 'NC_057794.1_389620362_A_T', 'NC_057794.1_499257611_A_G',
                     'NC_057794.1_375255632_T_G', 'NC_057795.1_101223400_A_C', 'NC_057794.1_543210171_C_G',
                     'NC_057799.1_653873678_G_A', 'NC_057798.1_742093620_T_G', 'NC_057797.1_31426927_T_C',
                     'NC_057797.1_31312062_T_C', 'NC_057798.1_39276851_C_T', 'NC_057799.1_577425838_G_C',
                     'NC_057797.1_780482719_T_C', 'NC_057798.1_707527775_T_A', 'NC_057796.1_22807639_T_A',
                     'NC_057794.1_390155410_A_C', 'NC_057800.1_29087188_C_G', 'NC_057796.1_115311363_A_G',
                     'NC_057798.1_796943707_G_A', 'NC_057795.1_53994106_G_C', 'NC_057798.1_745110651_C_T',
                     'NC_057799.1_634294266_T_C', 'NC_057797.1_718681635_C_T', 'NC_057798.1_46436525_C_T',
                     'NC_057800.1_720353693_C_T', 'NC_057800.1_720352186_A_G', 'NC_057800.1_10043345_C_A',
                     'NC_057798.1_749702178_G_A', 'NC_057795.1_64243914_G_A', 'NC_057799.1_604655662_T_C',
                     'NC_057795.1_66346392_G_A', 'NC_057796.1_466179244_A_G', 'NC_057798.1_2286301_G_A',
                     'NC_057794.1_464109702_G_C', 'NC_057799.1_11322168_A_G', 'NC_057796.1_114930761_G_C',
                     'NC_057797.1_71049089_T_C', 'NC_057800.1_98918258_T_C', 'NC_057795.1_323482362_G_C',
                     'NC_057794.1_400154487_T_G', 'NC_057797.1_31599223_T_C', 'NC_057797.1_5608219_G_C',
                     'NC_057797.1_5608168_T_G', 'NC_057797.1_5992484_A_G', 'NC_057799.1_513556598_A_C',
                     'NC_057798.1_749702184_T_C', 'NC_057795.1_27877106_C_G', 'NC_057799.1_622524713_G_A',
                     'NC_057798.1_751143141_A_T', 'NC_057801.1_843936446_G_A', 'NC_057797.1_747599397_T_C',
                     'NC_057794.1_299006559_G_A', 'NC_057802.1_521278368_T_C', 'NC_057797.1_153233734_G_A',
                     'NC_057799.1_90582495_C_T', 'NC_057799.1_651430296_T_C', 'NC_057798.1_567332823_T_G',
                     'NC_057802.1_375114877_C_T', 'NC_057798.1_763324449_A_G', 'NC_057794.1_498991064_C_G',
                     'NC_057794.1_499257598_C_A', 'NC_057794.1_516028266_A_G', 'NC_057796.1_458757_T_C',
                     'NC_057798.1_727563492_T_C', 'NC_057799.1_634294230_A_G', 'NC_057800.1_34524792_G_A',
                     'NC_057798.1_749703100_C_A', 'NC_057802.1_995137_T_C', 'NC_057797.1_29759478_T_C',
                     'NC_057795.1_46040339_T_C', 'NC_057796.1_491183243_G_A', 'NC_057795.1_45310856_T_C']

print(list(set(crop_yellow_score) & set(crop_brown_score)))
print(list(set(crop_height_score) & set(crop_yellow_score)))
print(list(set(crop_height_score) & set(crop_brown_score)))
for i in range(0, 30):
    crop_yellow_score[i] = crop_yellow_score[i].replace("_", "\_")
    crop_brown_score[i] = crop_brown_score[i].replace("_", "\_")
    crop_height_score[i] = crop_height_score[i].replace("_", "\_")

for i in range(0, 30, 3):
    print(f"${crop_height_score[i]}$ & ${crop_height_score[i + 1]}$ & ${crop_height_score[i + 2]}$ \\\ \hline")
print("#################")
for i in range(0, 30, 3):
    print(f"${crop_brown_score[i]}$ & ${crop_brown_score[i + 1]}$ & ${crop_brown_score[i + 2]}$ \\\ \hline")
print("#################")
for i in range(0, 30, 3):
    print(f"${crop_yellow_score[i]}$ & ${crop_yellow_score[i + 1]}$ & ${crop_yellow_score[i + 2]}$ \\\ \hline")

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

model_crop_height = "../combo_model/checkpoints/model_checkpoints/rand_cv_trained_model_iter_0.h5"
model_crop_brown = "../combo_model/checkpoints/model_checkpoints/model_saves/crop_brown/grid_cv_trained_model_iter0.h5"
# model_crop_yellow = "../combo_model/checkpoints/model_checkpoints/model_saves/crop_yellow/crop_yellow4/grid_cv_trained_model_iter4.h5"

model_height = tf.keras.models.load_model(model_crop_height,
                                          custom_objects={'custom_loss_mae': ComboModelTuner.custom_loss_mae},
                                          compile=False)
model_brown = tf.keras.models.load_model(model_crop_brown,
                                         custom_objects={'custom_loss_mae': ComboModelTuner.custom_loss_mae},
                                         compile=False)
model_yellow = tf.keras.models.load_model(model_crop_yellow,
                                          custom_objects={'custom_loss_mae': ComboModelTuner.custom_loss_mae},
                                          compile=False)
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

labels_height = df_wheat[["Высота.растений..см"]].to_numpy()
labels_crop = df_wheat[["Урожайность.зерна..г."]].to_numpy()
labels_brown = df_wheat[["Бурая.ржавчина..."]].to_numpy()
labels_yellow = df_wheat[["Желтая.ржавчина..."]].to_numpy()

# нормализация данных
labels_height = rank_based_transform(labels_height.flatten()[~np.isnan(labels_height.flatten())])
labels_crop = rank_based_transform(labels_crop.flatten()[~np.isnan(labels_crop.flatten())])
labels_brown = rank_based_transform(labels_brown.flatten()[~np.isnan(labels_brown.flatten())])
labels_yellow = rank_based_transform(labels_yellow.flatten()[~np.isnan(labels_yellow.flatten())])

# ошибка урожайность высота
pred_crop_height = model_height.predict([images, total_features])
err_crop = np.abs(labels_crop.flatten() - pred_crop_height[labels_crop.index][:, 0].flatten())
err_height = np.abs(labels_height.flatten() - pred_crop_height[labels_height.index][:, 1].flatten())
print(err_crop.mean())
print(err_height.mean())
print("###########")

# ошибка урожайность бурая ржавчина
pred_crop_brown = model_brown.predict([images, total_features])
err_crop = np.abs(labels_crop.flatten() - pred_crop_brown[labels_crop.index][:, 0].flatten())
err_brown = np.abs(labels_brown.flatten() - pred_crop_brown[labels_brown.index][:, 1].flatten())
print(err_crop.mean())
print(err_brown.mean())
print("###########")

# ошибка урожайность желтая ржавчина
pred_crop_yellow = model_yellow.predict([images, total_features])
err_crop = np.abs(labels_crop.flatten() - pred_crop_yellow[labels_crop.index][:, 0].flatten())
err_yellow = np.abs(labels_yellow.flatten() - pred_crop_yellow[labels_yellow.index][:, 1].flatten())
print(err_crop.mean())
print(err_yellow.mean())

print(np.corrcoef(labels_crop.flatten(), pred_crop_height[labels_crop.index][:, 0].flatten()))
print(np.corrcoef(labels_height.flatten(), pred_crop_height[labels_height.index][:, 1].flatten()))
print(np.corrcoef(labels_crop.flatten(), pred_crop_brown[labels_crop.index][:, 0].flatten()))
print(np.corrcoef(labels_brown.flatten(), pred_crop_brown[labels_brown.index][:, 1].flatten()))
print(np.corrcoef(labels_crop.flatten(), pred_crop_yellow[labels_crop.index][:, 0].flatten()))
print(np.corrcoef(labels_yellow.flatten(), pred_crop_yellow[labels_yellow.index][:, 1].flatten()))


