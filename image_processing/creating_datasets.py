import pandas as pd
from data_processing import vigna_to_table, delete_redundant_columns
import numpy as np
import sys
import allel
import zarr

# # creating vigna dataset for AIO generating
#
# df_gen = pd.read_csv("../datasets/vigna/full_tab_gwas.csv")
# df_meteo = pd.read_csv("../datasets/vigna/meteo_vigna.csv")
# df_date = pd.read_csv("../datasets/vigna/pheno_tab_muc.csv")
#
# # df_gen_with_date = merge_columns_to_df(df_gen, df_date, ["year", "geo_id", "doy"])
# # df_gen_with_date.to_csv("../tables_for_AIO/full_tab_gwas_with_date.csv", index=False)
#
# # creating total dataframe
# df_gen = delete_redundant_columns(df_gen, ["resp"])
# total_df = vigna_to_table(df_genes=df_gen, df_meteo=df_meteo, df_resp=df_date, days_before=5, days_after=20,
#                           colnames=["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "GWETTOP", "ALLSKY_SFC_LW_DWN", "dl"])
# total_df.to_csv("../tables_for_AIO/total_df_for_aio_vigna.csv", index=False)

# creating vigna dataset for AIO generating

# ffl = allel.read_vcf("../datasets/chickpea/VIR.VF.SNP.vcf.gz")
# print(ffl['samples'])

# read xlsx
# chickpea_grant_2016 = pd.read_excel("../datasets/chickpea/_БАЗА_НУТ_ГРАНТ_2016.xlsx")
# print(chickpea_grant_2016.head(5))
#
# meteo_chickpea = pd.read_csv("../datasets/chickpea/Meteo_chickpea_data.csv")
# print(meteo_chickpea.head(5))

# file 1
vir_1_path = "../datasets/chickpea/VIR.VF.SNP.vcf.gz"
# vir_1 = allel.read_vcf(vir_1_path)
# print(vir_1.keys())
#
# for key in vir_1.keys():
#     print(f"{key}:")
#     print(vir_1[key])
#     print(vir_1[key].shape)
#     print("############\n")
#
# genotypes = allel.GenotypeArray(vir_1['calldata/GT'])
# print(genotypes.is_het())
# print(genotypes.count_het())

zarr_path = vir_1_path.replace(".vcf.gz", ".zarr")

# пока в формате zarr все будет лежать, до тех пор, пока обрабатываю данные
# allel.vcf_to_zarr(vir_1_path, zarr_path, fields='*', log=sys.stdout, overwrite=True)

zarr_callset = zarr.open_group(zarr_path, mode='r')
# print(zarr_callset.tree(expand=True))
print(zarr_callset['samples'])
# total_df = pd.read_csv("../tables_for_AIO/total_df_for_aio_vigna.csv")
# print(total_df.head(3))
# files 2 and 3
# vir_2 = allel.read_vcf("../datasets/chickpea/VIR.VF.VIR.SNP.miCount3.vcf.gz")
# vir_3 = allel.read_vcf("../datasets/chickpea/VIR.VF.VIR.SNP.vcf.gz")
# print(len(vir_2))
# print(len(vir_3))