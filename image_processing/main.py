import pandas as pd
import image_creation as imcr
import data_processing as dp

import argparse

if __name__ == "__main__":

    # argument parser for path input/output paths parameters
    argsparser = argparse.ArgumentParser(description='Parser of parameters for AIO creation for plant datasets')

    # adding parameters

    argsparser.add_argument('path_wheat_df', type=str, help='Path to folder with total wheat dataset table')
    argsparser.add_argument('path_meteo_wheat', type=str, help='Path to folder with meteo table with data for wheat')
    argsparser.add_argument('path_model_AIO_wheat', type=str, help='Path to folder for AIO objects for wheat in format for user')
    argsparser.add_argument('path_control_AIO_wheat', type=str, help='Path to folder for AIO objects for wheat in format for user')

    # parsing arguments
    args = argsparser.parse_args()

    # creation of AIO set for wheat
    total_df_wheat = pd.read_csv(args.path_vigna_df)
    # save response values
    y = total_df_wheat["resp"]
    # remove redundant information from model
    total_df_wheat = dp.delete_redundant_columns(total_df_wheat, ["seqid", "doy", "geo_id", "year", "resp"])
    # print(total_df.shape)
    meteo_df_wheat = pd.read_csv(args.path_meteo_wheat)
    # print(meteo_df.columns)
    modified_meteo_df_wheat = dp.delete_redundant_columns(meteo_df_wheat, ["year", "doy", "geo_id"])
    meteo_names_wheat = modified_meteo_df_wheat.columns
    snp_colnames_wheat, weather_colnames_wheat = imcr.snp_weather_colnames(total_df_wheat.columns, meteo_names_wheat)
    images = imcr.create_image_dataset(total_df_wheat, 200, 200, snp_colnames_wheat, weather_colnames_wheat,
                                       args.path_model_AIO_wheat, args.path_control_AIO_wheat, multithread=False)
