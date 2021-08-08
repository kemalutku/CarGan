import os

import pandas as pd
from shutil import copyfile


def extract_single_mm(df_dir, mm_name, target_dir, extract_as='jpg'):
    df = pd.read_csv(df_dir, index_col=0)
    mm_df = df[df['label'] == mm_name]
    if extract_as == 'csv':
        pass
    elif extract_as == 'jpg':
        target_dir = os.path.join(target_dir, mm_name)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        counter = 0
        for _, row in mm_df.iterrows():
            if counter % 500 == 0:
                print(counter, '/', len(mm_df))
            counter += 1
            file_path = row['path']
            if not os.path.isfile(file_path):
                print("File not found:", file_path)
                continue

            target_path = os.path.join(target_dir, os.path.basename(file_path))
            copyfile(file_path, target_path)
    elif extract_as == 'jpgCrop':
        target_dir = os.path.join(target_dir, mm_name)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        counter = 0
        for _, row in mm_df.iterrows():
            if counter % 500 == 0:
                print(counter, '/', len(mm_df))
            counter += 1
            file_path = row['path']
            if not os.path.isfile(file_path):
                print("File not found:", file_path)
                continue

        target_path = os.path.join(target_dir, os.path.basename(file_path))

if __name__ == '__main__':
    dataframe_dir = r"D:\Dataset\MarkaModel\MarkaModelMerged.csv"
    output_dir = r"D:\Dataset\GanExperimentSets"
    make_model_name = "Volkswagen_Golf"
    extract_single_mm(dataframe_dir, make_model_name, output_dir)
