import os
import pandas as pd
from kaggle_automl import api
import zipfile


competition_title = None


def download_data(title):
    os.makedirs('data', exist_ok=True)
    global competition_title
    competition_title = title
    api.competition_download_files(title)
    data_folder = os.path.join('data', title)
    with zipfile.ZipFile(f'{title}.zip', 'r') as zip_ref:
        zip_ref.extractall(data_folder)
    os.remove(f'{title}.zip')
    return data_folder


def read_files(data_folder):
    files = os.listdir(data_folder)
    files = [f for f in files if not f.startswith('.')]

    if len(files) == 3 and 'train.csv' in files and 'test.csv' in files:
        print("----possible----\n")
        train_df = pd.read_csv(os.path.join(data_folder, 'train.csv'))
        test_df = pd.read_csv(os.path.join(data_folder, 'test.csv'))
        files.remove('train.csv')
        files.remove('test.csv')
        submission_df = pd.read_csv(os.path.join(data_folder, files[0]))
        return train_df, test_df, submission_df
    else:
        print("----impossible----\n")
        return False
