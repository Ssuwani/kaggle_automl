from kaggle_automl.dataset import download_data, read_files
from kaggle_automl.train import train_classification
from kaggle_automl.submission import make_submission

if __name__ == '__main__':

    competition_title = 'titanic'
    print(f"{competition_title} data downloading...")
    data_folder = download_data(competition_title)
    print("Data download complete!!")

    print("read files...")
    files = read_files(data_folder)
    print(f"files : {[file for file in files]}!!")
    if files:
        train_df = files[0]
        test_df = files[1]
        submission_df = files[2]

        predictions = train_classification(train_df, test_df)
        make_submission(submission_df, predictions)
        print("submission Complete")
