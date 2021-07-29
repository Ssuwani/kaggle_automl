from kaggle_automl import api


def make_submission(submission_df, predictions):
    from .train import target
    from .dataset import competition_title
    submission_file = f'submission_{competition_title}.csv'
    submission_df[target] = predictions['Label']
    submission_df.to_csv(submission_file, index=False)
    api.competition_submit(submission_file, "submission by kaggle_automl", competition_title)
