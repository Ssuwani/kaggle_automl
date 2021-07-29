from pycaret.classification import *

target = None


def train_classification(train_df, test_df):
    train_cols = train_df.columns
    test_cols = test_df.columns
    global target
    target = list((set(train_cols) | set(test_cols)) - (set(train_cols) & set(test_cols)))[0]

    clf = setup(data=train_df, target=target, silent=True)
    best_3 = compare_models(sort='AUC', n_select=3)
    blended = blend_models(estimator_list=best_3, fold=5, method='soft')
    pred_holdout = predict_model(blended)
    final_model = finalize_model(blended)
    predictions = predict_model(final_model, data=test_df)
    return predictions
