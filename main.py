import json
import logging
import warnings
from pprint import pprint

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# set up logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# env vars
CAT_VARS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
       'month', 'day_of_week', 'poutcome']
NUM_VARS = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate',
       'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

PRED_CUTOFFS = [0.15, 0.25, 0.30, 0.35, 0.40, 0.50]


def preprocess_data(df, categorical_cols, numeric_cols):
    # recode y as 0 or 1
    df["y"] = (df["y"] == "yes").astype("int")

    # recode pdays so 999 = -1
    df["pdays"] = df["pdays"].replace(999, -1)

    # impute unknown categoricals
    imp = SimpleImputer(missing_values="unknown", strategy='most_frequent')
    df[categorical_cols] = imp.fit_transform(df[categorical_cols])

    # recode cat vars using data dict
    for var in categorical_cols:
        dummy = pd.get_dummies(df[var], prefix=var, drop_first=True)
        df = pd.concat([df, dummy], axis=1).drop(var, axis=1)

    # split into train / test sets
    train, test = train_test_split(df, test_size=0.3, random_state=1990)

    X = train.loc[:, train.columns != "y"]
    y = train["y"]

    X_test = test.loc[:, test.columns != "y"]
    y_test = test["y"]

    # rescale num vars (using train only)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X[numeric_cols])
    X[numeric_cols] = scaler.transform(X[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X, y, X_test, y_test


def main():
    # load up dataset
    data = pd.read_csv("bank-additional/bank-additional-full.csv", delimiter=";")
    data.pop("duration")
    X, y, X_test, y_test = preprocess_data(data, categorical_cols=CAT_VARS, numeric_cols=NUM_VARS)

    # load params from model exploration
    with open("gradient_boost_params.json") as f:
        gb_params = json.load(f)["params"]

    with open("neural_net_params.json") as f:
        nn_params = json.load(f)["params"]

    # fit models
    gb = GradientBoostingClassifier(**gb_params)
    gb.fit(X, y)
    nn = MLPClassifier(**nn_params)
    nn.fit(X, y)

    # predict on dataset
    naive_stat = (y_test == 0).sum() / y_test.count()
    print(f"Naive rule (baseline accuracy) = {naive_stat}")
    cutoff_scores = {}
    for cutoff in PRED_CUTOFFS:
        preds = {}
        gb_prob = gb.predict_proba(X_test)[:,1]
        preds["gradient_boost"] = (gb_prob > cutoff).astype("int")
        nn_prob = nn.predict_proba(X_test)[:, 1]
        preds["neural_net"] = (nn_prob > cutoff).astype("int")

        # calculate scores for each model
        scores = {}
        for name, y_hat in preds.items():
            metrics = {}
            # scores
            metrics["accuracy"] = accuracy_score(y_test, y_hat)
            metrics["precision"] = precision_score(y_test, y_hat)
            metrics["recall"] = recall_score(y_test, y_hat)
            metrics["f1"] = f1_score(y_test, y_hat)
            metrics["passes_naive_test"] = metrics["accuracy"] > naive_stat
            metrics["roc_auc"] = roc_auc_score(y_test, y_hat)
            # confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_hat, labels=[0, 1]).ravel()
            metrics["true_negatives"] = tn
            metrics["false_positives"] = fp
            metrics["false_negatives"] = fn
            metrics["true_positives"] = tp

            scores[name] = metrics
            cutoff_scores[cutoff] = scores

    pprint(cutoff_scores)


if __name__ == "__main__":
    main()