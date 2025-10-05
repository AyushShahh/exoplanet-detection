import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.impute import KNNImputer
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np
import joblib
import matplotlib.pyplot as plt


def clean_data(path='kepler_exoplanet_data.csv'):
    df = pd.read_csv(path)

    drop_cols = ["rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition"]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # error_cols = [col for col in df.columns if "_err1" in col or "_err2" in col]
    # print(error_cols)
    # df.drop(columns=error_cols, inplace=True)

    disposition_map = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
    df["koi_disposition"] = df["koi_disposition"].map(disposition_map)

    df = df.select_dtypes(include="number").astype("float64")

    df.drop(columns=["koi_score"], errors="ignore", inplace=True)

    # save updated csv
    df.to_csv('kepler_exoplanet_data_cleaned.csv', index=False)

def load_data(path='kepler_exoplanet_data_cleaned.csv'):
    df = pd.read_csv(path)
    X, y = df.drop(columns=["koi_disposition"]), df["koi_disposition"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=23):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    imputer = KNNImputer(n_neighbors=7)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    return X_train, X_test, y_train, y_test

def hyper_param_tuning(X_train, y_train):
    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=23)

    def objective(params):
        model = xgb.XGBClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            min_child_weight=int(params['min_child_weight']),
            eval_metric='mlogloss',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        accuracy = np.mean(cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy', n_jobs=-1))
        return {'loss': -accuracy, 'status': STATUS_OK}

    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 500, 25),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'gamma': hp.uniform('gamma', 0, 2),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-6), np.log(1.5)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(0.05), np.log(3)),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1)
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=1000,
        trials=trials
    )
    return best

def save_best_params(best):
    with open("best_params.txt", "w") as f:
        for param, value in best.items():
            f.write(f"{param}={value},\n")

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

    print("Test set accuracy: {:.2f}%".format(acc * 100))
    print("Test set ROC AUC: {:.2f}".format(roc))

def get_best_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    best = hyper_param_tuning(X_train, y_train)
    # print("Best parameters:", best)
    save_best_params(best)

    best_model = xgb.XGBClassifier(
        n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),
        learning_rate=best['learning_rate'],
        gamma=best['gamma'],
        reg_alpha=best['reg_alpha'],
        reg_lambda=best['reg_lambda'],
        subsample=best['subsample'],
        colsample_bytree=best['colsample_bytree'],
        min_child_weight=int(best['min_child_weight']),
        eval_metric='mlogloss',
        early_stopping_rounds=10,
        n_jobs=-1
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]
    best_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    joblib.dump(best_model, 'xgb_best_model.pkl')
    axes = xgb.plot_importance(best_model)
    axes.figure.savefig('feature_importance.png')

    evaluate(best_model, X_test, y_test)
    return best_model


if __name__ == "__main__":
    clean_data()
    get_best_model()
