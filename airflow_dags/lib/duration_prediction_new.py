#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


def read_dataframe(year, month, colour):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{colour}_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    print(f"Read {len(df)} rows")

    if colour == 'yellow':
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    elif colour == 'green':
        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    else:
        print(f"Unknown colour: {colour}")

    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    if colour == 'green':
        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_X(df, dv=None, colour='green'):
    if colour == 'green':
        categorical = ['PU_DO']
        numerical = ['trip_distance']
        dicts = df[categorical + numerical].to_dict(orient='records')
    elif colour == 'yellow':
        categorical = ['PULocationID', 'DOLocationID']
        numerical = ['trip_distance']
        dicts = df[categorical + numerical].to_dict(orient='records')
    else:
        print(f"Unknown colour: {colour}")
        return None, None

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
        print(f"Feature matrix size: {X.shape}")
    else:
        X = dv.transform(dicts)
        print(f"Feature matrix size: {X.shape}")

    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id

def train_linear_model(X_train, y_train, dv):
    with mlflow.start_run() as run:
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        rmse = root_mean_squared_error(y_train, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("intercept_", float(lr.intercept_))

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b",  artifact_path="preprocessor")

        mlflow.sklearn.log_model(lr, artifact_path="linear_model")

        return run.info.run_id


def run(year, month, colour):
    df_train = read_dataframe(year=year, month=month, colour=colour)

    if colour == 'green':
        next_year = year if month < 12 else year + 1
        next_month = month + 1 if month < 12 else 1
        df_val = read_dataframe(year=next_year, month=next_month, colour=colour)

        X_train, dv = create_X(df_train, dv=None, colour=colour)
        X_val, _ = create_X(df_val, dv, colour=colour)

        target = 'duration'
        y_train = df_train[target].values
        y_val = df_val[target].values

    elif colour == 'yellow':
        X_train, dv = create_X(df_train, dv=None, colour=colour)
        target = 'duration'
        y_train = df_train[target].values


    if colour == 'green':
        run_id = train_model(X_train, y_train, X_val, y_val, dv)
    elif colour == 'yellow':
        run_id = train_linear_model(X_train, y_train, dv)
    else:
        print(f"Unknown colour: {colour}")
        return None

    print(f"Model training completed. Run ID: {run_id}")
    return run_id


if __name__ == '__main__':
    # use argparse to read year and month from command line
    import argparse
    parser = argparse.ArgumentParser(description='Train a model for duration prediction')
    parser.add_argument('--year', type=int, required=True, help='Year of the data')
    parser.add_argument('--month', type=int, required=True, help='Month of the data')
    parser.add_argument('--colour', type=str, default="green", required=False, help='colour of the taxi')
    args = parser.parse_args()

    run(args.year, args.month, args.colour)
