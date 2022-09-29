from sklearn.model_selection import KFold, cross_validate
import pandas as pd
import mlflow


def cv_eval(pipeline, X, y, n_splits: int):
    cv = KFold(n_splits=n_splits)
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=(
            "r2",
            "neg_root_mean_squared_error",
            "neg_median_absolute_error",
            "neg_mean_absolute_error",
            "neg_mean_absolute_percentage_error",
        ),
        n_jobs=(-1),
        return_train_score=True,
    )
    rmse = -scores["test_neg_root_mean_squared_error"]
    mae = -scores["test_neg_mean_absolute_error"]
    mape = -scores["test_neg_mean_absolute_percentage_error"]
    r2 = scores["test_r2"]
    print(f"RMSE: R$ {rmse.mean():,.2f} ± R$ {rmse.std():,.2f}")
    print(f"MAE: R$ {mae.mean():,.2f} ± R$ {mae.std():,.2f}")
    print(f"MAPE: {mape.mean() * 100:,.2f}% ± {mape.std() * 100:,.2f}%")
    print(f"R2: {r2.mean() * 100:,.2f}% ± {r2.std() * 100:,.2f}% \n")
    return (rmse, mae, mape, r2)


def train_cv(
    experiment_name: str,
    run_name: str,
    pipe_dict: dict,
    df: pd.DataFrame,
    params: dict,
    n_splits: int,
):
    X, y = df.drop("valor", axis=1), df.valor
    for pipe_name, pipe_info in pipe_dict.items():
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            print(f"> Model {pipe_name}:")
            mlflow.lightgbm.autolog()
            mlflow.set_tag("Range", f"{pipe_name}")
            rmse, mae, mape, r2 = cv_eval(pipe_info, X, y, n_splits=n_splits)
            mlflow.log_metric("rmse", rmse.mean())
            mlflow.log_metric("mae", mae.mean())
            mlflow.log_metric("mape", mape.mean())
            mlflow.log_metric("r2", r2.mean())
