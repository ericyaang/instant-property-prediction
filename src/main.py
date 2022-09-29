from src import data, utils, pipeline, train
from pathlib import Path
from config import config
import pandas as pd
import typer

app = typer.Typer()


@app.command()
def generate_splits():
    df_ = data.load_raw_data()
    df_ = data.process_pandas(df_)
    df_ = data.winsorizer(df_)
    train, valid, test = data.split_data(df_)
    return (train, valid, test)


@app.command()
def prepare_splits(train: pd.DataFrame, valid: pd.DataFrame):
    params_fp = Path(config.CONFIG_DIR, "params.json")
    params = utils.load_dict(filepath=params_fp)
    X_train, y_train = train.drop("valor", axis=1), train.valor
    X_valid, y_valid = valid.drop("valor", axis=1), valid.valor
    X_train, X_valid = data.prepare_cv(X_train, X_valid)
    X_train, X_valid = data.impute_data(X_train, X_valid)
    X_train = X_train[params["selected_cols"]]
    X_valid = X_valid[params["selected_cols"]]
    train = pd.concat([X_train, y_train], axis=1)
    valid = pd.concat([X_valid, y_valid], axis=1)
    return (train, valid)


@app.command()
def train_models(experiment_name, run_name, df):
    lgb_params = utils.load_dict(filepath=(Path(config.CONFIG_DIR, "lgbm_params.json")))
    preprocessor = pipeline.get_peprocessor()
    lgbm_median = pipeline.get_lgbm_pipeline(preprocessor, alpha=0.5, param=lgb_params)
    lgbm_lower = pipeline.get_lgbm_pipeline(preprocessor, alpha=0.25, param=lgb_params)
    lgbm_upper = pipeline.get_lgbm_pipeline(preprocessor, alpha=0.75, param=lgb_params)
    pipe_dict = {"Lower": lgbm_lower, "Median": lgbm_median, "Upper": lgbm_upper}
    train.train_cv(
        experiment_name=experiment_name,
        run_name=run_name,
        pipe_dict=pipe_dict,
        df=df,
        params=lgb_params,
        n_splits=5,
    )
    return pipe_dict
