from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from lightgbm import LGBMRegressor
import numpy as np
import scipy as sp
from src import utils
from pathlib import Path
from config import config


def get_peprocessor():
    params = utils.load_dict(filepath=(Path(config.CONFIG_DIR, "params.json")))
    numeric_trans = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=True, interaction_only=False),
        StandardScaler(),
    )
    return make_column_transformer(
        (numeric_trans, params["cont_cols"] + params["int_cols"] + params["ord_cols"]),
        (
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False),
            params["cat_cols"],
        ),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def get_lgbm_pipeline(preprocessor, alpha, param):
    params = utils.load_dict(filepath=(Path(config.CONFIG_DIR, "params.json")))
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=LGBMRegressor(
                        objective="quantile",
                        alpha=alpha,
                        **param,
                        **{"random_state": params["seed"]}
                    ),
                    func=(np.log10),
                    inverse_func=(sp.special.exp10),
                ),
            ),
        ]
    )
