import click
import joblib
import pandas as pd

from click import option
from mlrap.main.deal_data import load_data


@click.group()
def optim():
    """
    Perform Bayesian hyperparametric optimization on the model.
    """


@optim.command()
def getconfig():
    """
    Obtain hyperparameter optimization configuration file.
    """
    from mlrap.util import copy_config
    from mlrap.config.load_config import load_config_path

    source_file = load_config_path("optim-CONFIG")
    copy_file = "optim-CONFIG.yaml"
    copy_config(source_file, copy_file)

    click.echo("The default configuration file obtained is: optim-CONFIG.yaml")


@optim.command()
@option("-f", "--filename", type=click.Path(exists=True), default='SELECT.csv',
        help="Dataset name.", show_default=True)
@option("-cf", "--config-file", default="optim-CONFIG.yaml", show_default=True, 
        help="Select your hyperparameter search config file.")
@option("-mf", "--model-file", type=click.Path(exists=True), default="model.pkl",
        help="Select the model that needs to be search params.", show_default=True)
@option("-t", "--target", default="target", help="Target attributes.", show_default=True)
def search(filename, model_file, config_file, target):
    """
    Search for hyperparameters based on the search configuration file.
    """
    import os
    from mlrap.main.deal_data import identify_df
    from mlrap.main.optimization import optimization
    from mlrap.config.load_config import load_config_path

    df = load_data(filename)
    load_model = joblib.load(model_file)
    df, _ = identify_df(df)

    if not os.path.exists(config_file): 
        config_file = load_config_path("optim-CONFIG")

    optimization(
        X=df.drop(target, axis=1), 
        y=df[target], 
        model=load_model["model"], 
        config_file=config_file
    )


@optim.command()
@option("-f", "--filename", type=click.Path(exists=True), default='SELECT.csv',
        help="Dataset name.", show_default=True)
@option("-lf", "--log-file", default="bayes_log.json",
        help="Select the search log file.", show_default=True)
@option("-mf", "--model-file", type=click.Path(exists=True), default="model.pkl",
        help="The model file that requires the use of optimal parameters.", show_default=True)
@option("-t", "--target", default="target", help="Target attributes.", show_default=True)
def fitbest(filename, log_file, model_file, target):
    """
    Automatically select the best hyperparameters obtained from the search log and 
    train the model.
    """
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from mlrap.main.deal_data import identify_df

    with open(log_file, 'r') as file:
        params = json.load(file)

    sorted_data = sorted(params, key=lambda x: x['target'], reverse=True)
    best_param = sorted_data[0]

    df = load_data(filename)
    train_set, val_set = train_test_split(df, test_size=1/best_param["K-Fold"], 
                                    random_state=best_param["random state"])
    load_data("DATATRAIN.csv", df=train_set)
    load_data("DATAVAL.csv", df=val_set)
    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    data_train, data_train_other = identify_df(train_set)
    data_val, data_val_other = identify_df(val_set)
    train_X, train_y = data_train.drop(target, axis=1), data_train[target]
    val_X, val_y = data_val.drop(target, axis=1), data_val[target]

    load_model = joblib.load(model_file)
    load_model = {
        "model": load_model["model"].set_params(**best_param["params"]),
        "feature": list(train_X.columns)
    }
    load_model["model"].fit(train_X, train_y)

    pre_train = pd.DataFrame({"prediction": load_model["model"].predict(train_X)})
    pre_val = pd.DataFrame({"prediction": load_model["model"].predict(val_X)})

    score_train = r2_score(train_y, pre_train)
    score_val = r2_score(val_y, pre_val)
    click.echo("\nThe best parameters have been applied, and the score is:")
    click.echo(f"\tTrain: \t\t{score_train:.4f}\n\tValidation: \t{score_val:.4f}")
    train_pre = pd.concat([data_train_other, train_y, pre_train], axis=1)
    val_pre = pd.concat([data_val_other, val_y, pre_val], axis=1)
    load_data("PRETRAIN.csv", train_pre)
    load_data("PREVAL.csv", val_pre)
    click.echo("\nThe best parameters is: ")
    for key, value in best_param["params"].items():
        click.echo(f"\t{key}: \t{value:.4f}" if isinstance(value, float) else f"\t{key}: \t{value}")
    joblib.dump(load_model, 'model.pkl')