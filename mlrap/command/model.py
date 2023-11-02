import click
import joblib
import pandas as pd

from click import option
from mlrap.main.deal_data import load_data


@click.group()
def model():
    """
    Train a model.
    """


@model.command()
@option("-m", "--model-name", default="xgboost.XGBRegressor", show_default=True,
        help="Select the model algorithm to use.")
@option("-p", "--params", default=None, type=str,
        help="Set model hyperparameters. "
             "Example: 'n_estimators:10,int;learning_rate:0.05,float'")
def getmodel(model_name, params):
    """
    Select a model and set model parameters.
    """
    from mlrap.main.optimization import model_import

    type_dict = {
        "int": int,
        "float": float,
        "str": str
    }
    param_dict = {}
    if params is not None:
        p_list = params.split(";")
        for param in p_list:
            p_key, p_value_type = param.split(":")
            p_value, p_type = p_value_type.split(",")
            param_dict[p_key] = type_dict[p_type](p_value)
    choose_model = model_import(model_name, **param_dict)
    model_and_features = {"model": choose_model, 
                          "feature": {}}
    joblib.dump(model_and_features, 'model.pkl')
    click.echo("Your model was save in model.pkl.\n")
    click.echo(choose_model)


@model.command()
@option("-mf", "--model-file", type=click.Path(exists=True), default="model.pkl",
        help="Select the model that needs to update parameters.", show_default=True)
@option("-p", "--params", default=None, type=str,
        help="Set the hyperparameters that need to be updated."
             "Example: 'n_estimators:10,int;learning_rate:0.05,float'")
def updata(params, model_file):
    """
    Update model parameters.
    """
    type_dict = {
        "int": int,
        "float": float,
        "str": str
    }
    param_dict = {}
    if params is not None:
        p_list = params.split(";")
        for param in p_list:
            p_key, p_value_type = param.split(":")
            p_value, p_type = p_value_type.split(",")
            param_dict[p_key] = type_dict[p_type](p_value)
    load_model = joblib.load(model_file)
    load_model = {
        "model": load_model["model"].set_params(**param_dict),
        "feature": load_model["feature"]
    }
    joblib.dump(load_model, 'model.pkl')
    click.echo("Model parameters have been updated: ")
    click.echo(load_model["model"])


@model.command()
@option("-mf", "--model-file", type=click.Path(exists=True), default="model.pkl",
        help="Select the model that needs to fit.", show_default=True)
@option("-ft", "--file-train", type=click.Path(exists=True), default="DATATRAIN.csv",
        help="Training set file name.", show_default=True)
@option("-fv", "--file-val", type=click.Path(exists=True), default="DATAVAL.csv",
        help="Validation set file name.", show_default=True)
@option("-t", "--target", default="target", show_default=True,
        help="Target attributes.")
def train(file_train, file_val, model_file, target):
    """
    Training models based on datasets and model.pkl.
    """
    from mlrap.main.deal_data import identify_df
    from sklearn.metrics import r2_score

    data_train = load_data(file_train)
    data_val = load_data(file_val)
    load_model = joblib.load(model_file)

    data_train, data_train_other = identify_df(data_train)
    data_val, data_val_other = identify_df(data_val)
    train_X, train_y = data_train.drop(target, axis=1), data_train[target]
    val_X, val_y = data_val.drop(target, axis=1), data_val[target]

    load_model["model"].fit(train_X, train_y)
    load_model["feature"] = list(train_X.columns)

    pre_train = pd.DataFrame({"prediction": load_model["model"].predict(train_X)})
    pre_val = pd.DataFrame({"prediction": load_model["model"].predict(val_X)})

    score_train = r2_score(train_y, pre_train)
    score_val = r2_score(val_y, pre_val)
    click.echo("\nModel training is completed, and the R\u00b2 score is:")
    click.echo(f"\tTrain: \t\t{score_train:.4f}\n\tValidation: \t{score_val:.4f}")
    pre_train = pd.concat([data_train_other, train_y, pre_train], axis=1)
    pre_val = pd.concat([data_val_other, val_y, pre_val], axis=1)
    load_data("PRETRAIN.csv", pre_train)
    load_data("PREVAL.csv", pre_val)
    joblib.dump(load_model, 'model.pkl')


@model.command()
@option("-mf", "--model-file", type=click.Path(exists=True), default="model.pkl",
        help="Select the model that needs to be evaluated.", show_default=True)
@option("-f", "--filename", type=click.Path(exists=True), default='SELECT.csv',
        help="Dataset name.", show_default=True)
@option("-t", "--target", default="target", help="Target attributes.", show_default=True)
@option("-stratified", "stratified", flag_value="stratified",
        help="If set, stratified cross validation will be used.")
@option("-kf", "--k-fold", default=5, show_default=True,
        help="Set K-Fold for cross validation.")
@option("-sc", "--score", default="r2", show_default=True,
        help="Set up cross validation evaluation methods.")
@option("-shuffle", "shuffle", flag_value="shuffle",
        help="Whether to shuffle the dataset.")
@option("-rs", "--random-state", default=42, show_default=True,
        help="Set a random number seed when shuffling the dataset.")
def crossval(filename, model_file, target, stratified, k_fold, score, shuffle, random_state):
    """
    Cross validation of models based on datasets.
    """
    import numpy as np
    from mlrap.main.cross_validation import stratified_cv
    from mlrap.main.deal_data import identify_df
    from sklearn.model_selection import KFold, cross_val_score

    df = load_data(filename)
    load_model = joblib.load(model_file)
    df, df_other = identify_df(df)
    df_X, df_y = df.drop(target, axis=1), df[target]
    click.echo("Your model:")
    click.echo(load_model["model"])
    args = {}
    if shuffle: args = {"shuffle": True, "random_state": random_state}
    if stratified:
        score_list = stratified_cv(
            model=load_model["model"],
            X=df_X,
            y=df_y,
            scoring=score,
            cv=k_fold,
            **args
        )
    else:
        k_folds=KFold(n_splits=k_fold, **args)
        score_list = cross_val_score(estimator=load_model["model"], scoring=score,
                                     X=df_X, y=df_y, cv=k_folds)
    click.echo("\nThe score of the model is:")
    click.echo(f"\t{'_'*15}")
    click.echo("\tKFOLD\tSCORE")
    for i in range(len(score_list)):
        click.echo(f"\t{i+1}\t{score_list[i]:.4f}")
    click.echo(f"\n\tThe mean score is: {np.mean(score_list):.4f}")
    if shuffle: click.echo(f"\tRandom State: {random_state}")