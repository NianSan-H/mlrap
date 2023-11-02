import os
import click
import joblib
import pandas as pd
import yaml
from click import option


@click.group()
def run():
    """
    Global run base config file.
    """


@run.command()
def getconfig():
    """
    Prepare configuration files for global training.
    """
    from util import copy_config
    from mlrap.config.load_config import load_config_path

    source_file = load_config_path("run-CONFIG")
    copy_file = "run-CONFIG.yaml"
    copy_config(source_file, copy_file)

    click.echo("The default configuration file obtained is: run-CONFIG.yaml")


@run.command()
@option("-cf", "--config-file", default="run-CONFIG.yaml", show_default=True, 
        help="Select your global train config file.")
def train(config_file):
    """
    Global training, including feature generation and selection, hyperparameter tuning, model training 
    and saving, and image plotting.
    """
    import json
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from matplotlib import pyplot as plt
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
    from util import descriptor, build_name
    from mlrap.config.load_config import load_plot_style
    from mlrap.config.load_config import load_config_path
    from mlrap.main.optimization import model_import, optimization
    from mlrap.main.deal_data import load_data, correlation_select, identify_df, embedded_select
    from mlrap.main.draw import save_plot, plot_corr, sample_distribution, plot_model, plot_feature_importance

    if not os.path.exists(config_file): 
        config_file = load_config_path("run-CONFIG")

    # load config and dataset
    with open(config_file, "r", encoding='utf-8') as optim_config:
        config = yaml.safe_load(optim_config)
    if config["data path"] == None:
        config["data path"] = "DATA.csv"
    if config["save path"] == None:
        config["save path"] = "./"
    plot_path = os.path.join(config["save path"], "image")
    style_path = load_plot_style("current")
    os.makedirs(plot_path, exist_ok=True)

    df = load_data(config["data path"])
    model = model_import(config["choose model"])

    # general descriptor
    descriptor_list = config["descriptor type"].split(",")
    for i in descriptor_list:
        df = descriptor(df, i, "formula")
    df.columns = build_name(df)

    # select feature
    df, other_df = identify_df(df)
    y, X = df["target"], df.drop("target", axis=1)
    X = correlation_select(X, y, config["threshold"])
    X = embedded_select(model, X, y, 25)

    sfs = SFS(model,
              k_features='best',
              forward=False,
              floating=False,
              scoring='r2',
              cv=10,
              n_jobs=-1)

    sfs.fit(X, y)
    X = X[list(sfs.k_feature_names_)]
    df = pd.concat([X, y], axis=1, join="inner")
    # draw
    save_plot(plot_func=plot_corr, 
              style_path=style_path, 
              save_path=plot_path, 
              fig_name="corr",
              save_type="tiff",
              dataset=df,
              show=False)
    
    plt.style.use(style_path)
    plot_sfs(sfs.get_metric_dict(), kind='std_dev')
    plt.grid()
    plt.savefig(os.path.join(plot_path, "SFS.tiff"), dpi=600)
    plt.close()
    click.echo(f"figure was save in: {os.path.join(plot_path, 'SFS.tiff')}")

    # save data
    load_data(os.path.join(config["save path"], "CORR.csv"), df.corr(numeric_only=True))
    joblib.dump(sfs, "SFS.joblib")

    # split test set
    df = pd.concat([other_df, df], axis=1, join="inner")
    set_dict = {}
    if config["test set"] != None:
        df, df_test = train_test_split(df, test_size=0.1, shuffle=True, 
                                       random_state=config["random state"])
        set_dict["Test Set"] = df_test["target"]
        load_data(os.path.join(config["save path"], "DATATEST.csv"), df_test)

    # search params
    df, other_df = identify_df(df)
    optimization(
        X=df.drop("target", axis=1), 
        y=df["target"], 
        model=model, 
        config_file=config_file
    )

    # fit best
    with open("bayes_log.json", 'r') as file:
        params = json.load(file)

    sorted_data = sorted(params, key=lambda x: x['target'], reverse=True)
    best_param = sorted_data[0]
    train_set, val_set = train_test_split(df, test_size=1/best_param["K-Fold"], 
                                    random_state=best_param["random state"])
    
    load_data(os.path.join(config["save path"], "DATATRAIN.csv"), df=train_set)
    load_data(os.path.join(config["save path"], "DATAVAL.csv"), df=val_set)
    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    data_train, data_train_other = identify_df(train_set)
    data_val, data_val_other = identify_df(val_set)
    train_X, train_y = data_train.drop("target", axis=1), data_train["target"]
    val_X, val_y = data_val.drop("target", axis=1), data_val["target"]

    load_model = {
        "model": model.set_params(**best_param["params"]),
        "feature": list(train_X.columns)
    }
    load_model["model"].fit(train_X, train_y)

    # prediction
    pre_train = pd.DataFrame({"prediction": load_model["model"].predict(train_X)})
    pre_val = pd.DataFrame({"prediction": load_model["model"].predict(val_X)})

    score_train = r2_score(train_y, pre_train)
    score_val = r2_score(val_y, pre_val)
    click.echo("\nThe best parameters have been applied, and the R\u00b2 score is:")
    click.echo(f"\tTrain: \t\t{score_train:.4f}\n\tValidation: \t{score_val:.4f}")

    if config["test set"] != None:
        # if split test set
        test_set = load_data("DATATEST.csv")
        data_test, data_test_other = identify_df(test_set)
        test_X, test_y = data_test.drop("target", axis=1), data_test["target"]
        pre_test = pd.DataFrame({"prediction": load_model["model"].predict(test_X)})
        score_test = r2_score(test_y, pre_test)
        click.echo(f"\tTest: \t\t{score_test:.4f}")
        test_pre = pd.concat([data_test_other, test_y, pre_test], axis=1)
        load_data("PRETEST.csv", test_pre)

    train_pre = pd.concat([data_train_other, train_y, pre_train], axis=1)
    val_pre = pd.concat([data_val_other, val_y, pre_val], axis=1)
    load_data(os.path.join(config["save path"], "PRETRAIN.csv"), train_pre)
    load_data(os.path.join(config["save path"], "PREVAL.csv"), val_pre)
    click.echo("\nThe best parameters is: ")
    for key, value in best_param["params"].items():
        click.echo(f"\t{key}: \t{value:.4f}" if isinstance(value, float) else f"\t{key}: \t{value}")
    joblib.dump(load_model, 'model.pkl')

    # plot model performance and set distribution
    data_pre = {}
    data_dis = {}
    if config["test set"] != None:
        data_pre["test"] = (test_pre["target"].T.values.tolist(), test_pre["prediction"].T.values.tolist())
        data_dis["test"] = test_pre["target"]

    data_pre["train"] = (train_pre["target"].T.values.tolist(), train_pre["prediction"].T.values.tolist())
    data_pre["validation"] = (val_pre["target"].T.values.tolist(), val_pre["prediction"].T.values.tolist())
    data_dis["train"] = train_pre["target"]
    data_dis["validation"] = val_pre["target"]
    save_plot(plot_func=plot_model,
             style_path=style_path,
             save_path=plot_path,
             fig_name="pre-true",
             save_type="tiff",
             dict_data=data_pre,
             show=False)
    
    save_plot(plot_func=sample_distribution,
              save_path=plot_path,
              style_path=style_path,
              fig_name="distribution",
              save_type="tiff",
              samples=data_dis,
              show=False)

    # plot feature importance
    save_plot(plot_func=plot_feature_importance,
              save_path=plot_path,
              style_path=style_path,
              fig_name="feature-importance",
              save_type="tiff",
              importance=load_model["model"].feature_importances_,
              feature_names=load_model["feature"],
              show_feature=10,
              show=False)