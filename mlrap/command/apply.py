import click
import joblib
import numpy as np
import pandas as pd

from click import option
from mlrap.main.deal_data import load_data


@click.group()
def apply():
    """
    Utilize the model, including evaluation and prediction.
    """


@apply.command()
@option("-mf", "--model-file", type=click.Path(exists=True), default="model.pkl",
        help="Model file to be evaluated.", show_default=True)
@option("-ft", "--file-train", type=click.Path(exists=True), default="DATATRAIN.csv",
        help="Calculate the sample set for Shapley values (should be the training set filename).",
        show_default=True)
def eval(model_file, file_train):
    """
    Evaluate the trained model (returning feature importance and Shapley values).
    """
    import shap
    import warnings
    from tabulate import tabulate


    warnings.filterwarnings("ignore", category=UserWarning)
    load_model = joblib.load(model_file)

    df = load_data(file_train)[load_model["feature"]]
    explainer = shap.Explainer(load_model["model"], df)
    warnings.filterwarnings("default")
    shap_values = explainer(df)
    data = {"feature": load_model["feature"], 
            "feature importance": load_model["model"].feature_importances_,
            "shaply values": np.sum(abs(shap_values.values), axis=0) / len(df)}
    fea_imp = pd.DataFrame(data)
    load_data("EXPLANATION.csv", fea_imp)

    df_shap = fea_imp.sort_values(by='shaply values', ascending=False)
    df_imp = fea_imp.sort_values(by='feature importance', ascending=False)
    shap_remaining_sum = df_shap.iloc[5:]["shaply values"].sum()
    imp_remaining_sum = df_imp.iloc[5:]["feature importance"].sum()
    data = []
    headers = ["Rank", "SHAP Feature (SHAP Value)", "Importance Feature (Feature Importance)"]

    for i in range(5):
        shap_row = df_shap.iloc[i]
        imp_row = df_imp.iloc[i]
        rank = i + 1
        shap_feature = f"{shap_row['feature']} ({shap_row['shaply values']:.2f})"
        imp_feature = f"{imp_row['feature']} ({imp_row['feature importance']:.2f})"
        data.append([rank, shap_feature, imp_feature])

    data.append(["Total", f"Other {len(df_shap.iloc[5:])} Shap Sum ({shap_remaining_sum:.2f})", 
                 f"Other {len(df_imp.iloc[5:])} Importance Sum ({imp_remaining_sum:.2f})"])

    table = tabulate(data, headers, tablefmt="grid", maxcolwidths=[None, 30, 30])
    click.echo("\n")
    click.echo(table)


@apply.command()
@option("-mf", "--model-file", type=click.Path(exists=True), default="model.pkl",
        help="Model file for prediction.", show_default=True)
@option("-ft", "--file-test", type=click.Path(exists=True), default="DATATEST.csv",
        help="A dataset for predicting the target attribute.", show_default=True)
@option("-t", "--target", default="target", show_default=True, 
        help="Target attribute column.")
def predict(model_file, file_test, target):
    """
    Make predictions on the test set.
    """
    from sklearn.metrics import r2_score
    from mlrap.main.deal_data import identify_df

    load_model = joblib.load(model_file)
    df = load_data(file_test)
    df, df_other = identify_df(df)

    if target in df.columns:
        X, y = df[load_model["feature"]], df[target]
        df_other = pd.concat([df_other, y], axis=1)
        filename = "PRETEST.csv"
    else:
        X = df
        filename = "PRE-" + file_test

    pre_test = pd.DataFrame({"prediction": load_model["model"].predict(X)})
    if target in df.columns:
        score = r2_score(y, pre_test)
        click.echo(f"\nMLRAP have made predictions on {file_test}."
                f"\n\tScore: {score:.4f}")
        
    pre = pd.concat([df_other, pre_test], axis=1)
    load_data(filename, pre)

    click.echo(f"The prediction results have been saved in file {filename}.\n")