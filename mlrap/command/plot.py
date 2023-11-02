import os
import click
import joblib
import numpy as np
import pandas as pd

from click import option
from mlrap.config.load_config import load_plot_style
from mlrap.main.deal_data import load_data


@click.group()
def plot():
    """
    Draw images for model training data.
    """


@plot.command()
@option("-pp", "--plot-path", default=".\image", show_default=True,
        help="Image save location.")
@option("-distribution", "distribution", flag_value="distribution",
        help="Plotting Data Set Sample Distribution.")
@option("-corr", "corr", flag_value="corr",
        help="Plotting Pearson Correlation Coefficient.")
@option("-sfs", "sfs", flag_value="sfs",
        help="Plotting Sequential Feature Select Graphics.")
@option("-pretrue", "pretrue", flag_value="pretrue",
        help="Plotting Model Performance Graphics.")
@option("-importance", "importance", flag_value="importance",
        help="Plotting Model Feature Importance.")
@option("-fd", "--file-data", default="SELECT.csv", show_default=True,
        help="Dataset for plotting correlation.")
@option("-fp", "--file-pre", show_default=True,
        default="PRETRAIN.csv,train;PREVAL.csv,validation", 
        help="Dataset for plotting sample distribution or model performance. (dataset,set-type;...)")
@option("-mf", "--model-file", default="model.pkl", show_default=True,
        help="Model for plotting feature importance.")
@option("-sfsf", "--sfs-file", default="SFS.joblib", show_default=True,
        help="Choose SFS object.")
@option("-sf", "--show-feature", default=10, show_default=True,
        help="Number of features to display when plotting feature importance.")
@option("-px", "--pre-x", default="True Data", show_default=True,
        help="X-axis of model performance.")
@option("-py", "--pre-y", default="Predict Data", show_default=True,
        help="Y-axis of model performance.")
@option("-dx", "--distri-x", default="Range Of Sample", show_default=True,
        help="X-axis of sample distribution.")
@option("-dy", "--distri-y", default="Numbers", show_default=True,
        help="Y-axis of sample distribution.")
@option("-t", "--target", default="target", show_default=True,
        help="Target attribute.")
@option("-st", "--save-type", default="tiff", show_default=True,
        help="Image save file type.")
@option("-sp", "--style-path", default=None,
        help="Plot style path.")
def general(plot_path, corr, pretrue, file_data, file_pre, save_type, 
            importance, model_file, distribution, target, style_path,
            sfs, sfs_file, pre_x, pre_y, distri_x, distri_y, show_feature):
    """
    Draw all basic images.
    """
    from matplotlib import pyplot as plt
    from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
    from mlrap.main.deal_data import identify_df
    from mlrap.main.draw import plot_corr, plot_model, save_plot
    from mlrap.main.draw import plot_feature_importance, sample_distribution

    os.makedirs(plot_path, exist_ok=True)
    if corr:
        df = load_data(file_data)
        df, _ = identify_df(df)
        style_path = load_plot_style("current") if style_path is None else style_path
        save_plot(plot_func=plot_corr, 
                  style_path=style_path, 
                  save_path=plot_path, 
                  fig_name="corr",
                  save_type=save_type,
                  dataset=df,
                  show=False)
        
    if sfs:
        sfs_data = joblib.load(sfs_file)
        style_path = load_plot_style("current") if style_path is None else style_path
        save_path = os.path.join(plot_path, "SFS."+save_type)
        plt.style.use(style_path)
        plot_sfs(sfs_data.get_metric_dict(), kind='std_dev')
        plt.grid()
        plt.savefig(save_path, dpi=600)
        click.echo(f"figure was save in: {save_path}")

    if pretrue:
        set_all = file_pre.split(";")
        data = {}
        for one_set in set_all:
            set_path, set_name = one_set.split(",")
            df = load_data(set_path)
            data[set_name] = (df[target].T.values.tolist(), df["prediction"].T.values.tolist())
        style_path = load_plot_style("current") if style_path is None else style_path
        save_plot(plot_func=plot_model,
                  style_path=style_path,
                  save_path=plot_path,
                  fig_name="pre-true",
                  save_type=save_type,
                  dict_data=data,
                  xlabel=pre_x,
                  ylabel=pre_y,
                  show=False)

    if distribution:
        set_all = file_pre.split(";")
        data = {}
        for one_set in set_all:
            set_path, set_name = one_set.split(",")
            df = load_data(set_path)
            data[set_name] = df[target]
        style_path = load_plot_style("current") if style_path is None else style_path
        save_plot(plot_func=sample_distribution,
                  save_path=plot_path,
                  style_path=style_path,
                  fig_name="distribution",
                  save_type=save_type,
                  samples=data,
                  xlabel=distri_x,
                  ylabel=distri_y,
                  show=False)

    if importance:
        load_model = joblib.load(model_file)
        style_path = load_plot_style("current") if style_path is None else style_path
        save_plot(plot_func=plot_feature_importance,
                  save_path=plot_path,
                  style_path=style_path,
                  fig_name="feature-importance",
                  save_type=save_type,
                  importance=load_model["model"].feature_importances_,
                  feature_names=load_model["feature"],
                  show_feature=show_feature,
                  show=False)


@plot.command()
@option("-mf", "--model-file", type=click.Path(exists=True), default="model.pkl",
        help="Select the model file for shap analysis.", show_default=True)
@option("-ft", "--file-train", default="DATATRAIN.csv", show_default=True,
        help="Select the dataset used for the model fit.")
@option("-fa", "--file-analysis", default="DATAVAL.csv", show_default=True,
        help="Select the dataset for SHAP analysis.")
def getshap(model_file, file_train, file_analysis):
    """
    Generate a SHAP instance file for plotting.
    """
    import shap
    import warnings

    load_model = joblib.load(model_file)
    f_train = load_data(file_train)[load_model["feature"]]
    f_analysis = load_data(file_analysis)
    f_analysis_other = f_analysis.drop(load_model["feature"], axis=1)
    f_analysis = f_analysis[load_model["feature"]]
    warnings.filterwarnings("ignore", category=UserWarning)
    explainer = shap.Explainer(load_model["model"], f_train)
    warnings.filterwarnings("default")
    shap_values = explainer(f_analysis)

    shap_data = {
        "shap_values": shap_values,
        "data": pd.concat([f_analysis, f_analysis_other], axis=1)
    }

    joblib.dump(shap_data, "SHAP.joblib")
    click.echo("The SHAP instance has been saved to the file: SHAP.joblib")


@plot.command()
@option("-sf", "--show-feature", default=10, show_default=True,
        help="Number of features displayed in the plot.")
@option("-se", "--shap-explainer", default="SHAP.joblib", show_default=True,
        help="Shap Explainer instance generated by getshap.")
@option("-st", "--save-type", default="tiff", show_default=True,
        help="Image save file type.")
@option("-pp", "--plot-path", default=".\image", show_default=True,
        help="Image save location.")
@option("-sp", "--style-path", default=None,
        help="Plot style path.")
def shapglobal(show_feature, shap_explainer, save_type, style_path, plot_path):
    """
    Global SHAP image plotting (Including beeswarm and Global Bar).
    """
    import shap
    from mlrap.main.draw import save_plot

    os.makedirs(plot_path, exist_ok=True)
    shap_data = joblib.load(shap_explainer)
    style_path = load_plot_style("current") if style_path is None else style_path

    save_plot(plot_func=shap.plots.beeswarm,
              save_path=plot_path,
              style_path=style_path,
              fig_name="beeswarm",
              save_type=save_type,
              shap_values=shap_data["shap_values"],
              max_display=show_feature,
              show=False)
    save_plot(plot_func=shap.plots.bar,
              save_path=plot_path,
              style_path=style_path,
              fig_name="globalbar",
              save_type=save_type,
              shap_values=shap_data["shap_values"],
              max_display=show_feature,
              show=False)



@plot.command()
@option("-sf", "--show-feature", default=10, show_default=True,
        help="Number of features displayed in the plot.")
@option("-se", "--shap-explainer", default="SHAP.joblib", show_default=True,
        help="Shap Explainer instance generated by getshap.")
@option("-ic", "--id-cols", default="formula", show_default=True,
        help="Dataset column names corresponding to sample IDs.")
@option("-cs", "--choose-sample", default=None,
        help="Select the samples that need to be displayed separately.")
@option("-st", "--save-type", default="tiff", show_default=True,
        help="Image save file type.")
@option("-pp", "--plot-path", default=".\image", show_default=True,
        help="Image save location.")
@option("-sp", "--style-path", default=None,
        help="Plot style path.")
def shapsample(show_feature, shap_explainer, choose_sample, save_type, style_path, plot_path, id_cols):
    """
    Examining the shapley values for a specific sample.
    """
    import shap
    from matplotlib import pyplot as plt

    os.makedirs(plot_path, exist_ok=True)
    shap_data = joblib.load(shap_explainer)
    index = np.where(shap_data["data"][id_cols] == choose_sample)[0][0]
    style_path = load_plot_style("shap") if style_path is None else style_path

    plt.style.use(style_path)
    shap.plots.waterfall(shap_data["shap_values"][index], max_display=show_feature, show=False)
    plt.tight_layout()
    plot_path = os.path.join(plot_path, choose_sample+" waterfall."+save_type)
    plt.savefig(plot_path, dpi=600)
    click.echo(f"figure was save in: {plot_path}")


@plot.command()
@option("-cf", "--choose-feature", default=None,
        help="Select the feature that need to be displayed separately.")
@option("-se", "--shap-explainer", default="SHAP.joblib", show_default=True,
        help="Shap Explainer instance generated by getshap.")
@option("-st", "--save-type", default="tiff", show_default=True,
        help="Image save file type.")
@option("-pp", "--plot-path", default=".\image", show_default=True,
        help="Image save location.")
@option("-sp", "--style-path", default=None,
        help="Plot style path.")
def shapfea(choose_feature, shap_explainer, save_type, style_path, plot_path):
    """
    Examining the shapley values for a specific feature.
    """
    import shap
    from mlrap.main.draw import save_plot

    os.makedirs(plot_path, exist_ok=True)
    shap_data = joblib.load(shap_explainer)
    style_path = load_plot_style("current") if style_path is None else style_path

    save_plot(plot_func=shap.plots.scatter,
              save_path=plot_path,
              style_path=style_path,
              fig_name="distribution " + choose_feature,
              save_type=save_type,
              shap_values=shap_data["shap_values"][:, choose_feature],
              color=shap_data["shap_values"],
              show=False)