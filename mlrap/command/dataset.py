import click
import numpy as np
import pandas as pd

from click import option
from mlrap.main.deal_data import load_data
from mlrap.main.optimization import model_import


@click.group()
def dataset():
    """
    Deal dataset.
    """


@dataset.command()
@option("-f", "--filename", type=click.Path(exists=True), default='DATA.csv',
        help="Dataset name.", show_default=True)
@option("-save", "save", flag_value="save",
        help="If set, your column name will be output in txt format.")
@option("-n", "--name", default="COLNAMEs",
        help="After setting save parameters, it is useful to control your txt file name.")
def getcols(filename, save, name):
    """
    Obtain data column names.
    """
    df = load_data(filename)
    click.echo("\n" + "----------"*5)
    click.echo(f"The columns in {filename} is:")
    cols = list(df.columns)
    for col in cols:
        click.echo("\t" + col)
    
    click.echo(f"Show {len(df.columns)} column names.")
    if save:
        name = name + ".txt"
        with open(name, "w") as file:
            for col in cols:
                file.write(col + "\n")
        click.echo(f"\nThe column name information has been saved to {name}")


@dataset.command()
@option("-f", "--filename", type=click.Path(exists=True), default='DATA.csv',
        help="Dataset name.", show_default=True)
@option("-delete", "delete", flag_value="delete",
        help="Switch to delete mode.")
@option("-c", "--cols", default="formula,target",
        help="Data columns that need to be retained.")
@option("-save", "save", flag_value="save",
        help="If set, the code will create a new file with name 'retain filename'.")
def retain(filename, cols, delete, save):
    """
    Manually filter data columns.
    """
    df = load_data(filename)
    cols = cols.split(",")
    is_subset = all(col_name in df.columns for col_name in cols)
    if not is_subset:
        mismatch = [col for col in cols if col not in df.columns]
        raise ValueError(
            f"The data column {mismatch} is not in the data file {filename}."
        )
    if delete:
        df = df.drop(cols, axis=1)
    else:
        df = df[cols]
    if save:
        filename = "retain " + filename

    load_data(filename, df)
    click.echo("\n" + "----------"*5)
    click.echo(f"The following data columns are retained:")
    cols = list(df.columns)
    for col in cols:
        click.echo("\t" + col)

    click.echo(f"File was save in {filename}")


@dataset.command()
@option("-f", "--filename", type=click.Path(exists=True), default='SELECT.csv',
        help="Dataset name.", show_default=True)
@option("-r", "--ratio", default=[0.8, 0.2, 0], show_default=True, nargs=3,
        help="Set the proportion of each subset in the dataset.")
@option("-stratified", "stratified", flag_value="stratified",
        help="If set, stratified sampling will be applied.")
@option("-rs", "--random-state", default=42, show_default=True, 
        help="Set the random state seed used for shuffling the dataset.")
def split(filename, ratio, stratified, random_state):
    """
    Split dataset to train_set, validation_set and test_set (optional).
    """
    from sklearn.model_selection import train_test_split
    from mlrap.main.deal_data import stratified_split
    
    df = load_data(filename)
    num_samples = len(df)
    ratio = np.array(ratio)
    if sum(ratio) <= 1: 
        ratio = ratio * num_samples
        mid = ratio.astype(int)
        ratio = np.array([ratio[0]+sum(ratio - mid), mid[1], mid[2]])

    ratio = ratio.astype(int)
    if sum(ratio) > num_samples or not np.all(ratio >= 0):
        raise ValueError(
            f"Check if the --ratio {ratio} you provided is legal {num_samples}."
        )
    elif sum(ratio) < num_samples:
        click.echo(
            "\nWARNNING: The sum of the provided subset samples is less\n"
            "          than the total number of samples, and the excess\n"
            "          data will be allocated to the training set."
        )

    split_func = stratified_split if stratified else train_test_split
    if ratio[2] != 0:
        df, test_set = split_func(df, test_size=ratio[2], random_state=random_state)
        load_data("DATATEST.csv", df=test_set)
        click.echo("The test set data has been filtered: DATATEST.csv")

    train_set, val_set = split_func(df, test_size=ratio[1], random_state=random_state)

    load_data("DATATRAIN.csv", df=train_set)
    load_data("DATAVAL.csv", df=val_set)
    click.echo("The train set data has been filtered: \t\tDATATRAIN.csv")
    click.echo("The validation set data has been filtered: \tDATAVAL.csv")


@dataset.command()
@option("-f", "--filename", type=click.Path(exists=True), default='DATA.csv',
        help="Dataset name.", show_default=True)
@option("-manual", "manual", flag_value="manual",
        help="Manually change the mode. If this parameter is set, please note"
             "that the --cols matches the dataset.")
@option("-c", "--cols", default="formula,target",
        help="New column names for all columns.")
def rename(filename, cols, manual):
    """
    Rename the data column.
    """
    from tabulate import tabulate
    from util import build_name

    df = load_data(filename)
    if manual:
        cols = cols.split(",")

    else:
        cols = build_name(df)

    if len(cols) != len(df.columns):
        raise ValueError(
            f"The number of column names provided does not match the dataset {len(df.columns)}."
        )

    old_name = df.columns
    df.columns = cols
    table_data = []
    for old, new in zip(old_name, cols):
        table_data.append([old, new])

    headers = ["old name", "new name"]
    table = tabulate(table_data, headers, tablefmt="grid", maxcolwidths=[30, 30])

    click.echo(table)
    load_data(filename, df=df)


@dataset.command()
@option("-f", "--filename", type=click.Path(exists=True), default='DATA.csv',
        help="Dataset name.", show_default=True)
@option("-t", "--target", default="target", show_default=True,
        help="Target attributes.")
@option("-th", "--threshold", default=0.75, show_default=True,
        help="The threshold for feature filtering (float or int).")
@option("-embedded", "embedded", flag_value="embedded",
        help="If set, mlrap will switch to embedded feature filtering.")
@option("-sfs", "sfs", flag_value="sfs", 
        help="Embedded feature selection defaults to using model feature "
             "importance ranking, and you can also switch to Sequential "
             "Feature Selector(SFS) mode with this command to automatically "
             "find the optimal feature subset.")
@option("-mt", "--model-type", default="sklearn.ensemble.RandomForestRegressor",
        help="Select a model for embedded filtering.", show_default=True)
@option("-save", "save", flag_value="save",
        help="If set, the code will overwrite the original file with retained.")
def select(filename, model_type, embedded, threshold, target, sfs, save):
    """
    Feature descriptor selection can be performed using corr-select(default) 
    or embedded methods. Feature selection based on correlation uses a float 
    threshold, while embedded feature selection can use both float and int 
    thresholds. Corr-select retains one feature when their correlation exceeds 
    the threshold. Embedded selection keeps features with importance higher 
    than the threshold or ranked above an integer threshold.
    """
    import joblib
    from mlrap.main.deal_data import embedded_select, correlation_select, identify_df
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS

    df = load_data(filename)
    df, other_df = identify_df(df)
    y, X = df[target], df.drop(target, axis=1)
    if embedded:
        # import model
        model = model_import(model_type)
        if sfs:
            sfs = SFS(model,
                      k_features='best',
                      forward=False,
                      floating=False,
                      scoring='r2',
                      cv=10,
                      n_jobs=-1)

            sfs.fit(X, y)
            df = df[list(sfs.k_feature_names_)]
            joblib.dump(sfs, "SFS.joblib")
            click.echo("\nSFS file has been saved to SFS.joblib.\n")
        else:
            threshold = threshold if threshold < 1 else int(threshold)
            df = embedded_select(model, X, y, threshold)
    else:
        df = correlation_select(X, y, threshold)

    if not save:
        filename = "SELECT.csv"

    corr_df = pd.concat([df, y], axis=1, join='inner')
    corr = corr_df.corr(numeric_only=True)
    df = pd.concat([other_df, corr_df], axis=1, join='inner')
    load_data(filename, df)
    load_data("CORR.csv", corr)
    click.echo("The data file has been saved.\n"
               f"\tFiltering Data:\t{filename}\n\tCorrelation:\tCORR.csv")


@dataset.command()
@option("-f", "--filename", type=click.Path(exists=True), default='DATA.csv',
        help="Dataset name.", show_default=True)
@option("-m", "--mode", default="write", type=click.Choice(["write", "export"]),
        help="Select the structure file operation mode.", show_default=True)
@option("-fp", "--file-path", default="./structure", show_default=True, 
        help="Specify the file path for writing when in write mode and for "
             "exporting when in export mode.")
@option("-ic", "--id-col", default="formula", show_default=True, 
        help="Select the data column corresponding to the structure file name.")
@option("-sc", "--structure-col", default="structure", show_default=True, 
        help="Select the data column containing structural objects.")
@option("-save", "save", flag_value="save",
        help="If set, the new dataset will be saved as a new CSV file.")
def struct(filename, mode, file_path, id_col, structure_col, save):
    """
    Convert a structure file to a structure object, or a structure object to a structure file.
    """
    import sys
    import warnings
    from pymatgen.core.structure import Structure
    from mlrap.main.featurize import to_structfile, to_structure

    warnings.filterwarnings("ignore", category=UserWarning)
    mode_func = {
        "write": to_structure,
        "export": to_structfile
    }
    df = load_data(filename)
    
    if mode == "export":
        df[structure_col] = df[structure_col].apply(lambda x: Structure.from_str(x, fmt="cif"))

    df = mode_func[mode](df=df, 
                         file_path=file_path, 
                         structure_col=structure_col, 
                         material_col=id_col)
    if df.empty: 
        sys.exit()
    if save:
        filename = "struct" + filename
    df["structure"] = df["structure"].apply(lambda x: x.to(fmt="cif"))
    warnings.filterwarnings("default")
    load_data(filename, df)


@dataset.command()
@option("-sn", "--set_name", default="elastic_tensor_2015", show_default=True, 
        help="Select the dataset to load from Matminer."
             "(https://hackingmaterials.lbl.gov/matminer/dataset_summary.html)")
def getset(set_name):
    """
    Load dataset from material database -- matminer.
    """
    import os
    from matminer.datasets import load_dataset

    data_home = None
    if set_name in ["elastic_tensor_2015", "dielectric_constant"]:
        data_home = os.path.join(os.path.dirname(__file__), "..\example")

    df = load_dataset(set_name, data_home=data_home)
    click.echo(df.head())
    if "structure" in df.columns:
        df["structure"] = df["structure"].apply(lambda x: x.to(fmt="cif"))
    load_data(set_name+".csv", df=df)