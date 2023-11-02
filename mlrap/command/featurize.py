import click

from click import option
from mlrap.main.deal_data import load_data


@click.command()
@option("-f", "--filename", type=click.Path(exists=True), default='DATA.csv',
        help="Dataset name.", show_default=True)
@option("-c", "--col-id", default="formula", show_default=True,
        help="Data column that is about to be featurize.")
@option(
    "-ft",
    "--fea-type", 
    default="magpie", show_default=True,
    type=click.Choice(
        ["xenonpy", "oliynyk", "jarvis", "magpie",  "mat2vec", "onehot", "random_200", 
         "deml", "matminer", "matscholar_el", "megnet_el", "structure"]
    ),
    help="Select a type of descriptor."
)
@option("-save", "save", flag_value="save",
        help="If set, the new dataset will be saved as a new CSV file.")
def featurize(filename, fea_type, col_id, save):
    """
    Featurize the dataset. 
    """
    import warnings
    from util import descriptor
    
    warnings.filterwarnings("ignore", category=UserWarning)
    df = load_data(filename)
    df = descriptor(df, fea_type, col_id)
    if save:
        filename = "FEATURIZE" + filename
    warnings.filterwarnings("default")
    df = load_data(filename, df)
    click.echo(f"The descriptor has been generated and file {filename} has been updated.")