import click

from mlrap.command.plot import plot
from mlrap.command.dataset import dataset
from mlrap.command.apply import apply
from mlrap.command.optim import optim
from mlrap.command.model import model
from mlrap.command.featurize import featurize


@click.group()
def subrun():
    """
    Run step by step.
    """

subrun.add_command(dataset)
subrun.add_command(model)
subrun.add_command(optim)
subrun.add_command(apply)
subrun.add_command(plot)
subrun.add_command(featurize)