import click
from mlrap.command.run import run
from mlrap.command.subrun import subrun


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    """
    Machine learning regression analyse packages\n
    \b
    \t\t███╗   ███╗██╗     ██████╗  █████╗ ██████╗ 
    \t\t████╗ ████║██║     ██╔══██╗██╔══██╗██╔══██╗
    \t\t██╔████╔██║██║     ██████╔╝███████║██████╔╝
    \t\t██║╚██╔╝██║██║     ██╔══██╗██╔══██║██╔═══╝ 
    \t\t██║ ╚═╝ ██║███████╗██║  ██║██║  ██║██║     
    \t\t╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     
    """


cli.add_command(run)
cli.add_command(subrun)