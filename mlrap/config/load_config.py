import os


def load_config_path(config_name):
    """
    return congfig path
    """
    module_directory = os.path.dirname(__file__)
    source_file = os.path.join(module_directory, config_name + ".yaml")

    return source_file


def load_plot_style(style):
    """
    return path of plot style
    """
    module_directory = os.path.dirname(__file__)
    source_file = os.path.join(module_directory, style + ".mplstyle")

    return source_file