"""Save figures"""

import matplotlib.pyplot as plt
import cmc_pylog as pylog
import platform

if platform.system() == 'Linux':
    save_folder = "../../../Report/figures/"
else:
    save_folder = "..\\..\\..\\Report\\figures\\"


def save_figure(figure, name=None, **kwargs):
    """ Save figure """
    for extension in kwargs.pop("extensions", ["pdf"]):
        fig = figure.replace(" ", "_").replace(".", "dot")
        if name is None:
            name = "{}.{}".format(fig, extension)
        else:
            name = "{}.{}".format(name, extension)
        name = save_folder + name
        plt.figure(figure)
        plt.savefig(name, bbox_inches='tight')
        pylog.debug("Saving figure {}...".format(name))


def save_figures(**kwargs):
    """Save_figures"""
    figures = [str(figure) for figure in plt.get_figlabels()]
    pylog.debug("Other files:\n    - " + "\n    - ".join(figures))
    for name in figures:
        save_figure(name, extensions=kwargs.pop("extensions", ["png"]))

