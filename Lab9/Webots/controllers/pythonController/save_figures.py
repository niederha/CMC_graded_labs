"""Save figures"""

import matplotlib.pyplot as plt
import cmc_pylog as pylog


def save_figure(figure, name=None, **kwargs):
    """ Save figure """
    for extension in kwargs.pop("extensions", ["png"]):
        fig = figure.replace(" ", "_").replace(".", "dot")
        if name is None:
            name = "{}.{}".format(fig, extension)
        else:
            name = "{}.{}".format(name, extension)
        plt.figure(figure)
        plt.savefig(name, bbox_inches='tight')
        pylog.debug("Saving figure {}...".format(name))


def save_figures(**kwargs):
    """Save_figures"""
    figures = [str(figure) for figure in plt.get_figlabels()]
    pylog.debug("Other files:\n    - " + "\n    - ".join(figures))
    for name in figures:
        save_figure(name, extensions=kwargs.pop("extensions", ["png"]))

