from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display

def use_svg_display():
    """Use the svg format to display a plot in Jupyter"""
    backend_inline.set_matplotlib_formats('svg')


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:  
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes_0 = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, [0,1.5], xscale, yscale, legend[:2])
        self.config_axes_1 = lambda: self.set_axes(
            self.axes[1], xlabel, ylabel, xlim, [0,1], xscale, yscale, legend[2:])
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib."""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        self.axes[1].cla()
        for x, y, fmt in zip(self.X[:2], self.Y[:2], self.fmts[:2]):
            self.axes[0].plot(x, y, fmt)
        for x, y, fmt in zip(self.X[2:], self.Y[2:], self.fmts[2:]):
            self.axes[1].plot(x, y, fmt)
        self.config_axes_0()
        self.config_axes_1()
        display.display(self.fig)
        display.clear_output(wait=True)