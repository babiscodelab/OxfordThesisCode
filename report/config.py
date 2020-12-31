import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import pyplot

font = {'family' : 'normal',
        'size': 10}

matplotlib.rc('font', **font)

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

outp_file_format = "pdf"

mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amsfonts}']