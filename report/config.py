import matplotlib

font = {'family' : 'normal',
        'size': 10}

matplotlib.rc('font', **font)

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

outp_file_format = "pdf"