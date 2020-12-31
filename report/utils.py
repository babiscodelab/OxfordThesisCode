import os
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def get_nonexistant_path(fname_path):
    """
    Get the path to a filename which does not exist by incrementing path.

    Examples
    --------
    >>> get_nonexistant_path('/etc/issue')
    '/etc/issue-1'
    >>> get_nonexistant_path('whatever/1337bla.py')
    'whatever/1337bla.py'
    """

    directory = os.path.dirname(fname_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(fname_path):
        return fname_path
    filename, file_extension = os.path.splitext(fname_path)
    i = 1
    new_fname = "{}-{}{}".format(filename, i, file_extension)
    while os.path.exists(new_fname):
        i += 1
        new_fname = "{}-{}{}".format(filename, i, file_extension)


    return new_fname



def savefig_metadata(file_path, fmt, fig, meta_data=None):

    def line_prepender2(filename, line):

        with open(filename, 'r+') as f:
            lines = f.readlines()
            lines[1] = line + "\n"

        with open(file_path, "w") as file:
            for line in lines:
                file.write(line)




    def line_prepender(filename, line):
        with open(filename, 'r+') as f:
            content = f.read()
            f.seek(0, 1)
            f.write(line.rstrip('\r\n') + '\n' + content)


    meta_data_str = str(dict(meta_data))

    if (fmt == "eps"):
        fig.savefig(file_path, format=fmt)
        meta_data_str = "%%" + meta_data_str
        line_prepender2(file_path, meta_data_str)
    elif (fmt =="pdf"):

        pdffig = PdfPages(file_path)

        pdffig.savefig(fig)
        metadata = pdffig.infodict()
        metadata['Title'] = 'Example'
        pdffig.close()