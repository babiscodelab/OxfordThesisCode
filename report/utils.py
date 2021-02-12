import os
import json
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from subprocess import Popen
from datetime import datetime

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
    fig.tight_layout()

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
        meta_data_str = "%" + meta_data_str
        line_prepender2(file_path, meta_data_str)
    elif (fmt =="pdf"):

        pdffig = PdfPages(file_path)

        pdffig.savefig(fig)
        metadata = pdffig.infodict()
        metadata['Keywords'] = meta_data_str
        pdffig.close()



def splitpath(path, maxdepth=20):
 ( head, tail ) = os.path.split(path)
 return splitpath(head, maxdepth - 1) + [ tail ] \
     if maxdepth and head and head != path \
     else [ head or tail ]



tmp_path = os.path.join(os.path.realpath(__file__), "../../tmp/")


def open_with_excel(df, fn=""):
    dt = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    file_path = os.path.join(tmp_path, "tmp_" + fn + dt + ".xlsx")

    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    df.to_excel(file_path, index=False)
    p = Popen(file_path, shell=True)

    pass


def read_results(file_directory, key=None):

    all_df = []
    for file in os.listdir(file_directory):
        df = pd.read_hdf(os.path.join(file_directory, file), key=key)
        all_df.append(df)
    return pd.concat(all_df)



def read_results_walk(file_directory, key=None):

    all_df = []
    for root, dirs, files in os.walk(file_directory):
        for file in files:
            df = pd.read_hdf(os.path.join(root, file), key=key)
            all_df.append(df)
    if all_df:
        return pd.concat(all_df)
    else:
        return pd.DataFrame()
