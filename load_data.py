from os import scandir

import numpy as np
import scipy.io as sio
from pandas import DataFrame

from trial import Trial


def load_all_trials(folder):
    # TODO make general on all trials
    scandir()


def load_single_trial(classes, filename='raw_data/natimg2800_M161025_MP030_2017-05-29.mat'):
    mt = sio.loadmat(filename)
    resp = mt['stim'][0]['resp'][0]
    istim = mt['stim'][0]['istim'][0]
    spatial = mt['med']
    return Trial(spatial, resp, istim, classes, filename)


def load_classes(filename='raw_data/stimuli_class_assignment_confident.mat'):
    classes = sio.loadmat(filename)
    df_cls = DataFrame()
    names_dict = {}

    df_cls["Numbers"] = classes['class_assignment'][0]

    for i, name in enumerate(np.hstack(classes['class_names'][0])):
        names_dict[i] = name
    df_cls['Names'] = df_cls.apply(lambda row: names_dict[row.Numbers], axis=1)
    df_cls.drop(['Numbers'], axis=1, inplace=True)
    df_cls.loc[2800] = ['gray']
    return df_cls.astype('str')

