import sys
sys.path.append("..")
from utils.getfilepaths import getfilepaths
from teo_cb_auto_env import TeagorExtractor
import numpy as np


def get_teo_for_day(zone, date, teo, delta):
    feature = []
    infiles = getfilepaths(zone, date)
    for i, file in enumerate(infiles):
        print(i, len(infiles))
        res = teo.process(file, delta)
        feature.append(res)

    # print(f"Feature size: {len(feature)}")
    f = np.vstack((feature[0], feature[1]))
    for ft in feature[2:]:
        f = np.vstack((f, ft))
    
    return f


zone = 1
date = "2018_08_13"
delta = 2

teo = TeagorExtractor()

res = get_teo_for_day(zone, date, teo, delta)
print(res.shape)

