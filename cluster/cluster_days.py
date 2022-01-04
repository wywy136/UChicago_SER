import sys 
sys.path.append("..")
from utils.getfilepaths import getfilepaths
from feature.teo_cb_auto_env import TeagorExtractor
import pydub
import numpy as np
from sklearn.cluster import KMeans
from pydub import AudioSegment
import io


zone = 1
date1 = ["2018_08_20", "2018_08_14", "2018_08_22", "2018_08_09", "2018_08_10", "2018_08_11", "2018_08_19"]
date4 = ["2018_08_13", "2018_08_14", "2018_08_22", "2018_08_16", "2018_08_10", "2018_08_11", "2018_08_12"]
delta = 2
features = []

teo = TeagorExtractor()
km = KMeans(n_clusters=3, random_state=666)

for d in date4:
    print("=======================")
    print(d)
    feature = []
    infiles = getfilepaths(zone, d)
    # print(len(infiles))
    # for file in infiles:
    #     sound = AudioSegment.from_mp3(file)
    #     d = sound.duration_seconds
    #     data = np.array(sound.get_array_of_samples())
    #     print(data.shape)
    #     # a = AudioSegment.from_mp3(file)
    for i, file in enumerate(infiles):
        print(i)
    #     print(file)
        # sound = pydub.AudioSegment.from_wav(file)
        res = teo.process(file, delta)
        feature.append(res)
    
    print(f"Feature size: {len(feature)}")
    f = np.vstack((feature[0], feature[1]))
    for ft in feature[2:]:
        f = np.vstack((f, ft))

    print(f.shape)
    res_feature = np.mean(f, axis=0)
    print(res_feature.shape)
    features.append(res_feature)


labels = km.fit_predict(features)
print(labels)
