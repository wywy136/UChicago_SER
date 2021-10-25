import pydub
import librosa
from scipy import signal
from mutagen.mp3 import MP3
import numpy as np
from numpy import sign, trapz
from teager_py import Teager
import statsmodels.api as sm


class TeagorExtractor:
    def __init__(self):
        self.audio_data = None
        self.filted_data = []
        self.sampling_rate = 0
        self.max_frequency = 0.
        self.num_filter = 16
        self.teo_feature = None
        self.acf_feature = []
        self.envelope = []
        self.area = []
        self.critical_bands = [
            [100, 200],
            [200, 300],
            [300, 400],
            [400, 510],
            [510, 630],
            [630, 770],
            [770, 920],
            [920, 1080],
            [1080, 1270],
            [1270, 1480],
            [1480, 1720],
            [1720, 2000],
            [2000, 2320],
            [2320, 2700],
            [2700, 3150],
            [3150, 3700]
        ]


    def get_teo_feature(self):
        self.teo_feature = Teager(self.filted_data, 'horizontal', 1)
        if self.teo_feature is None:
            raise AssertionError


    def trans_mp3_to_wav(self, filepath, wav_filepath):
        song = pydub.AudioSegment.from_mp3(filepath)
        song.export(wav_filepath, format="wav")
        self.audio_data = np.array(song.get_array_of_samples()).astype(np.int64)


    def get_sampling_rate(self, filepath):
        audio = MP3(filepath)
        return audio.info.sample_rate


    def get_max_frequency(self, wav_filepath, sampling_rate):
        wav_data, _ = librosa.load(wav_filepath, sr=sampling_rate)
        # print(type(wav_data))
        frequencies, times, spectrogram = signal.spectrogram(wav_data, sampling_rate)
        # print(type(frequencies))
        return max(frequencies)


    def cb_filter(self):
        for i in range(self.num_filter):
            wn1, wn2 = 2.9 * self.critical_bands[i][0] / self.max_frequency, 2.9 * self.critical_bands[i][1] / self.max_frequency
            b, a = signal.butter(1, [wn1, wn2], 'bandpass')
            self.filted_data.append(signal.filtfilt(b, a, self.audio_data))
        self.filted_data = np.array(self.filted_data)
        if self.filted_data.shape != (self.num_filter, len(self.audio_data)):
            raise AssertionError
        # print(self.filted_data.shape)

    
    def get_acf_feature(self):
        for i in range(self.num_filter):
            self.acf_feature.append(sm.tsa.stattools.acf(self.teo_feature[i]))
        self.acf_feature = np.array(self.acf_feature)
        # print(self.acf_feature.shape)

    
    def get_envelop(self):
        for i in range(self.num_filter):
            y_upper, y_lower = self.envelope_extraction(self.acf_feature[i])
            self.envelope.append([y_upper, y_lower])


    def get_area(self):
        for i in range(self.num_filter):
            area = []
            for j in range(2):
                area.append(trapz(self.envelope[i][j]))
            self.area.append(area)
        self.area = np.array(self.area)
        # print()


    def process(self, filepath):
        wav_filepath = './201812281230-119004-27730' + '.wav'
        self.trans_mp3_to_wav(filepath, wav_filepath)
        self.sampling_rate = self.get_sampling_rate(filepath)
        # print(self.sampling_rate)
        self.max_frequency = self.get_max_frequency(filepath, self.sampling_rate)
        # print(self.max_frequency)
        self.cb_filter()
        print('Teo feature ...')
        self.get_teo_feature()
        print(f'Teo feature size: {self.teo_feature.shape}')
        # print(self.teo_feature.shape)
        print('Acf feature ...')
        self.get_acf_feature()
        print(f'Acf feature size: {self.acf_feature.shape}')
        print('Envelope ...')
        self.get_envelop()
        print('Area ...')
        self.get_area()
        print(f'Area feature size: {self.area.shape}')

        return self.area


    def envelope_extraction(self, signal):
        s = signal.astype(float)
        q_u = np.zeros(s.shape)
        q_l = np.zeros(s.shape)

        u_x = [0, ] 
        u_y = [s[0], ]  

        l_x = [0, ] 
        l_y = [s[0], ]

        for k in range(1, len(s) - 1):
            if (sign(s[k] - s[k - 1]) == 1) and (sign(s[k] - s[k + 1]) == 1):
                u_x.append(k)
                u_y.append(s[k])

            if (sign(s[k] - s[k - 1]) == -1) and ((sign(s[k] - s[k + 1])) == -1):
                l_x.append(k)
                l_y.append(s[k])

        u_x.append(len(s) - 1)
        u_y.append(s[-1])  

        l_x.append(len(s) - 1)
        l_y.append(s[-1])  

        upper_envelope_y = np.zeros(len(signal))
        lower_envelope_y = np.zeros(len(signal))

        upper_envelope_y[0] = u_y[0]
        upper_envelope_y[-1] = u_y[-1]
        lower_envelope_y[0] = l_y[0]  
        lower_envelope_y[-1] = l_y[-1]

        last_idx, next_idx = 0, 0
        k, b = self.general_equation(u_x[0], u_y[0], u_x[1], u_y[1])
        for e in range(1, len(upper_envelope_y) - 1):
            if e not in u_x:
                v = k * e + b
                upper_envelope_y[e] = v
            else:
                idx = u_x.index(e)
                upper_envelope_y[e] = u_y[idx]
                last_idx = u_x.index(e)
                next_idx = u_x.index(e) + 1
                k, b = self.general_equation(u_x[last_idx], u_y[last_idx], u_x[next_idx], u_y[next_idx])

        last_idx, next_idx = 0, 0
        k, b = self.general_equation(l_x[0], l_y[0], l_x[1], l_y[1])
        for e in range(1, len(lower_envelope_y) - 1):

            if e not in l_x:
                v = k * e + b
                lower_envelope_y[e] = v
            else:
                idx = l_x.index(e)
                lower_envelope_y[e] = l_y[idx]
                last_idx = l_x.index(e)
                next_idx = l_x.index(e) + 1
                k, b = self.general_equation(l_x[last_idx], l_y[last_idx], l_x[next_idx], l_y[next_idx])

        return upper_envelope_y, lower_envelope_y


    def general_equation(self, first_x, first_y, second_x, second_y):
        A = second_y - first_y
        B = first_x - second_x
        C = second_x * first_y - first_x * second_y
        k = -1 * A / B
        b = -1 * C / B
        return k, b


if __name__ == "__main__":
    t = TeagorExtractor()
    teo_cb_auto_env = t.process("/project/graziul/data/Zone1/2018_12_28/201812281230-119004-27730.mp3")
    print(teo_cb_auto_env)