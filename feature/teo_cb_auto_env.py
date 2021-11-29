import pydub
import librosa
from scipy import signal
from mutagen.mp3 import MP3
import numpy as np
from numpy import sign, trapz
from teager_py import Teager
import statsmodels.api as sm
import wave


class TeagorExtractor:
    def __init__(self):
        self.audio_data = None
        self.audio_data_all = None
        self.filted_data = []
        self.sampling_rate = 0
        self.max_frequency = 0.
        self.num_filter = 16
        self.delta = 0.
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
        self.all_features = []


    def clear_data(self):
        self.filted_data = []
        self.acf_feature = []
        self.envelope = []
        self.area = []

    
    def get_teo_feature(self):
        self.teo_feature = Teager(self.filted_data, 'horizontal', 1)
        if self.teo_feature is None:
            raise AssertionError


    def split_wav(self, filepath, wav_filepath):
        song = pydub.AudioSegment.from_wav(filepath)
        duration = song.duration_seconds * 1000
        # print(f"Wav file length: {duration}")
        length = 0
        self.audio_data_all = []
        while length + int(self.delta * 1000) <= duration:
            song_piece = song[length:length + int(self.delta * 1000)]
            self.audio_data_all.append(np.array(song_piece.get_array_of_samples()).astype(np.int64))
            length += int(self.delta * 1000)
        # print(len(self.audio_data_all))

    @staticmethod
    def get_sampling_rate():
        return 22050

    @staticmethod
    def get_max_frequency(wav_filepath, sampling_rate):
        wav_data, _ = librosa.load(wav_filepath, sr=sampling_rate)
        frequencies, times, spectrogram = signal.spectrogram(wav_data, sampling_rate)
        return max(frequencies)


    def cb_filter(self):
        for i in range(self.num_filter):
            wn1, wn2 = 2.9 * self.critical_bands[i][0] / self.max_frequency, 2.9 * self.critical_bands[i][1] / self.max_frequency
            b, a = signal.butter(1, [wn1, wn2], 'bandpass')
            self.filted_data.append(signal.filtfilt(b, a, self.audio_data))
        self.filted_data = np.array(self.filted_data)
        if self.filted_data.shape != (self.num_filter, len(self.audio_data)):
            raise AssertionError

    
    def get_acf_feature(self):
        for i in range(self.num_filter):
            self.acf_feature.append(sm.tsa.stattools.acf(self.teo_feature[i]))
        self.acf_feature = np.array(self.acf_feature)

    
    def get_envelope(self):
        for i in range(self.num_filter):
            y_upper, y_lower = self.envelope_extraction(self.acf_feature[i])
            self.envelope.append([y_upper, y_lower])


    def get_area(self):
        for i in range(self.num_filter):
            self.area.append(trapz(self.envelope[i][0]))
        self.area = np.array(self.area)
        # print()


    def process(self, filepath, delta):
        self.delta = delta
        wav_filepath = filepath
        self.split_wav(filepath, wav_filepath)
        self.sampling_rate = self.get_sampling_rate()
        self.max_frequency = self.get_max_frequency(filepath, self.sampling_rate)
        print("Computing TEO Features ...")
        nums = len(self.audio_data_all)
        for i, audio in enumerate(self.audio_data_all):
            if i % 5000 == 0:
                print(i, nums)
            self.audio_data = audio
            self.clear_data()
            self.cb_filter()
            self.get_teo_feature()
            self.get_acf_feature()
            self.get_envelope()
            self.get_area()
            self.all_features.append(self.area)

        return np.nan_to_num(np.array(self.all_features))


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
    teo_cb_auto_env = t.process("/project/graziul/ra/team_ser/data/new-201808120005-475816-27730.wav", 2)
    print(teo_cb_auto_env.shape)
    # print(teo_cb_auto_env)
    np.savetxt("./200.feature", teo_cb_auto_env, delimiter=',')
