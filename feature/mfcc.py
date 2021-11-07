from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav


class MFCC_Feature_Extractor:
    def __init__(self):
        self.nfft = 1024

    def extract(self, filepath):
        (rate, sig) = wav.read(filepath)
        mfcc_feat = mfcc(sig, rate, nfft=self.nfft)
        fbank_feat = logfbank(sig, rate, nfft=self.nfft)

        return fbank_feat


if __name__ == "__main__":
    f = MFCC_Feature_Extractor()
    print(f.extract("./201808120932-28710-27730-000221252.wav").shape)


