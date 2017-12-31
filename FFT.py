import struct
import wave

import numpy as np

def fftFreq(chemin,nbHarmoniques):
    wav = wave.open(chemin,'rb')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    frames = wav.readframes(nframes)

    data = struct.unpack('%sh' % (nframes * nchannels), frames)
    w = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(w))

    freqArray = []

    for i in range(nbHarmoniques):
        idx=np.argmax(np.abs(w)**2)
        #print(abs(w[idx]))
        freq=freqs[idx]
        frequence=abs(freq*framerate)
        freqArray.append(frequence)
        w = np.delete(w, idx)
    return freqArray

#print(fftFreq("VoiceRecord/homme/test_Leon.wav"))
#print(fftFreq("VoiceRecord/homme/test_Loic.wav"))


#print(fftFreq("VoiceRecord/femme/test_Alex.wav"))
#print(fftFreq("VoiceRecord/femme/test_maman.wav"))
