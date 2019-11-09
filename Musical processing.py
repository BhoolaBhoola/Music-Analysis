# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:47:47 2019

@author: Kamil
"""

import librosa

#analysis for 1 song
#location of audio
audio_path = r"C:\Users\Kamil\Desktop\Frank Ocean - Endless (iTunes 320)\01 At Your Best (You Are Love).mp3"
#loading the audio

x, sr = librosa.load(audio_path)

#This returns an audio time series as a numpy array with a default 
#sampling rate(sr) of 22KHZ mono
#sampling rate is the number of audio samples per second. The higher the better...
print(type(x), type(sr))

import IPython.display as ipd
ipd.Audio(audio_path)


#plotting the song in wave form.
%matplotlib inline
import matplotlib.pyplot as plt
import librosa.display


plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)

#plotting as a spectrogram
#visual representation of the spectrum of frequencies of sound or other signals as they 
#vary with time. for example a sonograph is sound bouncing off something to produce a signal.

X = librosa.stft(x)
#converting amplitude to decibel spectrogram
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14,5))
librosa.display.specshow(Xdb,sr=sr, x_axis = "time", y_axis = "hz")
plt.colorbar()

#We can observe most of the frequencies are taking place towards the bottom of the graph. We can apply a log transformation
#
X = librosa.stft(x)
#converting amplitude to decibel spectrogram
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14,5))
librosa.display.specshow(Xdb,sr=sr, x_axis = "time", y_axis = "hz")
plt.colorbar()

#Feature extraction
"""
in python an audio signal is a numpy array

a sound is composed of an amplitude and frequency. sound = vibration
amplitude is essentially how loud the sound is. (amp in music is to apply amplitude transformation to a sound)

frequency is the number of waves in a time period, which corresponds to pitch.
a higher pitch would mean a higher frequency.

Audio signals consists of lots of many features, we will try to attempt to extract them
relevant to the problem we are trying to solve.

1. Zero crossing rate

rate where sound signal crosses zero. +ve to -ve. Used heavily in speech recognition and 
music feature retrieval

what does it mean when a sound signal crosses 0?

2. Central spectroid

indicates where the center of mass is located on a spectrum. perceptively has a connection
with the "brightness" of a sound. Calculated as the weighted mean of frequencies present in the 
signal determined by Fourier transform (what is this), with magnitudes as weights.
x(n) represents the weighted frequency value, or magnitude, of bin number n, and f(n) represents the center frequency of that bin.

e.g for blues an metal, there are generally 2 different center of masses, blues is generally
the same throughout the length of the song, in metal it would be towards the end.

3. Spectral rolloff

measures the shape of the signal. represents the frequency below which a specified % of the 
spectral energy lies e.g 85%

4. Mel-Frequency Cepstral coefficients

(MFCCs) of a signal are a small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope. It models the characteristics of the human voice.


5. Chroma frequencies

Chroma features are an interesting and powerful representation for music audio in which the entire spectrum is projected onto 12 bins 
representing the 12 distinct semitones (or chroma) of the musical octave.

"""

#examining zero crossing rate.
#in the waveplot generated above, we will zoom in.
n0 = 900
n1 = 910
plt.figure(figsize=(15,5))
#plotting the first 910 milliseconds?
plt.plot(x[n0:n1])
plt.grid()

#there looks to be 7 zero crossings here. We can verify with librosa

zero_crossings = librosa.zero_crossings(x[n0:n1], pad = False)
print(sum(zero_crossings))


#central spectroid
spectral_centroids = librosa.feature.spectral_centroid(x,sr=sr)[0]

#time variable for visualisation
frames= range(len(spectral_centroids))
t = librosa.time_to_frames(frames)

#normalizing the spectral centroid for visualisation
from sklearn.preprocessing import minmax_scale

def normalize(x, axis = 0):
    return minmax_scale(x, axis=axis)

#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')


#spectral rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')

#Mel-Frequency
mfccs = librosa.feature.mfcc(x, sr=sr)

#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')


#We can also perform feature scaling such that each coefficient dimension has zero mean and unit variance:

from sklearn.preprocessing import scale


mfccs = scale(mfccs, axis=1)
print(mfccs.mean(axis=1))
print(mfccs.var(axis=1))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')


#Chroma scale
hop_length = 512
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')


#Can also compute RMSE 


#WE NEED TO APPLY THIS TO EAST COAST AND WEST COAST MUSIC.
from pydub import AudioSegment
sound = AudioSegment.from_file("2Pac - 02 - Trapped.mp3")

one_third = len(sound) // 3
two_thirds = len(sound) * (2/3)
first_third = sound[:one_third]
second_third = sound[one_third:two_thirds]
third_third = sound[two_thirds:]

# create a new file "first_half.mp3":
first_half.export(r"C:\Users\Kamil\Downloads\2Pac Discography [2007]\--- Studio Albums ---\1991 - 2Pacalypse Now\trapped pt 2.mp3", format="mp3")

#split
NEED TO COMPUTE THE FEATURES FOR EACH TRACK.
mel
chromo
central spectroid
spectral rollofff
zero crossing
energy.

#can use naieve bayes. ann, cnn, svm, naieve bayes, etc...

#are there any other features we can compute to measure an audio clip?

https://github.com/aubio/aubio
https://musicinformationretrieval.com/energy.html
https://www.ee.columbia.edu/~dpwe/pubs/ismir05-svm.pdf

