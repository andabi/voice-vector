# -*- coding: utf-8 -*-
#!/usr/bin/env python

import librosa
import datetime
import scikits.audiolab


filename = '/Users/avin/git/voice-vector/datasets/LibriSpeech/train-clean-100/19/198/19-198-0000.flac'  # sample rate = 16,000 Hz

# same sample rate
s = datetime.datetime.now()
wav, _ = librosa.load(filename, mono=True, sr=16000)
wave, sr, _ = scikits.audiolab.flacread(filename)
print(sr)

e = datetime.datetime.now()
diff = e - s
print(diff.microseconds)

# different sample rate (22,050 Hz)
s = datetime.datetime.now()
wav, _ = librosa.load(filename, mono=True)
e = datetime.datetime.now()
diff = e - s
print(diff.microseconds)
