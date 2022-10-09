from src.train_cnn import VGG
import librosa
import numpy as np
from src.data import mel_spectrogram
import pickle
import torch

def classify_gender(signal, sr=24000, model='svm'):
    """
    :param signal: signal to classify, will be cutted/padded to 2 sec
    :param model: 'svm' or 'vgg'
    :return: class - 'F' for female, 'M' for male
    """

    # changing sampling rate if needed
    if sr != 24000:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=24000)
        sr = 24000

    idx2label = {0: 'M', 1: 'F'}
    # preprocessing signal
    duration = 2 * sr
    sig = signal[:duration]
    if sig.shape[0] < duration:
        sig = np.pad(sig, (0, duration - sig.shape[0]))

    # mel-spectrogram
    mel_spectr = mel_spectrogram(sig, sr=sr, fmin=20, fmax=4000)

    # predicting
    if model == 'svm':
        svm_model = pickle.load(open('../models/svm.pkl', 'rb'))
        pred = svm_model.predict(mel_spectr.flatten().reshape(1, -1))
        return idx2label[pred[0]]

    if model == 'vgg':
        vgg_model = model = VGG('VGG16')
        model.load_state_dict(torch.load('../models/vgg16.pkl'))
        X = torch.tensor(mel_spectr[None, None, :, :])
        pred = model(X).cpu().detach().numpy()[0][0]
        pred = round(pred)
        return idx2label[pred]
