from pathlib import Path
import librosa
from tqdm import tqdm
import numpy as np
import scipy

"""
Функции, используемые для чтения и предобработки данных.
Все они заточены под конкретный вид папок и файлов датасета LibriTTS
"""


def dataset_wav_paths(path):
    """
    returns paths to wav files for all the speakers for some part of dataset

    :param path: path to the folder with dataset 
    (the folder containing folders for all speakers in the set)
    :return: dict(speaker_ids: [wav_paths])
    """
    path = Path(path)
    speakers_id_folders = list(path.iterdir())
    all_paths = {}
    for folder in speakers_id_folders:
        speaker_id = folder.stem
        wav_paths = [str(x) for x in list(folder.glob('*/*.wav'))]
        all_paths[speaker_id] = wav_paths
    return all_paths


def mel_spectrogram(sig, sr=24000, fmin=20, fmax=4000):
    """
    :param sig: signal
    :param sr: sampling rate
    :param fmax: highest frequency (in Hz)
    :return: mel-spectrogram for this signal (with default librosa parameters, max freq=fmax)
    """
    sgram = librosa.stft(sig)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sr, fmin=fmin, fmax=fmax)
    return mel_scale_sgram


def a_spectrum(sig):
    """
    :param sig: signal
    :return: amplitude spectrum of the signal
    """
    return np.abs(scipy.fft.rfft(sig))


def data_features(speakers_files, id_data, max_files_per_speaker=None, sr=24000, max_sig_duration=5, mode='mel',
                  fmin=20, fmax=4000):
    """
    Perform full preprocessing of the data for further analysis and training
    :param speakers_files: dict(speaker_ids: [wav_paths]) (paths to wav files for all the speakers for
    some part of dataset)
    :param max_files_per_speaker: if not None, defines the maximum number of wav files taken for one speaker.
    :param id_data: pd.DataFrame with columns "id" and "gender"
    :param sr: sampling rate (=24000 in all the files
    :param max_sig_duration: max duration of a signal in seconds (None, if cutting is not neccessary)
    :param mode: mel - mel-spectrogram, amp_spectrum - amplitude spectrum
    :param fmax: highest frequency (in Hz)
    :return: Xs - list with mel-spectrograms for all the files, ys - list with gender of speaker for each file,
    ids - list with speaker ids for all the files
    """
    Xs = []
    ys = []
    ids = []
    id_data['id'] = id_data['id'].astype(int)
    # for each speaker
    for speaker in tqdm(speakers_files):
        file_nums = 0
        # identifying gender of the speaker
        y = id_data[id_data['id'] == int(speaker)]['gender'].item()
        # for each file of one speaker
        for file in speakers_files[speaker]:
            file_nums += 1
            # if we already took max_files_per_speaker files, then stop
            if max_files_per_speaker is not None and file_nums > int(max_files_per_speaker):
                break
            sig, sr = librosa.load(file, sr=sr)
            # cutting the file if needed
            if max_sig_duration is not None:
                sig = sig[:max_sig_duration * sr]
                if sig.shape[0] < int(max_sig_duration * sr):
                    sig = np.pad(sig, (0, int(max_sig_duration * sr - sig.shape[0])))
                assert sig.shape[0] == int(max_sig_duration * sr)

            # calculate mel-spectrogram
            sig_mel = mel_spectrogram(sig, sr=sr, fmin=fmin, fmax=fmax)
            Xs.append(sig_mel)

            ys.append(y)
            ids.append(speaker)

    return Xs, ys, ids
