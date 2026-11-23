import numpy as np
import librosa


def extract_features(audio, sr):
    """
    Extract 86 audio features:
    - 40 MFCC means
    - 40 MFCC standard deviations
    - Spectral centroid mean and std
    - Zero-crossing rate mean and std
    - RMS energy mean and std
    """

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    zcr = librosa.feature.zero_crossing_rate(audio)

    rms = librosa.feature.rms(y=audio)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        [np.mean(centroid), np.std(centroid)],
        [np.mean(zcr), np.std(zcr)],
        [np.mean(rms), np.std(rms)]
    ])

    return features.astype(np.float32)


def extract_features_from_file(file_path):
    """
    Load an audio file and extract its 86 features.

    Preserve the original sample rate to match
    the feature extraction used during training.
    """

    audio, sr = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    return extract_features(audio, sr)
