import numpy as np
from src.data import data_features, dataset_wav_paths
from sklearn import svm
from sklearn import metrics
import pandas as pd
import pickle
import argparse

np.random.seed(10)


def main(train_path, dev_path, speaker_data_path, max_files_per_speaker, max_sig_duration, output_model_file):

    # preparing train and val data
    train_files = dataset_wav_paths(train_path)
    dev_files = dataset_wav_paths(dev_path)
    id_data = pd.read_csv(speaker_data_path,
                       skiprows=[0],
                       header=None,
                       sep='\t')
    id_data.columns = ['id', 'gender', 'subset', 'name']

    X_train, y_train, ids_train = data_features(train_files, id_data, max_files_per_speaker=max_files_per_speaker,
                                                sr=24000, max_sig_duration=max_sig_duration)

    X_dev, y_dev, ids_dev = data_features(dev_files, id_data, sr=24000, max_sig_duration=max_sig_duration)

    X_train = np.stack([x.flatten() for x in X_train], axis=0)
    X_dev = np.stack([x.flatten() for x in X_dev], axis=0)

    print(f"X_train size: {X_train.shape}, X_dev size: {X_dev.shape}")

    label2idx = {'M': 0, 'F': 1}
    y_train = [label2idx[x] for x in y_train]
    y_dev = [label2idx[x] for x in y_dev]

    # defining the model and hyperparameters
    kernels = ["linear", "poly", "rbf"]
    Cs = [1, 10, 100, 1000]

    # training the models and finding the best hyperparameters
    best_score, best_C, best_kernel = 0, None, None
    best_model = None
    for kernel in kernels:
        for C in Cs:
            model = svm.SVC(kernel=kernel, C=C)
            model.fit(X_train, y_train)
            print(f"kernel={kernel}, C={C}")
            print(metrics.classification_report(y_dev, model.predict(X_dev)))
            acc_score = metrics.accuracy_score(y_dev, model.predict(X_dev))
            if acc_score > best_score:
                best_score = acc_score
                best_C, best_kernel = C, kernel
                best_model = model

    print(f"best results with accuracy={best_score} with kernel={best_kernel} and C={best_C}")

    # saving best model
    with open(output_model_file, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"saved to {output_model_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SVM and tune hyperparameters')
    parser.add_argument('--output-path',
                        help='Directory where a model will be saved')
    parser.add_argument('--train-folder',
                        help='Folder with train examples')
    parser.add_argument('--dev-folder',
                        help='Folder with dev examples')
    parser.add_argument('--speaker-data',
                        help='Data with speakers id and their genders')
    parser.add_argument('--max-files-per-speaker', default=10,
                        help='Maximum wav files taken for one speaker', required=False)
    parser.add_argument('--max-signal-duration', default=2,
                        help='Duration of one speech', required=False)
    args = parser.parse_args()

    main(train_path=args.train_folder, dev_path=args.dev_folder, speaker_data_path=args.speaker_data,
              max_files_per_speaker=args.max_files_per_speaker, max_sig_duration=args.max_signal_duration,
              output_model_file=args.output_path)

