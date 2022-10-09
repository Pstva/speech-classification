import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from src.data import data_features, dataset_wav_paths
import argparse
from tqdm import tqdm


torch.manual_seed(10)

# layers value for different configurations of VGG model
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """
    General VGG model for working with multi-channel input
    """
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.sigmoid(self.classifier(out))
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.5)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class AudioDataset(Dataset):
    """
    Dataset for speeches from LibriTTS
    """
    def __init__(self, speaker_data_path, files_path, max_files_per_speaker=None, sr=24000, max_sig_duration=2):
        # loading paths to the wav files, to the data with speaker's ids and gender
        wav_paths = dataset_wav_paths(files_path)
        id_data = pd.read_csv(speaker_data_path,
                              skiprows=[0],
                              header=None,
                              sep='\t')
        id_data.columns = ['id', 'gender', 'subset', 'name']
        # getting features for signals (mel-spectrograms)
        features, classes, ids = data_features(wav_paths, id_data, max_files_per_speaker=max_files_per_speaker,
                      sr=sr, max_sig_duration=max_sig_duration)

        label2idx = {'M': 0, 'F': 1}
        y = [label2idx[x] for x in classes]
        self.X = features
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx][None, :, :]), torch.tensor(self.y[idx]).float()


def train(train_dataloader, val_dataloader, model, optimizer, bce_loss, device, epochs, patience, output_model_file, args):
    """
    training the model for epochs=epochs, with early stopping with patience=patience
    saving best model to file=output_model_file (best by accuracy on val data)
    """
    train_acc, val_acc = [], []
    num_epochs_no_improvement = 0
    best_val_acc, best_val_epoch = 0, 0

    for epoch in range(epochs):
        model.train()
        true_pred, num_examples = 0, 0
        train_loss = 0
        # first part - training on train data
        for X, y in tqdm(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = bce_loss(pred.squeeze(), y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            num_examples += X.shape[0]
            true_pred += (torch.round(pred).squeeze() == y).type(torch.float).sum().item()

        train_acc.append(true_pred / num_examples)

        true_pred, num_examples = 0, 0
        # second part - evaluating on val data
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            model.eval()
            pred = model(X)
            num_examples += X.shape[0]
            true_pred += (torch.round(pred).squeeze() == y).type(torch.float).sum().item()


        val_acc.append(true_pred / num_examples)

        print(f"Epoch: {epoch+1}, train accuracy:{train_acc[-1]}, train loss:{train_loss}, val accuracy:{val_acc[-1]}")

        # checking whether early stopping is needed
        if val_acc[-1] > best_val_acc:
            num_epochs_no_improvement = 0
            best_val_acc = val_acc[-1]
            best_val_epoch = epoch
            # saving the model if the best accuracy on val on this step
            print(f"New best result on epoch {epoch+1}")
            torch.save(model.state_dict(), output_model_file)
        else:
            num_epochs_no_improvement += 1
            if num_epochs_no_improvement >= patience:
                print(f"Stopping at epoch {epoch+1}, no improvement during {patience} epochs ")
                break

    print(f"Best result on epoch {best_val_epoch + 1} with accuracy on val data: {best_val_acc}")
    print(f"Best model was saved to {output_model_file}")


def main(train_path, dev_path, speaker_data_path, max_files_per_speaker, max_sig_duration, output_model_file,
         batch_size):

    # creating datasets for train and validation
    train_dataset = AudioDataset(speaker_data_path, train_path,  max_files_per_speaker=max_files_per_speaker,
                                 max_sig_duration=max_sig_duration)
    val_dataset = AudioDataset(speaker_data_path, dev_path,  max_files_per_speaker=None,
                                 max_sig_duration=max_sig_duration)

    print(f"Train dataset size:{len(train_dataset)}")

    # creating dataloaders for train and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # defining the model, optimizer and loss
    model = VGG('VGG16')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    bce_loss = nn.BCELoss()

    # defining other parameters for training
    # device - cpu/gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)
    # maximum number of epochs
    # and patience - the number of epochs after each the model stops the training
    # if there were no improvement during these epochs
    epochs, patience = 50, 10

    # training
    train(train_dataloader, val_dataloader, model, optimizer, bce_loss, device, epochs, patience, output_model_file, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CNN model and tune hyperparameters')
    parser.add_argument('--output-path',
                        help='Directory where a model will be saved')
    parser.add_argument('--train-folder',
                        help='Folder with train examples')
    parser.add_argument('--dev-folder',
                        help='Folder with dev examples')
    parser.add_argument('--speaker-data',
                        help='Data with speakers id and their genders')
    parser.add_argument('--max-files-per-speaker', default=None,
                        help='Maximum wav files taken for one speaker', required=False)
    parser.add_argument('--max-signal-duration', default=2, type=int,
                        help='Duration of one speech', required=False)
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size', required=False)
    args = parser.parse_args()

    main(train_path=args.train_folder, dev_path=args.dev_folder, speaker_data_path=args.speaker_data,
              max_files_per_speaker=args.max_files_per_speaker, max_sig_duration=args.max_signal_duration,
              output_model_file=args.output_path, batch_size=args.batch_size)

