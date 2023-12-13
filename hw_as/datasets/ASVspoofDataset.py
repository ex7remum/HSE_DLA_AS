from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import numpy as np


class ASVSpoofDataset(Dataset):
    def __init__(self, root, is_train=True, max_flac_len=64000, limit=None):
        self.root = root
        self.is_train = is_train

        if is_train:
            protocol_path = "LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
        else:
            protocol_path = "LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

        data = np.genfromtxt(os.path.join(self.root, protocol_path), dtype=str)

        self.flac_names = data[:, 1]
        self.attack_type = data[:, 3]
        self.labels = data[:, 4]

        self.max_wav_len = max_flac_len
        if limit is not None:
            self.flac_names = self.flac_names[:limit]
            self.labels = self.labels[:limit]

    def __len__(self):
        return len(self.flac_names)

    def __getitem__(self, idx):
        name = self.flac_names[idx]

        if self.is_train:
            full_path = "LA/LA/ASVspoof2019_LA_train/flac/{}.flac".format(name)
        else:
            full_path = "LA/LA/ASVspoof2019_LA_eval/flac/{}.flac".format(name)

        audio, sr = torchaudio.load(os.path.join(self.root, full_path))
        print(audio.shape)

        assert sr == 16000

        if len(audio) < self.max_wav_len:
            n_repeat = self.max_wav_len // len(audio) + 1
            audio = audio.repeat(n_repeat)

        audio = audio[:self.max_wav_len]
        label = int(self.labels[idx] == "bonafide")
        attack = self.attack_type[idx]

        return {
            "audio": audio,
            "label": label,
            "attack": attack
        }
