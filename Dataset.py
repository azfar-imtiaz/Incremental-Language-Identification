from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, padded_sequences, lang_labels):
        self.padded_sequences = padded_sequences
        self.lang_labels = lang_labels

    def __len__(self):
        return len(self.padded_sequences)

    def __getitem__(self, index):
        x = self.padded_sequences[index]
        y = self.lang_labels[index]
        return x, y
