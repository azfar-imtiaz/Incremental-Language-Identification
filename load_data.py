from torch import Tensor
from torch.utils import data

from Dataset import Dataset


def load_data(x_file, y_file, languages, lang_label_to_int_mapping, clip_length=100, clip_sents=False, padding=False):
    X = []
    y = []

    x_data = open(x_file, 'r').read().split('\n')
    y_data = open(y_file, 'r').read().split('\n')
    for index in range(len(x_data)):
        lang_label = y_data[index]
        if lang_label in languages:
            if clip_sents is True:
                sent = x_data[index][:clip_length]
            else:
                sent = x_data[index]

            # this will add 100 sentences for each sent
            if padding is True:
                for index in range(len(sent)):
                    X.append([x for x in sent[:index + 1]])
                    y.append(lang_label_to_int_mapping[lang_label])
            # this will add just the sent
            else:
                X.append([x for x in sent])
                y.append(lang_label_to_int_mapping[lang_label])

    return X, y


def get_numeric_representations_sents(sents):
    # using vocabulary to get word-to-integer mapping and creating numeric representations through that
    vocabulary = list(set(sum(sents, [])))
    char_to_int_mapping = {char: i + 1 for i, char in enumerate(vocabulary)}
    print(char_to_int_mapping)
    sent_vectors = [[char_to_int_mapping[char]
                     for char in list(sent)] for sent in sents]

    sent_vectors_tensors = [Tensor(vec) for vec in sent_vectors]
    return sent_vectors_tensors, vocabulary


def create_one_hot_vectors(sequences, vocabulary):
    # one_hot_vectors = torch.Tensor((len(sequences), len(vocabulary)))
    one_hot_vectors = []
    for seq in sequences:
        # one_hot_vec = zeros(len(vocabulary))
        one_hot_vec = [0.0] * len(vocabulary)
        for char_int in seq:
            char_int = int(char_int.item())
            if char_int == -1:
                break
            one_hot_vec[char_int - 1] = 1.0
        one_hot_vectors.append(one_hot_vec)

    # one_hot_vectors = LongTensor(one_hot_vectors)
    return one_hot_vectors


def initialize_data_generator(padded_sequences, y, batch_size):
    # create training_generator here
    params = {
        'batch_size': batch_size,
        'shuffle': True
    }
    training_set = Dataset(padded_sequences, y)
    training_generator = data.DataLoader(training_set, **params)
    return training_generator
