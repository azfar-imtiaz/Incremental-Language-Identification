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

            # this will add just the sent
            X.append(sent)
            y.append(lang_label_to_int_mapping[lang_label])

    return X, y


def get_clipped_sentences(sents, labels):
    # this function returns 100 sentences for each sentence
    character_wise_sents = []
    proportioned_labels = []
    for sent, label in zip(sents, labels):
        # this will add 100 sentences for each sent
        for index in range(len(sent)):
            character_wise_sents.append([x for x in sent[:index + 1]])
            proportioned_labels.append(label)
    return character_wise_sents, proportioned_labels


def generate_vocabulary(sents):
    sents = [[x for x in sent] for sent in sents]
    vocabulary = list(set(sum(sents, [])))
    char_to_int_mapping = {char: i + 1 for i, char in enumerate(vocabulary)}
    return char_to_int_mapping, vocabulary


def get_numeric_representations_sents(sents, vocab_mapping):
    # using vocabulary to get word-to-integer mapping and creating numeric representations through that
    sent_vectors = [[vocab_mapping[char]
                     for char in sent] for sent in sents]

    sent_vectors_tensors = [Tensor(vec) for vec in sent_vectors]
    return sent_vectors_tensors


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
