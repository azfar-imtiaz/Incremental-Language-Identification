from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

import config


def load_data(x_file, y_file, languages, clip_length=100, clip_sents=False, padding=False):
    '''
            INPUT: This function takes as input the files containing the sentences and the labels, as well as the
            list of languages to be considered. It goes through the content of both files simultaneously, and for
            each sentence that belongs to one of the selected languages, it adds both the sentence and the language
            label to the respective lists.
            USAGE: This function can be called separately for both the training data and the testing data.
            RETURNS: This function returns the selected sentences and their language labels in two separate lists.
    '''
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
                    y.append(lang_label)
            # this will add just the sent
            else:
                X.append([x for x in sent])
                y.append(lang_label)

    return X, y


def pad_sents(sents):
    '''
            We first convert strings to vectors. To do that, I am using ord() here - can replace it by something else 
            later.
            Then we convert the vectors to tensors.
            Then we pad the sequences
    '''
    # using ord() to convert words to integers/numeric representation and creating numeric representations through that
    # vectors = []
    # for sent in sents:
    #     vec = [ord(ch) for ch in sent]
    #     vectors.append(vec)

    # using vocabulary to get word-to-integer mapping and creating numeric representations through that
    # sent_char_lists = [list(sent) for sent in sents]
    unique_chars = list(set(sum(sents, [])))
    char_to_int_mapping = {char: i for i, char in enumerate(unique_chars)}
    sent_vectors = [[char_to_int_mapping[char]
                     for char in list(sent)] for sent in sents]

    sent_vectors_tensors = [Tensor(vec) for vec in sent_vectors]

    sent_vectors_tensors = pad_sequence(sent_vectors_tensors)
    return sent_vectors_tensors


languages = ['eng', 'urd', 'fars']

X, y = load_data(config.x_file, config.y_file, languages, clip_length=100,
                 clip_sents=True, padding=True)
for index in range(0, 5):
    print(X[index])
    print(y[index])
    print("-" * 60)

sequences = pad_sents(X[:5])
print(sequences)
