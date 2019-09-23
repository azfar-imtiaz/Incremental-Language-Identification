from torch import Tensor, LongTensor, zeros


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

    lang_label_to_int_mapping = {
        lang_label: i for i, lang_label in enumerate(languages)
    }

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

    # convert y to one-hot encoded vectors
    # y_one_hot = []
    # for elem in y:
    #     yoh = [0.0] * len(languages)
    #     yoh[elem] = 1.0
    #     y_one_hot.append(yoh)
    # return X, y_one_hot
    return X, y


def get_numeric_representations_sents(sents):
    '''
        We first convert strings to vectors. To do that, I am using ord() here - can replace
        it by something else later.
        Then we convert the vectors to tensors.
    '''
    # using ord() to convert words to integers/numeric representation and creating numeric representations through that
    # vectors = []
    # for sent in sents:
    #     vec = [ord(ch) for ch in sent]
    #     vectors.append(vec)

    # using vocabulary to get word-to-integer mapping and creating numeric representations through that
    # sent_char_lists = [list(sent) for sent in sents]
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
        one_hot_vec = [-1.0] * len(vocabulary)
        for char_int in seq:
            char_int = int(char_int.item())
            if char_int == -1:
                break
            one_hot_vec[char_int - 1] = 1.0
        one_hot_vectors.append(one_hot_vec)

    # one_hot_vectors = LongTensor(one_hot_vectors)
    return one_hot_vectors


# def pad_sents(sents):
#     '''
#             Padding the sequences using pad_sequences
#     '''

#     sent_vectors_tensors = pad_sequence(sent_vectors_tensors)
#     return sent_vectors_tensors
