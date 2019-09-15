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
                    X.append([ord(x) for x in sent[:index + 1]])
                    y.append(lang_label)
            # this will add just the sent
            else:
                X.append([ord(x) for x in sent])
                y.append(lang_label)

    return X, y


x_file = 'wili-2018/x_train.txt'
y_file = 'wili-2018/y_train.txt'
languages = ['eng', 'urd', 'fars']

X, y = load_data(x_file, y_file, languages, clip_length=100,
                 clip_sents=True, padding=True)
for index in range(0, 5):
    print(X[index])
    print(y[index])
    print("-" * 60)

# TODO: Convert sentences to vectors, vectors to tensors, get padded versions of all sentences
