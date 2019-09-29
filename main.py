import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
# import random

import config
from GRUNet import GRUNet
from load_data import load_data, get_numeric_representations_sents, initialize_data_generator, generate_vocabulary, get_clipped_sentences


def initialize_network(vocab_size, seq_len, input_size, hidden_size, output_size, num_layers, dropout, learning_rate):
    # initializing the network
    model = GRUNet(vocab_size=vocab_size, seq_len=seq_len, input_size=input_size,
                   hidden_size=hidden_size, output_size=output_size, num_layers=num_layers,
                   dropout=dropout)

    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


def train_model(training_generator, model, criterion, optimizer):
    # training the model
    for epoch in range(config.NUM_EPOCHS):
        print("Epoch: %d" % (epoch + 1))
        '''
        # This is for batch size 1 - need to manually shuffle the data
        zipped_data = list(zip(padded_sequences, y))
        random.shuffle(zipped_data)
        padded_sequences, y = zip(*zipped_data)
        '''
        epoch_loss = 0.0
        for local_batch, local_labels in training_generator:
            # for index, (input_seq, output_seq) in enumerate(zip(padded_sequences, y)):  --> This is for batch size 1
            optimizer.zero_grad()
            # output = model(torch.stack([input_seq]).long())  --> This is for batch size 1
            output = model(local_batch.long())
            # loss = criterion(output, torch.LongTensor([output_seq]))  --> This is for batch size 1
            loss = criterion(output, local_labels.long())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print("Loss at epoch %d: %.7f" % (epoch + 1, epoch_loss))
    return model


def test_model(model, vocab_mapping, X_test, Y_test):
    total_predictions = 0
    correct_predictions = 0

    for test_sent, test_label in zip(X_test, Y_test):
        # get 100 clipped sents for test_sent
        clipped_sents, _ = get_clipped_sentences(
            [test_sent], [test_label])
        # get numeric representation of test sentence
        numeric_sents = get_numeric_representations_sents(
            clipped_sents, vocab_mapping)
        # get 100 padded sequences
        padded_sequences = pad_sequence(
            numeric_sents, batch_first=True, padding_value=0.0)

        for padded_seq in padded_sequences:
            total_predictions += 1
            output = model(torch.stack([padded_seq]).long())
            _, prediction = torch.max(output.data, dim=1)
            if prediction == test_label:
                correct_predictions += 1
                break

            # for local_batch, local_labels in testing_generator:
            #     outputs = model(local_batch.long())
            #     # the first output of torch.max is the max value, the second output is the index of mac value
            #     _, predicted = torch.max(outputs.data, dim=1)
            #     total_predictions += local_labels.size(0)
            #     correct_predictions += (predicted == local_labels).sum().item()

    print("Accuracy of model: {}".format(
        (correct_predictions / total_predictions) * 100))


if __name__ == '__main__':
    languages = ['urd', 'fars', 'ara']
    lang_label_to_int_mapping = {
        lang_label: i for i, lang_label in enumerate(languages)
    }

    print("Loading data...")
    X, Y = load_data(config.x_file, config.y_file, languages,
                     lang_label_to_int_mapping, clip_length=100,
                     clip_sents=True)
    X = X[:500]
    Y = Y[:500]

    print("Creating train-test split...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=True)

    print("Generating character-level vocabulary...")
    vocab_mapping, vocabulary = generate_vocabulary(X)

    print("Getting clipped sentences...")
    X_train, Y_train = get_clipped_sentences(X_train, Y_train)

    print("Converting characters to numbers through vocabulary...")
    numeric_sequences_train = get_numeric_representations_sents(
        X_train, vocab_mapping)

    print("Padding sequences...")
    padded_sequences_train = pad_sequence(
        numeric_sequences_train, batch_first=True, padding_value=0.0)

    # print("Generating one hot representations...")
    # one_hot_vec_sequences = create_one_hot_vectors(
    #     padded_sequences, vocabulary)

    print("Creating training and testing data generators...")
    print("Total training instances: {}".format(len(padded_sequences_train)))
    training_generator = initialize_data_generator(
        padded_sequences_train, Y_train, config.BATCH_SIZE)

    print("Initializing the network...")
    vocab_size = len(vocabulary) + 1
    output_size = len(languages)
    seq_len = len(padded_sequences_train[0])
    model, criterion, optimizer = initialize_network(
        vocab_size, seq_len, config.INPUT_SIZE, config.HIDDEN_SIZE,
        output_size, config.GRU_NUM_LAYERS, config.DROPOUT, config.LEARNING_RATE)

    print("Training the model...")
    model = train_model(training_generator, model, criterion, optimizer)

    print("Testing the model...")
    test_model(model, vocab_mapping, X_test, Y_test)
