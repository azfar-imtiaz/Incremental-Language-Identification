import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
import joblib
import argparse
from sklearn.model_selection import train_test_split
# import random

import config
from GRUNet import GRUNet, CharMinimizationNet
from load_data import load_data, get_numeric_representations_sents, initialize_data_generator, generate_vocabulary, get_clipped_sentences


def initialize_network(vocab_size, seq_len, input_size, hidden_size, output_size, num_layers, dropout, learning_rate):
    # initializing the network
    gru_model = GRUNet(vocab_size=vocab_size, seq_len=seq_len, input_size=input_size,
                       hidden_size=hidden_size, output_size=output_size, num_layers=num_layers,
                       dropout=dropout)

    print(gru_model)

    char_min_model = CharMinimizationNet()
    # reduction='none' means that you don't average out the loss of a batch, but rather get the loss
    # as a list of loss values - the loss of each individual loss in the batch
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = Adam(gru_model.parameters(), lr=learning_rate)
    return gru_model, char_min_model, criterion, optimizer


def train_model(training_generator, gru_model, criterion, optimizer, num_epochs, loss_type=1):
    # training the model
    for epoch in range(num_epochs):
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
            output = gru_model(local_batch.long())
            # loss = criterion(output, torch.LongTensor([output_seq]))  --> This is for batch size 1
            loss = criterion(output, local_labels.long())
            if loss_type != 1:
                # get the number of characters in each sequence in the batch
                char_lengths = []
                for t in local_batch:
                    non_zero_indices = torch.nonzero(t)
                    char_lengths.append(non_zero_indices.size(0))

                if loss_type == 2:
                    # multiply the losses of the batch with the character lengths
                    loss *= torch.Tensor(char_lengths)
                elif loss_type == 3:
                    loss += torch.Tensor(char_lengths)
                # take mean of the loss
                loss = loss.mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print("Loss at epoch %d: %.7f" % (epoch + 1, epoch_loss))
    return gru_model


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

    parser = argparse.ArgumentParser(
        description="Train a recurrent network for language identification")
    parser.add_argument("-X", "--train_x", dest="x_file", type=str,
                        help="Specify the name of the file to load training sentences from")
    parser.add_argument("-Y", "--train_y", dest="y_file", type=str,
                        help="Specify the name of th efile to load training labels from")
    parser.add_argument("-E", "--epochs", dest="num_epochs", type=int,
                        help="Specify the number of epochs for training the model")
    parser.add_argument("-L", "--loss", dest="loss_function_type", type=int,
                        help="Specify the loss function to be used. 1=CrossEntropyLoss, 2=CrossEntropy with character length multiplied, 3=CrossEntropy with character length added.")
    args = parser.parse_args()

    languages = config.LANGUAGES
    lang_label_to_int_mapping = {
        lang_label: i for i, lang_label in enumerate(languages)
    }

    print("Loading data...")
    X, Y = load_data(args.x_file, args.y_file, languages,
                     lang_label_to_int_mapping, clip_length=100, clip_sents=True)
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

    print("Creating training data generator...")
    print("Total training instances: {}".format(len(padded_sequences_train)))
    training_generator = initialize_data_generator(
        padded_sequences_train, Y_train, config.BATCH_SIZE)

    print("Initializing the network...")
    vocab_size = len(vocabulary) + 1
    output_size = len(languages)
    seq_len = len(padded_sequences_train[0])
    gru_model, char_min_model, criterion, optimizer = initialize_network(
        vocab_size, seq_len, config.INPUT_SIZE, config.HIDDEN_SIZE,
        output_size, config.GRU_NUM_LAYERS, config.DROPOUT, config.LEARNING_RATE)

    print("Training the model...")
    gru_model = train_model(
        training_generator, gru_model, criterion, optimizer, args.num_epochs, args.loss_function_type)

    print("Saving model to disk...")
    joblib.dump(gru_model, config.GRU_MODEL_PATH)
    joblib.dump(lang_label_to_int_mapping, config.LANG_LABEL_MAPPING)

    # print("Testing the model...")
    # test_model(gru_model, vocab_mapping, X_test, Y_test)
