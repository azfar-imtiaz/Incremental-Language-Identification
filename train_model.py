import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
import joblib
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
# import random

import config
from GRUNet import GRUNet
from load_data import load_data, get_numeric_representations_sents, initialize_data_generator, generate_vocabulary, get_clipped_sentences
from test_model import test_model

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def initialize_network(vocab_size, seq_len, input_size, hidden_size, output_size, num_layers, dropout, learning_rate, loss_function_type, dev):
    # initializing the network
    gru_model = GRUNet(vocab_size=vocab_size, seq_len=seq_len, input_size=input_size,
                       hidden_size=hidden_size, output_size=output_size, num_layers=num_layers,
                       dropout=dropout, dev=dev)

    print(gru_model)

    # char_min_model = CharMinimizationNet()
    # reduction='none' means that you don't average out the loss of a batch, but rather get the loss
    # as a list of loss values - the loss of each individual loss in the batch. Need this for the custom losses
    if loss_function_type == 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = Adam(gru_model.parameters(), lr=learning_rate)
    # scheduler = StepLR(optimizer, step_size=int(num_epochs / 3), gamma=0.1)
    return gru_model, criterion, optimizer


def train_model(training_generator, gru_model, criterion, optimizer, num_epochs, hidden_layer_size, dev='cpu', loss_type=1):
    # move the model to device
    gru_model.train()
    gru_model = gru_model.to(dev)
    loss_values = []
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
        hidden_layer = gru_model.init_hidden(hidden_layer_size)
        for i, (local_batch, local_labels) in enumerate(training_generator):
            # move local_batch and local_labels to device
            local_batch = local_batch.to(dev)
            local_labels = local_labels.to(dev)
            # for index, (input_seq, output_seq) in enumerate(zip(padded_sequences, y)):  --> This is for batch size 1
            optimizer.zero_grad()

            # get data of hidden layer
            hidden_layer = hidden_layer.data
            # output = model(torch.stack([input_seq]).long())  --> This is for batch size 1
            output, hidden_layer = gru_model(local_batch.long(), hidden_layer)
            # loss = criterion(output, torch.LongTensor([output_seq]))  --> This is for batch size 1
            loss = criterion(output, local_labels.long())
            if loss_type != 1:
                # get the number of characters in each sequence in the batch
                char_lengths = []
                for t in local_batch:
                    non_zero_indices = torch.nonzero(t)
                    char_lengths.append(non_zero_indices.size(0) / 100.0)
                char_lengths = torch.Tensor(char_lengths)
                char_lengths = char_lengths.to(dev)

                if loss_type == 2:
                    # multiply the losses of the batch with the character lengths
                    loss *= char_lengths
                elif loss_type == 3:
                    # add the losses of the batch with the character lengths
                    loss += char_lengths
                # take mean of the loss
                loss = loss.mean()
            # if loss.item() <= 0.0001:
            #     loss.item = 0.0001

            if loss.item() != np.nan:
                loss.backward()
                # clip the gradients to avoid exploding gradients
                nn.utils.clip_grad_norm_(
                    gru_model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
        loss_values.append(epoch_loss)
        print("Loss at epoch %d: %.7f" % (epoch + 1, epoch_loss))

    return gru_model, loss_values


def plot_loss(loss_values, loss_function_type):
    plt.plot(loss_values)
    # plt.show()
    plt.savefig("./loss_values_plot_%d.png" % loss_function_type)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train a recurrent network for language identification")
    parser.add_argument("-X", "--train_x", dest="x_file", type=str,
                        help="Specify the name of the file to load training sentences from")
    parser.add_argument("-Y", "--train_y", dest="y_file", type=str,
                        help="Specify the name of the file to load training labels from")
    parser.add_argument("-E", "--epochs", dest="num_epochs", type=int,
                        help="Specify the number of epochs for training the model")
    parser.add_argument("-L", "--loss", dest="loss_function_type", type=int,
                        help="Specify the loss function to be used. 1=CrossEntropyLoss, 2=CrossEntropy with character length multiplied, 3=CrossEntropy with character length added.")
    parser.add_argument("-F", "--force-vocab-gen", dest="force_vocab_gen", type=int, default=0,
                        help="Force the vocabulary and vocabulary to index mapping to be generated. 1=Generate new vocab_mapping and vocabulary, 0=Do not generate new vocab_mapping and vocabulary")
    args = parser.parse_args()

    languages = config.LANGUAGES
    lang_label_to_int_mapping = {
        lang_label: i for i, lang_label in enumerate(languages)
    }

    print("Loading data...")
    X, Y = load_data(args.x_file, args.y_file, languages,
                     lang_label_to_int_mapping, clip_length=100, clip_sents=True)

    print("Creating train-test split...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=True)

    # if vocabulary mapping already exists on disk and force_vocab_gen is False, load vocab_mapping from disk
    # additionally, initialize vocabulary using keys of vocab_mapping
    if args.force_vocab_gen not in [0, 1]:
        print("Please specify either 0 or 1 for -F command line parameter")
        exit(1)
    args.force_vocab_gen = bool(args.force_vocab_gen)

    if os.path.exists(config.VOCAB_MAPPING) and args.force_vocab_gen is False:
        print("Loading vocab_mapping from disk!")
        vocab_mapping = joblib.load(config.VOCAB_MAPPING)
        vocabulary = list(vocab_mapping.keys())
    # else, create vocabulary and vocab_mapping from scratch
    else:
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
    dev = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    gru_model, criterion, optimizer = initialize_network(
        vocab_size, seq_len, config.INPUT_SIZE, config.HIDDEN_SIZE,
        output_size, config.GRU_NUM_LAYERS, config.DROPOUT, config.LEARNING_RATE,
        args.loss_function_type, dev)

    print("Training the model...")
    gru_model, loss_values = train_model(
        training_generator, gru_model, criterion, optimizer, args.num_epochs,
        config.BATCH_SIZE, dev, args.loss_function_type)
    # padded_sequences_train.size(1), dev, args.loss_function_type)

    print("Plotting loss values...")
    plot_loss(loss_values, args.loss_function_type)

    print("Evaluating on validation data...")
    lang_int_to_label_mapping = {y: x for x,
                                 y in lang_label_to_int_mapping.items()}
    test_model(gru_model, vocab_mapping,
               lang_int_to_label_mapping, X_test, Y_test, dev)

    print("Saving model to disk...")
    joblib.dump(gru_model, "{}_{}.pkl".format(
        config.GRU_MODEL_PATH, args.loss_function_type))
    joblib.dump(vocab_mapping, config.VOCAB_MAPPING)
    joblib.dump(lang_label_to_int_mapping, config.LANG_LABEL_MAPPING)

    # print("Testing the model...")
    # test_model(gru_model, vocab_mapping, X_test, Y_test)
