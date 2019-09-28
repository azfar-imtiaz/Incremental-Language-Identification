# import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
# import random

import config
from GRUNet import GRUNet
from load_data import load_data, get_numeric_representations_sents, initialize_data_generator


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


if __name__ == '__main__':
    languages = ['urd', 'fars', 'ara', 'srp', 'bos']
    lang_label_to_int_mapping = {
        lang_label: i for i, lang_label in enumerate(languages)
    }

    print("Loading data...")
    X, y = load_data(config.x_file, config.y_file, languages,
                     lang_label_to_int_mapping, clip_length=100,
                     clip_sents=True, padding=True)
    X = X[:2500]
    y = y[:2500]

    print("Converting characters to numbers, generating vocabulary...")
    numeric_sequences, vocabulary = get_numeric_representations_sents(X)

    print(vocabulary)

    print("Padding sequences...")
    padded_sequences = pad_sequence(
        numeric_sequences, batch_first=True, padding_value=0.0)

    # print("Generating one hot representations...")
    # one_hot_vec_sequences = create_one_hot_vectors(
    #     padded_sequences, vocabulary)

    print("Creating training data generator...")
    training_generator = initialize_data_generator(
        padded_sequences, y, config.BATCH_SIZE)

    print("Initializing the network...")
    model, criterion, optimizer = initialize_network(len(vocabulary) + 1, len(
        padded_sequences[0]), 200, 300, len(languages), 2, 0.2, config.LEARNING_RATE)

    print("Training the model...")
    model = train_model(training_generator, model, criterion, optimizer)
