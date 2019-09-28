import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
import random

import config
from GRUNet import GRUNet
from load_data import load_data, get_numeric_representations_sents


if __name__ == '__main__':
    languages = ['urd', 'fars', 'ara', 'srp', 'bos']
    lang_label_to_int_mapping = {
        lang_label: i for i, lang_label in enumerate(languages)
    }

    lang_int_to_label_mapping = {
        i: lang_label for i, lang_label in enumerate(languages)
    }

    print("Loading data...")
    X, y = load_data(config.x_file, config.y_file, languages,
                     lang_label_to_int_mapping, clip_length=100, clip_sents=True, padding=True)
    X = X[:500]
    y = y[:500]

    print("Converting characters to numbers, generating vocabulary...")
    numeric_sequences, vocabulary = get_numeric_representations_sents(X)

    print(vocabulary)
    print(len(vocabulary))

    print("Padding sequences...")
    padded_sequences = pad_sequence(
        numeric_sequences, batch_first=True, padding_value=0.0)
    # for index in range(0, 5):
    #     print(padded_sequences[index])
    #     print(len(padded_sequences[index]))

    # print("Generating one hot representations...")
    # one_hot_vec_sequences = create_one_hot_vectors(
    #     padded_sequences, vocabulary)
    # for index in range(len(one_hot_vec_sequences)):
    #     print(one_hot_vec_sequences[index])
    #     print(one_hot_vec_sequences[index].eq(1).sum().item())

    # initializing the network
    print("Initializing the network...")
    model = GRUNet(vocab_size=len(vocabulary) + 1, seq_len=len(
        padded_sequences[0]), input_size=200, hidden_size=300,
        output_size=len(languages), num_layers=2, dropout=0.2)

    print(model)
    criterion = nn.CrossEntropyLoss()
    learning_rate = config.LEARNING_RATE
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print("Training the model...")
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        print("Epoch: %d" % (epoch + 1))
        zipped_data = list(zip(padded_sequences, y))
        random.shuffle(zipped_data)
        padded_sequences, y = zip(*zipped_data)
        epoch_loss = 0.0
        # for index, sequence in enumerate(one_hot_vec_sequences):
        for index, (input_seq, output_seq) in enumerate(zip(padded_sequences, y)):
            optimizer.zero_grad()
            output = model(torch.stack([input_seq]).long())
            loss = criterion(output, torch.LongTensor([output_seq]))
            loss.backward()
            optimizer.step()

            # if index % 500 == 0 and index > 0:
            #     print(loss.item())
            epoch_loss += loss.item()
        print("Loss at epoch {}: {}".format(epoch + 1, epoch_loss))
