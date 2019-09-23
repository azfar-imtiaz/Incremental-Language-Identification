import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

import config
from GRUNet import GRUNet
from load_data import load_data, get_numeric_representations_sents, create_one_hot_vectors


if __name__ == '__main__':
    languages = ['eng', 'urd', 'fars']

    print("Loading data...")
    X, y = load_data(config.x_file, config.y_file, languages, clip_length=100,
                     clip_sents=True, padding=True)
    X = X[:5000]
    y = y[:5000]
    # for index in range(0, 5):
    #     print(X[index])
    #     print(y[index])
    #     print("-" * 60)

    print("Converting characters to numbers, generating vocabulary...")
    numeric_sequences, vocabulary = get_numeric_representations_sents(X)

    print(vocabulary)
    print(len(vocabulary))

    print("Padding sequences...")
    padded_sequences = pad_sequence(
        numeric_sequences, batch_first=True)
    for index in range(0, 5):
        print(padded_sequences[index])
        print(len(padded_sequences[index]))

    print("Generating one hot representations...")
    one_hot_vec_sequences = create_one_hot_vectors(
        padded_sequences, vocabulary)
    # for index in range(len(one_hot_vec_sequences)):
    #     print(one_hot_vec_sequences[index])
    #     print(one_hot_vec_sequences[index].eq(1).sum().item())

    # print(one_hot_vec_sequences.size())
    print(len(y))

    # initializing the network
    print("Initializing the network...")
    model = GRUNet(input_size=len(one_hot_vec_sequences[0]), hidden_size=300,
                   output_size=3, num_layers=2, batch_size=config.BATCH_SIZE, dropout=0.0)
    print(model)
    criterion = nn.CrossEntropyLoss()
    learning_rate = config.LEARNING_RATE
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print("Training the model...")
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        print("Epoch: %d" % (epoch + 1))
        model.zero_grad()
        for index, sequence in enumerate(one_hot_vec_sequences):
            try:
                sequences_batch = [one_hot_vec_sequences[i]
                                   for i in range(index, index + config.BATCH_SIZE)]
                outputs_batch = [y[i]
                                 for i in range(index, index + config.BATCH_SIZE)]
            except IndexError as e:
                print("We have reached the end of the dataset, break!")
                print(str(e))
                break
            optimizer.zero_grad()
            # output = model(torch.FloatTensor([[sequence]]))
            # print(np.asarray([sequences_batch]).shape)
            output = model(torch.FloatTensor([sequences_batch]))
            # print("Output size: {}".format(output.size()))
            # print("Label size: {}".format(np.asarray(outputs_batch).shape))
            loss = criterion(output, torch.Tensor([outputs_batch]).long())
            loss.backward()
            optimizer.step()

            if index % 500 == 0 and index > 0:
                print(loss.item())
