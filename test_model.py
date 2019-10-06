import torch
import joblib
import argparse
from torch.nn.utils.rnn import pad_sequence

import config
from load_data import load_data, get_numeric_representations_sents, get_clipped_sentences


def test_model(model, vocab_mapping, X_test, Y_test, dev):
    total_predictions = 0
    correct_predictions = 0
    num_chars_until_hit_score_list = []

    for test_sent, test_label in zip(X_test, Y_test):
        print("Testing instance: %s" % test_sent)
        # get 100 clipped sents for test_sent
        clipped_sents, _ = get_clipped_sentences(
            [test_sent], [test_label])
        # get numeric representation of test sentence
        numeric_sents = get_numeric_representations_sents(
            clipped_sents, vocab_mapping)
        # get 100 padded sequences
        padded_sequences = pad_sequence(
            numeric_sents, batch_first=True, padding_value=0.0)
        correct_predictions_per_instance = 0
        num_chars_until_hit_score = 100

        for padded_seq in padded_sequences:
            total_predictions += 1
            input = torch.stack([padded_seq]).long()
            input = input.to(dev)
            output = model(input)
            _, prediction = torch.max(output.data, dim=1)
            if prediction == test_label:
                correct_predictions_per_instance += 1
                if num_chars_until_hit_score == 0:
                    num_chars_until_hit_score = torch.nonzero(
                        padded_seq).size(0)
                    # add this hit score to list of num_char_until_hit_score for averaging later
                    num_chars_until_hit_score_list.append(
                        num_chars_until_hit_score)
                correct_predictions += 1

        print("Most probable class was correct for a total of %d character prefix lengths for this instance" %
              correct_predictions_per_instance)
        print("Number of characters until hit score for this instance: %d" %
              num_chars_until_hit_score)
        print()

        # for local_batch, local_labels in testing_generator:
        #     outputs = model(local_batch.long())
        #     # the first output of torch.max is the max value, the second output is the index of mac value
        #     _, predicted = torch.max(outputs.data, dim=1)
        #     total_predictions += local_labels.size(0)
        #     correct_predictions += (predicted == local_labels).sum().item()

    avg_chars_until_hit_score = sum(
        num_chars_until_hit_score_list) / len(num_chars_until_hit_score_list)
    overall_accuracy = (correct_predictions / total_predictions) * 100

    print("Average number of characters until hit score: {}".format(
        avg_chars_until_hit_score))
    print("Overall Accuracy of model: {}".format(overall_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Load the trained model and get predictions on test data")
    parser.add_argument("-X", "--test_x", dest="x_file", type=str,
                        help="Specify the name of the file to load training sentences from")
    parser.add_argument("-Y", "--test_y", dest="y_file", type=str,
                        help="Specify the name of the file to load training labels from")
    parser.add_argument("-M", "--model", dest="model_path", type=str,
                        help="Specify the path to the trained model")
    args = parser.parse_args()
    # dev = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    dev = torch.device("cpu")

    languages = config.LANGUAGES

    print("Loading model...")
    lang_label_to_int_mapping = joblib.load(config.LANG_LABEL_MAPPING)
    vocab_mapping = joblib.load(config.VOCAB_MAPPING)
    gru_model = joblib.load(args.model_path)
    gru_model = gru_model.to(dev)

    print("Loading testing data...")
    X, Y = load_data(args.x_file, args.y_file, languages,
                     lang_label_to_int_mapping, clip_sents=True)

    print("Getting accuracy on testing data...")
    test_model(gru_model, vocab_mapping, X, Y, dev)
