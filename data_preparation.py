import config


def read_data_from_files(filename_x, filename_y, languages):
    X = []
    y = []
    with open(filename_x, 'r') as rfile:
        x_data = rfile.read().split('\n')
    with open(filename_y, 'r') as rfile:
        y_data = rfile.read().split('\n')
    for index in range(len(x_data)):
        lang_label = y_data[index]
        if lang_label in languages:
            X.append(x_data[index])
            y.append(lang_label)

    return X, y


def write_data_to_files(X, y, filename_x, filename_y):
    wfile_x = open(filename_x, 'w')
    wfile_y = open(filename_y, 'w')
    for index in range(len(X)):
        wfile_x.write(X[index])
        # add line break only if we are not at last record of file
        if index < len(X) - 1:
            wfile_x.write('\n')

        wfile_y.write(y[index])
        # add line break only if we are not at last record of file
        if index < len(X) - 1:
            wfile_y.write('\n')

    wfile_x.close()
    wfile_y.close()


if __name__ == '__main__':
    languages = config.LANGUAGES
    path_to_files = "../../../usr/local/courses/lt2316-h19/a1/"

    x_train_filename = "{}x_train.txt".format(path_to_files)
    y_train_filename = "{}y_train.txt".format(path_to_files)
    X, y = read_data_from_files(x_train_filename, y_train_filename, languages)

    x_train_subset_filename = "x_train_subset_langs.txt"
    y_train_subset_filename = "y_train_subset_langs.txt"
    write_data_to_files(X, y, x_train_subset_filename, y_train_subset_filename)

    x_test_filename = "{}x_test.txt".format(path_to_files)
    y_test_filename = "{}y_test.txt".format(path_to_files)
    X, y = read_data_from_files(x_test_filename, y_test_filename, languages)

    x_test_subset_filename = "x_test_subset_langs.txt"
    y_test_subset_filename = "y_test_subset_langs.txt"
    write_data_to_files(X, y, x_test_subset_filename, y_test_subset_filename)
