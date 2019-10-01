x_file = 'x_train_subset_langs.txt'
y_file = 'y_train_subset_langs.txt'
LANGUAGES = ['urd', 'ara', 'fas', 'swe', 'fin']

# number of epochs for training
NUM_EPOCHS = 5
BATCH_SIZE = 400
INPUT_SIZE = 200
HIDDEN_SIZE = 300
GRU_NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.01

GRU_MODEL_PATH = "gru_model.pkl"
LANG_LABEL_MAPPING = "lang_label_mapping.pkl"