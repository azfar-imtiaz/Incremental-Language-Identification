x_file = 'x_train_subset_langs.txt'
y_file = 'y_train_subset_langs.txt'
LANGUAGES = ['urd', 'ara', 'fas', 'swe',
             'fin', 'arz', 'dan', 'deu', 'eng', 'hin']

# number of epochs for training
BATCH_SIZE = 500
INPUT_SIZE = 350
HIDDEN_SIZE = 500
GRU_NUM_LAYERS = 1
DROPOUT = 0.0
LEARNING_RATE = 0.0005
DEVICE = 'cuda:1'

GRU_MODEL_PATH = "gru_model"
VOCAB_MAPPING = "vocab_mapping.pkl"
LANG_LABEL_MAPPING = "lang_label_mapping.pkl"
