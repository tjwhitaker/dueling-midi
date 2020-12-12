from models import LSTM, GRU, CNN, LSTMEncoder, LSTMDecoder, GRUEncoder, GRUDecoder
import time
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Timing model initialization")

# First time we load a model, python takes a lot longer.
lstm = LSTM().to(device)
lstm.load_state_dict(torch.load("../models/lstm.model"))

gru = GRU().to(device)
gru.load_state_dict(torch.load("../models/gru.model"))

cnn = CNN().to(device)
cnn.load_state_dict(torch.load("../models/cnn.model"))

lstmencoder = LSTMEncoder().to(device)
lstmdecoder = LSTMDecoder().to(device)

lstmencoder.load_state_dict(torch.load("../models/lstmencoder.model"))
lstmdecoder.load_state_dict(torch.load("../models/lstmdecoder.model"))

gruencoder = GRUEncoder().to(device)
grudecoder = GRUDecoder().to(device)

gruencoder.load_state_dict(torch.load("../models/gruencoder.model"))
grudecoder.load_state_dict(torch.load("../models/grudecoder.model"))

# So we start timing here

lstm_times = []
gru_times = []
cnn_times = []
lstm_enc_dec_times = []
gru_enc_dec_times = []

for i in range(1):
    start = time.time()
    lstm = LSTM().to(device)
    lstm.load_state_dict(torch.load("../models/lstm.model"))
    lstm_times.append(time.time() - start)

    start = time.time()
    gru = GRU().to(device)
    gru.load_state_dict(torch.load("../models/gru.model"))
    gru_times.append(time.time() - start)

    start = time.time()
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load("../models/cnn.model"))
    cnn_times.append(time.time() - start)

    start = time.time()
    lstmencoder = LSTMEncoder().to(device)
    lstmdecoder = LSTMDecoder().to(device)

    lstmencoder.load_state_dict(torch.load("../models/lstmencoder.model"))
    lstmdecoder.load_state_dict(torch.load("../models/lstmdecoder.model"))
    lstm_enc_dec_times.append(time.time() - start)

    start = time.time()
    gruencoder = GRUEncoder().to(device)
    grudecoder = GRUDecoder().to(device)

    gruencoder.load_state_dict(torch.load("../models/gruencoder.model"))
    grudecoder.load_state_dict(torch.load("../models/grudecoder.model"))
    gru_enc_dec_times.append(time.time() - start)

# Model sizes
# for model in [cnn, lstm, gru, gruencoder, grudecoder, lstmencoder, lstmdecoder]:
#     print(sum(p.numel() for p in model.parameters()))

print(lstm_times)
print(gru_times)
print(cnn_times)
print(lstm_enc_dec_times)
print(gru_enc_dec_times)
