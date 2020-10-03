import numpy as np
import torch
from torch import distributions, nn, optim, squeeze, functional
from models import MelodyLSTM
from utils import load_training_set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get Data. Use first track for now.
data = load_training_set()
data = torch.tensor(data[0]["pitches"]).to(device)
data = torch.unsqueeze(data, dim=1)

num_pitches = 128
hidden_size = 256
hidden_layers = 2
sequence_length = 64

epochs = 10
model = MelodyLSTM(num_pitches, num_pitches,
                   hidden_size, hidden_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# Just do pitches for now and only train on the first track
for i in range(epochs):
    ptr = 0
    running_loss = 0
    hidden_state = model.init_hidden(1)

    while True:
        input_seq = data[ptr:ptr+sequence_length]
        target_seq = data[ptr+1:ptr+1+sequence_length]

        hidden_state = tuple([each.data for each in hidden_state])

        output, hidden_state = model(input_seq, hidden_state)

        loss = criterion(squeeze(output), squeeze(target_seq))
        running_loss += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        ptr += sequence_length

        if ptr+sequence_length+1 > len(data):
            break

    print(f"Epoch: {i} \t Loss: {running_loss}")

    # Sample Melody Every Epoch
    # ptr = 0

    # rand_index = np.random.randint(len(data))
    # input_seq = data[rand_index:rand_index+4]

    # Start with random sequence
    input_seq = torch.tensor([0, 66, 66, 0]).unsqueeze(1).to(device)

    # print("---------------------------------------")

    # print(input_seq, end=' ')

    with torch.no_grad():

        output, hidden_state = model(input_seq, hidden_state)

    #     print(index, end=' ')

    #     ptr += 1

    #     if ptr > sequence_length:
    #         break

    # print("\n---------------------------------------")
