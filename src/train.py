import numpy as np
import torch
from models.notelstm import NoteLSTM, NoteCNN
import utils

epochs = 1
sequence_length = 64
batch_size = 64
num_pitches = 128
hidden_size = 256
hidden_layers = 2

#####################
# LSTM Training Code
#####################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NoteLSTM(num_pitches, num_pitches,
                 hidden_size, hidden_layers).to(device)

training_set = utils.get_training_set(sequence_length)
loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for i in range(epochs):
    print(f"EPOCH {i}")
    print("------------------------")

    total_loss = 0

    for i, (inputs, targets) in enumerate(loader):
        hidden_state = model.init_hidden(inputs.size()[0])

        # (batch_size, sequence_length)
        inputs = inputs.to(device)
        targets = targets.to(device)

        output, hidden_state = model(inputs, hidden_state)

        # Loss function expects (batch_size, feature_dim, sequence_length)
        output = output.permute(0, 2, 1)

        loss = criterion(output, targets)
        total_loss += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {total_loss}\n")

# Save model
torch.save(model.state_dict(), "models/checkpoints/notelstm.model")


#####################
# CNN Training Code
#####################

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = NoteCNN().to(device)

# training_set = utils.get_training_set(sequence_length)
# loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size)

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters())

# for i in range(epochs):
#     print(f"EPOCH {i}")
#     print("------------------------")

#     total_loss = 0

#     for i, (inputs, targets) in enumerate(loader):
#         # (batch_size, sequence_length)
#         inputs = inputs.to(device)
#         targets = targets.to(device)

#         output = model(inputs, hidden_state)

#         loss = criterion(output, targets)
#         total_loss += loss.item()

#         model.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Loss: {total_loss}\n")

# # Save model
# torch.save(model.state_dict(), "models/checkpoints/notelstm.model")
