import numpy as np
import torch
from models import NoteCNN
import utils

epochs = 20
sequence_length = 64
batch_size = 64


#####################
# CNN Training Code
#####################

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

model = NoteCNN().to(device)

training_set = utils.get_training_set(sequence_length)
loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for i in range(epochs):
    print(f"EPOCH {i}")
    print("------------------------")

    total_loss = 0

    for i, (inputs, targets) in enumerate(loader):

        inputs = inputs.to(device)

        # CNN Doesn't batch targets like lstm.
        # Need to create single note targets instead of sequence
        targets = targets[0:sequence_length, -1].to(device)

        output = model(inputs)

        loss = criterion(output, targets)
        total_loss += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {total_loss}\n")

# Save model
torch.save(model.state_dict(), "../models/notecnn.model")
