import numpy as np
import torch
from models import NoteGRU
import utils

epochs = 20
sequence_length = 32
batch_size = 32

#####################
# GRU Training Code
#####################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NoteGRU().to(device)

dataset = utils.get_training_set(sequence_length)

# Split data into random 80/20 train/test
indices = list(range(len(dataset)))
split = int(np.floor(0.2 * len(dataset)))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=test_sampler)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for i in range(epochs):
    print(f"EPOCH {i}")
    print("------------------------")

    train_loss = 0
    test_loss = 0

    # Train
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        hidden_state = model.init_hidden(inputs.size()[0])

        inputs = inputs.to(device)
        targets = targets.to(device)

        output, hidden_state = model(inputs, hidden_state)

        # Loss function expects (batch_size, feature_dim, sequence_length)
        output = output.permute(0, 2, 1)

        loss = criterion(output, targets)
        train_loss += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()

    # Test
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            hidden_state = model.init_hidden(inputs.size()[0])

            inputs = inputs.to(device)
            targets = targets.to(device)

            output, hidden_state = model(inputs, hidden_state)

            # Loss function expects (batch_size, feature_dim, sequence_length)
            output = output.permute(0, 2, 1)

            loss = criterion(output, targets)
            test_loss += loss.item()

    print(f"Train Loss: {train_loss}")
    print(f"Test Loss: {test_loss}\n")

# Save model
torch.save(model.state_dict(), "../models/notegru.model")
