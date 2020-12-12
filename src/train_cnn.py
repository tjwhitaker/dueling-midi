import numpy as np
import torch
from models import CNN
import utils
from time import time

epochs = 10
sequence_length = 32
batch_size = 32

#####################
# CNN Training Code
#####################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)

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

epoch_times = []

for i in range(epochs):
    start_time = time()

    print(f"EPOCH {i}")
    print("------------------------")

    train_loss = 0
    test_loss = 0

    # Train
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)

        # CNN Doesn't batch targets like lstm
        # Need to create single note targets instead of sequence
        # Should probably do this at the dataset level instead of here
        targets = targets[0:, -1].to(device)

        output = model(inputs)

        loss = criterion(output, targets)
        train_loss += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets[0:, -1].to(device)

            output = model(inputs)

            loss = criterion(output, targets)
            test_loss += loss.item()

    print(f"Train Loss: {train_loss}")
    print(f"Test Loss: {test_loss}\n")

    epoch_times.append(time() - start_time)

# Save model
torch.save(model.state_dict(), "../models/cnn.model")

print(epoch_times)
