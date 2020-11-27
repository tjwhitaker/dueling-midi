import numpy as np
import torch
from models import Encoder, Decoder
import utils

epochs = 20
sequence_length = 32
batch_size = 32

#####################
# Seq2Seq Training Code
#####################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder().to(device)
decoder = Decoder().to(device)

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

encoder_optimizer = torch.optim.Adam(encoder.parameters())
decoder_optimizer = torch.optim.Adam(decoder.parameters())


for i in range(epochs):
    print(f"EPOCH {i}")
    print("------------------------")

    train_loss = 0
    test_loss = 0

    # Train
    encoder.train()
    decoder.train()

    for i, (inputs, targets) in enumerate(train_loader):
        hidden_state = encoder.init_hidden(inputs.size()[0])

        inputs = inputs.to(device)
        targets = torch.cat((targets, torch.zeros((targets.shape[0], 1))),
                            1).long().to(device)

        decoder_inputs = torch.roll(
            targets, shifts=1, dims=1).long().to(device)

        encoder_output, hidden_state = encoder(inputs, hidden_state)
        decoder_output, hidden_state = decoder(decoder_inputs, hidden_state)

        # Loss function expects (batch_size, feature_dim, sequence_length)
        decoder_output = decoder_output.permute(0, 2, 1)

        loss = criterion(decoder_output, targets)
        train_loss += loss.item()

        encoder.zero_grad()
        decoder.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    # Test
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            hidden_state = encoder.init_hidden(inputs.size()[0])

            inputs = inputs.to(device)
            targets = torch.cat(
                (targets, torch.zeros((targets.shape[0], 1))), 1).long().to(device)

            decoder_inputs = torch.roll(
                targets, shifts=1, dims=1).long().to(device)

            encoder_output, hidden_state = encoder(inputs, hidden_state)
            decoder_output, hidden_state = decoder(
                decoder_inputs, hidden_state)

            # Loss function expects (batch_size, feature_dim, sequence_length)
            decoder_output = decoder_output.permute(0, 2, 1)

            loss = criterion(decoder_output, targets)
            test_loss += loss.item()

    print(f"Train Loss: {train_loss}")
    print(f"Test Loss: {test_loss}\n")

# Save model
torch.save(encoder.state_dict(), "../models/encoder.model")
torch.save(decoder.state_dict(), "../models/decoder.model")
