from torch import nn, optim
from models import MelodyLSTM

# Get Data
# Folder of midi files

# Preprocess data if needed
# Embeddings?

epochs = 10
model = MelodyLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam()

model = MelodyLSTM()

for _ in range(epochs):
    for signal in data:
        model.zero_grad()

        x, y = process_midi(signal)

        result = model(x)

        loss = criterion(result, y)

        loss.backward()
        optimizer.step()
