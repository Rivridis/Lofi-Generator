import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from music21 import converter, instrument, note, chord


class MusicDataset(Dataset):
    def __init__(self, notes, sequence_length):
        self.sequence_length = sequence_length
        self.notes = notes
        self.pitchnames = sorted(set(item for item in notes))
        self.note_to_int = dict((note, number) for number, note in enumerate(self.pitchnames))
        self.network_input, self.network_output = self.create_sequences()

    def __len__(self):
        return len(self.network_input)

    def __getitem__(self, idx):
        return self.network_input[idx], self.network_output[idx]

    def create_sequences(self):
        network_input, network_output = [], []
        for i in range(0, len(self.notes) - self.sequence_length, 1):
            in_seq = self.notes[i:i + self.sequence_length]
            out_seq = self.notes[i + self.sequence_length]
            network_input.append([self.note_to_int[char] for char in in_seq])
            network_output.append(self.note_to_int[out_seq])
        network_input = np.reshape(network_input, (len(network_input), self.sequence_length, 1))
        return (
            torch.tensor(network_input, dtype=torch.float32),
            torch.tensor(network_output, dtype=torch.long),
        )

class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MusicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Take only the last output
        out = self.fc1(out)
        return out
    
class MusicLightning(LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.model = MusicLSTM(input_size, hidden_size, num_layers, output_size)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.RMSprop(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        return {"loss": loss}
# Load and preprocess the data
notes = []
for file in glob.glob("venv/Data/*.mid"):
    midi = converter.parse(file)
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)
    if parts: # file has instrument parts
        notes_to_parse = parts[0].recurse()
    else: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

sequence_length = 100
dataset = MusicDataset(notes, sequence_length)
model = MusicLightning(input_size=1, hidden_size=256, num_layers=3, output_size=len(dataset.pitchnames))

# Load the saved weights
model.load_state_dict(torch.load('my_model.pt'))

# Assuming 'model' is your PyTorch/Lightning model and 'dataset' is your MusicDataset instance
model.eval()  # Set the model to evaluation mode

start = torch.randint(0, len(dataset.network_input)-1, (1,))
int_to_note = dict((number, note) for number, note in enumerate(dataset.pitchnames))
pattern = dataset.network_input[start]
print(pattern)

prediction_output = []

# Generate 500 notes
for note_index in range(500):
    prediction_input = pattern

    # Add a check to ensure pattern is not empty
    if prediction_input.numel() > 0:
    
        prediction_input = prediction_input / len(dataset.pitchnames)  # Assuming n_vocab is the number of unique notes

        # Pass the input through the model
        with torch.no_grad():
            prediction = model(prediction_input)

        # Get the index of the predicted note
        index = torch.argmax(prediction)

        # Convert index to note
        result = int_to_note[int(index.item())]
        prediction_output.append(result)

        # Update the pattern for the next iteration
        new_note = torch.tensor([[[index]]])  # Convert index to tensor
        pattern = torch.cat((pattern, new_note), dim=1)
        pattern = pattern[:, 1:]
    else:
        # Handle the case where pattern is empty (e.g., with a default action or break the loop)
        print("empty")

print(prediction_output)




