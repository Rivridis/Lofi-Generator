import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from music21 import converter, instrument, note, chord

torch.set_float32_matmul_precision('medium')

def to_categorical(y, num_classes=None, dtype="float32"):

    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def get_notes():

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
    return notes

def prepare_sequences(notes, n_vocab):
    sequence_length = 32
    pitchnames = sorted(set(item for item in notes))
    global note_to_int 
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # Create input sequences and corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.array(network_input)
    network_output = np.array(network_output)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab) # type: ignore
    network_output = to_categorical(network_output)
    
    return network_input, network_output


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import numpy as np

class MusicGenerationModel(pl.LightningModule):
    def __init__(self, n_vocab):
        super(MusicGenerationModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=512, batch_first=True, dropout=0.3, bidirectional=False)
        self.lstm2 = nn.LSTM(512, 512, batch_first=True, dropout=0.3, bidirectional=False)
        self.lstm3 = nn.LSTM(512, 512, batch_first=True, dropout=0.3, bidirectional=False)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512, 256)
        self.activation = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, n_vocab)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]  # Only take the output of the last time step
        x = self.batchnorm1(x)
        x = self.batchnorm2(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.batchnorm3(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class MusicGenerationLightning(pl.LightningModule):
    def __init__(self, network_input, network_output, n_vocab):
        super(MusicGenerationLightning, self).__init__()
        self.model = MusicGenerationModel(n_vocab)
        self.network_input = network_input
        self.network_output = network_output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # Use CrossEntropyLoss for multi-class classification
        loss = nn.functional.cross_entropy(y_hat, y)

        return {'loss': loss}

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters())

    def train_dataloader(self):
        dataset = TensorDataset(self.network_input, self.network_output)
        return DataLoader(dataset, batch_size=64, shuffle=True)


def train_network():
    notes = get_notes()
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab)

    network_input = torch.Tensor(network_input).float()  # Convert to torch.Tensor and float type
    network_output = torch.Tensor(network_output).float()  # Convert to torch.Tensor and long type

    print(n_vocab)



    pl_model = MusicGenerationLightning(network_input, network_output, n_vocab)

    trainer = Trainer(max_epochs=50)
    trainer.fit(pl_model)


    torch.save(pl_model.state_dict(), 'my_model.pt')

def generate_sequence(model, start_sequence, int_to_note, n_vocab, seq_length=500):
    pattern = start_sequence
    prediction_output = []

    for _ in range(seq_length):
        prediction_input = torch.from_numpy(pattern).unsqueeze(0).unsqueeze(2).float()  # Assuming pattern is a numpy array
        prediction_input /= float(n_vocab)
        prediction = model(prediction_input)
        
        index = torch.argmax(prediction).item()
        result = int_to_note[index]
        prediction_output.append(result)

        # Append index instead of the actual note
        pattern = np.append(pattern, index)
        pattern = pattern[1:]

    return prediction_output

if __name__ == '__main__':
    #train_network()
    notes = get_notes()
    n_vocab = len(set(notes))
    model = MusicGenerationModel(n_vocab)
    model.load_state_dict(torch.load('my_model.pt'),strict=False)
    model.eval()  # Set the model to evaluation mode

    network_input,out = prepare_sequences(notes,n_vocab)
    network_input = torch.tensor(network_input, dtype=torch.float32)

    int_to_note = dict((number, note) for number, note in enumerate(range(n_vocab)))

    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]

    prediction_output = []

    # Generate 500 notes
    for note_index in range(500):
        prediction_input = pattern

        # Add a check to ensure pattern is not empty
        if prediction_input.numel() > 0:
        
            prediction_input = prediction_input / len(range(n_vocab))  # Assuming n_vocab is the number of unique notes

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