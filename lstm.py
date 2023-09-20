# Imports
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from music21 import converter, instrument, note, chord
import torch.nn.functional as F

# Functions and classes
def parse_notes():
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

def note_to_int(notes):
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    return(note_to_int)
    
def create_seq(notes):
    sequence = []

    for i in notes:
        l = note_to_int(notes)
        k = l.get(i)
        sequence.append(k)
    return sequence

def params():
    notes = parse_notes()
    pitchnames = sorted(set(item for item in notes))
    seq_length = 100  
    input_size = 1
    hidden_size = 128
    num_layers = 2
    output_size = len(pitchnames)
    batch_size = 64
    num_epochs = 100
    return seq_length,input_size,hidden_size,num_layers,output_size,batch_size,num_epochs

class MusicDataset(Dataset):
    def __init__(self, sequence, seq_length):
        self.sequence = sequence
        self.seq_length = seq_length

    def __len__(self):
        return len(self.sequence) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequence[idx:idx+self.seq_length]),
            torch.tensor(self.sequence[idx+self.seq_length])
        )

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm = nn.BatchNorm1d(hidden_size) 

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.batchnorm(out)
        out = F.relu(out)
        out = self.fc(out)
        return out

