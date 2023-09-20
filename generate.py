# Imports
import lstm
import torch
import random
from music21 import instrument, note, chord
import music21.stream.base as stream

# Global Variables
notes = lstm.parse_notes()
sequence = lstm.create_seq(notes)
pitchnames = sorted(set(item for item in notes))
seq_length,input_size,hidden_size,num_layers,output_size,batch_size,num_epochs = lstm.params()
offset = 0
output_notes = []
prediction_output = []

# Temperature
def sample_with_temperature(logits, temperature=0.8):
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def cust_seq():
    lst = ['A5','E5','D5','G5','C6','E6','D6','A5','G5','C6']
    lst2 = []
    for i in lst:
        l = lstm.note_to_int(notes)
        k = l.get(i)
        lst2.append(k)
    return lst2

# Generate
model = lstm.LSTMModel(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('lstm_model.pth')) #change to pt
model.eval()

with torch.no_grad():
    random_start_idx = random.randint(0, len(sequence) - seq_length - 1)
    start_sequence = torch.tensor(sequence[random_start_idx:random_start_idx+seq_length]).unsqueeze(0).unsqueeze(2).float()
    #start_sequence = torch.tensor(cust_seq()).unsqueeze(0).unsqueeze(2).float()

    for _ in range(200): 
        output = model(start_sequence)
        predicted_note = sample_with_temperature(output[0]).item()
        start_sequence = torch.cat((start_sequence[:, 1:, :], torch.tensor([[[predicted_note]]]).float()), 1)
        val = pitchnames[int(predicted_note)]
        prediction_output.append(val)

print(prediction_output)

# Convert notes to music
for pattern in prediction_output:
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='test_output.mid')
