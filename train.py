import pretty_midi
import os
import numpy as np


folder_path = r"B:\Github\Lofi-Generator\venv\Training Data"
midi_data_list = []

for filename in os.listdir(folder_path):
    if filename.endswith('.mid'):
        midi_data = pretty_midi.PrettyMIDI(os.path.join(folder_path, filename))
        midi_data_list.append(midi_data)

all_notes = []
all_durations = []

for midi_data in midi_data_list:
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        if 'piano' in instrument.name.lower():
            notes = [note.pitch for note in instrument.notes]
            durations = [note.end - note.start for note in instrument.notes]
            all_notes.extend(notes)
            all_durations.extend(durations)

print(all_notes)
print(all_durations)

sequence_length = 16  # Adjust as needed
input_sequences = []
output_notes = []
output_durations = []

for i in range(0, len(all_notes) - sequence_length):
    input_seq = all_notes[i:i + sequence_length]
    output_note = all_notes[i + sequence_length]
    output_duration = all_durations[i + sequence_length]
    input_sequences.append(input_seq)
    output_notes.append(output_note)
    output_durations.append(output_duration)

# Create mappings to integer values
note_to_int = {note: i for i, note in enumerate(sorted(set(all_notes)))}
duration_to_int = {duration: i for i, duration in enumerate(sorted(set(all_durations)))}

# Map your sequences to integers
input_sequences = [[note_to_int[note] for note in seq] for seq in input_sequences]
output_notes = [note_to_int[note] for note in output_notes]
output_durations = [duration_to_int[duration] for duration in output_durations]


input_sequences = np.array(input_sequences)
output_notes = np.array(output_notes)
output_durations = np.array(output_durations)
