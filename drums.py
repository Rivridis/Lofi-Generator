# Imports
import wave
import numpy as np
import random

# Global variables
bpm = 50
beats_per_measure = 4 
measure_duration = 60.0 / bpm 
time_between_beats = measure_duration / beats_per_measure

# Opening files
Ch = wave.open("B://Github//Lofi-Generator//sounds//CH.wav", 'rb') #0
Oh = wave.open("B://Github//Lofi-Generator//sounds//OH.wav", 'rb') #1
kick = wave.open("B://Github//Lofi-Generator//sounds//kick.wav", 'rb') #2
perc = wave.open("B://Github//Lofi-Generator//sounds//perc.wav", 'rb') #3
snare = wave.open("B://Github//Lofi-Generator//sounds//snare.wav", 'rb') #4

# Output file
output_filename = "output.wav"
output_wave = wave.open(output_filename, 'wb')
output_wave.setparams(Ch.getparams())


# Play only x seconds of each beat
sample_rate = Ch.getframerate()
num_frames = int(time_between_beats * sample_rate)

# Silence
num_frames_silence = int(time_between_beats * sample_rate)
silence_data = np.zeros((num_frames_silence, Ch.getnchannels()), dtype=np.int16)
silence = silence_data.tobytes()

# Getting audio data
Ch_data = Ch.readframes(num_frames)
Oh_data = Oh.readframes(num_frames)
kick_data = kick.readframes(num_frames)
perc_data = perc.readframes(num_frames)
snare_data = snare.readframes(num_frames)

# Drum Beats

# beats = [2,0,0,0,2,0,0,0,2,0,0,0,0,0,2,0,0,0,0,0,2,0,0,0,0,0,2,0,0,0,0,0]
# for i in range(len(beats)):
#     if beats[i]!= 2:
#         beats[i] = random.randint(0,5)

def generate_drum_beat_with_markov(transition_matrix, sequence_length):
    current_note = 2
    beat = [current_note]

    for _ in range(sequence_length - 1):
        next_note = random.choices(
            list(transition_matrix[current_note].keys()),
            weights=transition_matrix[current_note].values())[0]
        beat.append(next_note)
        current_note = next_note

    return beat

# Define the transition matrix
transition_probabilities = {
    0: {0: 0.4, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1},
    1: {0: 0.2, 1: 0.4, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1},
    2: {0: 0.1, 1: 0.1, 2: 0.4, 3: 0.1, 4: 0.2, 5: 0.1},
    3: {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.4, 4: 0.2, 5: 0.1},
    4: {0: 0.1, 1: 0.1, 2: 0.2, 3: 0.2, 4: 0.3, 5: 0.1},
    5: {0: 0.1, 1: 0.1, 2: 0.2, 3: 0.1, 4: 0.1, 5: 0.4}
}

sequence_length = 32
beats = generate_drum_beat_with_markov(transition_probabilities, sequence_length)
# print(beats)

# Writing
for i in beats:
    if i == 0:
        output_wave.writeframes(Ch_data)
    elif i == 1:
        output_wave.writeframes(Oh_data)
    elif i == 2:
        output_wave.writeframes(kick_data)
    elif i == 3:
        output_wave.writeframes(perc_data)
    elif i == 4:
        output_wave.writeframes(snare_data)
    elif i == 5:
        output_wave.writeframes(silence)
print("Beat Created")

# Closing 
Ch.close()
Oh.close()
kick.close()
perc.close()
snare.close()
output_wave.close()
