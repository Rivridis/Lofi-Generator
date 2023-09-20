# Imports
import lstm
import torch

# Variables
notes = lstm.parse_notes()
sequence = lstm.create_seq(notes)
seq_length,input_size,hidden_size,num_layers,output_size,batch_size,num_epochs = lstm.params()

dataset = lstm.MusicDataset(sequence, seq_length)
dataloader = lstm.DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = lstm.LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = lstm.nn.CrossEntropyLoss()
optimizer = lstm.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(num_epochs):
    loss = []
    for batch_input, batch_target in dataloader:
        optimizer.zero_grad()
        output = model(batch_input.unsqueeze(2).float())
        loss = criterion(output, batch_target)
        loss.backward()
        optimizer.step()
        if epoch%10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'lstm_model.pt')