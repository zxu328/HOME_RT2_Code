# LSTM auto-encoder model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# LSTM encoder class
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        

    def forward(self, input):
        encoded_input, hidden = self.lstm(input)
        return encoded_input

# LSTM decoder class
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(output_size, output_size)
        self.act = nn.LeakyReLU()
        
        
    def forward(self, encoded_input):
        
        decoded_output, hidden = self.lstm(encoded_input)    
        
        return decoded_output

# LSTM variational autoencoder
class LSTMVAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer_en, num_layer_de):
        super(LSTMVAE, self).__init__()
        self.encoderMean = EncoderRNN(input_size, hidden_size, num_layer_en)
        self.encoderVar = EncoderRNN(input_size, hidden_size, num_layer_en)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layer_de)
        
    def reparameter(self, mu, logVar):
        std = torch.exp(0.5 * logVar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, input):
        mu = self.encoderMean(input)
        logVar = self.encoderVar(input)
        encoded_input = self.reparameter(mu, logVar)
        decoded_output = self.decoder(encoded_input)
        return decoded_output, mu, logVar
    
    
# LSTM autoencoder
class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer_en, num_layer_de):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layer_en)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layer_de)
        
    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output
    
    