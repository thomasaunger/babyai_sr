import torch
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
import torch.nn as nn
import torch.nn.functional as F
import math

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ExpertControllerFiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        
        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)
        
        self.apply(initialize_parameters)
    
    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)
        out = self.bn2(out)
        out = F.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, embedding_size, enc_dim, num_symbols):
        super().__init__()
        self.lstm = nn.LSTM(num_symbols, enc_dim, batch_first=True)
    
    def forward(self, inputs):
        h, c = self.lstm(inputs)
        
        msg = h[:, -1, :]
        
        return msg

class Decoder(nn.Module):
    def __init__(self, embedding_size, dec_dim, len_msg, num_symbols):
        super().__init__()
        self.lstm   = nn.LSTM(embedding_size, dec_dim, batch_first=True)
        self.linear = nn.Linear(dec_dim, num_symbols)
        
        self.embedding_size = embedding_size
        self.len_msg        = len_msg
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        
        h, c   = self.lstm(inputs.expand(self.len_msg, batch_size, self.embedding_size).transpose(0, 1))
        logits = self.linear(h)
        
        dists_speaker = OneHotCategorical(logits=F.log_softmax(logits, dim=2))
        
        return dists_speaker

class ACModel(nn.Module):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128, enc_dim=128, dec_dim=128,
                 len_msg=8, num_symbols=8):
        super().__init__()
        
        self.image_dim    = image_dim
        self.memory_dim   = memory_dim
        self.instr_dim    = instr_dim
        self.enc_dim      = enc_dim
        self.dec_dim      = dec_dim
        self.len_msg      = len_msg
        self.num_symbols  = num_symbols
        
        self.obs_space = obs_space

        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Define instruction embedding
        self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
        gru_dim = self.instr_dim
        self.instr_rnn = nn.GRU(
            self.instr_dim, gru_dim, batch_first=True,
            bidirectional=False
        )
        self.final_instr_dim = self.instr_dim
        
        # Define memory
        self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)
        
        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        
        mod0 = ExpertControllerFiLM(
            in_features=self.final_instr_dim+self.enc_dim,
            out_features=128, in_channels=128, imm_channels=128
        )
        mod1 = ExpertControllerFiLM(
            in_features=self.final_instr_dim+self.enc_dim, out_features=self.image_dim,
            in_channels=128, imm_channels=128
        )
        self.controllers = [mod0, mod1]
        self.add_module('FiLM_Controler_0', mod0)
        self.add_module('FiLM_Controler_1', mod1)
        
        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )
        
        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Define encoder
        self.encoder = Encoder(self.embedding_size, self.enc_dim, self.num_symbols)
        
        # Define decoder
        self.decoder = Decoder(self.embedding_size, self.dec_dim, self.len_msg, self.num_symbols)
        
        # Initialize parameters correctly
        self.apply(initialize_parameters)
    
    @property
    def memory_size(self):
        return 2 * self.semi_memory_size
    
    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, msg=None):
        device = torch.device("cuda" if obs.instr.is_cuda else "cpu")
        batch_size = obs.image.size(0)
        
        # pad image to be large enough for convolution
        n = obs.image.size(1)
        m = obs.image.size(2)
        if n < 7 or m < 7:
            # place image into middle-right of padded image
            if n < 7 and m < 7:
                image_new = torch.zeros((batch_size, 7, 7, 3), device=device)
                offset = math.floor((7-n)/2)
            elif n < 7:
                image_new = torch.zeros((batch_size, 7, m, 3), device=device)
                offset = math.floor((7-n)/2)
            else: # m < 7
                image_new = torch.zeros((batch_size, n, 7, 3), device=device)
                offset = 0
            image_new[:, offset:offset+n, -m:, :] = obs.image
            obs.image = image_new
        
        instr_embedding = self._get_instr_embedding(obs.instr)

        if msg is None:
            msg = torch.zeros(batch_size, self.len_msg, self.num_symbols, device=device)

        msg_embedding = self.encoder(msg)
        
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        
        x = self.image_conv(x)
        for controler in self.controllers:
            x = controler(x, torch.cat((instr_embedding, msg_embedding), dim=-1))
        x = F.relu(self.film_pool(x))
            
        # take mean
        x = x.mean(-1).mean(-1)
        
        x = x.reshape(x.shape[0], -1)
        
        hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        hidden = self.memory_rnn(x, hidden)
        embedding = hidden[0]
        memory = torch.cat(hidden, dim=1)
        
        x = self.actor(embedding)
        
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)
        
        dists_speaker = self.decoder(embedding)
        
        return {"dist": dist, "dists_speaker": dists_speaker, "value": value, "memory": memory}

    def _get_instr_embedding(self, instr):
        _, hidden = self.instr_rnn(self.word_embedding(instr))
        return hidden[-1]

    def speaker_log_prob(self, dists_speaker, msg):
        return dists_speaker.log_prob(msg).sum(-1)
    
    def speaker_entropy(self, dists_speaker):
        return dists_speaker.entropy().sum(-1)
