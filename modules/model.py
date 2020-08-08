import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .init_layer import *
from .transformer import *
from utils.utils import get_mask_from_lengths
import logging
import torch
import hparams as hp


class Duration(nn.Module):            #same as Variance Predictor for FastSpeech2
    def __init__(self, hp):
        super(Duration, self).__init__()
        self.conv1 = Conv1d(hp.hidden_dim,
                            hp.duration_dim,
                            kernel_size=3,
                            padding=1)
        
        self.conv2 = Conv1d(hp.duration_dim,
                            hp.duration_dim,
                            kernel_size=3,
                            padding=1)
        
        self.ln1 = nn.LayerNorm(hp.duration_dim)
        self.ln2 = nn.LayerNorm(hp.duration_dim)
        self.dropout = nn.Dropout(0.5)                             #as per the paper 
        
        self.linear = nn.Linear(hp.duration_dim, 1)

    def forward(self, hidden_states):
        x = F.relu(self.conv1(hidden_states))
        x = self.dropout(self.ln1(x.transpose(1,2)))
        x = F.relu(self.conv2(x.transpose(1,2)))
        x = self.dropout(self.ln2(x.transpose(1,2)))
        out = self.linear(x)
        
        return out.squeeze(-1)

    
class Model(nn.Module):
    def __init__(self, hp):
        super(Model, self).__init__()
        self.hp = hp
        self.Embedding = nn.Embedding(hp.n_symbols, hp.symbols_embedding_dim)
        
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.register_buffer('pe', PositionalEncoding(hp.hidden_dim).pe)
        self.dropout = nn.Dropout(0.2)                                                       #as per the paper

        self.Encoder = nn.ModuleList([TransformerEncoderLayer(d_model=hp.hidden_dim,
                                                              nhead=hp.n_heads,
                                                              dim_feedforward=hp.ff_dim)
                                      for _ in range(hp.n_layers)])
        
        self.Decoder = nn.ModuleList([TransformerEncoderLayer(d_model=hp.hidden_dim,
                                                              nhead=hp.n_heads,
                                                              dim_feedforward=hp.ff_dim)
                                      for _ in range(hp.n_layers)])
        
        self.Duration = Duration(hp)
        self.Pitch = Duration(hp)
        self.Energy = Duration(hp)
        self.Projection = Linear(hp.hidden_dim, hp.n_mel_channels)
        
    def outputs(self, text, durations, text_lengths, mel_lengths):
        #print(text.device, durations.device, text_lengths.device, mel_lengths.device)
        ### Size ###
        B, L, T = text.size(0), text.size(1), mel_lengths.max().item()                                 #alignments.size(1)
        #print("Batch",B,"\nTime Length",L,"\nMax Number of Frames",T)
        
        ### Prepare Inputs ###
        encoder_input = self.Embedding(text).transpose(0,1)
        encoder_input += self.alpha1*(self.pe[:L].unsqueeze(1))
        encoder_input = self.dropout(encoder_input)                                                   # [L,B,256]
        ##print(encoder_input.shape, "Shape of Encoder Input")
        
        ### Prepare Masks ###
        text_mask = get_mask_from_lengths(text_lengths)
        mel_mask = get_mask_from_lengths(mel_lengths)
        #print(text_mask.device, mel_mask.device)
        ### Speech Synthesis ###
        hidden_states = encoder_input
        for layer in self.Encoder:
            hidden_states, _ = layer(hidden_states,
                                     src_key_padding_mask=text_mask)
        ##print(hidden_states.shape,"Shape of Encoder Output")                                          # [L,B,256]
        #durations = self.align2duration(alignments, mel_mask)
        
        duration_out = self.Duration(hidden_states.permute(1,2,0))              #[B,L] passing the encoder output to Duration Module
        ##print(duration_out.shape, "Shape of predicted duration")
        
        hidden_states_expanded = self.LR(hidden_states, durations)              #[T, B, 256]
        ##print(hidden_states_expanded.shape, "Shape LR output")
        
        pitch_out = self.Pitch(hidden_states_expanded.permute(1,2,0))            #passing the expanded hidden states into the Pitch and Energy Modules
        energy_out  = self.Energy(hidden_states_expanded.permute(1,2,0))         # [B,T]
        
        #print(pitch_out.shape,energy_out.shape, "Pitch and Energy Shape" )
        #print(pitch_out.dtype,energy_out.dtype, "Pitch and Energy dtype" )
        
        pitch_one_hot = pitch_to_one_hot(pitch_out)                                  # [B,T,256]
        energy_one_hot = energy_to_one_hot(energy_out)                               # [B,T,256]  
        #print(pitch_one_hot.shape,energy_one_hot.shape, "Pitch and Energy One Hot Shape")
        
        #print(hidden_states_expanded.device, pitch_one_hot.device, energy_one_hot.device, "HS, p, e, device") all cpu as ngpu = 0 in hparams
        
        hidden_states_expanded = hidden_states_expanded + pitch_one_hot.transpose(1,0) + energy_one_hot.transpose(1,0)    #adding all the outputs to collect the decoder input
        
        #print(hidden_states_expanded.shape,"Decoder Input Shape")                # [T, B, 256]
        
        hidden_states_expanded += self.alpha2*(self.pe[:T].unsqueeze(1))
        hidden_states_expanded = self.dropout(hidden_states_expanded)
        
        for layer in self.Decoder:
            hidden_states_expanded, _ = layer(hidden_states_expanded,
                                              src_key_padding_mask=mel_mask)
        
        #print(hidden_states_expanded.shape,"Decoder Output Shape")                 #[T,B,256]
        
        mel_out = self.Projection(hidden_states_expanded.transpose(0,1)).transpose(1, 2)
        
        #print(mel_out.shape,"Output Mel Shape")          #[10,80,833] [B, num_mel, T]                         
        
        return mel_out, duration_out, durations, pitch_out, energy_out    
     
    
    def forward(self, text, melspec, durations, text_lengths, mel_lengths, pitch, energy, criterion):
        
	#print(text.device, melspec.device, durations.device, text_lengths.device, mel_lengths.device, pitch.device, energy.device)
	### Size ###
        text = text[:,:text_lengths.max().item()]                        #no need for this maybe [B, L]
        melspec = melspec[:,:,:mel_lengths.max().item()]                #no need for this maybe [B, 80, T]
        ##alignments = alignments[:,:mel_lengths.max().item(),:text_lengths.max().item()]
	#print(text.device, durations.device, text_lengths.device, mel_lengths.device)
        mel_out, duration_out, durations, pitch_out, energy_out = self.outputs(text.cuda(), durations.cuda(), text_lengths.cuda(), mel_lengths.cuda())
        mel_loss, duration_loss, pitch_loss, energy_loss = criterion((mel_out, duration_out, pitch_out, energy_out),
                                            (melspec, durations, pitch, energy),
                                            (text_lengths, mel_lengths))
        
        return mel_loss, duration_loss, pitch_loss, energy_loss
    
    
    def inference(self, text, alpha=1.0):
        
        #input - text_sequence (ndarray) (109,)
        
        
        ### Prepare Inference ###
        text_lengths = torch.tensor([text.shape[0]])       # [L]                              #torch.tensor([1, text.size(1)])
        text = text.unsqueeze(0) # .from_numpy(text)         #[1, L]
        print(text.shape)
        ### Prepare Inputs ###
        encoder_input = self.Embedding(text).transpose(0,1)
        encoder_input += self.alpha1*(self.pe[:text.size(1)].unsqueeze(1))
        print(encoder_input.shape)
        ### Speech Synthesis ###
        hidden_states = encoder_input
        text_mask = text.new_zeros(1,text.size(1)).to(torch.bool)
        for layer in self.Encoder:
            hidden_states, _ = layer(hidden_states,
                                     src_key_padding_mask=text_mask)
        print(hidden_states.shape, "output of Encoder")
    
        ### Duration Predictor ###
        durations = self.Duration(hidden_states.permute(1,2,0))
        print(durations.shape, "shape of durations")
        hidden_states_expanded = self.LR(hidden_states, durations, alpha, inference=True)
        print(hidden_states_expanded.shape, "hidden states expanded")
        pitch = self.Pitch(hidden_states_expanded.permute(1,2,0))
        energy = self.Energy(hidden_states_expanded.permute(1,2,0))
        print(pitch.shape, "P S")
        print(energy.shape, "E S")
        pitch_one_hot = pitch_to_one_hot(pitch, False)
        energy_one_hot = energy_to_one_hot(energy, False)
        print(hidden_states_expanded.shape, "hidden states expanded after e and p encoding")
        
        hidden_states_expanded = hidden_states_expanded + pitch_one_hot.transpose(1,0) + energy_one_hot.transpose(1,0)       #check for all device attributes
        print(hidden_states_expanded.shape, pitch_one_hot.shape, energy_one_hot.shape)
        hidden_states_expanded += self.alpha2*self.pe[:hidden_states_expanded.size(0)].unsqueeze(1)
        
        print(hidden_states_expanded.shape, "hidden states expanded after positional encoding")
        
        mel_mask = text.new_zeros(1, hidden_states_expanded.size(0)).to(torch.bool)
        print(mel_mask.shape, "Shape of mel Mask")
        print(hidden_states_expanded.shape)
        for layer in self.Decoder:
            hidden_states_expanded, _ = layer(hidden_states_expanded,
                                              src_key_padding_mask=mel_mask)
        
        mel_out = self.Projection(hidden_states_expanded.transpose(0,1)).transpose(1,2)
        
        return mel_out, durations, pitch, energy

    
    '''def align2duration(self, alignments, mel_mask):
        ids = alignments.new_tensor( torch.arange(alignments.size(2)) )
        max_ids = torch.max(alignments, dim=2)[1].unsqueeze(-1)
        
        one_hot = 1.0*(ids==max_ids)
        one_hot.masked_fill_(mel_mask.unsqueeze(2), 0)
        
        durations = torch.sum(one_hot, dim=1)

        return durations
    '''

    def LR(self, hidden_states, durations, alpha=1.0, inference=False):
        L, B, D = hidden_states.size()
        durations = torch.round(durations*alpha).to(torch.long)
        if inference:
            durations[durations<=0]=1
        T=int(torch.sum(durations, dim=-1).max().item())
        print(T, "Number of mel frames")
        expanded = hidden_states.new_zeros(T, B, D)
        
        for i, d in enumerate(durations):
            mel_len = torch.sum(d).item()
            expanded[:mel_len, i] = torch.repeat_interleave(hidden_states[:,i],
                                                            d,
                                                            dim=0)
        print(expanded.shape, "Shape of expanded after LR")
        return expanded

def energy_to_one_hot(e, is_inference = False, is_log_output = False, offset = 1):                                        #check for scale
    # e = de_norm_mean_std(e, hp.e_mean, hp.e_std)
    # For pytorch > = 1.6.0
    bins = torch.linspace(hp.e_min, hp.e_max, steps=255).to(torch.device("cuda" if hp.ngpu > 0 else "cpu"))
    if is_inference and is_log_output:
        e = torch.clamp(torch.round(e.exp() - offset), min=0).long()
        
    e_quantize = bucketize(e.to(torch.device("cuda" if hp.ngpu > 0 else "cpu")), bins)

    return F.one_hot(e_quantize.long(), 256).float()
    
    
def pitch_to_one_hot(f0, is_inference = False, is_log_output = False, offset = 1):
    # Required pytorch >= 1.6.0
    # f0 = de_norm_mean_std(f0, hp.f0_mean, hp.f0_std)
    bins = torch.exp(torch.linspace(np.log(hp.p_min), np.log(hp.p_max), 255)).to(torch.device("cuda" if hp.ngpu > 0 else "cpu"))
    if is_inference and is_log_output:
        f0 = torch.clamp(torch.round(f0.exp() - offset), min=0).long()
        
    p_quantize = bucketize(f0.to(torch.device("cuda" if hp.ngpu > 0 else "cpu")), bins)
    #p_quantize = p_quantize - 1  # -1 to convert 1 to 256 --> 0 to 255
    return F.one_hot(p_quantize.long(), 256).float()

def bucketize(tensor, bucket_boundaries):
    result = torch.zeros_like(tensor, dtype=torch.int64)
    for boundary in bucket_boundaries:
        result += (tensor > boundary).int()
    return result
