import torch
import torch.nn as nn
from utils.utils import get_mask_from_lengths

class TransformerLoss(nn.Module):
    def __init__(self):
        super(TransformerLoss, self).__init__()
        
    def forward(self, pred, target, lengths):
        mel_out, duration_out, pitch_out, energy_out = pred
        mel_target, duration_target, pitch_target, energy_target = target
        text_lengths, mel_lengths = lengths
        
        assert(mel_out.shape == mel_target.shape)                  #check for the shape
        assert(duration_out.shape == duration_target.shape)
        assert(pitch_out.shape == pitch_out.shape)
        assert(energy_out.shape == energy_target.shape)
        
        
        mel_mask = ~get_mask_from_lengths(mel_lengths)        # same for pitch and energy
        duration_mask = ~get_mask_from_lengths(text_lengths)
        
        #print(mel_mask.shape,duration_mask.shape, "Shape of mel mask and duration mask")
        #print(mel_mask.unsqueeze(1).shape)      # [B, 1, T]
        
        mel_target = mel_target.masked_select(mel_mask.unsqueeze(1))
        mel_out = mel_out.masked_select(mel_mask.unsqueeze(1))

        duration_target = duration_target.masked_select(duration_mask)
        duration_out = duration_out.masked_select(duration_mask)
        
        pitch_target = pitch_target.masked_select(mel_mask.unsqueeze(1))
        pitch_out = pitch_out.masked_select(mel_mask.unsqueeze(1))
        
        energy_target = energy_target.masked_select(mel_mask.unsqueeze(1))
        energy_out = energy_out.masked_select(mel_mask.unsqueeze(1))
        
        mel_loss = nn.L1Loss()(mel_out, mel_target)
        duration_loss = nn.MSELoss()(duration_out, duration_target)
        pitch_loss = nn.MSELoss()(pitch_out, pitch_target)
        energy_loss = nn.MSELoss()(energy_out, energy_target)
        
        return mel_loss, duration_loss, pitch_loss, energy_loss