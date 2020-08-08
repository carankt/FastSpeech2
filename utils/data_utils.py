import random
import numpy as np
import hparams
import torch
import torch.utils.data
import os

def load_filepaths_and_text(metadata, data_path, split="|"):
    filepaths_and_text = []
    with open(metadata, encoding='utf-8') as f:  #read data from the csv file
        for line in f:
            file_name, text1, text2 = line.strip().split('|')
            if os.path.exists(f'{data_path}/energy/{file_name}.npy'):  #check for the npy file if it exists in the energy folder, assuming exact same files in all e,o,m folders
                filepaths_and_text.append( (file_name, text1, text2) )
    return filepaths_and_text


class TMDPESet(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text, hparams.data_path)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self.data_type=hparams.data_type

    def get_TMDPE_pair(self, audiopath_and_text):
        # separate filename and text
        file_name = audiopath_and_text[0][0:10]
        #set path for all the features
        seq = os.path.join(hparams.data_path, 'symbol')  #updated the path
        m = os.path.join(hparams.data_path, 'mels')
        d = os.path.join(hparams.data_path, 'alignment')
        p = os.path.join(hparams.data_path, 'pitch')
        e = os.path.join(hparams.data_path, 'energy')
        #load data corresponding to each file_name
        text = np.load(f'{seq}/{file_name}.npy')
        mel = np.load(f'{m}/{file_name}.npy')
        duration = np.load(f'{d}/{file_name}.npy')
        pitch = np.load(f'{p}/{file_name}.npy')
        energy = np.load(f'{e}/{file_name}.npy')
        
        '''
        with open(f'{seq}/{file_name}_sequence.pkl', 'rb') as f:
            text = pkl.load(f)
        
        if hparams.distillation==True:
            with open(f'{hparams.teacher_dir}/targets/{file_name}.pkl', 'rb') as f:
                mel = pkl.load(f)
        else:
            with open(f'{mel}/{file_name}_melspectrogram.pkl', 'rb') as f:
                mel = pkl.load(f)
                
        with open(f'{hparams.teacher_dir}/alignments/{file_name}.pkl', 'rb') as f:
            alignments = pkl.load(f)
        '''
        return (text, mel, duration, pitch, energy) #return the tuple containing TMDPE

    def __getitem__(self, index):
        return self.get_TMDPE_pair(self.audiopaths_and_text[index])  #index of the file you want the feature vector of

    def __len__(self):
        return len(self.audiopaths_and_text)


class TMDPECollate():
    def __init__(self):
        return

    def __call__(self, batch):
        
        # type = Tensor, dtype = torch.float32 for all the elements in the tuple
        # Right zero-pad all one-hot symbol sequences to max input length
        
        
        input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([len(x[2]) for x in batch]),dim=0, descending=True)
        max_input_len = input_lengths[0]
        text_padded = torch.zeros(len(batch), max_input_len, dtype=torch.int64)
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.shape[0]] = torch.tensor(text) # Tensor (batch size, max_input_len )
        
        
        #Mel 2-D padding
        num_mels = batch[0][1].shape[0]     #num of mel filters
        max_target_len = max([x[1].shape[1] for x in batch])   #max mel frames in the batch
        mel_padded = torch.zeros(len(batch), num_mels, max_target_len, dtype=torch.float32) #(batch size, num of filter banks, max_target_len)
        output_lengths = torch.LongTensor(len(batch))  
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]                       
            mel_padded[i, :, :mel.shape[1]] = torch.tensor(mel)             
            output_lengths[i] = mel.shape[1]    #retain the original number of frames in mel
        mel_pad = mel_padded.view(-1,num_mels, max_target_len)
        
        #alignment padding
        input_duration_lengths, _ = torch.sort(torch.LongTensor([len(x[2]) for x in batch]),dim=0, descending=True)
        max_duration_input_len = input_duration_lengths[0]   #max duration
        duration_padded = torch.zeros(len(batch), max_duration_input_len, dtype=torch.float32)
        for i in range(len(ids_sorted_decreasing)):
            duration = batch[ids_sorted_decreasing[i]][2]
            duration_padded[i, :duration.shape[0]] = torch.tensor(duration)  # Tensor (batch size, max duration)
            
        #pitch padding
        pitch_padded = torch.zeros(len(batch), max_target_len, dtype=torch.float32)
        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]][3]
            pitch_padded[i, :pitch.shape[0]] = torch.tensor(pitch)  # Tensor (batch size, max frames or max_target_len)
            
        #energy padding
        energy_padded = torch.zeros(len(batch), max_target_len, dtype=torch.float32)
        for i in range(len(ids_sorted_decreasing)):
            energy = batch[ids_sorted_decreasing[i]][4]
            energy_padded[i, :energy.shape[0]] = torch.tensor(energy) # Tensor (batch size, max frames or max_target_len) 
        
        assert(energy_padded.shape == pitch_padded.shape)
        
        return text_padded, input_lengths, mel_pad, output_lengths, duration_padded, pitch_padded, energy_padded