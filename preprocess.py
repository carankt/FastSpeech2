import os
import glob
import tqdm
import torch
import argparse
import numpy as np
import hparams as hp
from stft import TacotronSTFT
from utils.utils import read_wav_np
from audio_processing import pitch
from text import phonemes_to_sequence

def main(args):
    stft = TacotronSTFT(filter_length=hp.n_fft,
                        hop_length=hp.hop_length,
                        win_length=hp.win_length,
                        n_mel_channels=hp.n_mels,
                        sampling_rate=hp.sampling_rate,
                        mel_fmin=hp.fmin,
                        mel_fmax=hp.fmax)
    # wav_file loacation 
    wav_files = glob.glob(os.path.join(args.wav_root_path, '**', '*.wav'), recursive=True)
    
    #Define all the paths correesponding to the feature
    text_path = os.path.join(hp.data_path, 'text')
    mel_path = os.path.join(hp.data_path, 'mels')
    duration_path = os.path.join(hp.data_path, 'alignment')
    energy_path = os.path.join(hp.data_path, 'energy')
    pitch_path = os.path.join(hp.data_path, 'pitch')
    symbol_path = os.path.join(hp.data_path, 'symbol')
    
    # create directory if doesnt exist
    os.makedirs(text_path,exist_ok = True)
    os.makedirs(duration_path, exist_ok = True)
    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(energy_path, exist_ok=True)
    os.makedirs(pitch_path, exist_ok=True)
    os.makedirs(symbol_path, exist_ok=True)
    
    for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to mel, energy, and pitch'):
        sr, wav = read_wav_np(wavpath)
        p = pitch(wav)  # [T, ] T = Number of frames
        wav = torch.from_numpy(wav).unsqueeze(0)      
        mel, mag = stft.mel_spectrogram(wav) # mel [1, 80, T]  mag [1, num_mag, T]
        mel = mel.squeeze(0) # [num_mel, T]
        mag = mag.squeeze(0) # [num_mag, T]
        e = torch.norm(mag, dim=0) # [T, ]
        p = p[:mel.shape[1]]
        p = np.array(p, dtype='float32')
        id = os.path.basename(wavpath).split(".")[0]
        
        # save the features
        np.save('{}/{}.npy'.format(mel_path,id), mel.numpy(), allow_pickle=False)
        np.save('{}/{}.npy'.format(energy_path, id), e.numpy(), allow_pickle=False)
        np.save('{}/{}.npy'.format(pitch_path, id), p , allow_pickle=False)
        
        
    
    with open(hp.filelist_alignment_dir + "alignment.txt", encoding='utf-8') as f:      #add all 13100 examples to filelist.txt 
        for lines in f:
            content = lines.split('|')
            id = content[4].split()[0].split('.')[0]
            if os.path.exists(os.path.join(args.wav_root_path, id + '.wav')):
                text = content[0]
                duration = content[2]
                duration = duration.split()
                dur = np.array(duration, dtype = 'float32')         #done
                phoneme = content[3]
                symbol_sequence = phonemes_to_sequence(phoneme)      
            
                np.save('{}/{}.npy'.format(text_path, id), (text, phoneme), allow_pickle=False) #what is the input text or phonemen???
                np.save('{}/{}.npy'.format(duration_path, id), dur, allow_pickle=False)
                np.save('{}/{}.npy'.format(symbol_path, id), symbol_sequence, allow_pickle=False)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--wav_root_path', type=str, required=True,
                        help="root directory of wav files")
    args = parser.parse_args()

    main(args)
