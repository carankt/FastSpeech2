from text import valid_symbols

################################
# MFA File Preparation        #
################################
wav_path = lab_path = './wavs'
csv_path = 'metadata.csv'
dict_path = 'cmudict.txt'



################################
# Experiment Parameters        #
################################
seed=1234
n_gpus= ngpu = 1
output_directory = 'training_log'
log_directory = 'fastspeech2-phone'
data_path = data_dir = './preprocessed/'
filelist_alignment_dir=teacher_dir = './MFA_filelist/'

training_files='filelists/ljs_audio_text_train_filelist.txt'
validation_files='filelists/ljs_audio_text_val_filelist.txt'
text_cleaners=['english_cleaners']


################################
# Audio Parameters             #
################################
sample_rate = sampling_rate=22050
filter_length=1024
hop_length=256
win_length=1024
n_mels=n_mel_channels=80
mel_fmin=0.0
mel_fmax=8000.0
n_fft = 1024
fmin = 0.0
fmax = 8000.0


################################
# Model Parameters             #
################################
n_symbols=len(valid_symbols)
data_type='phone_seq'
symbols_embedding_dim=256
hidden_dim=256
dprenet_dim=256
postnet_dim=256
ff_dim=1024
duration_dim=256
n_heads=4
n_layers=4 #as per the paper 
e_min = 0
e_max = 314.9619140625
p_min = 0
p_max = 795.7948608398438

################################
# Optimization Hyperparameters #
################################
lr=384**-0.5                        # ~384^-0.5 = 0.05
warmup_steps=4000
grad_clip_thresh=1.0
batch_size=8
accumulation=2
iters_per_validation=2000
iters_per_checkpoint=10000
train_steps = 200000
distillation=True
pretrained_embedding=False
