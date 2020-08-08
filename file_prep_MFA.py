from g2p_en import G2p
import os
g2p = G2p()
from text.cleaners import english_cleaners, punctuation_removers
import hparams

class MFA():
  def __init__(self, hparams):
    return

  def MFA_file_prep(self, hparams):
    ''' Will Create .lab files for all existing wav files in the hparams.wav_path folder.
    The .lab files will contain the cleaned or normalized text (expanded abbreviations, remove punctuations, expand number, to upper case) from csv_file
    The function will also check if the words in content are there in dict or not
    And will create a dictionary {alien_words: phoneme(alien_words)} using g2p_en
      input - hparams
      output - new_word_dictionary
    '''

    filenames = []
    content = []
    update_words = {}

    with open(hparams.csv_path,  encoding='utf-8') as f:
      for lines in f:
        filenames.append(lines.split("|")[0])
        content.append(lines.split('|')[1])
    words = self.load_words_from_dict(hparams)

    for i in range(0, len(filenames)):
        if os.path.exists(f'{hparams.lab_path}/{filenames[i]}.wav'):
          path = os.path.join(hparams.lab_path, filenames[i] + ".lab")
          clean_content = english_cleaners(content[i])
          clean_content = punctuation_removers(clean_content) # add remove punctuations
          f = open(path, 'w+')
          f.write(clean_content.upper())
          f.close()
          alien = set(clean_content.upper().split()) - set(words)
          alien_update = {i: (g2p(i)) for i in list(alien)}
          update_words = {**update_words, **alien_update }
        
    if update_words:
      print("update your dictionary using update_dict() of this class")
    else:
      print("No dictionary update required")
    return update_words #words to update
  
  def update_dict(self, hparams, alien_update):
    '''will append the new words and their pronounciations at the end of the old dictionary'''
    with open(hparams.dict_path, 'a') as f:
      for k, v in alien_update.items():
        print(k, *v, file = f)
      f.close()
    return

  def load_words_from_dict(self, hparams):
    words = []
    with open(hparams.dict_path, 'r') as f:
      for lines in f:
        w = lines.split()[0]
        words.append(w)
    return words