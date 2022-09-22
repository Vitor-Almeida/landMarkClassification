
from utils.definitions import ROOT_DIR
import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers import pre_tokenizers
from tokenizers import normalizers
from tokenizers.models import WordPiece
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
import numpy as np

def read_experiments(fileName,type):

    path = os.path.join(ROOT_DIR,'lawclassification',fileName)

    df = pd.read_csv(path)

    df = df[df['type']==type]

    return df.to_dict(orient='records')

def hug_tokenizer(vocab_size:int):

    bertTokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    bertTokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
    bertTokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Punctuation('removed'),pre_tokenizers.Whitespace()])

    bertTokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    trainer = WordPieceTrainer(
        vocab_size = vocab_size, 
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        #special_tokens=[],
        min_frequency = 0, 
        show_progress = True, 
        initial_alphabet  = [],
        #continuing_subword_prefix = '##'
        continuing_subword_prefix = ''
    )

    return bertTokenizer, trainer

class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print('early stop count =',self.counter)
            if self.counter >= self.patience:
                return True
        return False