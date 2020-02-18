import os
import torch


class Config(object):
    MAX_LENGTH = 10     # Maximum sentence length to consider
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_name = "cornell movie-dialogs corpus"
    corpus = os.path.join("data", corpus_name)
    # Default word tokens
    PAD_token = 0   # Used for padding short sentences
    SOS_token = 1   # Start-of-sentence token
    EOS_token = 2   # End-of-sentence token
    MIN_COUNT = 3   # Minimum word count threshold for trimming
    teacher_forcing_ratio = 1.0
    save_dir = None
