import torch
import random
import pickle

class Tokenizer():
    def __init__(self, pheno_list, tokenizer_path = None, max_len = 256, save_path = None):

        if tokenizer_path:
            try:
                with open(tokenizer_path, 'rb') as f:
                    self = pickle.load(f)
                    return
            except:
                print('TOKENIZER NOT FOUND, INITIALIZING NEW ONE')

        self.key_value = {
            'CLS_KEY': 0,
            'PAD_KEY': 1,
            'UNK': 2
        }

        self.max_len = max_len
        self.phenotype_list = pheno_list
        for idx, pheno in enumerate(pheno_list):
            self.key_value[pheno] = idx + 3

        self.value_key = {v: k for k, v in self.key_value.items()}
        if save_path:
            self.save_tokenizer(save_path)

    def get_key(self, pheno):
        try:
            return self.key_value[pheno]
        except:
            return self.key_value['UNK']

    def save_tokenizer(self, tokenizer_path):
        print(f'SAVING TOKENIZER AT: {tokenizer_path}')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self, f)

    def _tokenize(self, pheno_list):
        return [self.key_value[pheno] if pheno in self.key_value else self.key_value['UNK'] for pheno in pheno_list]

    def pad(self, pheno_list):
        pheno_list = ['CLS_KEY'] + pheno_list
        delta = max(0, (self.max_len - len(pheno_list)))
        if len(pheno_list) < self.max_len:
            pheno_list += ['PAD_KEY'] * (self.max_len - len(pheno_list))
        else:
            pheno_list = pheno_list[:self.max_len]

        assert len(pheno_list) == self.max_len
        return pheno_list, delta

    def tokenize(self, pheno_list):
        # 2 things
        # 1. Tokenized Pheno List
        # 2. Pad Mask
        padded_pheno_list, n_padded = self.pad(pheno_list)
        tokenized_pheno_list = torch.tensor(self._tokenize(padded_pheno_list)).to(torch.long)
        pad_mask = torch.tensor([1] * (self.max_len - n_padded) + [0] * n_padded).to(torch.long)

        # Verification
        assert len(tokenized_pheno_list) == self.max_len
        assert len(pad_mask) == self.max_len

        return tokenized_pheno_list, pad_mask

    def reverse_tokenize(self, tokenized_pheno_list):
        return [self.value_key[token] for token in tokenized_pheno_list]