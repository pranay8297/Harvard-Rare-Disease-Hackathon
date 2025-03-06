import random
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class DS(Dataset):
    def __init__(self, data_dict, tokenizer, save_path = None):
        self.data = data_dict

        self.X = list(data_dict.values())
        self.Y = list(data_dict.keys())
        self.Y_values = {y: i for i, y in enumerate(self.Y)}
        self.tokenizer = tokenizer

        if save_path:
            self.save_dataset(save_path)

    def save_dataset(self, save_path):
        print(f'SAVING DATASET AT: {save_path}')
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if len(x) > self.tokenizer.max_len:
            # sample tokenizer.max_len number of items out of this list randomly
            sampled_indices = random.sample(range(len(x)), self.tokenizer.max_len)
            x = [x[i] for i in sampled_indices]

        elif len(x) < self.tokenizer.max_len:
            # sample somewhere between 30% to 70% of x randomly
            sample_percentage = random.uniform(0.2, 0.99)
            sample_size = max(1, int(len(x) * sample_percentage))
            sampled_indices = random.sample(range(len(x)), sample_size)
            x = [x[i] for i in sampled_indices]

        tokens, mask = self.tokenizer.tokenize(x)
        return (tokens, mask), self.Y_values[y]

# Lets first create a dataset
# Actor Network and Critic Network - Symptom and Disease

class POCDS(Dataset):
    def __init__(self, data_dict, tokenizer, Y_values, Y_embeddings):
        self.data = data_dict

        self.X = list(data_dict.values())
        self.Y = list(data_dict.keys())
        self.Y_values = Y_values
        self.Y_embeddings = Y_embeddings

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.Y[idx]

        random.shuffle(x)

        if len(x) >= self.tokenizer.max_len - 1:
            x = x[:self.tokenizer.max_len - 1]

        y_val = self.Y_values[y]
        y_emb = self.Y_embeddings[y_val]

        tokens, mask = self.tokenizer.tokenize(x)
        return (tokens, mask), y_emb

# TODO
class Batch:

    '''
    stores T - current time step
    State - Current X
    All X - Remaining X
    '''
    def __init__(self, phenos: list[list], embs: torch.Tensor,  tokenizer: Tokenizer, max_t = 50):
        import copy

        self.X = copy.deepcopy(phenos)
        self.Y = embs

        self.state_x = None
        self.max_t = max_t
        self.tokenizer = tokenizer

    def get_sample(self):

        for phenotypes, present_state in zip(self.X, self.state_x):
            if len(phenotypes) > 0:
                random.shuffle(phenotypes)
                present_state.append(phenotypes.pop(0))

        # breakpoint()
        tokenized_state_x = []
        tokenized_state_mask = []

        for state in self.state_x:
            state_tokenized, state_mask = self.tokenizer.tokenize(state)
            tokenized_state_x.append(state_tokenized)
            tokenized_state_mask.append(state_mask)

        tokenized_state_x = torch.stack(tokenized_state_x)
        tokenized_state_mask = torch.stack(tokenized_state_mask)
        return (self.t, tokenized_state_x, tokenized_state_mask), self.Y

    def get_next(self):

        if hasattr(self, 't'):
            self.t += 1
        else:
            self.t = 0
            self.state_x = [[] for _ in range(len(self.X))]

        eps_done = self.t > self.max_t
        (current_t, tokenized_state_x, tokenized_state_mask), y = self.get_sample()

        return_obj = namedtuple('StateBatch', ['eps_done', 'current_t', 'tokenized_state_x', 'tokenized_state_mask', 'y'])
        return return_obj(eps_done, current_t, tokenized_state_x, tokenized_state_mask, y)
