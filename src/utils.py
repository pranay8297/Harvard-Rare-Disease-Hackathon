import random
import torch
from data import Batch

def get_batch_obj(size = 32): 
    with open('/content/drive/MyDrive/hrd_hack/disease_to_hpo_with_parents.json', 'r') as f:
        data = json.load(f)

    with open('/content/drive/MyDrive/hrd_hack/ds_test.pkl', 'rb') as f:
        original_ds = pickle.load(f)

    disease_embs = torch.load('/content/drive/MyDrive/hrd_hack/disease_embs.pt')
    disease_embs = disease_embs.detach().cpu()

    new_data = {}
    keys = list(data.keys())
    for i in range(size):
        new_data[keys[i]] = data[keys[i]]

    poc_phenotypes = list(new_data.values())

    y_values = original_ds.Y_values
    diseases = list(new_data.keys())
    disease_ids = [y_values[disease] for disease in diseases]
    poc_disease_embs = disease_embs[disease_ids]

    tokenizer = original_ds.tokenizer
    replay_buffer = Batch(poc_phenotypes, poc_disease_embs, tokenizer)
    return replay_buffer

class RewardFn():
    '''
     -alpha(t) * l2_distance

    where alpha(t) = α_base + α_scale × (t / t_max)
    '''
    def __init__(self, max_steps = 22, threshold = 0.5):
        self.max_steps = max_steps
        self.alpha_base = 0.01
        self.alpha_scale = 0.05
        self.threshold = threshold

    def calculate_reward(self, emb_hat, emb, t):
        distances = torch.dist(emb_hat, emb, p = 2)

        done = distances < self.threshold
        alpha = torch.ones_like(distances, device = emb_hat.device) * self.alpha_base
        emb = emb.to(emb_hat.device)
        
        not_done_mask = ~done
        done_mask = done
        
        alpha[not_done_mask] += self.alpha_scale * (t / self.max_steps)
        
        # alpha[done_mask] += 1.0
        reward = -alpha * distances
        
        return reward, done