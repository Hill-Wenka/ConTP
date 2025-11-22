import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConTP(nn.Module):
    def __init__(self, hidden_dim, out_dim, esm_dim=1280, dropout=0.1, temperature=0.1):
        super(ConTP, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.esm_dim = esm_dim

        self.fc1 = nn.Linear(esm_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.temperature = temperature

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x

    def sup_con_hard_loss(self, model_emb, n_pos, temp=None):
        '''
        return the SupCon-Hard loss
        model_emb:
            model output embedding, dimension [bsz, n_all, out_dim],
            where bsz is batch size,
            n_all is anchor, pos, neg (n_all = 1 + n_pos + n_neg)
            and out_dim is embedding dimension
        temp:
            temperature
        n_pos:
            number of positive examples per anchor
        '''
        temp = temp if temp is not None else self.temperature
        # l2 normalize every embedding
        features = F.normalize(model_emb, dim=-1, p=2)
        # features_T is [bsz, outdim, n_all], for performing batch dot product
        features_T = torch.transpose(features, 1, 2)
        # anchor is the first embedding
        anchor = features[:, 0]
        # anchor is the first embedding
        anchor_dot_features = torch.bmm(anchor.unsqueeze(1), features_T) / temp
        # anchor_dot_features now [bsz, n_all], contains
        anchor_dot_features = anchor_dot_features.squeeze(1)
        # deduct by max logits, which will be 1/temp since features are L2 normalized
        logits = anchor_dot_features - 1 / temp
        # the exp(z_i dot z_a) excludes the dot product between itself
        # exp_logits is of size [bsz, n_pos+n_neg]
        exp_logits = torch.exp(logits[:, 1:])
        exp_logits_sum = n_pos * torch.log(exp_logits.sum(1))  # size [bsz], scale by n_pos
        pos_logits_sum = logits[:, 1:n_pos + 1].sum(1)  # sum over all (anchor dot pos)
        log_prob = (pos_logits_sum - exp_logits_sum) / n_pos
        loss = - log_prob.mean()
        return loss

    def compute_cluster_center(self, cache_file, return_sample_embed=False):
        class_embed_dict = np.load(cache_file, allow_pickle=True).item()

        unique_labels = sorted(class_embed_dict.keys())
        concat_embed = [v for k, v in class_embed_dict.items()]
        concat_embed = np.concatenate(concat_embed, axis=0)
        concat_embed = torch.from_numpy(concat_embed).to(device=self.fc1.weight.device)
        concat_embed = self.forward(concat_embed)

        if return_sample_embed:
            cluster_labels = []
            start_idx = 0
            for i, label in enumerate(unique_labels):
                end_idx = start_idx + len(class_embed_dict[label])
                cluster_labels.extend([label] * len(class_embed_dict[label]))
                start_idx = end_idx
            cluster_labels = np.array(cluster_labels)
            return concat_embed, cluster_labels
        else:
            cluster_embeddings = torch.zeros([max(unique_labels) + 1, self.out_dim], device=concat_embed.device)
            start_idx = 0
            # some labels are missing, e.g. label=1 is missing
            # to avoid the position shift, use label as index
            for i, label in enumerate(unique_labels):
                end_idx = start_idx + len(class_embed_dict[label])
                cluster_emb = concat_embed[start_idx:end_idx]
                cluster_embeddings[label] = cluster_emb.mean(0)
                start_idx = end_idx
            return cluster_embeddings

    def find_nearest_cluster(self, predict_embed, cluster_embed, return_dist=False):
        predict_embed = F.normalize(predict_embed, dim=-1, p=2)
        cluster_embed = F.normalize(cluster_embed, dim=-1, p=2)
        dist = torch.cdist(predict_embed, cluster_embed, p=2)
        pred = torch.argmin(dist, dim=1).cpu().detach().tolist()
        if return_dist:
            return pred, dist
        else:
            return pred
