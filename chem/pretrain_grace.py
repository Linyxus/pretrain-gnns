import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dropout_adj

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from tensorboardX import SummaryWriter


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)

class GRACE(nn.Module):
    def __init__(self, gnn, mlp):
        super(GRACE, self).__init__()
        self.gnn = gnn
        self.mlp = mlp


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def drop_edge(edge_index: torch.LongTensor, edge_attr: torch.LongTensor, drop_prob: float = 0.2):
    return dropout_adj(edge_index, edge_attr, p=drop_prob)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 batch_size: int, temperature: float):
    def _similarity(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return z1 @ z2.t()
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / temperature)
    indices = torch.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        batch_mask = indices[i * batch_size: (i + 1) * batch_size]
        intra_similarity = f(_similarity(z1[batch_mask], z1))  # [B, N]
        inter_similarity = f(_similarity(z1[batch_mask], z2))  # [B, N]

        positives = inter_similarity[:, batch_mask].diag()
        negatives = intra_similarity.sum(dim=1) + inter_similarity.sum(dim=1) \
                    - intra_similarity[:, batch_mask].diag()

        losses.append(-torch.log(positives / negatives))

    return torch.cat(losses)


def train(args, model, device, loader, optimizer):
    model.train()

    train_loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        optimizer.zero_grad()
        batch = batch.to(device)
        edge_index1, edge_attr1 = drop_edge(batch.edge_index, batch.edge_attr, drop_prob=0.2)
        x1 = drop_feature(batch.x, drop_prob=0.2)
        node_emb1 = model.gnn(x1, edge_index1, edge_attr1)

        edge_index2, edge_attr2 = drop_edge(batch.edge_index, batch.edge_attr, drop_prob=0.2)
        x2 = drop_feature(batch.x, drop_prob=0.2)
        node_emb2 = model.gnn(x2, edge_index2, edge_attr2)

        z1 = model.mlp(node_emb1)
        z2 = model.mlp(node_emb2)

        loss1 = nt_xent_loss(z1, z2, batch_size=1024, temperature=0.1)
        loss2 = nt_xent_loss(z2, z1, batch_size=1024, temperature=0.1)
        loss = (loss1.mean() + loss2.mean()) / 2.0

        loss.backward()

        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())

    return train_loss_accum/step


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=3,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    mlp = torch.nn.Sequential(torch.nn.Linear(args.emb_dim, 2 * args.emb_dim), torch.nn.BatchNorm1d(2 * args.emb_dim),
                              torch.nn.ELU(), torch.nn.Linear(2 * args.emb_dim, args.emb_dim))

    model = GRACE(gnn, mlp)
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train_loss = train(args, model, device, loader, optimizer)

        print(train_loss)


    if not args.output_model_file == "":
        torch.save(gnn.state_dict(), args.output_model_file + ".pth")

if __name__ == "__main__":
    main()
