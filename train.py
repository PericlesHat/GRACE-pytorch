# ---- coding: utf-8 ----
# @author: Ziyang Zhang
# @description: GRACE training script


import argparse
import torch
import numpy as np
import warnings
import os

os.environ["OMP_NUM_THREADS"] = "11"
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster")

from sklearn.cluster import KMeans
from torch.optim import Adam, RMSprop

from graph import Graph
from model import GRACE
from tqdm import tqdm
from utils import f1_community, jc_community



def train_model(args):

    # read data
    base_dir = args.base_dir + str(args.dataset) + '/'
    feature_file = base_dir + 'feature.txt'
    edge_file = base_dir + 'edge.txt'
    cluster_file = base_dir + 'cluster.txt'
    graph = Graph(base_dir, feature_file, edge_file, cluster_file, args.alpha, args.lambda_)

    # create model
    model = GRACE(args, graph, graph.cluster.shape[1]).to(args.device)

    # init optimizer
    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    elif args.optimizer == 'RMSProp':
        optimizer = RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    else:
        raise NotImplementedError()


    if args.transition_function == 'RI':
        model.RI = torch.Tensor(graph.RI).to(args.device)
    elif args.transition_function == 'RW':
        model.RW = torch.Tensor(graph.RW).to(args.device)
    else:
        raise NotImplementedError()

    # pre-train
    losses = []
    with tqdm(total=args.pre_epoch, desc='Pre-Training Epochs') as pbar:
        for epoch in range(args.pre_epoch):
            loss = model.build_loss_r(model.decode())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            pbar.update(1)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.pre_epoch + 1), losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Function Over Iterations')
    plt.grid(True)
    plt.show() if args.plot else None

    # init Q by K-means
    Z = model.get_embedding().cpu().detach().numpy()
    kmeans = KMeans(n_clusters=graph.cluster.shape[1]).fit(Z)
    model.init_mean(torch.Tensor(kmeans.cluster_centers_).to(args.device))


    # train
    losses = []
    with tqdm(total=args.epoch, desc='Training Epochs') as pbar:
        for epoch in range(args.epoch):
            loss = model.build_loss()
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            pbar.update(1)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epoch + 1), losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Function Over Iterations')
    plt.grid(True)
    plt.show() if args.plot else None

    prediction = model.predict().cpu()

    prediction, ground_truth = np.transpose(prediction), np.transpose(graph.cluster)

    f1score = f1_community(prediction, ground_truth).item()
    jcscore = jc_community(prediction, ground_truth).item()

    print(f"F1 Score: {f1score}")
    print(f"JC Score: {jcscore}")


    return 0

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for GRACE')

    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--base_dir', type=str, default='data/', help='Data dir')
    parser.add_argument('--feat_dim', type=int, default=-1, help='Feature dimension')
    parser.add_argument('--embed_dim', type=int, default=728, help='Embedding dimension')
    parser.add_argument('--encoder_hidden', nargs='+', type=int, default=[728], help='Encoder hidden layer dimension')
    parser.add_argument('--decoder_hidden', nargs='+', type=int, default=[728], help='Decoder hidden layer dimension')
    parser.add_argument('--transition_function', type=str, default='RI', help='Transition function [T, RI, RW]')
    parser.add_argument('--random_walk_step', type=int, default=0, help=None)
    parser.add_argument('--keep_prob', type=float, default=0.5, help='Keep probability of dropout')
    parser.add_argument('--alpha', type=float, default=0.9, help='Damping coefficient for propagation process')
    parser.add_argument('--lambda_', type=float, default=0.1)
    parser.add_argument('--BN', type=bool, default=False, help='Apply batch normalization')
    parser.add_argument('--lambda_r', type=float, default=1.0, help='Reconstruct loss coefficient')
    parser.add_argument('--lambda_c', type=float, default=0.1, help='Clustering loss coefficient')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer [Adam, Momentum, GradientDescent, RMSProp, Adagrad]')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help=None)
    parser.add_argument('--pre_epoch', type=int, default=1000, help=None)
    parser.add_argument('--epoch', type=int, default=1500, help=None)
    parser.add_argument('--epsilon', type=float, default=1.0, help='Annealing hyperparameter for cluster assignment')
    parser.add_argument('--dataset', type=str, default='citeseer', help=None)
    parser.add_argument('--plot', type=bool, default=False, help=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
