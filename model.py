# ---- coding: utf-8 ----
# @author: Ziyang Zhang
# @description: GRACE implementation in PyTorch



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Encoder(nn.Module):
    def __init__(self, input_dim, encoder_hidden, embed_dim, keep_prob):
        super(Encoder, self).__init__()

        # Initialize parameters and build layers
        self.encoder_hidden = encoder_hidden + [embed_dim]
        self.keep_prob = keep_prob

        # Creating fully connected layers
        self.layers = nn.ModuleList()
        for i in range(len(self.encoder_hidden)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, self.encoder_hidden[i]))
            else:
                self.layers.append(nn.Linear(self.encoder_hidden[i - 1], self.encoder_hidden[i]))

    def forward(self, x):
        hidden = x
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden.detach().clone())
            hidden = F.dropout(hidden.detach().clone(), p=self.keep_prob)
        return hidden


class Decoder(nn.Module):
    def __init__(self, embed_dim, decoder_hidden, feat_dim, keep_prob):
        super(Decoder, self).__init__()

        # Initialize parameters and build layers
        self.decoder_hidden = decoder_hidden
        self.feat_dim = feat_dim
        self.keep_prob = keep_prob

        # Creating fully connected layers
        self.layers = nn.ModuleList()
        for i in range(len(self.decoder_hidden)):
            if i == 0:
                self.layers.append(nn.Linear(embed_dim, self.decoder_hidden[i]))
            else:
                self.layers.append(nn.Linear(self.decoder_hidden[i - 1], self.decoder_hidden[i]))
        self.output_layer = nn.Linear(self.decoder_hidden[-1], self.feat_dim)


    def forward(self, x):
        hidden = x
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            hidden = F.dropout(hidden.clone(), p=self.keep_prob)
        return self.output_layer(hidden)


class GRACE(nn.Module):
    def __init__(self, paras, graph, num_cluster):
        super(GRACE, self).__init__()
        
        self.paras = paras
        self.X = torch.Tensor(graph.feature).to(paras.device)
        # Initialize parameters and build layers
        self.feat_dim = graph.feature.shape[1]
        
        # Store the tensors for random walk outgoing and influence propagation
        self.T_indices = torch.LongTensor(graph.indices).to(paras.device)
        self.T_values = torch.FloatTensor(graph.T_values).to(paras.device)
        self.RI = torch.Tensor(graph.RI).to(paras.device)
        self.RW = torch.Tensor(graph.RW).to(paras.device)
        
        # Mean tensor and cluster assignment matrix, using HE initialization
        self.mean = nn.Parameter(torch.Tensor(num_cluster, self.paras.embed_dim).to(paras.device))
        init.kaiming_normal_(self.mean, mode='fan_in', nonlinearity='relu')

        # Initialize Encoder and Decoder
        self.encoder = Encoder(input_dim=self.feat_dim,
                               encoder_hidden=self.paras.encoder_hidden,
                               embed_dim=self.paras.embed_dim,
                               keep_prob=self.paras.keep_prob).to(paras.device)
        
        self.decoder = Decoder(embed_dim=self.paras.embed_dim,
                               decoder_hidden=self.paras.decoder_hidden,
                               feat_dim=self.feat_dim,
                               keep_prob=self.paras.keep_prob).to(paras.device)
        # self.Z = self.encode()
        # self.Z_transform = self.transform()
        # self.Q = self.build_Q()



    def encode(self):
        hidden = self.encoder(self.X)
        return hidden

    def decode(self):
        hidden = self.transform()

        return self.decoder(hidden)

    def transform(self):
        Z = self.encode()
        transition_function = self.paras.transition_function
        if transition_function == 'T':
            for i in range(self.paras.random_walk_step):
                Z = torch.sparse.mm(self.T, Z.t()).t()
        elif transition_function in ['RI', 'RW']:
            Z = torch.matmul(getattr(self, transition_function).T.float(), Z)
        else:
            raise ValueError('Invalid transition function')

        if self.paras.BN:
            bn = nn.BatchNorm1d(Z.shape[1]).to(self.paras.device)
            Z = bn(Z)

        return Z

    """ calculate Q: init and each ? """
    def build_Q(self):
        Z = self.Z_transform.unsqueeze(1)  # Shape: [N, 1, dim]
        diff = Z - self.mean  # Broadcasting subtraction
        squared_norm = torch.sum(diff ** 2, dim=2)  # Shape: [N, K]
        Q = torch.pow(squared_norm / self.paras.epsilon + 1.0, -(self.paras.epsilon + 1.0) / 2.0)
        return Q / torch.sum(Q, dim=1, keepdim=True)

    """ loss for reconstruction of X """
    def build_loss_r(self, X_p):
        loss_r = F.binary_cross_entropy_with_logits(X_p, self.X)
        return loss_r

    """ loss for clustering """
    def build_loss_c(self, P):
        loss_c = torch.mean(P * torch.log(P / self.Q))
        return loss_c

    """ overall loss """
    def build_loss(self):
        self.Z_transform = self.transform()
        self.Q = self.build_Q()
        X_p = self.decode()
        loss_r = self.build_loss_r(X_p)
        loss_c = self.build_loss_c(self.get_P())
        return self.paras.lambda_r * loss_r + self.paras.lambda_c * loss_c

    """ get embedding Z' """
    def get_embedding(self):
        return self.transform()

    """ init mean from KMEANS """
    def init_mean(self, mean):
        self.mean = nn.Parameter(mean).to(self.paras.device)

    """ calculate P: each epoch, only called outside """
    def get_P(self):
        f_k = torch.sum(self.Q, dim=0)
        numerator = self.Q**2 / f_k
        denominator_terms = self.Q ** 2 / f_k.unsqueeze(0)
        denominator = torch.sum(denominator_terms, dim=1, keepdim=True)
        return numerator / denominator

    """ predict cluster label """
    def predict(self):
        indices = torch.argmax(self.Q, dim=1)
        one_hot = F.one_hot(indices, num_classes=self.Q.shape[1])
        return one_hot

