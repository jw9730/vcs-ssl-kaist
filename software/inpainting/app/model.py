# pylint: disable=line-too-long
from argparse import ArgumentParser

import pytorch_lightning as pl

from embedding import GraphEmbedding
from autoencoder import GraphAutoEncoder


class Model(pl.LightningModule):
    def __init__(self, n_layers, dim_hidden, dim_qk, dim_v, dim_ff, n_heads, input_dropout_rate, dropout_rate,
                 weight_decay, dataset_name, warmup_updates, tot_updates, peak_lr, end_lr, beta1, beta2):
        super().__init__()
        self.save_hyperparameters()
        node_classes = 63
        node_attributes = 209
        edge_attributes = 209

        self.embedding = GraphEmbedding(node_classes, node_attributes, edge_attributes, dim_hidden)
        self.network = GraphAutoEncoder(node_classes, node_attributes, edge_attributes, dim_hidden,
                                        n_layers, dim_qk, dim_v, dim_ff, n_heads, input_dropout_rate, dropout_rate)

        self.dataset_name = dataset_name

        self.evaluator = NotImplemented

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.betas = (beta1, beta2)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, batch):
        G_input, G_target = self.embedding(batch)
        loss_dict = self.network(G_input, G_target)
        return loss_dict

    def predict(self, batch):
        G_input, G_target = self.embedding(batch)
        G_recon = self.network.predict(G_input)
        return G_recon, G_target

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Graph Autoencoder")
        parser.add_argument('--n_layers', type=int, default=4)
        parser.add_argument('--dim_hidden', type=int, default=256)
        parser.add_argument('--dim_qk', type=int, default=256)
        parser.add_argument('--dim_v', type=int, default=256)
        parser.add_argument('--dim_ff', type=int, default=256)
        parser.add_argument('--n_heads', type=int, default=16)
        parser.add_argument('--input_dropout_rate', type=float, default=0.)
        parser.add_argument('--dropout_rate', type=float, default=0.)
        parser.add_argument('--weight_decay', type=float, default=0.)
        parser.add_argument('--checkpoint_path', type=str, default='')
        parser.add_argument('--warmup_updates', type=int, default=6000)
        parser.add_argument('--tot_updates', type=int, default=100000)
        parser.add_argument('--val_check_interval', type=int, default=None)
        parser.add_argument('--check_val_every_n_epoch', type=int, default=None)
        parser.add_argument('--peak_lr', type=float, default=2e-4)
        parser.add_argument('--end_lr', type=float, default=1e-9)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.999)
        parser.add_argument('--validate', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)
        return parent_parser
