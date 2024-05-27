# pylint: disable=line-too-long
import torch
from torch import nn

from hot_pytorch.batch.sparse import make_batch_concatenated


class GraphEmbedding(nn.Module):
    def __init__(self, node_classes, node_attributes, edge_attributes, dim_hidden):
        super().__init__()
        self.node_cls_encoder = nn.Embedding(node_classes + 1, dim_hidden // 2)
        self.node_attr_encoder = nn.Linear(node_attributes + 1, dim_hidden // 2, bias=False)
        self.node_bbox_encoder = nn.Linear(4, dim_hidden // 2, bias=False)
        self.edge_attr_encoder = nn.Linear(edge_attributes, dim_hidden // 2, bias=False)

    def forward(self, batch):
        G_target = make_batch_concatenated(
            torch.cat((batch.x[:, None], batch.x_attr, batch.x_bbox), dim=-1),
            batch.edge_index,
            batch.edge_attr,
            batch.node_num,
            batch.edge_num
        )
        G_input = make_batch_concatenated(
            self.node_cls_encoder(batch.masked_x) + self.node_attr_encoder(batch.masked_x_attr) + self.node_bbox_encoder(batch.masked_x_bbox),
            batch.masked_edge_index,
            self.edge_attr_encoder(batch.masked_edge_attr),
            batch.masked_node_num,
            batch.masked_edge_num
        )
        return G_input, G_target
