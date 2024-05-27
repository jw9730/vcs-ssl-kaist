# pylint: disable=line-too-long,not-callable
import torch
from torch import nn

from hot_pytorch.models import Encoder
from hot_pytorch.batch.sparse import Batch as S, make_batch
from hot_pytorch.batch.dense import Batch as D, apply, batch_like


class GraphAutoEncoder(nn.Module):
    def __init__(self, node_classes, node_attributes, edge_attributes, dim_hidden,
                 n_layers, dim_qk, dim_v, dim_ff, n_heads, input_dropout_rate, dropout_rate):
        super().__init__()
        self.node_classes = node_classes
        self.node_attributes = node_attributes
        self.edge_attributes = edge_attributes
        ord_hidden = [2] * n_layers
        self.encoder = Encoder(2, 1, ord_hidden, dim_hidden, dim_hidden, dim_hidden, dim_qk, dim_v, dim_ff, n_heads,
                               0, 0, 0, 'default', 'default', input_dropout_rate, dropout_rate, sparse=True)
        self.decoder = Encoder(1, 2, ord_hidden, dim_hidden, dim_hidden, dim_hidden, dim_qk, dim_v, dim_ff, n_heads,
                               0, 0, 0, 'default', 'default', input_dropout_rate, dropout_rate, sparse=False)
        self.output = nn.Linear(dim_hidden, 1 + node_classes + node_attributes + 4 + edge_attributes)

    @staticmethod
    def _loss_adj(G_logits: D, G_target: S) -> D:
        bsize, n, device = G_logits.A.size(0), G_logits.A.size(1), G_logits.device
        adj_logits = G_logits.A[..., 0]  # [B, N, N]
        adj_target = torch.zeros(bsize, n, n, device=device)  # [B, N, N]
        for idx, (i, m) in enumerate(zip(G_target.indices, G_target.mask)):  # [|E|, 2],  [|E|,]
            m = m & (i[:, 0] != i[:, 1])  # [|E|,]
            adj_target[idx, i[m, 0], i[m, 1]] = 1
        loss = nn.functional.binary_cross_entropy_with_logits(adj_logits, adj_target, reduction='none')  # [B, N, N]
        loss = loss[..., None] * (1 - torch.eye(n, device=device))[None, :, :, None]
        return batch_like(G_logits, loss)

    def _loss_node_cls(self, G_logits: D, G_target: S) -> D:
        bsize, n, device = G_logits.A.size(0), G_logits.A.size(1), G_logits.device
        node_logits = G_logits.A[..., 1:1 + self.node_classes].permute(0, 3, 1, 2)  # [B, node classes, N, N]
        node_target = torch.zeros(bsize, n, n, dtype=torch.long, device=device)  # [B, N, N]
        for idx, (i, v, m) in enumerate(zip(G_target.indices, G_target.values[..., 0], G_target.mask)):  # [|E|, 2], [|E|,], [|E|,]
            m = m & (i[:, 0] == i[:, 1])  # [|E|,]
            node_target[idx, i[m, 0], i[m, 1]] = v[m].long()
        loss = nn.functional.cross_entropy(node_logits, node_target, reduction='none')  # [B, N, N]
        loss = loss[..., None] * torch.eye(n, device=device)[None, :, :, None]
        return batch_like(G_logits, loss)

    def _loss_node_attr(self, G_logits: D, G_target: S) -> D:
        bsize, n, device = G_logits.A.size(0), G_logits.A.size(1), G_logits.device
        node_logits = G_logits.A[..., 1 + self.node_classes:1 + self.node_classes + self.node_attributes]  # [B, N, N, node attributes]
        node_target = torch.zeros(bsize, n, n, self.node_attributes, device=device)  # [B, N, N, node attributes]
        for idx, (i, v, m) in enumerate(zip(G_target.indices, G_target.values[..., 1:1+self.node_attributes], G_target.mask)):  # [|E|, 2], [|E|, node attributes], [|E|,]
            m = m & (i[:, 0] == i[:, 1])  # [|E|,]
            node_target[idx, i[m, 0], i[m, 1]] = v[m].float()
        loss = nn.functional.binary_cross_entropy_with_logits(node_logits, node_target, reduction='none')  # [B, N, N, node attributes]
        loss = loss * torch.eye(n, device=device)[None, :, :, None]
        return batch_like(G_logits, loss)

    def _loss_node_bbox(self, G_logits: D, G_target: S) -> D:
        bsize, n, device = G_logits.A.size(0), G_logits.A.size(1), G_logits.device
        node_logits = G_logits.A[..., 1 + self.node_classes + self.node_attributes:1 + self.node_classes + self.node_attributes + 4]
        node_target = torch.zeros(bsize, n, n, 4, device=device)  # [B, N, N, 4]
        for idx, (i, v, m) in enumerate(zip(G_target.indices, G_target.values[..., 1 + self.node_attributes:1 + self.node_attributes + 4], G_target.mask)):  # [|E|, 2], [|E|, 4], [|E|,]
            m = m & (i[:, 0] == i[:, 1])  # [|E|,]
            node_target[idx, i[m, 0], i[m, 1]] = v[m].float()
        loss = nn.functional.mse_loss(node_logits.sigmoid(), node_target, reduction='none')  # [B, N, N, 4]
        loss = loss * torch.eye(n, device=device)[None, :, :, None]
        return batch_like(G_logits, loss)

    def _loss_edge_attr(self, G_logits: D, G_target: S) -> D:
        bsize, n, device = G_logits.A.size(0), G_logits.A.size(1), G_logits.device
        edge_logits = G_logits.A[..., -self.edge_attributes:]  # [B, N, N, edge attributes]
        edge_target = torch.zeros(bsize, n, n, self.edge_attributes, device=device)  # [B, N, N, edge attributes]
        for idx, (i, v, m) in enumerate(zip(G_target.indices, G_target.values[..., -self.edge_attributes:], G_target.mask)):  # [|E|, 2], [|E|, edge attributes], [|E|,]
            m = m & (i[:, 0] != i[:, 1])  # [|E|,]
            edge_target[idx, i[m, 0], i[m, 1]] = v[m].float()
        loss = nn.functional.binary_cross_entropy_with_logits(edge_logits, edge_target, reduction='none')  # [B, N, N, edge attributes]
        loss = loss * (1 - torch.eye(n, device=device))[None, :, :, None]
        return batch_like(G_logits, loss)

    def _compute_loss(self, G_logits: D, G_target: S):
        loss_adj = self._loss_adj(G_logits, G_target)  # [B, N, N, 1]
        loss_node_cls = self._loss_node_cls(G_logits, G_target)  # [B, N, N, 1]
        loss_node_attr = self._loss_node_attr(G_logits, G_target)  # [B, N, N, node attributes]
        loss_node_bbox = self._loss_node_bbox(G_logits, G_target)  # [B, N, N, 4]
        loss_edge_attr = self._loss_edge_attr(G_logits, G_target)  # [B, N, N, edge attributes]
        loss_adj = loss_adj.A.mean(dim=0).sum()
        loss_node_cls = loss_node_cls.A.mean(dim=0).sum()
        loss_node_attr = loss_node_attr.A.mean(dim=0).sum()
        loss_node_bbox = loss_node_bbox.A.mean(dim=0).sum()
        loss_edge_attr = loss_edge_attr.A.mean(dim=0).sum()
        return {
            'loss_adj': loss_adj,
            'loss_node_cls': loss_node_cls,
            'loss_node_attr': loss_node_attr,
            'loss_node_bbox': loss_node_bbox,
            'loss_edge_attr': loss_edge_attr,
            'loss': loss_adj + loss_node_cls + loss_node_attr + loss_node_bbox + loss_edge_attr
        }

    def forward(self, G: S, G_target: S):
        h = self.encoder(G)
        h = self.decoder(D(h.values, h.n_nodes))
        G_logits = apply(h, self.output)
        return self._compute_loss(G_logits, G_target)

    @staticmethod
    def _decode_adj(G_logits: D) -> D:
        adj_logits = G_logits.A[..., 0]
        n, device = adj_logits.size(1), adj_logits.device
        adj = adj_logits.sigmoid().round()
        adj = adj[..., None] * (1 - torch.eye(n, device=device, dtype=torch.long))[None, :, :, None]
        return batch_like(G_logits, adj)

    def _decode_node_cls(self, G_logits: D) -> D:
        node_logits = G_logits.A[..., 1:1 + self.node_classes]
        n, device = node_logits.size(1), node_logits.device
        node_cls = node_logits.argmax(dim=-1)
        node_cls = node_cls[..., None] * torch.eye(n, device=device, dtype=torch.long)[None, :, :, None]
        return batch_like(G_logits, node_cls)

    def _decode_node_attr(self, G_logits: D) -> D:
        node_logits = G_logits.A[..., 1 + self.node_classes:1 + self.node_classes + self.node_attributes]
        n, device = node_logits.size(1), node_logits.device
        node_attr = torch.nn.functional.one_hot(node_logits.argmax(dim=-1), num_classes=node_logits.size(-1)) + node_logits.sigmoid().round()
        node_attr = node_attr.clamp(0, 1)
        node_attr = node_attr * torch.eye(n, device=device, dtype=torch.long)[None, :, :, None]
        return batch_like(G_logits, node_attr)

    def _decode_node_bbox(self, G_logits: D) -> D:
        node_logits = G_logits.A[..., 1 + self.node_classes + self.node_attributes:1 + self.node_classes + self.node_attributes + 4]
        node_bbox = node_logits.sigmoid()
        return batch_like(G_logits, node_bbox)

    def _decode_edge_attr(self, G_logits: D) -> D:
        edge_logits = G_logits.A[..., -self.edge_attributes:]
        n, device = edge_logits.size(1), edge_logits.device
        edge_attr = torch.nn.functional.one_hot(edge_logits.argmax(dim=-1), num_classes=edge_logits.size(-1)) + edge_logits.sigmoid().round()
        edge_attr = edge_attr.clamp(0, 1)
        edge_attr = edge_attr * (1 - torch.eye(n, device=device, dtype=torch.long))[None, :, :, None]
        return batch_like(G_logits, edge_attr)

    @staticmethod
    def _combine(adj: D, node: D, edge: D) -> S:
        indices_list = [a.squeeze(-1).to_sparse().coalesce().indices() for a in adj.A]
        node_list = [x[:n, :n].diagonal(dim1=0, dim2=1).t() for n, x in zip(node.n_nodes, node.A)]
        edge_list = [e[i[0], i[1]] for i, e in zip(indices_list, edge.A)]
        return make_batch(node_list, indices_list, edge_list)

    def _decode(self, G_logits: D) -> S:
        adj = self._decode_adj(G_logits)
        node_cls = self._decode_node_cls(G_logits)
        node_attr = self._decode_node_attr(G_logits)
        node_bbox = self._decode_node_bbox(G_logits)
        edge_attr = self._decode_edge_attr(G_logits)
        node = batch_like(G_logits, torch.cat([node_cls.A, node_attr.A, node_bbox.A], dim=-1))
        return self._combine(adj, node, edge_attr)

    def predict(self, G: S) -> S:
        h = self.encoder(G)
        h = self.decoder(D(h.values, h.n_nodes))
        G_logits = apply(h, self.output)
        return self._decode(G_logits)

    def decode_G(self, G: S, idx: int):
        n = G.n_nodes[idx]
        device = G.device

        x = torch.zeros(n, n, device=device, dtype=torch.long)
        x_attr = torch.zeros(n, n, self.node_attributes, device=device, dtype=torch.long)
        x_bbox = torch.zeros(n, n, 4, device=device)
        edge_index = torch.zeros(n, n, device=device, dtype=torch.long)
        edge_attr = torch.zeros(n, n, self.edge_attributes, device=device, dtype=torch.long)

        G_indices = G.indices[idx]
        G_values = G.values[idx]
        G_mask = G.mask[idx]

        G_node_mask = G_mask & (G_indices[:, 0] == G_indices[:, 1])
        G_edge_mask = G_mask & (G_indices[:, 0] != G_indices[:, 1])

        x[G_indices[G_node_mask, 0], G_indices[G_node_mask, 1]] = G_values[G_node_mask, 0].long()
        x_attr[G_indices[G_node_mask, 0], G_indices[G_node_mask, 1]] = G_values[G_node_mask, 1:1+self.node_attributes].long()
        x_bbox[G_indices[G_node_mask, 0], G_indices[G_node_mask, 1]] = G_values[G_node_mask, 1+self.node_attributes:1+self.node_attributes+4].float()

        x = torch.diagonal(x, dim1=0, dim2=1)
        x_attr = torch.diagonal(x_attr, dim1=0, dim2=1).t()
        x_bbox = torch.diagonal(x_bbox, dim1=0, dim2=1).t()

        offset = 10
        edge_index[G_indices[G_edge_mask, 0], G_indices[G_edge_mask, 1]] = 1
        edge_attr[G_indices[G_edge_mask, 0], G_indices[G_edge_mask, 1]] = G_values[G_edge_mask, -self.edge_attributes:].long() + offset

        edge_index = edge_index.to_sparse(2).coalesce()
        edge_attr = edge_attr.to_sparse(2).coalesce()
        assert torch.allclose(edge_index.indices(), edge_attr.indices())
        edge_index = edge_attr.indices()
        edge_attr = edge_attr.values() - offset

        return x, x_attr, x_bbox, edge_index, edge_attr
