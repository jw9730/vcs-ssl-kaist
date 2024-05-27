import torch


class Batch:
    def __init__(
            self,
            x,
            x_attr,
            x_bbox,
            edge_index,
            edge_attr,
            node_num,
            edge_num,
            masked_x,
            masked_x_attr,
            masked_x_bbox,
            masked_edge_index,
            masked_edge_attr,
            masked_node_num,
            masked_edge_num
        ):
        super().__init__()
        assert node_num == masked_node_num
        self.x = x
        self.x_attr = x_attr
        self.x_bbox = x_bbox
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.node_num = node_num
        self.edge_num = edge_num
        self.masked_x = masked_x
        self.masked_x_attr = masked_x_attr
        self.masked_x_bbox = masked_x_bbox
        self.masked_edge_index = masked_edge_index
        self.masked_edge_attr = masked_edge_attr
        self.masked_node_num = masked_node_num
        self.masked_edge_num = masked_edge_num

    def to(self, device):
        self.x = self.x.to(device)
        self.x_attr = self.x_attr.to(device)
        self.x_bbox = self.x_bbox.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.masked_x = self.masked_x.to(device)
        self.masked_x_attr = self.masked_x_attr.to(device)
        self.masked_x_bbox = self.masked_x_bbox.to(device)
        self.masked_edge_index = self.masked_edge_index.to(device)
        self.masked_edge_attr = self.masked_edge_attr.to(device)
        return self

    def __len__(self):
        return len(self.node_num)


def collator(items):
    xs, x_attrs, x_bboxes, edge_indices, edge_attrs = zip(*[
        (
            item.x,
            item.x_attr,
            item.x_bbox,
            item.edge_index,
            item.edge_attr,
        )
        for item in items
    ])
    masked_xs, masked_x_attrs, masked_x_bboxes, masked_edge_indices, masked_edge_attrs = zip(*[
        (
            item.masked_x,
            item.masked_x_attr,
            item.masked_x_bbox,
            item.masked_edge_index,
            item.masked_edge_attr,
        )
        for item in items
    ])
    batch = Batch(
        x=torch.cat(xs),
        x_attr=torch.cat(x_attrs).float(),
        x_bbox=torch.cat(x_bboxes),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_attr=torch.cat(edge_attrs).float(),
        node_num=[i.size(0) for i in xs],
        edge_num=[i.size(1) for i in edge_indices],
        masked_x=torch.cat(masked_xs),
        masked_x_attr=torch.cat(masked_x_attrs).float(),
        masked_x_bbox=torch.cat(masked_x_bboxes),
        masked_edge_index=torch.cat(masked_edge_indices, dim=1),
        masked_edge_attr=torch.cat(masked_edge_attrs).float(),
        masked_node_num=[i.size(0) for i in masked_xs],
        masked_edge_num=[i.size(1) for i in masked_edge_indices]
    )
    return batch
