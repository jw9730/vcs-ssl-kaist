import random
import json
from argparse import ArgumentParser
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image, ImageDraw, ExifTags
import numpy as np
import torch
from torch_geometric.data import Data
import pytorch_lightning as pl

from model import Model
from collator import collator

logging.getLogger('PIL').setLevel(logging.WARNING)


def adjust_exif_orientation(image: Image) -> Image:
    orientation = None
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError, TypeError):
        pass
    return image


def get_label_info(decode=False):
    root = Path('metadata_v2' if decode else 'metadata_v1')
    with open(root / 'object_class_id_to_name.json', 'r', encoding='utf-8') as f:
        object_class_id_to_name = json.load(f)
    with open(root / 'semantic_attribute_id_to_name.json', 'r', encoding='utf-8') as f:
        semantic_attribute_id_to_name = json.load(f)
    with open(root / 'semantic_attribute_id_to_type.json', 'r', encoding='utf-8') as f:
        semantic_attribute_id_to_type = json.load(f)
    with open(root / 'geometric_attribute_id_to_name.json', 'r', encoding='utf-8') as f:
        geometric_attribute_id_to_name = json.load(f)
    with open(root / 'semantic_relation_id_to_name.json', 'r', encoding='utf-8') as f:
        semantic_relation_id_to_name = json.load(f)
    with open(root / 'geometric_relation_id_to_name.json', 'r', encoding='utf-8') as f:
        geometric_relation_id_to_name = json.load(f)
    node_class_id_to_name = {int(k): v for k, v in object_class_id_to_name.items()}
    node_attr_id_to_name = {int(k): v for k, v in list(semantic_attribute_id_to_name.items()) + list(geometric_attribute_id_to_name.items())}
    edge_attr_id_to_name = {int(k): v for k, v in list(semantic_relation_id_to_name.items()) + list(geometric_relation_id_to_name.items())}
    return node_class_id_to_name, node_attr_id_to_name, edge_attr_id_to_name


def parse_scene_graph(img, masked_img, json_data, json_sample_data):
    # label info
    node_class_id_to_name, node_attr_id_to_name, edge_attr_id_to_name = get_label_info()
    node_class_ids = list(node_class_id_to_name.keys())
    node_attr_ids = list(node_attr_id_to_name.keys())
    edge_attr_ids = list(edge_attr_id_to_name.keys())

    # parse masked object
    masked_object = json_sample_data['masked_object']
    masked_object_id, masked_class_id, masked_class_name, masked_object_bbox = masked_object['object_id'], masked_object['class_id'], masked_object['class_name'], masked_object['object_bbox']

    # parse image
    img = adjust_exif_orientation(img)
    masked_img = adjust_exif_orientation(masked_img)
    assert img.size == masked_img.size

    # parse scene graph
    image_id, image_name = json_data['image_id'], json_data['image_name']
    objects, predicates = json_data['objects'], json_data['predicates']

    # parse objects
    n = len(objects)
    node_class = torch.zeros(n, dtype=torch.long)  # categorical
    node_attr = torch.zeros(n, len(node_attr_id_to_name), dtype=torch.long)  # binary
    node_bbox = torch.zeros(n, 4, dtype=torch.float)  # xcen, ycen, width, height (-1 to +1 standardized)
    node_mask = torch.zeros(n, dtype=torch.bool)  # mask information. True is masked(hidden).
    object_id_to_idx = {}
    out_of_bounds = False
    for idx, obj in enumerate(objects):
        object_id, class_id, class_name, object_bbox = obj['object_id'], obj['class_id'], obj['class_name'], obj['object_bbox']
        assert node_class_id_to_name[class_id] == class_name
        object_id_to_idx[object_id] = idx
        node_class[object_id_to_idx[object_id]] = node_class_ids.index(class_id)

        # parse bounding box
        x, y, width, height = object_bbox['x'], object_bbox['y'], object_bbox['width'], object_bbox['height']
        assert width > 0 and height > 0
        if not (0 <= x <= img.size[0] - width and 0 <= y <= img.size[1] - height):
            out_of_bounds = True
        # standardize to [0, 1] as center and size
        x /= img.size[0]
        y /= img.size[1]
        width /= img.size[0]
        height /= img.size[1]
        xcen = x + width / 2
        ycen = y + height / 2
        node_bbox[object_id_to_idx[object_id], :] = torch.tensor([xcen, ycen, width, height], dtype=torch.float)

        # set the mask flag
        # if mask_image.getpixel((int(xcen * img.size[0]), int(ycen * img.size[1]))) == 255:
        #     print(f'masking {class_name} with id {object_id} at {(int(xcen * img.size[0]), int(ycen * img.size[1]))}')
        #     node_mask[object_id_to_idx[object_id]] = True
        if object_id == masked_object_id:
            assert masked_class_id == class_id
            assert masked_class_name == class_name
            assert masked_object_bbox == object_bbox
            node_mask[object_id_to_idx[object_id]] = True

    # parse predicates
    edge_list = []
    for predicate in predicates:
        predicate_name, predicate_id = predicate['predicate_name'], predicate['predicate_id']
        subject_id, object_id = predicate['subject_id'], predicate['object_id']
        assert subject_id in object_id_to_idx
        if object_id == -1:
            # attribute
            assert node_attr_id_to_name[predicate_id] == predicate_name
            node_attr[object_id_to_idx[subject_id], node_attr_ids.index(predicate_id)] = 1
        else:
            # relation
            assert object_id in object_id_to_idx
            assert edge_attr_id_to_name[predicate_id] == predicate_name
            edge_list.append((object_id_to_idx[subject_id], object_id_to_idx[object_id], predicate_id))

    # coalesce edges
    n_edges = len(edge_list)
    edge_index = torch.zeros(2, n_edges, dtype=torch.long)
    edge_attr = torch.zeros(n_edges, len(edge_attr_id_to_name), dtype=torch.float)
    for idx, (i, j, predicate_id) in enumerate(edge_list):
        edge_index[0, idx] = i
        edge_index[1, idx] = j
        edge_attr[idx, edge_attr_ids.index(predicate_id)] = 1
    edge = torch.sparse_coo_tensor(edge_index, edge_attr, (n, n, len(edge_attr_id_to_name))).coalesce()
    edge_index = edge.indices()
    edge_attr = edge.values()

    # create data
    data = Data(
        x=node_class,
        x_attr=node_attr,
        x_bbox=node_bbox,
        x_mask=node_mask,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    # filter out of bounds
    assert not out_of_bounds

    # return result
    return data, img, masked_img, image_id, image_name


@torch.no_grad()
def scene_graph_to_json_dict(data, img, image_id, image_name):
    json_data = {}
    json_data['image_id'] = image_id
    json_data['image_name'] = image_name
    json_data['objects'] = []
    json_data['predicates'] = []

    # label info
    node_class_id_to_name, node_attr_id_to_name, edge_attr_id_to_name = get_label_info()
    node_class_ids = list(node_class_id_to_name.keys())
    node_attr_ids = list(node_attr_id_to_name.keys())
    # edge_attr_ids = list(edge_attr_id_to_name.keys())
    node_class_names = list(node_class_id_to_name.values())
    node_attr_names = list(node_attr_id_to_name.values())
    edge_attr_names = list(edge_attr_id_to_name.values())
    _, _, decode_edge_attr_id_to_name = get_label_info(decode=True)
    decode_edge_attr_ids = list(decode_edge_attr_id_to_name.keys())
    decode_edge_attr_names = list(decode_edge_attr_id_to_name.values())

    # parse data
    img_width, img_height = img.size
    x = data.x.cpu()
    x_attr = data.x_attr.cpu()
    x_bbox = data.x_bbox.cpu()
    x_mask = data.x_mask.cpu()
    edge_index = data.edge_index.cpu()
    edge_attr = data.edge_attr.cpu()
    assert not x_mask.any(), 'there should be no masked nodes'

    # parse objects
    n = x.shape[0]
    for idx in range(n):
        obj = {}
        obj['object_id'] = idx + 1
        obj['class_id'] = int(node_class_ids[x[idx].item()])
        obj['class_name'] = node_class_names[x[idx].item()]
        obj['object_bbox'] = {}
        obj['object_bbox']['x'] = int((x_bbox[idx, 0].item() - x_bbox[idx, 2].item() / 2) * img_width)
        obj['object_bbox']['y'] = int((x_bbox[idx, 1].item() - x_bbox[idx, 3].item() / 2) * img_height)
        obj['object_bbox']['width'] = int(x_bbox[idx, 2].item() * img_width)
        obj['object_bbox']['height'] = int(x_bbox[idx, 3].item() * img_height)
        json_data['objects'].append(obj)
        for attr in np.where(x_attr[idx].cpu().numpy()==1)[0]:
            predicate = {}
            predicate['attribute_name'] = node_attr_names[attr]
            predicate['attribute_id'] = int(node_attr_ids[attr])
            predicate['subject_id'] = int(idx + 1)
            predicate['object_id'] = -1
            json_data['predicates'].append(predicate)

    # parse relations
    subjects, objects = edge_index[0,:].cpu().numpy(), edge_index[1,:].cpu().numpy()
    for idx, (subj_idx, obj_idx) in enumerate(zip(subjects, objects)):
        for attr in np.where(edge_attr[idx].cpu().numpy()==1)[0]:
            predicate = {}
            predicate['predicate_name'] = edge_attr_names[attr]
            predicate['predicate_id'] = int(decode_edge_attr_ids[decode_edge_attr_names.index(predicate['predicate_name'])])
            predicate['subject_id'] = int(subj_idx + 1)
            predicate['object_id'] = int(obj_idx + 1)
            json_data['predicates'].append(predicate)

    # return result
    return json_data


@torch.no_grad()
def visualize_scene_graph(data, img, show_mask=False):
    # label info
    node_class_id_to_name, node_attr_id_to_name, edge_attr_id_to_name = get_label_info()
    node_class_names = list(node_class_id_to_name.values())
    node_attr_names = list(node_attr_id_to_name.values())
    edge_attr_names = list(edge_attr_id_to_name.values())

    # parse data
    x = data.x.cpu()
    x_attr = data.x_attr.cpu()
    x_bbox = data.x_bbox.cpu()
    x_mask = data.x_mask.cpu()
    edge_index = data.edge_index.cpu()
    edge_attr = data.edge_attr.cpu()
    if not show_mask:
        x_mask = torch.zeros_like(x_mask)

    # visualize
    fig, ax = plt.subplots()
    fig.set_dpi(1000)
    # adjust position, [0, +1] to pixelspace
    image_width, image_height = img.size
    image_center = (image_width / 2, image_height / 2)
    transformed_bbox = torch.zeros_like(x_bbox)
    x_bbox = x_bbox.clone()
    x_bbox[:, :2] = x_bbox[:, :2] * 2 - 1
    x_bbox[:, 2:] = x_bbox[:, 2:] * 2
    transformed_bbox[:, 0] = x_bbox[:, 0] * image_width / 2 + image_center[0]
    transformed_bbox[:, 1] = x_bbox[:, 1] * image_height / 2 + image_center[1]
    transformed_bbox[:, 2] = x_bbox[:, 2] * image_width / 2
    transformed_bbox[:, 3] = x_bbox[:, 3] * image_height / 2
    ax.imshow(img, alpha=0.7)

    subjects, objects = edge_index[0,:].cpu().numpy(), edge_index[1,:].cpu().numpy()
    for i, (subj_idx, obj_idx) in enumerate(zip(subjects, objects)):
        edge_attrs = np.where(edge_attr[i].cpu().numpy()==1)
        x1, y1 = transformed_bbox[subj_idx, 0], transformed_bbox[subj_idx, 1]
        x2, y2 = transformed_bbox[obj_idx, 0], transformed_bbox[obj_idx, 1]
        # Draw the arrowhead using an arrowstyle
        arrowprops = dict(arrowstyle='->', color='yellow', lw=2)
        if x_mask[subj_idx] or x_mask[obj_idx]:
            arrowprops['linestyle'] = 'dotted'
            arrowprops['lw'] = 1
            arrowprops['arrowstyle'] = '-|>'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrowprops)
        for num, attr in enumerate(edge_attrs[0]):
            ax.annotate(edge_attr_names[attr], ((x1+x2)/2, (y1+y2)/2+10*(num+1)), color='grey' if x_mask[obj_idx] or x_mask[subj_idx] else 'black', ha='center', va='center', fontsize=6, fontweight='bold')

    # Draw nodes as circles
    for i, (pos_x, pos_y, width, height) in enumerate(transformed_bbox.cpu().numpy()):
        attrs = np.where(x_attr[i].cpu().numpy()==1)
        if x_mask[i]:
            # mark x-shaped instead
            ax.annotate('X', (pos_x, pos_y), color='red', ha='center', va='center', fontsize=8)
            ax.annotate(node_class_names[x[i].item()], (pos_x, pos_y+15), color='skyblue', ha='center', va='center', fontsize=8)
        else:
            # draw bbox
            ax.add_patch(patches.Rectangle((pos_x-width/2, pos_y-height/2), width, height, linewidth=1, edgecolor='red', facecolor='none', alpha=0.5))

            ax.add_patch(patches.Circle((pos_x, pos_y), radius=2, color='red'))
            ax.add_patch(patches.Circle((pos_x, pos_y), radius=1, color='white'))
            ax.annotate(node_class_names[x[i].item()], (pos_x, pos_y+10), color='blue', ha='center', va='center', fontsize=8)
            for num, attr in enumerate(attrs[0]):
                ax.annotate(node_attr_names[attr], (pos_x, pos_y+10+10*(num+1)), color='purple', ha='center', va='center', fontsize=6)

    # display the image
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)
    plt.axis('off')
    plt.show()


def mask_random_nodes(data, p=0.5):
    # reset mask
    n = data.x.shape[0]
    for i in range(n):
        if random.random() < p:
            print(f'masking id {i}')
            data.x_mask[i] = True
    return data


@torch.no_grad()
def mask(data):
    # copy tensors to avoid in-place modification
    x = data.x.clone()
    x_attr = data.x_attr.clone()
    x_bbox = data.x_bbox.clone()
    x_mask = data.x_mask.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone()
    # randomly choose a single node to mask
    masked_node_indices = torch.where(x_mask)[0].tolist()
    if len(masked_node_indices) == 0:
        print('no nodes to mask, adding one')
        masked_node_indices = [len(x)]
        x = torch.cat([x, torch.zeros_like(x[:1])], dim=0)
        x_attr = torch.cat([x_attr, torch.zeros_like(x_attr[:1])], dim=0)
        x_bbox = torch.cat([x_bbox, torch.zeros_like(x_bbox[:1])], dim=0)
        x_mask = torch.cat([x_mask, torch.zeros_like(x_mask[:1])], dim=0)
        x_bbox[-1, :2] = torch.rand_like(x_bbox[-1, :2])
        x_mask[-1] = True
        data.x = x
        data.x_attr = x_attr
        data.x_bbox = x_bbox
        data.x_mask = x_mask
    # choose edges connected to masked nodes
    edge_indices_to_keep = []
    for idx, (u, v) in enumerate(edge_index.t().tolist()):
        if not (u in masked_node_indices or v in masked_node_indices):
            edge_indices_to_keep.append(idx)
    # mask nodes and edges
    x = x + 1
    x[masked_node_indices] = 0
    x_attr[masked_node_indices] = 0
    x_attr = torch.cat([x_attr, torch.zeros_like(x_attr[:, :1])], dim=1)
    x_attr[masked_node_indices, -1] = 1
    # for bbox, mask width and height
    x_bbox[masked_node_indices][:2] += 0.01 * torch.randn_like(x_bbox[masked_node_indices][:2])
    x_bbox[masked_node_indices][2:] = 0
    edge_index = edge_index[:, edge_indices_to_keep]
    edge_attr = edge_attr[edge_indices_to_keep]
    # return
    data.masked_x = x
    data.masked_x_attr = x_attr
    data.masked_x_bbox = x_bbox
    data.masked_edge_index = edge_index
    data.masked_edge_attr = edge_attr
    return data


@torch.no_grad()
def predict(model, data):
    # prepare batch
    batch = collator([data]).to('cuda')
    # predict
    G_recon, G_target = model.predict(batch)
    x, x_attr, x_bbox, edge_index, edge_attr = model.network.decode_G(G_target, 0)
    x_pred, x_attr_pred, x_bbox_pred, edge_index_pred, edge_attr_pred = model.network.decode_G(G_recon, 0)
    # choose masked node
    masked_node_indices = torch.where(data.x_mask)[0].tolist()
    masked_node_index = random.choice(masked_node_indices)
    # recombine
    x = x.clone()
    x_attr = x_attr.clone()
    x_bbox = x_bbox.clone()
    x[masked_node_index] = x_pred[masked_node_index]
    x_attr[masked_node_index] = x_attr_pred[masked_node_index]
    x_bbox[masked_node_index] = x_bbox_pred[masked_node_index]
    edge_attr = edge_attr[(edge_index[0] != masked_node_index) & (edge_index[1] != masked_node_index)]
    edge_attr_pred = edge_attr_pred[(edge_index_pred[0] == masked_node_index) | (edge_index_pred[1] == masked_node_index)]
    edge_index = edge_index[:, (edge_index[0] != masked_node_index) & (edge_index[1] != masked_node_index)]
    edge_index_pred = edge_index_pred[:, (edge_index_pred[0] == masked_node_index) | (edge_index_pred[1] == masked_node_index)]
    edge_attr = torch.cat([edge_attr, edge_attr_pred], dim=0)
    edge_index = torch.cat([edge_index, edge_index_pred], dim=1)
    # unmask
    x_mask = data.x_mask.clone()
    assert x_mask[masked_node_index]
    x_mask[masked_node_index] = False
    # return
    data = Data(
        x=x,
        x_attr=x_attr,
        x_bbox=x_bbox,
        x_mask=x_mask,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    return data


def inpaint(img, masked_img, json_data, json_sample_data):
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')
    model = Model.load_from_checkpoint(
        'checkpoints/mae.ckpt',
        strict=False,
        n_layers=6,
        dim_hidden=384,
        dim_qk=384,
        dim_v=384,
        dim_ff=1536,
        n_heads=6,
        input_dropout_rate=0.,
        dropout_rate=0.,
        dataset_name='scenegraphs'
    )
    model.cuda()
    model.eval()
    print('model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    data, img, masked_img, image_id, image_name = parse_scene_graph(img, masked_img, json_data, json_sample_data)

    # visualize_scene_graph(data, img)
    # plt.savefig('_original.png')

    while data.x_mask.any():
        data = predict(model, mask(data))

    # visualize_scene_graph(data, img)
    # plt.savefig('_inpaint.png')

    json_data = scene_graph_to_json_dict(data, masked_img, image_id, image_name)
    return json_data
