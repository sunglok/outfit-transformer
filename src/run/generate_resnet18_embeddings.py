import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

from ..data import collate_fn
from ..data.datasets import polyvore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--polyvore_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weights', required=True)
    args = parser.parse_args()

    dataset = polyvore.PolyvoreItemDataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn.item_collate_fn)
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    model.fc = torch.nn.Identity()
    model.cuda().eval()
    transform = weights.transforms()

    ids, outputs = [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc='ResNet-18 embeddings'):
            images = torch.stack([transform(item.image.convert('RGB')) for item in batch]).cuda()
            outputs.append(model(images).cpu().numpy().astype(np.float32))
            ids.extend(str(item.item_id) for item in batch)

    output_dir = os.path.join(args.polyvore_dir, 'precomputed_clip_embeddings')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'polyvore_0.pkl'), 'wb') as f:
        pickle.dump({'ids': ids, 'embeddings': np.concatenate(outputs)}, f)


if __name__ == '__main__':
    main()
