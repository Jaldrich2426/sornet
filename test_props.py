'''
MIT License

Copyright (c) 2022 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# from datasets import CLEVRDataset

# from PROPS_Relation_Dataset.src.props_relation_dataset.PropsRelationDataset import PROPSRelationDataset
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from props_relation_dataset.PropsRelationDataset import PROPSRelationDataset
from networks import EmbeddingNet, ReadoutNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--max_nobj', type=int, default=10)
    parser.add_argument('--img_h', type=int, default=480)
    parser.add_argument('--img_w', type=int, default=640)
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--d_hidden', type=int, default=512)
    parser.add_argument('--n_relation', type=int, default=4)
    # Evaluation
    parser.add_argument('--checkpoint')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_worker', type=int, default=1)
    args = parser.parse_args()

    data = PROPSRelationDataset("val", "objects", args, rand_patch=False, resize=True)
    loader = DataLoader(data, args.batch_size, num_workers=args.n_worker)

    model = EmbeddingNet(
        (args.img_w, args.img_h), args.patch_size, args.max_nobj,
        args.width, args.layers, args.heads
    )
    head = ReadoutNet(args.width, args.d_hidden, 0, args.n_relation)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    head.load_state_dict(checkpoint['head'])
    model = model.cuda().eval()
    head = head.cuda().eval()

    correct = 0
    total = 0
    correct_matrix = torch.zeros(1,360)
    total_matrix = torch.zeros(1,360)
    for img, obj_patches, target, mask in tqdm(loader):
        img = img.cuda()
        obj_patches = obj_patches.cuda()
        with torch.no_grad():
            emb, attn = model(img, obj_patches)
            logits = head(emb)
            pred = (logits > 0).int().cpu()
        target = target.int()
        mask = mask.bool()

        correct += (pred[mask] == target[mask]).sum().item()
        correct_matrix[mask] += pred[mask] == target[mask]
        total_matrix[mask] += 1
        total += mask.sum().item()

    print('Total', total)
    print('Accuracy', correct / total * 100)

    accuracy_matrix = correct_matrix / total_matrix * 100

    accuracy_matrix=accuracy_matrix.reshape(4,90)
    diag_mask = torch.ones(10, 10).bool() ^ torch.eye(10).bool()
    # print(diag_mask)
    flat_mask= diag_mask.reshape(-1)
    # print(flat_mask.shape)
    full_accuracy_matrix = torch.ones(4,100)*-1
    for i in range(4):
        full_accuracy_matrix[i,flat_mask] = accuracy_matrix[i,:]
    full_accuracy_matrix=full_accuracy_matrix.reshape(4,10,10)

    # print('Accuracy Matrix',full_accuracy_matrix)
    # average over first dim
    print('Average Accuracy Matrix, row is obj1, col is obj2:')
    print(torch.mean(full_accuracy_matrix,dim=0))

