#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

print('current working dir:', os.getcwd())
sys.path.append(os.getcwd())

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint

from model.ConTP_data_module import ConTPDataModule
from model.ConTP_module import ConTPModule
from utils.lightning import LitModelInference
from utils.file import read_fasta
from utils.wrapper.ESM import ESMWrapper


# ============================================================
# Multi-label prediction function
# ============================================================
def multi_label_from_distance(dist, threshold=0.035):
    """
    Multi-label prediction based on the distance matrix:
    1) Apply softmax to -dist to obtain probabilities
    2) Select classes whose probability > threshold
    3) If no class is selected for a sample, take the Top-1 class
    """

    sorted_probs, sorted_indices = torch.sort(
        torch.softmax(-dist, dim=1), descending=True
    )
    probs = sorted_probs
    preds = sorted_indices

    mask = probs > threshold
    indices = mask.nonzero(as_tuple=False)  # (K, 2)

    N, _ = probs.shape
    multi_label_pred = [[] for _ in range(N)]

    # Threshold-based selection
    for sample_id, class_id in indices.tolist():
        multi_label_pred[sample_id].append(class_id)

    # If empty → use top1
    top1_ids = torch.argmax(probs, dim=1).tolist()
    for i in range(N):
        if len(multi_label_pred[i]) == 0:
            multi_label_pred[i].append(top1_ids[i])

    # Map back to original class ids
    final_preds = []
    for i, idx_list in enumerate(multi_label_pred):
        cls_list = [preds[i, idx].item() for idx in idx_list]
        final_preds.append(cls_list)

    return final_preds


# ============================================================
# Inference main function
# ============================================================
def run_inference(args):
    # ====== Load query sequences ======
    query_seqs, query_headers = read_fasta(args.query_fasta)
    num_query = len(query_seqs)
    print(f"Loaded {num_query} query sequences.")

    if args.task == 'substrate':
        ckpt_path = args.ckpt_substrate
        label_map = pd.read_csv(args.substrate_map)
        temp_dir = "./temp/inference_substrate/"
    else:  # tc
        ckpt_path = args.ckpt_tc
        label_map = pd.read_csv(args.tc_map)
        temp_dir = "./temp/inference_tc/"

    # ====== Initialize model ======
    os.makedirs(temp_dir, exist_ok=True)
    predictor = LitModelInference(ConTPModule, ConTPDataModule, ckpt_path)

    # ====== Load class embeddings ======
    cache_file = f'{temp_dir}/class_embeddings.pth'
    if os.path.exists(cache_file):
        class_embeddings = torch.load(cache_file, weights_only=False)
        select_cluster = np.load(f'{temp_dir}/select_cluster.npy')
        print("Loaded cached class embeddings.")
    else:
        raise FileNotFoundError(f"Missing class embeddings at {cache_file}")

    # ====== Initialize ESM ======
    esm = ESMWrapper(args.esm_cache, device=args.device)
    esm.__init_submodule__()

    # ====== Compute ESM embeddings ======
    batch_size = args.batch_size
    num_batches = (num_query // batch_size) + (0 if num_query % batch_size == 0 else 1)

    query_esm = []
    for i in tqdm(range(num_batches), desc='Computing ESM Embeddings'):
        batch_seqs = query_seqs[i * batch_size: (i + 1) * batch_size]
        batch_embed = esm.forward(batch_seqs)['mean_representations']
        query_esm.append(batch_embed)

    query_esm = torch.concat(query_esm, dim=0)

    # ====== Forward pass to obtain embeddings ======
    query_X = predictor.ckpt_model.model.forward(query_esm.to(args.device))

    # ====== Find nearest cluster ======
    raw_pred, dist = predictor.ckpt_model.model.find_nearest_cluster(
        query_X, class_embeddings, return_dist=True
    )

    # ====== Multi-label or single-label output ======
    if args.task == 'substrate':
        preds = multi_label_from_distance(dist, threshold=args.threshold)
        final_pred = [
            [label_map.loc[pred_y, 'substrate'] for pred_y in pred_list]
            for pred_list in preds
        ]
    else:
        preds = np.array([select_cluster[i] for i in raw_pred])
        final_pred = [
            label_map.loc[pred, 'tcid'] for pred in preds
        ]

    # ====== Save prediction ======
    pred_df = pd.DataFrame(final_pred)
    if args.task == 'substrate':
        pred_df.columns = [f'substrate_top{col + 1}' for col in pred_df.columns]
    else:
        pred_df.columns = ['tcid']

    columns = pred_df.columns.tolist()
    pred_df['header'] = query_headers
    pred_df = pred_df[['header'] + columns]

    os.makedirs(args.out_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.query_fasta))[0]
    save_path = os.path.join(args.out_dir, f"{basename}_pred.csv")
    pred_df.to_csv(save_path, index=False)
    print(f"\nSaved prediction to: {save_path}")

    if args.save_dist:
        # Convert dist → numpy
        dist_np = dist.detach().cpu().numpy()  # shape (num_query, num_classes)

        # Get original class labels
        if args.task == "substrate":
            class_names = label_map["substrate"].tolist()
        else:
            class_names = [label_map.loc[i, 'tcid'] for i in select_cluster]

        # Construct DataFrame
        dist_df = pd.DataFrame(dist_np, columns=class_names)

        # Insert FASTA headers as first column
        dist_df.insert(0, "header", query_headers)

        # Save
        dist_save_path = os.path.join(args.out_dir, f"{basename}_dist.csv")
        dist_df.to_csv(dist_save_path, index=False)
        print(f"Saved distance matrix to: {dist_save_path}")


# ============================================================
# argparse configuration
# ============================================================
def build_args():
    parser = argparse.ArgumentParser(description="ConTP Inference Script")

    # -------- Basic arguments --------
    parser.add_argument("--query_fasta", type=str, required=True,
                        help="Query FASTA file to run inference on.")

    parser.add_argument("--task", type=str, choices=["substrate", "tc"], default="substrate",
                        help="Prediction task: substrate classification or TC classification.")

    parser.add_argument("--threshold", type=float, default=0.034,
                        help="Threshold for multi-label substrate prediction.")

    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--batch_size", type=int, default=20,
                        help="Batch size used for ESM embedding extraction.")

    parser.add_argument("--save_dist", action="store_true",
                        help="If set, save the distance matrix for all queries.")

    # -------- Directory arguments --------
    parser.add_argument("--out_dir", type=str, default="./temp/output/")
    parser.add_argument("--esm_cache", type=str, default="./temp/esm/")

    # -------- Model checkpoint --------
    parser.add_argument("--ckpt_substrate", type=str,
                        default="/home/hew/python/contp/ckpt/lightning_logs/substrate/checkpoints/last.ckpt")
    parser.add_argument("--ckpt_tc", type=str,
                        default="/home/hew/python/contp/ckpt/lightning_logs/tc/checkpoints/last.ckpt")

    # -------- Label mapping --------
    parser.add_argument("--substrate_map", type=str, default="./data/substrate_mapping.csv")
    parser.add_argument("--tc_map", type=str, default="./data/tc_mapping.csv")

    return parser.parse_args()


# ============================================================
# Main entry
# ============================================================
if __name__ == "__main__":
    args = build_args()

    print()
    print("=" * 40 + ' Arguments ' + "=" * 40)
    pprint(vars(args))
    print("=" * 91)
    print()

    run_inference(args)
