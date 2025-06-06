import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch_geometric.loader import LinkLoader
import csv

def cosine_sim(z, edge_index):
    src, dst = edge_index
    return F.cosine_similarity(z[src], z[dst], dim=1)

def link_prediction(z, threshold=0.85):
    # first calculate cosine similarity of all
    # edges in the graph
    edge_list = []
    num_nodes = z.size(0)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_list.append([i, j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    scores = cosine_sim(z, edge_index)
    # find the top k similar edges above the threshold and return the list
    inferred = []
    for k in range(edge_index.size(1)):
        if scores[k] > threshold:
            i, j = edge_index[:, k]
            inferred.append((i.item(), j.item(), scores[k].item()))
    return inferred

# function to plot AUC and AP curves
def plot_curves(auc, ap, test=False):
    if test:
        epochs = list(range(len(auc)))
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, auc, label='AUC')
        plt.plot(epochs, ap, label='Average Precision')
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("AUC and Average Precision over Epochs: Test Set")
        plt.legend()
        plt.grid(True)
        plt.savefig("test_auc_ap_curves.png")
    else:
        epochs = list(range(len(auc)))
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, auc, label='AUC')
        plt.plot(epochs, ap, label='Average Precision')
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("AUC and Average Precision over Epochs: Validation Set")
        plt.legend()
        plt.grid(True)
        plt.savefig("val_auc_ap_curves.png")

def save_link_preds(inferred_links, sentence_lookup, sentences_dict):
    with open("inferred_links.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Node1", "Node2", "CosineSimilarity", "Sentence1", "Sentence2"])
        # uncomment out next line to have sorted and limited similarities returned
        #inferred_links = sorted(inferred_links, key=lambda x: -x[2])[:100]
        inferred_links = [
            (i, j, sim) for i, j, sim in inferred_links
            if sentence_lookup[i][0] != sentence_lookup[j][0]
        ]

        for i, j, sim in inferred_links:
            doc1, idx1 = sentence_lookup[i]
            doc2, idx2 = sentence_lookup[j]

            s1 = sentences_dict[doc1][idx1]
            s2 = sentences_dict[doc2][idx2]

            writer.writerow([i, j, f"{sim:.4f}", s1, s2])

def run_eval_and_test(data, model, device):
    data_loader = LinkLoader(
        data=data,
        link_label_index=data.edge_label_index,
        link_label=data.edge_label,
        batch_size=256,
        shuffle=False,
        num_neighbors=[15, 10],  # Same as training for consistency
    )
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            z = model(batch.x, batch.edge_index)
            score = cosine_sim(z, batch.link_label_index)

            all_preds.append(score)
            all_labels.append(batch.link_label)

    return all_preds, all_labels

def run_inference(data, candidate_edges, model, device):
    inference_loader = LinkLoader(
        data=data,
        link_label_index=candidate_edges,
        link_label=None,
        batch_size=256,
        shuffle=False,
        num_neighbors=[15, 10],
    )
    all_scores = []
    all_links = []

    with torch.no_grad():
        for batch in inference_loader:
            batch = batch.to(device)
            z = model(batch.x, batch.edge_index)
            scores = cosine_sim(z, batch.link_label_index)

            # Apply threshold
            mask = scores > 0.85
            filtered_edges = batch.link_label_index[:, mask]
            all_links.append(filtered_edges)

    all_links = torch.cat(all_links, dim=1)



# function to create snapshot of graph
#def create_graph_image(G, x, edge_index):

    








