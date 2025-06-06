"""
This file runs the primary Graph Neural Network on the
Holocaust testimonies. There are three steps in the code below:
(1) Accessing, reading, and embedding the Holocaust testimonies.
(2) Constructing the graph with the embeddings.
(3) Running the message-passing neural network over the graph. The
GNN class is found in the file network.py


"""
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import holocaust_utils
import torch.nn as nn
import pickle
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import LinkLoader



"""
Function for loading the data, extracting the relevant
sentences, and converting them to a format useful for
the HuggingFace Transformer. The data is in CSV format of: 
FileID, Questions, Answers
where each answer begins with the initials of the interviewee and a colon
(e.g., AB: [answer]). Only the text following the colon is extracted for use.
Note: The questions will be dropped from the dataframe, as seen below.
"""
def get_sentences(file_path):
    print("Parsing file...")
    df = pd.read_csv(file_path, sep="\t")
    df['Answers'] = df['Answers'].str.replace(r'^[A-Z]{1,5}:\s*', '', regex=True)
    df_edited = df.drop('Questions', axis=1) # drop questions column
    # test to make sure that the data is now only fileid and answers
    print(df_edited.head())
    sentences_dict = df_edited.groupby('FileID')['Answers'].apply(list).to_dict() # convert to dictionary
    print("File parsed successfully.")
    return sentences_dict


"""
The following code between the lines of pound signs
was adapted from the instructions for running HuggingFace's
all-MiniLM-L6-v2 SentenceTransformer: 
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
"""
####################
def get_embeddings(sentences):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    return embeddings
####################

"""
Function for converting the answers of a given testimony to 
embeddings using get_embeddings. Important to note that 
embeds will be a matrix for each fileID, not a list of vectors
hence, the conversion to a list.

Each embedding will be a vector of length 384. 
"""
def convert_sentences(sentences_dict):
    print("Creating embeddings...")
    embeddings_dict = {}
    for key, sentences in sentences_dict.items():
        embeds = get_embeddings(sentences)
        embeddings_dict[key] = embeds.tolist() 
    print("Embeddings created successfully.")
    return embeddings_dict

def build_graph(embeddings_dict):
# build node_features matrix from Transformer encodings
    print("Building graph...")
    x = []
    for embeds in embeddings_dict.values():
        x.append(torch.tensor(embeds))  # typo: was "appened"
    node_features = torch.cat(x, dim=0)

    # build edge_index 
    edge_index = []
    sentence_lookup = {}
    node_idx = 0

    for doc_id, embeds in embeddings_dict.items():
        num_nodes = len(embeds)
        # used later to map the embeddings back to the original sentences
        for i in range(num_nodes):
            sentence_lookup[node_idx + i] = (doc_id, i)
        pairs = torch.combinations(torch.arange(num_nodes), r=2, with_replacement=False)
        edges = torch.cat([
            (pairs + node_idx).T,                      # forward edges
            (pairs[:, [1, 0]] + node_idx).T            # reverse edges
        ], dim=1)


        edge_index.append(edges)
        node_idx += num_nodes

    edge_index = torch.cat(edge_index, dim=1)

    # build graph
    G = Data(x=node_features, edge_index=edge_index)
    print("Graph built successfully.")
    return G, sentence_lookup, edge_index

"""
Manual implementation of the Graph Isomorphism Network
primarily to control global pooling (not useful for 
link prediction). This implementation allows for a sub_graph
flag to be turned on.
"""
class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=None, num_layers=3, sub_graph=False):
        super().__init__()
        self.sub_graph = sub_graph
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            layers = nn.Sequential(
                nn.Linear(in_channels if i == 0 else hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(layers))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        if sub_graph and out_channels is not None:        
            self.final = nn.Linear(hidden_channels, out_channels)
        elif sub_graph:
            raise ValueError("Out_channels must be defined when sub_graph=True.")


    def forward(self, x, edge_index, batch=None):
        layers = zip(self.convs, self.bns)
        for conv, bn in layers:
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        if self.sub_graph:
            if batch is None:
                raise ValueError("Batch argument must be defined when sub_graph=True")
            else:
                return self.final(x)

        return x


def main(get_embeds=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if get_embeds:
        file_path = "ht_questions_answers.tsv"
        sentences_dict = get_sentences(file_path)
        embeddings_dict = convert_sentences(sentences_dict)
        # save raw embeddings in case of bugs or for later reuse
        doc_lengths = [len(v) for v in embeddings_dict.values()]
        doc_ids = list(embeddings_dict.keys())
        all_embeds = np.vstack([np.array(v) for v in embeddings_dict.values()])
        np.savez("sentence_embeddings.npz", features=all_embeds, lengths=doc_lengths, doc_ids=np.array(doc_ids))
    else:
        file_path = "sentence_embeddings.npz"
        data = np.load(file_path, allow_pickle=True)
        features = data["features"]
        lengths = data["lengths"]
        doc_ids = data["doc_ids"]

        embeddings_dict = {}
        idx = 0
        for doc_id, length in zip(doc_ids, lengths):
            doc_id = doc_id.decode() if isinstance(doc_id, bytes) else doc_id
            embeddings_dict[doc_id] = features[idx:idx + length]
            idx += length

    G, sentence_lookup, edge_index = build_graph(embeddings_dict)
    NUM_EPOCHS = 49
    print(type(G))
    print(G)
    # split data into train, validation, and test sets
    transform = RandomLinkSplit(is_undirected=True, split_labels=True)
    train_data, val_data, test_data = transform(G)
    print("Transform complete")
    print(train_data)

    train_loader = LinkLoader(
        data=train_data,
        edge_label_index=train_data.edge_label_index,
        link_label=train_data.edge_label,
        batch_size=128,
        shuffle=True,
        num_neighbors=[15, 10],  # sample 15 neighbors in 1st hop, 10 in 2nd hop
    )
    model = GIN(in_channels=G.num_node_features, hidden_channels=128, sub_graph=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training the model...")
    for i in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = model(batch.x, batch.edge_index)
            score = cosine_sim(z, batch.link_label_index)

            loss = F.binary_cross_entropy_with_logits(score, batch.link_label.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {i}, Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), "gin_model.pt")

    
    # run evaluation
    model.eval()
    val_preds, val_labels = run_eval_and_test(val_data)

    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)
    val_auc = roc_auc_score(val_labels.cpu(), val_preds.cpu())
    val_ap = average_precision_score(val_labels.cpu(), val_preds.cpu())

    # run test
    model.eval()
    test_preds, test_labels = run_eval_and_test(test_data)

    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)
    test_auc = roc_auc_score(test_labels.cpu(), test_preds.cpu())
    test_ap = average_precision_score(test_labels.cpu(), test_preds.cpu())

    plot_curves(val_auc, val_ap, test=False)
    plot_curves(test_auc, test_ap, test=True)

    # run inference using the existing model by re-initializing
    # the graph (raw data) and then applying the model for link prediction
    print("Running inference...")
    sentences_dict = get_sentences(file_path)
    embeddings_dict = convert_sentences(sentences_dict)
    G, sentence_lookup, edge_index = build_graph(embeddings_dict)
    num_nodes = G.x.size(0)
    candidate_edges = torch.combinations(torch.arange(num_nodes), r=2).T

    model = GIN(in_channels=384, hidden_channels=128, sub_graph=False).to(device)
    model.load_state_dict(torch.load("gin_model.pt"))
    model.eval()
    inferred_links = run_inference(G, candidate_edges, model, device)
    save_link_preds(inferred_links, sentence_lookup, sentences_dict)
    print("Predicted links saved and process complete.")

if __name__ == "__main__":
    main()











# overall goal of the GNN: link prediction for nodes with attributes that incorporate space and time;
# also neighborhood matching to see what kind of responses are grouped together


# use HDBSCAN after using node2vec to explore thematic groupings
# use Graph Autoencoder for link prediction?
# use Deep Graph Infomax for learning node embeddings



# use in-context learning (!!!) to explore the construction of the testimonies and how
# space and time fit in to the overall narrative structure