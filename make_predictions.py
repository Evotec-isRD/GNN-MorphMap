import sys
import pandas as pd
import pickle
from itertools import islice
import pandas as pd
import numpy as np
from itertools import islice
import torch
from torch_geometric.data import Data
import sys
from typing import Dict
import torch
import itertools
from evord_evognn_lib.utils.plots import plot_roc_curve
from collections import defaultdict
from collections import Counter
import yaml
import os
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch.nn as nn
import networkx as nx
from tqdm import tqdm
import umap
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import degree
import seaborn as sns
import pickle
import logging as logger

RANDOM_SEED = 1
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def subset_dict(input_dict, n=5):
    """Get the first n key-value pairs from a dictionary"""
    dict_items = input_dict.items()
    return list(islice(dict_items, n))

def get_pos_neg_index(output_transform, tag = None):
    if tag=="train_data":
        edge_index = output_transform.get(f"{tag}.edge_index")
                
        pos = output_transform['set_of_pos_train_edges']
        neg = output_transform['set_of_neg_train_edges']
        pos_train = output_transform[f'{tag}.pos_edge_label_index']        
        neg_train = output_transform[f'{tag}.neg_edge_label_index']
        print(f"total_edges {tag} (train_data.edge_index): {len(edge_index[0])}")
        print(f"set_of_pos_train_edges: {len(pos)}")
        print(f"set_of_neg_train_edges: {len(neg)}")
        print("\n")
        return edge_index, pos, neg
    else:
        if f"{tag}" in output_transform.keys():
            edge_index = output_transform[f"{tag}"]

        if f"{tag}.edge_index" in output_transform.keys():
            edge_index = output_transform[f"{tag}.edge_index"]         
        
        try:
            pos = output_transform[f'{tag}.pos_edge_label_index'].T        
            neg = output_transform[f'{tag}.neg_edge_label_index'].T  
        
            print(f"total_edges {tag}: {len(edge_index[0])}")
            print(f"{tag}.pos_edge_label_index: {len(pos.T[0])}")
            print(f"{tag}.neg_edge_label_index: {len(neg.T[0])}")
            print("\n")
            return edge_index, pos, neg
        except:
            print("The shape of edge_index should be [2, num_nodes]")

def create_feature_matrix(num_nodes):    
    return torch.sparse.torch.eye(num_nodes)

def create_data_object(
        num_nodes, edges_mp, edges_src_target_pos, edges_src_target_neg
    ):
        dataset = Data()
        dataset.num_nodes = num_nodes
        dataset.edge_index = torch.tensor(edges_mp, dtype=torch.int64)
        dataset.x = create_feature_matrix(num_nodes)
        
        dataset.pos_edge_label = torch.ones(
            len(edges_src_target_pos), dtype=torch.int64
        )
        dataset.pos_edge_label_index = edges_src_target_pos.T
        
        dataset.neg_edge_label = torch.zeros(
            len(edges_src_target_neg), dtype=torch.int64
        )
        dataset.neg_edge_label_index = edges_src_target_neg.T
        return dataset

def compute_ranks(scores, source_nodes, target_nodes, labels):
    """
    Given source nodes, target nodes, and prediction score, compute the rank associated with every positive triple

    :param predictionsDF: Dataframe of triplet, and prediction score
    :return: Computed ranks for each positive triple
    """

    ranks = {}
    source_nodes_new = []

    mask_neg = labels == "N"
    scores_neg = scores[mask_neg]
    sources_neg = source_nodes[mask_neg]
    irx = np.argsort(-scores_neg)
    scores_neg = scores_neg[irx]
    sources_neg = sources_neg[irx]

    scores_neg_dict = {}
    sources_neg_dict = {}
    for i, sn in tqdm(enumerate(sources_neg)):
        v = sources_neg_dict.get(sn, [])
        v.append(i)
        sources_neg_dict[sn] = v
    unique_sources = list(sources_neg_dict.keys())

    mask_pos = np.logical_not(mask_neg)
    scores_pos = scores[mask_pos]
    sources_pos = source_nodes[mask_pos]
    for i in tqdm(range(sum(mask_pos))):
        score = scores_pos[i]
        sourceNode = sources_pos[i]
        scores_negative = scores_neg[sources_neg_dict[sourceNode]]
        rank = np.sum(scores_negative > score) + 1

        # detect ties and put the rank in the middle of the tie
        irx = np.where(scores_negative == score)[0]
        if len(irx) > 1:
            rank = int((irx[0] + irx[-1]) / 2)

        if sourceNode not in source_nodes_new:
            ranks[sourceNode] = []
            source_nodes_new.append(sourceNode)
        ranks[sourceNode].append(rank)
    return ranks

def visualize_umap_embeddings(
    model,
    data,
    title,
    perplexity=30.0,
    labeled=False,
    labels=[],
    sizes=None,
    data_path=None,
    number_of_diseases=519,
    number_of_genes=7294,
    removeFirstPC=0,
):
    """Visualizes node embeddings in 2D space with t-SNE.

    Args: model, pass in the trained or untrained model
          data, Data object, where we assume the first 519 datapoints are disease
            nodes and the rest are gene nodes
          title, title of the plot
          perplexity, t-SNE hyperparameter for perplexity
    """
    model.eval()
    reducer = umap.UMAP(metric="cosine")
    x = data.x
    z = model.encode(x, data.edge_index).detach().cpu().numpy()
    if removeFirstPC > 0:
        pca = PCA()
        u = pca.fit_transform(z)
        z = u[:, removeFirstPC:]

    embedding = reducer.fit_transform(z)
    number_of_others = data.num_nodes-number_of_genes-number_of_diseases
    return embedding, z

def get_train_val_test_splits(output_transform_path=None):
    try:
        if output_transform_path:
            output_transform = torch.load(output_transform_path, map_location=torch.device('cpu'))
            train_edge_index, train_pos, train_neg = get_pos_neg_index(output_transform, tag = "train_data")
            val_edge_index, val_pos, val_neg = get_pos_neg_index(output_transform, tag = "val_data")
            test_edge_index, test_pos, test_neg = get_pos_neg_index(output_transform, tag = "test_data")

            num_nodes = len(output_transform['mapping_dict'][0])

            train_dataset = create_data_object(num_nodes, train_edge_index, train_pos, train_neg)
            val_dataset = create_data_object(num_nodes, val_edge_index, val_pos, val_neg)
            test_dataset = create_data_object(num_nodes, test_edge_index, test_pos, val_neg)

            return train_dataset, val_dataset, test_dataset, output_transform
    except:
        print("Please provide 'output_transform file path'")

class VariationalGCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_size, cached=False)
        self.conv_mu = GCNConv(hidden_size, out_channels, cached=False)
        self.conv_logstd = GCNConv(hidden_size, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.explain = False

    def set_explain(self, bool_val):
        self.explain = bool_val

    def forward(self, x, edge_index):
        #x = x.to_dense()
        x_temp1 = self.conv1(x, edge_index).relu()
        x_temp2 = self.dropout(x_temp1)
        mu = self.conv_mu(x_temp2, edge_index)
        logstd = self.conv_logstd(x_temp2, edge_index)
        if self.explain:
            return mu
        else:
            return mu, logstd

### --------------------------
root_folder = ''
link_to_predict = sys.argv[1]
print('link_to_predict=', link_to_predict)

# Get train/val/test splits, and output_transform mapping
train_dataset, val_dataset,test_dataset, output_transform= get_train_val_test_splits("trained_models/"+link_to_predict+"/output_transform.pt")
num_nodes = len(output_transform['mapping_dict'][0])

with open('configurations/prediction_tasks/'+link_to_predict+'.yaml', 'r') as file:
    config = yaml.safe_load(file)
config = DictConfig(config)
print(config)

weight_file = "trained_models/"+link_to_predict+"/vgae_model_weights_bestHitsAt10.pt"
data_object_file = 'trained_models/'+link_to_predict+'/'+link_to_predict+'_data_metadata.pt'
list_of_function_file = 'trained_models/'+link_to_predict+'/'+link_to_predict+'_function.tsv'
node_mapping_file = "trained_models/"+link_to_predict+"/"+link_to_predict+"_node_mapping_file.pkl"
prediction_file = 'trained_models/'+link_to_predict+'/'+link_to_predict+'_predictions.tsv'

if link_to_predict.startswith("gene_mf"):
    function_prefix = 'Molecular Function::'
if link_to_predict.startswith("gene_pathway"):
    function_prefix = 'Pathway::'
if link_to_predict.startswith("gene_cc"):
    function_prefix = 'Cellular Component::'
if link_to_predict.startswith("gene_bp"):
    function_prefix = 'Biological Process::'

# loading mappings
dz_mapping = {}
gene_mapping = {}
rest_mapping = {}

for key, value in output_transform['mapping_dict'][0].items():
    if key.startswith(function_prefix):
        if value<12000: # 12000, an arbitrary value, to selecect only nodes used for supervision
            dz_mapping[key] = value
        else:
            rest_mapping[key] = value
    if key.startswith('Gene'):
        gene_mapping[key] = value
    if not ((key.startswith(function_prefix)) or (key.startswith('Gene'))):
        rest_mapping[key] = value

inverse_gene_mapping = {gene_mapping[g]:g for g in gene_mapping}
inverse_dz_mapping = {dz_mapping[g]:g for g in dz_mapping}
inverse_rest_mapping = {rest_mapping[g]:g for g in rest_mapping}

dfgene2hugo = pd.read_csv(root_folder+'tests/geneid2hugo.tsv',sep='\t')
gene2hugo = {str(row['GeneID']):row['HUGO'] for i,row in dfgene2hugo.iterrows()}
hugo2geneid = {row['HUGO']:str(row['GeneID']) for i,row in dfgene2hugo.iterrows()}


df_ensembl = pd.read_csv(root_folder+'tests/ensembl_hugo_entrez_conversion.tsv',sep='\t')
human_entrez = set([str(int(row["NCBIgeneID"])) for i,row in df_ensembl.iterrows() if not np.isnan(row["NCBIgeneID"])])
print('Number of human GeneIDs:',len(human_entrez))


### ----------------------
# Loading the function names
if not link_to_predict=='gene_pathway':
    if os.path.exists(root_folder+'data/goid2name.pkl'):
        with open(root_folder+'data/goid2name.pkl','rb') as f:
            [goid2name,goid2definition] = pickle.load(f)
    else:
            df_goannot = pd.read_csv(root_folder+'data/GOannot.tsv',sep='\t')
            goid2name = {row['GO term accession']:row['GO term name'] for i,row in tqdm(df_goannot.iterrows())}
            goid2definition = {row['GO term accession']:row['GO term definition'] for i,row in tqdm(df_goannot.iterrows())}
            with open(root_folder+'data/goid2name.pkl','wb') as f:
                pickle.dump([goid2name,goid2definition],f)
else:
    if os.path.exists(root_folder+'data/pathwayid2name.pkl'):
        with open(root_folder+'data/pathwayid2name.pkl','rb') as f:
            goid2name = pickle.load(f)
    else:
            df_pathwayannot = pd.read_csv(root_folder+'tests/pathways.tsv',sep='\t')
            goid2name = {row['identifier']:row['name'] for i,row in tqdm(df_pathwayannot.iterrows())}
            with open(root_folder+'data/pathwayid2name.pkl','wb') as f:
                pickle.dump(goid2name,f)


### ----------------------
# create networkx graph for DRKG, creating map to nodetype

DRKG_graph = nx.Graph()
lines = []
with open(root_folder+"data/drkg.tsv") as f:
    lines = f.readlines()
lines = lines[1:]
print('Number of lines',len(lines))
for l in tqdm(lines):
    parts = l.split('\t')
    DRKG_graph.add_edge(parts[0].strip(),parts[2].strip())
print('Nodes:',DRKG_graph.number_of_nodes())
print('Edges:',DRKG_graph.number_of_edges())


if os.path.exists(root_folder+'data/id2nodetype.pkl'):
    with open(root_folder+'data/id2nodetype.pkl','rb') as f:
        id2nodetype = pickle.load(f)
else:
    # Making global graph id mapping
    id2nodetype = {str(n.replace('::','@').split('@')[1]):str(n.replace('::','@').split('@')[0]) for n in DRKG_graph.nodes}
    with open('data/id2nodetype.pkl','wb') as f:
        pickle.dump(id2nodetype,f)


head_degrees = np.zeros(len(dz_mapping))
for node,deg in nx.degree(DRKG_graph):
    if node in dz_mapping:
        head_degrees[dz_mapping[node]] = deg

genehead_degrees = {}
for node,deg in nx.degree(DRKG_graph):
    genehead_degrees[node] = deg

print(subset_dict(dz_mapping, 4))
print(subset_dict(gene_mapping, 4))

if os.path.exists(list_of_function_file):
    with open(list_of_function_file,'r') as f:
        lines = f.readlines()
    lines = [line.split("\n")[0] for line in lines]
    irx = np.array([dz_mapping[function_prefix+l] for l in lines])
else:
    irx = np.where((head_degrees<500)&(head_degrees>=20))[0]
    head_node_selection = [inverse_dz_mapping[i] for i in irx]
    if link_to_predict!='gene_pathway':
        head_node_selection = [s[s.index('GO:'):] for s in head_node_selection]
    else:
        head_node_selection = [s[s.index('Pathway::'):] for s in head_node_selection]
    with open(list_of_function_file,'w') as f:
        f.write('\n'.join(head_node_selection))
    print('Number of functions in selection:',len(irx),len(set(head_node_selection)))


### ----------------------
number_of_diseases = len(dz_mapping)
number_of_genes = len(gene_mapping)
number_of_others = num_nodes-number_of_genes-number_of_diseases

irx_dz = np.arange(number_of_diseases)
irx_dz = irx
irx_gene = np.arange(number_of_diseases,number_of_diseases+number_of_genes)
irx_other = np.arange(number_of_diseases+number_of_genes,number_of_diseases+number_of_genes+number_of_others)
irx_all = np.array(list(irx_dz)+list(irx_gene)+list(irx_other))

#degrees = degree(drkg_object_dict["data_object"].edge_index[0]).numpy()
degrees = degree(train_dataset.edge_index[0]).numpy() # TODO: Assuming all nodes are present in the train data itself


nodetypes = ['None']*len(degrees)
for d in inverse_dz_mapping:
    if (d in inverse_dz_mapping)&(inverse_dz_mapping[d][len(function_prefix):] in id2nodetype):
        ntype = function_prefix[:-2]#'Function'#id2nodetype[inverse_dz_mapping[d]]
        nodetypes[d] = ntype
    else:
        print('head',d,inverse_dz_mapping[d])
for g in inverse_gene_mapping:
    if (g in inverse_gene_mapping)&(inverse_gene_mapping[g][len('Gene::'):] in id2nodetype):    
        ntype = 'Gene'#id2nodetype[inverse_gene_mapping[g]]
        nodetypes[g] = ntype
    else:
        print('tail',g,inverse_gene_mapping[g])
for r in inverse_rest_mapping:
    if ':' in inverse_rest_mapping[r]:
        nodetypes[r] = inverse_rest_mapping[r].split(':')[0]
    if (r in inverse_rest_mapping)&(inverse_rest_mapping[r] in id2nodetype):    
        ntype = id2nodetype[inverse_rest_mapping[r]].split(':')[0]
        nodetypes[r] = ntype

nodenames = []
for d in inverse_dz_mapping:
    id = inverse_dz_mapping[d]
    nodenames.append(id)
for g in inverse_gene_mapping:
    id = inverse_gene_mapping[g]
    nodenames.append(gene2hugo.get(id[len('Gene::'):],id))
for r in inverse_rest_mapping:
    id = inverse_rest_mapping[r]
    nodenames.append(id)

print(Counter(nodetypes))


### ----------------------
################### LOADING MODEL AND DEFINING THE SPLIT
with open('configurations/prediction_tasks/'+link_to_predict+'.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open('configurations/config.yaml', 'r') as file:
    config_all = yaml.safe_load(file)

config = DictConfig(config)
print(config)

config_all = DictConfig(config_all)
print(config_all)

print('Producing the training/test split')
Encoder = VariationalGCNEncoder(
                num_nodes,
                config.NN_settings.HIDDEN_SIZE,
                config.NN_settings.OUT_CHANNELS,
                dropout=config.NN_settings.P_DROPOUT,
            )
Decoder = None
vgae_model = VGAE(encoder=Encoder, decoder=Decoder)  
print('Loading model weights')
vgae_model.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
vgae_model.eval()
print('train_dataset:',train_dataset)
print('test_dataset:',test_dataset)
print('val_dataset:',val_dataset)

noise_level = 0.01
data = test_dataset

plot_roc_curve(vgae_model, data)
plt.savefig("trained_models/"+link_to_predict+"/"+link_to_predict+"_roc.png")
plt.show()


model = vgae_model
model.eval()
x = data.x
z = model.encode(x, data.edge_index)
pos_preds = model.decode(z, data.pos_edge_label_index, sigmoid=False)
neg_preds = model.decode(z, data.neg_edge_label_index, sigmoid=False)
preds = torch.cat([pos_preds, neg_preds], dim=0)
preds = preds.detach().cpu().numpy()
preds += np.random.rand(len(preds)) * noise_level
labels = torch.cat((data.pos_edge_label, data.neg_edge_label), dim=0)
labels = labels.detach().cpu().numpy()

print(pos_preds.shape, neg_preds.shape)
# plt.hist(preds,bins=100)
# plt.title('Sigmoid(dot product), predictions')
# plt.show()

plt.hist(preds, bins=50)
plt.title("Sigmoid(Dot product)")
plt.savefig("trained_models/"+link_to_predict+"/"+link_to_predict+"_scorehist.png")
plt.show()

df = pd.DataFrame(data={"score": preds, "label": labels})
sns.displot(df, x="score", hue="label", kind="kde")
plt.title("sigmoid(dot product), test dataset")
plt.savefig("trained_models/"+link_to_predict+"/"+link_to_predict+"_distplot.png")
plt.show()
del(df)

### ----------------------
# Compute UMAP

embedding, z = visualize_umap_embeddings(
    vgae_model,
    train_dataset,
    "Trained VGAE: train set embeddings UMAP",
    labeled=False,
    sizes=degrees,
    number_of_diseases=len(dz_mapping),
    number_of_genes=len(gene_mapping),
    data_path=None,
    removeFirstPC=0,
)

with open(root_folder+'trained_models/'+link_to_predict+'/'+ link_to_predict+'_embed_UMAP.pkl','wb') as f:
    pickle.dump(embedding,f)
with open(root_folder+'trained_models/'+link_to_predict+'/'+ link_to_predict+'_embed.pkl','wb') as f:
    pickle.dump(z,f)


### ----------------------
# Redraw the UMAP plot
sizes = degrees.copy()

# higlight_genes = ['84274','84266','132158']
higlight_genes = []
nodecolors = np.array(nodetypes)

for hg in higlight_genes:
    nodecolors[gene_mapping[hg]] = 'HIGHLIGHT'
    sizes[gene_mapping[hg]] = 1000000
    print(gene2hugo[hg],gene_mapping[hg])

nodecolors = nodecolors[irx_all]
nodesizes = list(sizes[irx_dz]*100+100)+list(sizes[irx_gene]*5+100)+list(np.ones(number_of_others)*500+50)

df = {'UMAP1':embedding[irx_all, 0],
      'UMAP2':embedding[irx_all, 1],
      #'color': ["functions"] * number_of_diseases + ["genes"] * number_of_genes + ['other'] * number_of_others,
      'color':nodecolors,
      #'size': np.array(list(sizes[:number_of_diseases] * 20) + list(sizes[number_of_diseases:-number_of_others] * 5) + list(sizes[:number_of_others] * 5))[irx_all],
      'size': nodesizes,
      'degree':degrees[irx_all],
      'nodename':np.array(nodenames)[irx_all],
      }

sns.scatterplot(df,x='UMAP1',y='UMAP2',hue='color', size='size')
plt.savefig("trained_models/"+link_to_predict+"/"+link_to_predict+"_umap.png")
plt.show()

with open(root_folder+'trained_models/'+link_to_predict+'/'+ link_to_predict+'_embed_nodetypes.pkl','wb') as f:
    pickle.dump(nodecolors,f)

assert len(np.array(nodenames)) == len(irx_all)

### ----------------------
# Produce the prediction list
# Use all possible negative edges for predictions and estimating hits@k

print('Creating list of all possible edges:')

alledges_list = []
alledges_iter = itertools.product(list(irx),list(irx_gene))
for ep in tqdm(alledges_iter):
    alledges_list.append(list(ep))
alledges_array = np.array(alledges_list).astype(np.int32).T
alledges_tensor = torch.Tensor(alledges_array).int()

x = train_dataset.x
z = vgae_model.encode(x, train_dataset.edge_index)
#all_preds = model.decode(z, alledges_tensor, sigmoid=False)

### ----------------------
# writing down the prediction table memory-efficient

zn = z.detach().cpu().numpy()
del(z)
del(x)
del(vgae_model)

pred_functions_id = []
pred_functions = []
pred_genes_id = []
pred_genes = []
dot_products = []
pred_status = []
dz_connectivity = []
gene_connectivity = []

postrain = set()
postest = set()
posval = set()
for i in range(len(train_dataset.pos_edge_label_index.T)):
    ep = train_dataset.pos_edge_label_index[:,i]
    ep = (int(ep[1]),int(ep[0]))
    postrain.add(ep)
for i in range(len(test_dataset.pos_edge_label_index.T)):
    ep = test_dataset.pos_edge_label_index[:,i]
    ep = (int(ep[1]),int(ep[0]))
    postest.add(ep)
for i in range(len(val_dataset.pos_edge_label_index.T)):
    ep = val_dataset.pos_edge_label_index[:,i]
    ep = (int(ep[1]),int(ep[0]))
    posval.add(ep)

print('Length of postrain:',len(postrain))
print('Length of postest:',len(postest))
print('Length of posval:',len(posval))

with open('trained_models/'+link_to_predict+'/'+link_to_predict+'_testset.tsv','w') as f:
    for e in postest:
        f.write(str(e[0]))
        f.write('\t')
        f.write(str(e[1]))
        f.write('\n')

print('Deleting the data object from memory')
del(data)
del(train_dataset)
del(test_dataset)
del(val_dataset)
del(embedding)

with open(prediction_file,'w') as f:
    f.write("\tDISEASE_ID\tDISEASE\tGENE_ID\tGENE\tDISEASE_DEGREE\tGENE_DEGREE\tDotProduct\tStatus\n")
    for i,ep in tqdm(enumerate(alledges_list)):    
        pred_function_id = inverse_dz_mapping[ep[0]]
        pred_function = goid2name.get(inverse_dz_mapping[ep[0]][len(function_prefix):],inverse_dz_mapping[ep[0]])
        pred_gene_id = inverse_gene_mapping[ep[1]]
        pred_gene = gene2hugo.get(inverse_gene_mapping[ep[1]][len('Gene::'):],inverse_gene_mapping[ep[1]])
        dp = np.dot(zn[ep[0],:],zn[ep[1],:])
        dot_product = dp
        st = 'None'
        e = (ep[1],ep[0])
        if e in postrain:
            st = 'Train'
        if e in postest:
            st = 'Test'
        if e in posval:
            st = 'Validation'                
        pred_status = st
        gene_connectivity = degrees[e[1]]
        dz_connectivity = degrees[e[0]]
        #f.write(f"{i}\t{ep[0]}\t{ep[1]}\t{pred_function_id}\t{pred_function}\t{pred_gene_id}\t{pred_gene}\t{dz_connectivity}\t{gene_connectivity}\t{dot_product}\t{pred_status}\n")
        f.write(f"{i}\t{pred_function_id}\t{pred_function}\t{pred_gene_id}\t{pred_gene}\t{dz_connectivity}\t{gene_connectivity}\t{dot_product}\t{pred_status}\n")