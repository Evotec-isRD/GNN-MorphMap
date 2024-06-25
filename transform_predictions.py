import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

link_to_predict = sys.argv[1]
print('link_to_predict=', link_to_predict)

# Transforming the table into gene-function format: takes several minutes
print('Starting to transform prediction...')
prediction_file = 'trained_models/'+link_to_predict+'/'+link_to_predict+'_predictions.tsv'
# Transforming the table into gene-function format: takes several minutes
df_predictions = pd.read_csv(prediction_file,sep='\t',index_col=0)

genes = list(df_predictions['GENE_ID'])
diseases = list(df_predictions['DISEASE_ID'])
dps = list(df_predictions['DotProduct'])
del(df_predictions)
unique_genes = list(set(genes))
unique_dz = list(set(diseases))
gene2id = {unique_genes[i]:i for i in range(len(unique_genes))}
print(len(unique_genes),len(unique_dz))
Xfd = {}
for i,g in tqdm(enumerate(genes)):
    d = diseases[i]
    v = Xfd.get(d,np.zeros(len(unique_genes)))
    v[gene2id[g]] = dps[i]
    Xfd[d] = v
df_gene_function = pd.DataFrame(data=Xfd)
df_gene_function.index = unique_genes
df_gene_function.index = unique_genes
#display(df_gene_function)
df_gene_function.to_csv("trained_models/"+link_to_predict+"/"+link_to_predict+"_genefunction.tsv",sep='\t')