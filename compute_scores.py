import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba
from numba import jit
import sklearn.metrics
import sys

score_type = sys.argv[1]
print('score_type=',score_type)

path_to_orf_scores = 'data/'+score_type+'_scores_clean.zip'
# link_to_predict_list = ['gene_mf','gene_bp','gene_pathway']
link_to_predict_list = ['gene_mf__go', 'gene_bp__go', 'gene_pathway']
#link_to_predict_list = ['gene_mf','gene_mf__go','gene_mf__go__enriched_0.30__QC', 'gene_mf__go__enriched_0.40__QC', 'gene_mf__go__enriched_0.50__QC', 'gene_mf__go__enriched_0.50__QC',
#                        'gene_bp','gene_bp__go', 'gene_bp__go__enriched_0.30__QC', 'gene_bp__go__enriched_0.40__QC', 'gene_bp__go__enriched_0.50__QC', 'gene_bp__go__enriched_0.50__QC',
#                        'gene_pathway']

pdtemp = pd.read_csv('tests/biomart_hugogeneid.tsv',sep='\t')
hugo2geneid = {row['HUGO']:row['GeneID'] for i,row in pdtemp.iterrows()}
geneid2hugo = {row['GeneID']:row['HUGO'] for i,row in pdtemp.iterrows()}


@jit
def pvalscore(vec1,vec2,random_dpvals):
     # centered dot product
     x = vec1-np.mean(vec1)
     y = vec2-np.mean(vec2)
     dst = np.dot(x,y)

     # centered correlation
     #dst = np.dot(vec1-np.mean(vec1),vec2-np.mean(vec2))/np.std(vec1-np.mean(vec1))/np.std(vec2-np.mean(vec2))
     return dst

def sigmoid_norm(z):
    #zn = z/np.std(z)
    zn = z
    zn = 1/(1 + np.exp(-zn))
    zn = (zn-0.5)*2.0
    return zn

def compute_scores_list():
     scores_list = []
     for link_to_predict in link_to_predict_list:
        print(link_to_predict)
        scores = compute_scores(dict_geneids2index[link_to_predict],dict_Xf[link_to_predict],dict_random_dpvals[link_to_predict])
        scores_list.append(scores)
     return scores_list

def compute_scores(geneids2index,Xf,random_dpvals):
        scores = {}
        k = 0
        for gs,gt in tqdm(orfsimilarities):
            if (gs in hugo2geneid)&(gt in hugo2geneid):
                orfsim = np.abs(orfsimilarities[(gs,gt)])
                if (k%1 == 0)|(orfsim>0.35): # for testing, we use only each 100th value
                    i1 = geneids2index.get(hugo2geneid[gs],0)
                    i2 = geneids2index.get(hugo2geneid[gt],0)
                    if i1*i2>0:
                        vec1 = Xf[i1,:]
                        vec2 = Xf[i2,:]
                        score = pvalscore(vec1,vec2,random_dpvals)
                        #pval = np.sum(random_dpvals>dst)/len(random_dpvals)
                        scores[(gs,gt)] = score
            k+=1
        return scores


dict_geneids2index = {}
dict_Xf = {}
dict_random_dpvals = {}
dict_func_similarity = {}

for link_to_predict in link_to_predict_list:
    func_similarity = pd.read_csv('trained_models/'+link_to_predict+'/'+link_to_predict+'_genefunction.tsv',sep='\t')
    dict_func_similarity[link_to_predict] = func_similarity
    Xf = func_similarity[func_similarity.columns[1:]].to_numpy()
    number_of_genes = Xf.shape[0]

    # Build null distribution of dotproducts
    num_of_samples = 10000
    random_dpvals = np.zeros(num_of_samples)
    for i in range(num_of_samples):
        i1 = np.random.randint(number_of_genes)
        i2 = np.random.randint(number_of_genes)
        #random_dpvals[i] = np.linalg.norm(Xf[i1,:]-Xf[i2,:])
        #random_dpvals[i] = np.dot(Xf[i1,:],Xf[i2,:])/(np.linalg.norm(Xf[i1,:])*np.linalg.norm(Xf[i2,:]))
        random_dpvals[i] = np.corrcoef(Xf[i1,:],Xf[i2,:])[0,1]

    plt.hist(random_dpvals,bins=200)
    plt.title(link_to_predict)
    plt.xlabel("Corr")
    plt.show()


    t = np.array(func_similarity[func_similarity.columns[0]])
    geneids = [int(g[6:]) for g in t]
    geneids2index = {}
    for i,gid in enumerate(geneids):
        k = -1
        if 'Gene::'+str(gid) in t:
            k= np.where(t=='Gene::'+str(gid))[0][0]
        geneids2index[gid] = k

    dict_geneids2index[link_to_predict] = geneids2index
    dict_Xf[link_to_predict] = Xf
    dict_random_dpvals[link_to_predict] = random_dpvals

  
orf_pairs = pd.read_csv(path_to_orf_scores,sep=',')
print(len(orf_pairs))

sources = list(orf_pairs['SOURCE_NAME'])
print('red sources')
targets = list(orf_pairs['TARGET_NAME'])
print('red targets')
cosim = list(orf_pairs['SIMILARITY_SCORE'])
print('red cosims')
orfsimilarities = {(sources[i],targets[i]):cosim[i] for i in tqdm(range(len(sources)))}


scores_list = compute_scores_list()
# normalization
spread_coefficient = 1.5
if True:
    for score in scores_list:
        xd = np.array([score[k] for k in score])
        std = np.std(xd)
        print(xd.shape,std)
        for key in score:
            score[key] = sigmoid_norm(score[key]/(spread_coefficient*std))
            
with open('data/'+score_type+'_scores_merged.tsv',"w") as f:
    f.write('GENE1\tGENE2\t'+score_type.upper()+'_SIM\tABS_'+score_type.upper()+'_SIM')
    for i,link_to_predict in enumerate(link_to_predict_list):
        f.write('\t'+link_to_predict)
    f.write('\tunsupervised_max')
    f.write('\n')
    for gs,gt in tqdm(orfsimilarities):
        if (gs in hugo2geneid)&(gt in hugo2geneid):
            found = True
            for sc in scores_list:
                if not (gs,gt) in sc:
                    found = False
            if found:
                f.write(gs+'\t'+gt+'\t'+'{:2.3f}'.format(orfsimilarities[(gs,gt)])+'\t'+'{:2.3f}'.format(np.abs(orfsimilarities[(gs,gt)])))
                maxval = -100
                for i,link_to_predict in enumerate(link_to_predict_list):
                    val = scores_list[i][(gs,gt)]
                    f.write('\t{:2.3f}'.format(val))
                    if val>maxval:
                        maxval = val
                f.write('\t{:2.3f}'.format(maxval))                
                f.write('\n')
