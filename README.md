# GNN-MorphMap

This repository hosts knowledge graph analysis scripts associated with our paper titled 'Title'. 
The parent repository can be found at: [2024_Chandrasekaran_Morphmap](https://github.com/jump-cellpainting/2024_Chandrasekaran_Morphmap).

## Installation

```bash
conda env create -f environment.yml
conda activate morphmap
poetry install
```

### Usage:
	python make_predictions.py gene_mf
	python transform_predictions.py gene_mf
	python make_predictions.py gene_bp
	python transform_predictions.py gene_bp
	python make_predictions.py gene_pathway
	python transform_predictions.py gene_pathway

This will generate prediction table, gene-function table, and some png files for every model type

	python compute_scores.py orf
	python compute_scores.py crispr
	
This will generate *orf_scores_merged.tsv* file and *crispr_scores_merged.tsv* in the data folder

### Required files in the data/ folder : 

	drkg.tsv
	GOannot.tsv
	pathways.tsv
	orf_scores_clean.zip
	crispr_scores_clean.zip
	supervised_weak/weakly_supervised_orf__similarity.csv
	supervised_strong/strongly_supervised_orf__similarity.csv
	supervised_weak/weakly_supervised_crispr__similarity.csv
	supervised_strong/strongly_supervised_crispr__similarity.csv

Each trained model file should be accompanied by the data object
	
	trained_model/gene_mf/gene_mf_data_object.pt
	trained_model/gene_bp/gene_bp_data_object.pt
	trained_model/gene_pathway/gene_pathway_data_object.pt

### Notebooks

1. [ModelAccuracy.ipynb](ModelAccuracy.ipynb) : evaluates the accuracy of model predictions, also allows browsing the model prediction file
2. [Scatterplot.ipynb](Scatterplot.ipynb) : compares model predictions (KG score) and the MorphMap similarities, produces conditional density plots and the plot of explained fraction of gene-gene pairs
3. [Heatmap.ipynb](Heatmap.ipynb) : for a set of genes, produces a clustergram based on MorphMap similarities, with KG score indicated on top of the cells
4. [SLCOR.ipynb](SLCOR.ipynb) : produces a set of figures illustrating the connection between Solute Carriers and Olfactory Receptors

#### ORF and CRISPR data (Shared on April 4th, 2024)

Here are the ORF and CRISPR profiles

ORF: https://drive.google.com/file/d/1Xe3VMqkOtMNAeoZnxl-XZTFEHQSqe0O0/view?usp=drive_link

CRISPR: https://drive.google.com/file/d/1yW1vZydttyvb7ebfTotuHFP8O_ZDIr_w/view?usp=drive_link 

 and the cosine similarity matrices

ORF: https://drive.google.com/file/d/1Uo27o3jUVMb0u4wN2cTCpOjG51nWZL6P/view?usp=drive_link

CRISPR: https://drive.google.com/file/d/1QBPSXm9LG2kT25EGXAqgtwPVkdFf5dbR/view?usp=drive_link