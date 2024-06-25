#!/bin/bash
#SBATCH --job-name=run_morphmap
#SBATCH --output=logs/train_output_bp_mf_pathway.txt
#SBATCH --error=logs/train_error_bp_mf_pathway.txt
#SBATCH --partition=standard.q
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --verbose

clear

module load cuda11.2
module load cudnn8.1-cuda11.2

export LD_PRELOAD=""

export PYTHONPATH=${PYTHONPATH}:morphmap_analysis
cd morphmap_analysis

# Predict
## Original DRKG
python morphmap_analysis/make_predictions.py gene_mf
python morphmap_analysis/make_predictions.py gene_bp
python morphmap_analysis/make_predictions.py gene_pathway

## test only GO
python morphmap_analysis/make_predictions.py gene_mf__go
python morphmap_analysis/make_predictions.py gene_bp__go

### Enriched GO 0.30
python morphmap_analysis/make_predictions.py gene_mf__go__enriched_0.30__QC
python morphmap_analysis/make_predictions.py gene_bp__go__enriched_0.30__QC

### Enriched GO 0.40
python morphmap_analysis/make_predictions.py gene_mf__go__enriched_0.40__QC
python morphmap_analysis/make_predictions.py gene_bp__go__enriched_0.40__QC

### Enriched GO 0.50
python morphmap_analysis/make_predictions.py gene_mf__go__enriched_0.50__QC
python morphmap_analysis/make_predictions.py gene_bp__go__enriched_0.50__QC


# ----------------------------------------
# Compute scores
python morphmap_analysis/compute_scores.py orf
python morphmap_analysis/compute_scores.py crispr


# ----------------------------------------
# Make prediction GO models, and x10 enriched models
###  gene_bp_gox10
python morphmap_analysis/make_predictions_traverse.py gene_bp__go__QC__x10 gene_bp__go

###  gene_bp_gox10_enriched
python morphmap_analysis/make_predictions_traverse.py gene_bp__go__enriched_0.50__QC__x10 gene_bp__go__enriched_0.50__QC

###  gene_mf_gox10
python morphmap_analysis/make_predictions_traverse.py gene_mf__go__QC__x10 gene_mf__go

###  gene_mf_gox10_enriched
python morphmap_analysis/make_predictions_traverse.py gene_mf__go__enriched_0.50__QC__x10 gene_mf__go__enriched_0.50__QC