!/usr/bin/bash

# Enable the conda environment
eval "$(conda shell.bash hook)"
conda activate annotation
PYTHONPATH=$PYTHONPATH:./

modeltype=earlyfusion
#earlyfusion
#baseline
#facebook/bart-base 
for modelbase in facebook/bart-base facebook/bart-large
do
python pn_bartmodel.py -only_eval -epochs 10 -ws 0 -wr 0.21 -bs 8 -mt $modeltype -mb $modelbase -output_path outputs
done
#_full_rated
#modelbase=facebook/bart-large
#for wr in 1e-1 25e-2 30e-2 21e-2 19e-2
#do
#python pn_bartmodel.py -epochs 10 -ws 0 -wr $wr -bs 8 -mt $modeltype -mb $modelbase -output_path outputs_full_rated$mr
#done