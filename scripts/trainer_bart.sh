!/usr/bin/bash

# Enable the conda environment
eval "$(conda shell.bash hook)"
conda activate annotation
PYTHONPATH=$PYTHONPATH:./

modeltype=earlyfusion
# t5-large
#t5-small t5-base
for modelbase in facebook/bart-large facebook/bart-base 
do
python pn_bartmodel.py -ws 800 -bs 8 -mt $modeltype -mb $modelbase -output_path outputs
done