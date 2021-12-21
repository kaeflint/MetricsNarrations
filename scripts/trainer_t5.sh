!/usr/bin/bash

# Enable the conda environment
eval "$(conda shell.bash hook)"
conda activate annotation
PYTHONPATH=$PYTHONPATH:./

modeltype=baseline
# t5-large
#t5-small t5-base
for modelbase in t5-small t5-base t5-large
do
python pn_t5model.py -epochs 10 -bs 4 -mt $modeltype -mb $modelbase -output_path outputs_full_rated
done