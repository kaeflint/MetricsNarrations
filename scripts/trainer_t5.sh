!/usr/bin/bash

# Enable the conda environment
eval "$(conda shell.bash hook)"
conda activate annotation
PYTHONPATH=$PYTHONPATH:./

#!/bin/bash

VAR1="t5-large"
VAR2="Linuxize"
if [ "$VAR1" = "t5-large" ]; then
    echo "Strings are equal."
else
    echo "Strings are not equal."
fi



modeltype=baseline
# t5-large
#t5-small t5-base
for modelbase in t5-large
do
echo $modelbase
if [ "$modelbase" = "t5-large" ]; then
    bs=4
else
    bs=8
fi
echo $bs
python pn_t5model.py -epochs 20 -bs $bs -mt $modeltype -mb $modelbase -output_path outputs_new
done
#outputs_full_raw -use_raw 
#outputs_full_rated
#
#