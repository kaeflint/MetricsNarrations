!/usr/bin/bash

# Enable the conda environment
eval "$(conda shell.bash hook)"
conda activate annotation
PYTHONPATH=$PYTHONPATH:./


# t5-large
#t5-small t5-base
#modelbase=t5-small
#for seedd in 128 456 3087 1984
#do
#python pn_t5model.py -sc -epochs 20 -bs 8 -mt $modeltype -mb $modelbase -output_path outputs_new --seed $seedd
#done
#outputs_full_raw -use_raw 
#outputs_full_rated


#modeltype=baseline
#43 128 456 3087 1984
for sed in 48 128 456 3087 1984
do 
    echo $sed
    for modelbase in facebook/bart-base facebook/bart-large
    do 
        echo $modelbase
        if [ "$modelbase" = "t5-large" ]; then
            bs=4
            epochs=10

        else
            bs=8
            epochs=20
        fi
        echo $bs
        for modeltype in baseline earlyfusion
        do
        echo $modeltype
        python pn_bartmodel.py -wr 0.21 -sc -epochs $epochs -bs $bs -mt $modeltype -mb $modelbase -output_path outputs_new --seed $sed
        done
    done
done 