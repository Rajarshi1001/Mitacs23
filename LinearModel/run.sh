#!/bin/bash

if [[-f id_bonds_clean_train.csv && -f id_bonds_clean_train.csv]]; then
    echo "training and validation file exists"
    python3 ABlinear_nn.py id_bonds_clean_train.csv id_bonds_clean_val.csv
else
    data_dir="ConvModel"
    mv ./../{data_dir}/id_bonds_clean_train.csv ./../{data_dir}/id_bonds_clean_val.csv .
    python3 ABlinear_nn.py id_bonds_clean_train.csv id_bonds_clean_val.csv
fi

# python3 ABlinear_nn_updated.py id_combined_init_updated.csv