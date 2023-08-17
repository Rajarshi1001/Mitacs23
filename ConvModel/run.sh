#!/bin/bash

if [[ -f id_bonds_clean_train.csv && -f id_bonds_clean_val.csv ]]; then
    echo "training and validation files exists"
    python3 ABlinear_nn2.py id_bonds_clean_train.csv id_bonds_clean_val.csv
else
    mv ./../LinearModel/id_bonds_clean_train.csv ./../LinearModel/id_bonds_clean_val.csv .
    python3 ABlinear_nn2.py id_bonds_clean_train.csv id_bonds_clean_val.csv
fi

# python3 ABlinear_nn2_updated.py id_combined_init_updated.csv