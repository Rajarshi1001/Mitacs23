import os
import csv
import numpy as np
import pandas as pd
from mp_api.client import MPRester

APIKEY = "ZCYEmnXt58eSTj0pMZJWslELPcqaQojc"

def prepare(file, outfile):
    data = pd.read_csv(file)
    for idx, row in data.iterrows():
        material_id = row["id"]
        atom1 = row["atom1"]
        atom2 = row["atom2"]
        target = row["target"]
        bondlen = row["AB-bondlen"]
        contents =[]
        contents.append(material_id)
        contents.append(target)
        contents.append(atom1)
        contents.append(atom2)
        contents.append(bondlen)

        with open(outfile, "a") as file:
            
            writer = csv.writer(file)
            writer.writerow(contents)


if __name__ == "__main__":
    if os.path.exists("id_bonds_clean_train.csv"):
        os.remove("id_bonds_clean_train.csv")
    if os.path.exists("id_bonds_clean_val.csv"):
        os.remove("id_bonds_clean_val.csv")
    prepare("clean_train.csv", "id_bonds_clean_train.csv")
    prepare("clean_val.csv", "id_bonds_clean_val.csv")