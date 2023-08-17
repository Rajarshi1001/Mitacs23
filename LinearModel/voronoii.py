import numpy as np
import pandas as pd
import os, csv
import pymatgen
from pymatgen.core.structure import Structure
import pymatgen.analysis.local_env
import json, shutil, glob, sys, warnings
from mp_api.client import MPRester

DATA_DIR = "combined_binary-cifs"
APIKEY = "ZCYEmnXt58eSTj0pMZJWslELPcqaQojc"
MAX_NUM_NBRS = 4
RADIUS = 8
# print(len(os.listdir(DATA_DIR)))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)   

# Reading all the material_ids from the text file and downloading the CIF structure files in a directory
def download_files(file, to_cif = False):

    material_ids = []
    with open(file, "r") as file:
        lines = file.readlines()
        for buffer in lines:
            id = str(buffer.strip().split(",")[0])
            material_ids.append(id)

    with MPRester(APIKEY) as mpr:
        docs = mpr.summary.search(material_ids = material_ids, fields = ["material_id", "structure", "band_gap"])       
        for i in docs:
            if to_cif is True:
                i.structure.to("{}.cif".format(i.material_id), "cif")
                shutil.move("{}.cif".format(i.material_id), os.path.join(DATA_DIR, "{}.cif".format(i.material_id)))
            print(i.material_id, i.band_gap, sep = ",")

    print("{} CIF files saved..".format(len(os.listdir(DATA_DIR))))

# fetching the atomic number of the species
def get_num(specie):
        try:
            return specie.number
        except AttributeError:
            if isinstance(specie, pymatgen.core.periodic_table.DummySpecies):
                try:
                    return - pymatgen.core.periodic_table.Specie(specie.symbol[2:]).number
                except:
                    return -1

# Retrieving the bond length of a given material_id via a CGCNN helper function
def getBondLength(material_id):

    ari = json.load(open("atom_init.json", "r"))

    idx = os.path.join(DATA_DIR, "{}.cif".format(material_id))
    crystal=Structure.from_file(idx,'cif')
    for i in range(len(crystal)):
        if get_num(crystal[i].specie) <= 92:
            atom_fea = np.vstack([ari[str(get_num(crystal[i].specie))]])
        else: 
            atom_fea = np.vstack([ari[str(-1)]])
    
        all_nbrs = crystal.get_all_neighbors_py(RADIUS, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    
    nbr_fea_idx, nbr_fea = [], []
    
    for nbr in all_nbrs:
        if len(nbr) < MAX_NUM_NBRS:
            warnings.warn('not find enough neighbors to build graph. '
                          'If it happens frequently, consider increase '
                          'radius.')
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (MAX_NUM_NBRS - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) + [RADIUS + 1.] * (MAX_NUM_NBRS - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:MAX_NUM_NBRS])))
            nbr_fea.append(list(map(lambda x: x[1], nbr[:MAX_NUM_NBRS])))

    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)

    return material_id, nbr_fea[0][0]

def getBandGaps():

    material_ids = []
    with MPRester(APIKEY) as mpr:
        with open("mpids_all.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                material_ids.append(str(line))
        docs = mpr.summary.search(material_ids = material_ids, fields = ["material_id", "band_gap", "elements"])
        for i in docs:
            with open("id_bonds_clean_train.csv", "a") as file:
                writer = csv.writer(file)
                contents = []
                mp_id, bondlen = getBondLength(i.material_id)
                contents.append(i.material_id)
                contents.append(np.round(i.band_gap, 4))
                print(i.elements, i.material_id)
                if len(i.elements) != 0:
                    for element in i.elements:
                        contents.append(element.symbol)
                contents.append(bondlen)

                writer.writerow(contents)


def calculateVoronoiiNN(crystal):

    nbr_fea_idx_v, nbr_fea_v = [], []
    try:
        voronoii = pymatgen.analysis.local_env.VoronoiNN(tol = 0.1, compute_adj_neighbors = False, cutoff = 8)
        all_nbrs_v = voronoii.get_all_nn_info(crystal)
        all_nbrs_v = [[(i["site_index"], i["poly_info"]["face_dist"]*2) for i in nbrs] for nbrs in all_nbrs_v]
        for nbr in all_nbrs_v:
            nbr_fea_idx_v.append(list(map(lambda x: x[0], nbr)))
            nbr_fea_v.append(list(map(lambda x: x[1], nbr)))
    except:
        pass
    
    return nbr_fea_v, nbr_fea_idx_v


def main():

    files = os.listdir(DATA_DIR)
    print("Calculated {} files...".format(len(files)))
    for file in files:
        crystal = Structure.from_file(os.path.join(DATA_DIR, file), "cif")
        face_dist, idxs = calculateVoronoiiNN(crystal)
        # print("calculated voronoii NN")
        comp = str(crystal.formula).replace(" ","")
        material_id = os.path.splitext(file)[0]

        with open("mpids_all.txt", "a") as file:
            file.write(material_id + "\n")

        with open("Voronoii.csv", "a") as csvfile:
            fieldnames = ["MP-id", "Formula"]
            row = []         
            row.append(material_id)
            row.append(comp)
            if len(idxs) == 0:
                pass
            else:
                for idx in idxs:
                    row.append(len(idx))

            writer = csv.writer(csvfile)
            writer.writerow(row)

if __name__ == "__main__":
    if os.path.exists("Voronoii.csv"):
        os.remove("Voronoii.csv")
    # if os.path.exists("mpids_all.txt"):
    #     os.remove("mpids_all.txt")
    # if os.path.exists("id_bonds_all_binaries.csv"):
    #     os.remove("id_bonds_all_binaries.csv")

    num = len(os.listdir(DATA_DIR))

    if num > 0:
        pass
    else: 
        download_files("all_2-elem.txt", to_cif = True)
    
    main()
    getBandGaps()


