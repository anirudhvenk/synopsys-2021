import matplotlib as plt
from pandas.core.common import flatten
import numpy as np
import time
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import Draw


def isValid(mol):
    if mol == None or len(mol) <= 3:
        return False
    mol = Chem.MolFromSmiles(mol)
    if mol == None:
        return False
    else:
        try:
            # if molecule is not drawable, the molecule is not valid
            Draw.MolToImage(mol)
            return True
        except:
            return False


def get_h_bond_donors(mol):
    idx = 0
    donors = 0
    while idx < len(mol)-1:
        if mol[idx].lower() == "o" or mol[idx].lower() == "n":
            if mol[idx+1].lower() == "h":
                donors += 1
        idx += 1
    return donors


def get_h_bond_acceptors(mol):
    acceptors = 0
    for i in mol:
        if i.lower() == "n" or i.lower() == "o":
            acceptors += 1
    return acceptors


def isDrugLike(mol):
    m = Chem.MolFromSmiles(mol)
    if get_h_bond_donors(mol) <= 5 and get_h_bond_acceptors(mol) <= 10 and Descriptors.MolWt(m) <= 500 and Descriptors.MolLogP(m) <= 5:
        return True
    else:
        return False


# filter garbage molecules
data = open("generated-molecules/Jan-08-2021_2144.txt", "r").read()
data = [data[y - 138:y] for y in range(138, len(data) + 138, 138)]
mol = [seq.split("\n") for seq in data]
mol = [list(filter(None, arr)) for arr in mol]
mol = [i[1:-1] for i in mol]
mol = list(flatten(mol))


t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)
file = open("./valid-molecules/" + timestamp + ".txt", "a")
validMol = 0
invalidMol = 0
for m in mol:
    if (isValid(m)):
        validMol += 1
        file.write(m + "\n")
    else:
        invalidMol += 1


print("{0:.0%}".format(validMol / len(mol)))
print(validMol)
print(invalidMol)
