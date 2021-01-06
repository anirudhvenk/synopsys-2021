from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps


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


mol = "CCCCn1cc(C(=O)NCC[C@H](c2ccccc2)c2nc3ccccc3c2=O)c1"
print(isValid(mol))
print(isDrugLike(mol))
