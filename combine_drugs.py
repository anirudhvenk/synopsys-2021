from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import BRICS
from rdkit.Chem.Draw import SimilarityMaps, IPythonConsole
import numpy as np


m1 = 'O=C(Cc1ccccc1C(=O)N[C@@H]1CCCC[C@@H]1C)c1ccc(F)cc1'


def fingerprints(molecule):  # https://towardsdatascience.com/a-practical-introduction-to-the-use-of-molecular-fingerprints-in-drug-discovery-7f15021be2b1
    m = Chem.MolFromSmiles(molecule)
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        m, radius=2, nBits=2048, bitInfo=bi)
    fp_arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    np.nonzero(fp_arr)
    fp_bits = fp.GetOnBits()
    prints = [(m, x, bi) for x in fp_bits]
    return Draw.DrawMorganBits(prints, molsPerRow=5, legends=[str(x) for x in fp_bits])


# result = fingerprints(m1)
Draw.IPythonConsole.addMolToView(Chem.MolFromSmiles(m1))
