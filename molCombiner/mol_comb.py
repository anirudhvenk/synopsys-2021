'''

Title: Molecule Combiner
Author: Anirudh Venkatraman
Availibility: https://github.com/anirudhvenkatraman/synopsys-2021

Class to combine two molecules together to create a child molecule optimizing for the highest IC-50 score
for the given FASTA sequence. Implements the neuraldecipher neural network to reverse engineer fingerprints back to
SMILES sequences.

'''


import importlib.util
from generationRNN.generate_molecules import GenerateMolecules
from affinityCNN.predict_affinity import AffinityPrediction
from rdkit import Chem
from molCombiner.mol_comb_helper import MolCombHelper
import numpy as np


class MolComb:
    def __init__(self, fasta):
        self.fasta = fasta

        spec = importlib.util.spec_from_file_location(
            "module", "./neuraldecipher/source/evaluation.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.evaluator = module.Evaluation()

        self.nd = self.evaluator.load_final_model()
        self.mol_comb_helper = MolCombHelper(
            3, 1024, self.get_affinity, 'generationRNN/data/100k_SMILES.txt')

    def get_affinity(self, mol):
        predictor = AffinityPrediction()
        return (predictor.predict_affinity(mol, self.fasta))

    def combine(self, mol1, mol2):
        fp = self.mol_comb_helper.reconstructMol(
            Chem.MolFromSmiles(mol1), Chem.MolFromSmiles(mol2))

        np.save("neuraldecipher/data/dfFold1024/fingerprint.npy", [fp[0]])
        final_smiles = self.evaluator.eval_wrapper(self.nd)
        return(final_smiles)
