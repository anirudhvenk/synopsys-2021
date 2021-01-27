from chembl_webresource_client.new_client import new_client
molecule = new_client.molecule
m1 = molecule.get('CHEMBL25')

print(m1["molecule_structures"]["canonical_smiles"])
