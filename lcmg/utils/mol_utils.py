import random

from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
from rdkit.Chem import Conformer
from rdkit.Geometry import Point3D

from lcmg.runtime_utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def str_to_bond_type(bond_type_str):
    bond_type_mapping = {
        "SINGLE": Chem.BondType.SINGLE,
        "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE,
        "AROMATIC": Chem.BondType.AROMATIC,
    }

    bond_type = bond_type_mapping.get(bond_type_str)

    if bond_type is None:
        raise ValueError(f"Unsupported bond type: {bond_type_str}")

    return bond_type


def build_mol(g, atom_types, charges, pos, bond_types):
    mol = Chem.RWMol()

    for atom_type, charge in zip(atom_types, charges):
        atom = Chem.Atom(atom_type)
        atom.SetFormalCharge(int(charge))
        mol.AddAtom(atom)

    us, vs = [i.tolist() for i in g.edges()]
    visited = set()
    for u, v, bond_type in zip(us, vs, bond_types):
        if bond_type == 'NONE':
            continue
        if u == v:
            continue
        if u < v:
            u, v = v, u
        if (u, v) in visited:
            continue
        visited.add((u, v))

        mol.AddBond(u, v, str_to_bond_type(bond_type))

    # mol = mol.GetMol()  # todo 需要吗？

    set_conformer(mol, pos)

    return mol


def _try_fix_atom_valence(atom):
    atom.UpdatePropertyCache(strict=False)
    if atom.GetAtomicNum() == 7 and atom.GetExplicitValence() == 4:  # N+
        atom.SetFormalCharge(1)
        return True
    return False


def sanitize_mol(mol):
    flag = True
    while flag:
        flag = False
        try:
            Chem.SanitizeMol(mol)
        except Chem.rdchem.AtomValenceException as ex:
            if _try_fix_atom_valence(mol.GetAtomWithIdx(ex.cause.GetAtomIdx())):
                flag = True
                continue
            raise ex


def mol2clean_smiles(mol, random=False):
    if mol is None:
        return None

    mol = Chem.Mol(mol)  # make a copy

    # clean isotope
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)

    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=not random, doRandom=random)  # clean stereochemistry


def set_conformer(mol, pos):
    assert len(pos) == mol.GetNumAtoms()

    conf = Conformer(len(pos))
    for i in range(len(pos)):
        x, y, z = [float(ii) for ii in pos[i]]
        conf.SetAtomPosition(i, Point3D(x, y, z))

    mol.RemoveAllConformers()
    mol.AddConformer(conf)

    return mol


def set_random_conformer(mol, fast=True):
    mol = Chem.Mol(mol)  # make a copy
    mol = Chem.AddHs(mol)

    if fast:
        try:
            code = AllChem.EmbedMolecule(mol, maxAttempts=10)
            if code == -1:
                return None
            AllChem.MMFFOptimizeMolecule(mol, maxIters=20)
        except:
            return None
        mol = Chem.RemoveHs(mol)
        return mol

    try:
        code = AllChem.EmbedMolecule(mol, maxAttempts=10)
        if code != -1:
            AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:
        log.debug(f'exception in set_random_conformer step0: {e}')
        code = -1

    if code == -1:
        log.debug(f'Default embedding failed!')
        mol = Chem.Mol(mol)
        mol = Chem.AddHs(mol)
        code = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=10)
        if code == -1:
            log.debug(f'set_random_conformer failed!')
            return None
        AllChem.UFFOptimizeMolecule(mol)

    mol = Chem.RemoveHs(mol)

    return mol


def random_break_brics_bonds(mol, num_bonds_to_cut=1, return_frag_mol=False, strict_cut=False):
    """
    Perform random breaking of BRICS bonds in a molecule and retrieve fragment information.

    :param mol: RDKit Mol object of the molecule to be fragmented.
    :param num_bonds_to_cut: Number of BRICS bonds to randomly cut.
    :param return_frag_mol: If True, return the fragmented RDKit Mol object along with fragment indices and anchor atom indices.
    :return: Tuple (fragment_idx, anchor_atom_idx) or (fragment_idx, anchor_atom_idx, fragmented_mol) depending on return_frag_mol.
             - fragment_idx: List of lists containing atom indices for each fragment.
             - anchor_atom_idx: List of lists containing atom indices of the anchor atoms connecting fragments.
             - fragmented_mol: Optional, RDKit Mol object of the fragmented molecule if return_frag_mol=True.
    """
    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol)

    # Find BRICS bonds
    brics_bonds = list(BRICS.FindBRICSBonds(mol))

    if len(brics_bonds) < num_bonds_to_cut:
        msg = f"Not enough BRICS bonds {len(brics_bonds)} to cut the molecule into {num_bonds_to_cut} parts"
        if strict_cut or len(brics_bonds) == 0:
            raise ValueError(msg)
        else:
            log.debug(msg)
    # Randomly select BRICS bonds to cut
    bonds_to_cut = random.sample(brics_bonds, num_bonds_to_cut)

    # Break selected BRICS bonds
    fragmented_mol = BRICS.BreakBRICSBonds(mol, bonds=bonds_to_cut)

    # Get fragments and record atom indices
    fragment_idx_list = Chem.GetMolFrags(fragmented_mol)

    virtual_atom_idx = {atom.GetIdx() for atom in fragmented_mol.GetAtoms() if atom.GetSymbol() == '*'}

    match len(fragment_idx_list):
        case 2:
            selected_idx = random.randint(0, 1)
            selected_frags = [set(fragment_idx_list[selected_idx])]
        case 3:
            # remove the linker
            fragment_idx_list = [set(i) for i in fragment_idx_list]
            selected_frags = [i for i in fragment_idx_list if len(i & virtual_atom_idx) == 1]
        case _:
            raise NotImplementedError()

    fragment_idx = []
    anchor_atom_idx = []
    for frag in selected_frags:
        _t = frag & virtual_atom_idx
        _t = fragmented_mol.GetAtomWithIdx(next(iter(_t))).GetNeighbors()
        assert len(_t) == 1
        fragment_idx.append(list(frag - virtual_atom_idx))
        anchor_atom_idx.append([i.GetIdx() for i in _t])

    if return_frag_mol:
        return fragment_idx, anchor_atom_idx, fragmented_mol
    else:
        return fragment_idx, anchor_atom_idx
