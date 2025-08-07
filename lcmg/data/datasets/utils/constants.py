from enum import Enum


class Properties(Enum):
    LIGAND_GEO = {'v_pos'}
    LIGAND_CHEM = {'v_atom_type', 'v_atom_charge', 'e_bond_type'}
    LIGAND = LIGAND_GEO | LIGAND_CHEM
    POCKET = {'v_feat', 'v_pos', 'v_geo_feat_inv'}
    ANY = LIGAND | POCKET

    def __get__(self, instance, owner):
        return self.value


# dgl hetero graph related
class NodeType(Enum):
    LIG = 'lig'
    PKT = 'pkt'
    PHAR = 'phar'

    def __get__(self, instance, owner):
        return self.value


class EdgeType(Enum):
    LIG = ('lig', 'with', 'lig')
    PKT = ('pkt', 'with', 'pkt')
    LIG_PKT = ('lig', 'to', 'pkt')
    PKT_LIG = ('pkt', 'to', 'lig')
    PHAR = ('phar', 'with', 'phar')
    PHAR_LIG = ('phar', 'to', 'lig')
    LIG_PHAR = ('lig', 'to', 'phar')

    def __init__(self, source, relation, target):
        self.source = source
        self.relation = relation
        self.target = target

    def __get__(self, instance, owner):
        return self.value
