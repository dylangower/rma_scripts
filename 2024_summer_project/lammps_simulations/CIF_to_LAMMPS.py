# This script converts the phana eigvec.dat file to a .phonon file to be used for post processing such as viewing phonon modes in JMol. Please edit as needed and contact me for any bug reports.
# Author Dylan Gower, dxg122@student.bham.ac.uk
# Last edit : 16/07/2023

import numpy as np
import argparse

element_labels = {'C':1, 'H':2, 'O':3, 'N':4}
element_labels_inv = {v: k for k, v in element_labels.items()}
element_masses = {'C':12.0107, 'H':1.00794, 'N':14.0067, 'O':15.9994}

# Source: UFF, a Full Periodic Table Force Field for MolecularMechanics and Molecular Dynamics Simulations - https://pubs.acs.org/doi/epdf/10.1021/ja00051a040
element_LJ = {'C':(3.851, 0.105, 12.73), 'H':(2.886, 0.044, 12.0), 'O':(3.500, 0.06, 14.085), 'H_HB':(2.886, 0.001, 12.0)}

# Add additional symmetry operations to this list and as a case to the apply_sym() function.
defined_sym_ops = ['x', 'y', 'z', '-x', '-y', '-z', 'x+1/2', 'y+1/2', 'z+1/2', 'x-1/2', 'y-1/2', 'z-1/2', '-x+1/2', '-y+1/2', '-z+1/2', '-x-1/2', '-y-1/2', '-z-1/2']

# Class to represent nodes of the graph, contains information about each atom such as position, element and which molecule it's part of.
class Atom(object):
    def __init__(self, name, element, position):
        self.__name  = name
        self.__element = element
        self.__bonds = set()
        self.__molecule = None
        self.__position = position
        self.__can_hbond = False  
    @property
    def name(self):
        return self.__name
    @property
    def element(self):
        return self.__element
    @property
    def molecule(self):
        return self.__molecule
    @property
    def bonds(self):
        return set(self.__bonds)
    @property
    def position(self):
        return self.__position
    @property
    def can_hbond(self):
        return self.__can_hbond
    def add_bond(self, other):
        self.__bonds.add(other)
        other.__bonds.add(self)
    def in_molecule(self, molecule):
        self.__molecule = molecule
    def set_can_hbond(self, can_hbond):
        self.__can_hbond = can_hbond

# Class to represent edges of the graph, contains information about each bond such as length and type of bond.
class Bond(object):
    def __init__(self, atom1, atom2):
        self.__atom1 = atom1
        self.__atom2 = atom2
        self.__length = np.linalg.norm(atom1.position - atom2.position)
        self.__type = None
    @property
    def atom1(self):
        return self.__atom1
    @property
    def atom2(self):
        return self.__atom2
    @property
    def length(self):
        return self.__length
    @property
    def type(self):
        return self.__type
    def set_type(self, type):
        self.__type = type


# Algorithm to find molecules by finding the connected components of the graph from https://breakingcode.wordpress.com/2013/04/08/finding-connected-components-in-a-graph/
def connected_components(nodes):
    result = []
    nodes = set(nodes)
    while nodes:
        n = nodes.pop()
        group = {n}
        queue = [n]
        while queue:
            n = queue.pop(0)
            neighbors = n.bonds
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)   
        result.append(group)
    return result

# Convert betweeen cell lengths and angles and reduced triclinic representation for LAMMPS https://docs.lammps.org/Howto_triclinic.html
def cell_parameters_to_structure_matrix(a, b, c, alpha, beta, gamma):
    lx = a
    xy = b*np.cos(gamma)
    xz = c*np.cos(beta)
    ly = np.sqrt(b*b - xy*xy)
    yz = (b*c*np.cos(alpha)-xy*xz)/ly
    lz = np.sqrt(c*c - xz*xz - yz*yz)

    structure_matrix = np.array([[lx,0,0], [xy,ly,0],[xz,yz,lz]])

    return structure_matrix

# Find new positions due to symmetries, add additional symmetry operations to the match-case statment. If this is not working, convert the CIFs to P1 symmetry to bypass this process.
def apply_sym(pos, sym, lengths):
    sym_pos = np.zeros(3)
    for i in range(3):
        match sym[i]:
            case 'x':
                sym_pos[i] = pos[0]
            case 'y':
                sym_pos[i] = pos[1]
            case 'z':
                sym_pos[i] = pos[2]
            case '-x':
                sym_pos[i] = lengths[i] - pos[0]
            case '-y':
                sym_pos[i] = lengths[i] - pos[1]
            case '-z':
                sym_pos[i] = lengths[i] - pos[2]
            case 'x+1/2':
                sym_pos[i] = lengths[i] * 0.5 + pos[0]
            case 'y+1/2':
                sym_pos[i] = lengths[i] * 0.5 + pos[1]
            case 'z+1/2':
                sym_pos[i] = lengths[i] * 0.5 + pos[2]
            case '-x+1/2':
                sym_pos[i] = lengths[i] * 0.5 - pos[0]
            case '-y+1/2':
                sym_pos[i] = lengths[i] * 0.5 - pos[1]
            case '-z+1/2':
                sym_pos[i] = lengths[i] * 0.5 - pos[2]
            case 'x-1/2':
                sym_pos[i] = lengths[i] * 0.5 + pos[0]
            case 'y-1/2':
                sym_pos[i] = lengths[i] * 0.5 + pos[1]
            case 'z-1/2':
                sym_pos[i] = lengths[i] * 0.5 + pos[2]
            case '-x-1/2':
                sym_pos[i] = lengths[i] * 0.5 - pos[0]
            case '-y-1/2':
                sym_pos[i] = lengths[i] * 0.5 - pos[1]
            case '-z-1/2':
                sym_pos[i] = lengths[i] * 0.5 - pos[2]

    return sym_pos


def read_atoms_from_CIF(filename, max_covelent_bond_length):
    
    with open(filename) as f:
        lines = f.readlines()

    CIF_data = dict()
    fract_positions = dict()
    sym_ops = list()
    loop_flag, start_pos, end_pos, start_sym, end_sym = 0, 0, 0, 0, 0
    element_counter = {'H':0,'C':0,'O':0}

    for i, line in enumerate(lines):
        if start_pos and loop_flag:
            try:
                if line.strip != '\n':
                    if line.strip()[0] in element_labels.keys():
                        line_segments = line.split()
                        
                        if len(line_segments[0])>1:
                            name = line_segments[0]
                        else:
                            element = line_segments[0]
                            element_counter[element] = element_counter[element] + 1
                            name = element + str(element_counter[element])
                            
                        fract_positions[name] = np.array([float(line_segments[pos_index].split('(')[0]), float(line_segments[pos_index+1].split('(')[0]), float(line_segments[pos_index+2].split('(')[0])])
                        end_pos = 1
                    elif end_pos:
                        loop_flag = 0
                        start_pos = 0
                        end_pos = 0
            except IndexError as e:
                print(f'Skipping line : {i}')

        elif loop_flag and start_sym:
            try:
                if 'x' in line and 'y' in line and 'z' in line:
                    sym_op = [transform.replace("'", "").strip() for transform in line.strip().split(',')]
                    if sym_op[0][0] in ['1','2','3','4','5','6','7','8','9'] and '/' not in sym_op[0]:
                        sym_op[0] = sym_op[0][1:].strip()
                    sym_ops.append(sym_op)
                    end_sym = 1
                elif end_sym:
                    loop_flag = 0
                    start_sym = 0
                    end_sym = 0
                    if line[0] == '_':
                        line_segments = line.split()
                        key = line_segments[0]
                        values = line_segments[1:]
                        CIF_data[key] = values
            except IndexError as e:
                print(f'Skipping line : {i}')

        
        elif loop_flag and line.strip() == '_atom_site_fract_x':
            start_pos = 1
            pos_index = i - loop_line - 1

        elif loop_flag and line.strip() == '_symmetry_equiv_pos_as_xyz':
            start_sym = 1
            sym_index = i - loop_line - 1
                
        elif line[0] == '_':
            line_segments = line.split()
            key = line_segments[0]
            values = line_segments[1:]
            CIF_data[key] = values

        elif line == 'loop_\n':
            loop_flag = 1
            loop_line = i

    a, b, c = np.array([CIF_data['_cell_length_a'][0].split('(')[0], CIF_data['_cell_length_b'][0].split('(')[0], CIF_data['_cell_length_c'][0].split('(')[0]]).astype(float)
    alpha, beta, gamma = np.array([CIF_data['_cell_angle_alpha'][0].split('(')[0], CIF_data['_cell_angle_beta'][0].split('(')[0], CIF_data['_cell_angle_gamma'][0].split('(')[0]]).astype(float) / 180 * np.pi

    structure_matrix = cell_parameters_to_structure_matrix(a, b, c, alpha, beta, gamma)


    # Convert fractional positions to x y z positions
    atoms_no_sym = list()
    for i, key in enumerate(fract_positions.keys()):
        fract_pos = fract_positions[key]
        atoms_no_sym.append(Atom(i+1, key[0], fract_pos @ structure_matrix))

    # Add atoms specified by symmetry operations
    atoms = list()
    lengths = np.sum(structure_matrix, axis=0)
    num_atom_no_sym = len(atoms_no_sym)
    for i, sym in enumerate(sym_ops):
        if (np.array([transform in defined_sym_ops for transform in sym]) == 0).any():
            raise NameError(f'There is an undefined symmetry operation : {sym}.')
        for j, atom in enumerate(atoms_no_sym):
            atoms.append(Atom(i*num_atom_no_sym + j + 1, atom.element, apply_sym(atom.position, sym, lengths)))

    # Check if two atoms are within the covelent bond cutoff and make a bond between them if they are
    bonds = list()
    for i, atom_1 in enumerate(atoms):
        for j, atom_2 in enumerate(atoms[i+1:]):
            distance = np.linalg.norm(atom_1.position - atom_2.position)
            if distance < max_covelent_bond_length:
                if distance < 0.01:
                    raise RuntimeError('Atom {atom_1.name} and atom {atom_2.name} have a seperation of {distance}! This will cause issues in the simulation.')
                atom_1.add_bond(atom_2)
                bonds.append(Bond(atom_1, atom_2))

                # Add the ability for partially positive hydrogens to hydrogen bond
                if atom_1.element == 'H' and (atom_2.element == 'O' or atom_2.element == 'N'):
                    atom_1.set_can_hbond = True
                elif (atom_1.element == 'O' or atom_1.element == 'N') and atom_2.element =='H':
                    atom_2.set_can_hbond = True

    # Find molecules from atoms
    connected_components_list = list()
    for i, components in enumerate(connected_components(atoms)):
        for atom in components:
            atom.in_molecule(i+1)
        names = sorted(atom.name for atom in components)
        connected_components_list.append(names)

    print(f'Found {len(connected_components_list)} molecules: ')
    values, counts = np.unique([len(component) for component in connected_components_list], return_counts=True)
    for i, value in enumerate(values):
        print(f'{counts[i]} molecules with {value} atoms.')

    return atoms, bonds, structure_matrix

def write_LAMMPS_data_file(atoms, bonds, structure_matrix):

    # Find the number of unique bond types
    bond_types = dict()
    for bond in bonds:
        if len(bond_types.keys()) == 0:
            bond.set_type(len(bond_types.keys())+1)
            bond_types[len(bond_types.keys())+1] = (sorted([bond.atom1.element, bond.atom2.element]), bond.length)
        else:
            for i, bond_type in enumerate(sorted(bond_types.keys())):
                if sorted([bond.atom1.element, bond.atom2.element]) == bond_types[bond_type][0] and np.isclose(bond.length, bond_types[bond_type][1], 0.01):
                    bond.set_type(i+1)
            if bond.type == None:
                bond.set_type(len(bond_types.keys())+1)
                bond_types[len(bond_types.keys())+1] = (sorted([bond.atom1.element, bond.atom2.element]), bond.length)

    num_atoms = len(atoms)
    num_bonds = len(bonds)
    
    num_atom_types = np.unique([atom.element for atom in atoms]).shape[0]
    num_bond_types = len(bond_types.keys())
    
    hbond_type = None
    if np.sum([atom.can_hbond for atom in atoms]) != 0:
        num_atom_types += 1
        hbond_type = num_atom_types

    structure_matrix = np.round(structure_matrix, 6)

    # Write all data in a form recognised by LAMMPS

    result = [f'LAMMPS data file. CGCMM style. atom_style full', f'{num_atoms} atoms', f'{num_bonds} bonds', f'0 angles',
            f'0 dihedrals', f'0 impropers', f'{num_atom_types} atom types', f'{num_bond_types} bond types',
            f'0 angle types', f'0 dihedral types', f'0 improper types types',
            f'0 {structure_matrix[0][0]} xlo xhi', f'0 {structure_matrix[1][1]} ylo yhi', f'0 {structure_matrix[2][2]} zlo zhi',
            f'{structure_matrix[1][0]} {structure_matrix[2][0]} {structure_matrix[2][1]} xy xz yz']

    result.append(f'\nBond Coeffs\n')
    for i in range(num_bond_types):
        result.append(f'{i+1} {100} {bond_types[i+1][1]} # {bond_types[i+1][0][0]} - {bond_types[i+1][0][1]}')

    result.append(f'\nMasses\n')
    for i in range(num_atom_types):
        if i == hbond_type:
            mass = element_masses[element_labels_inv['H']]
            label = 'H_HB'
        else:
            mass = element_masses[element_labels_inv[i+1]]#
            label = element_labels_inv[i+1]
        
        result.append(f'{i+1} {mass} # {label}')

    result.append(f'\nAtoms\n')
    for i in range(num_atoms):
        if atoms[i].can_hbond:
            atom_type = hbond_type
        else:
            atom_type = element_labels[atoms[i].element]

        result.append(f'{i+1} {atoms[i].molecule} {atom_type} 0.0 {' '.join(np.round(atoms[i].position, 6).astype(str))} # {atoms[i].element}')

    result.append(f'\nBonds\n')
    for i in range(num_bonds):
        if bonds[i].type == None:
            print(bonds[i].type, bonds[i].atom1.element, bonds[i].atom2.element)
        result.append(f'{i+1} {bonds[i].type} {bonds[i].atom1.name} {bonds[i].atom2.name}')

    output = '\n'.join(result)

    return output


if __name__ == '__main__':

    # Process command line arguments/
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input CIF file (if results are not making sense use a CIF with P1 symmetry).')
    parser.add_argument('-c', '--cutoff', default=1.7, help='Covelent bond cutoff length (default = 1.7 angstrom).')
    parser.add_argument('-o', '--output', default='lammps', help='Root of filename to save LAMMPS data file to.')
    parser.add_argument('-v', '--verbose', help='Print output to console.', action='store_true')
    args = parser.parse_args()

    # Read CIF file.

    atoms, bonds, structure_matrix = read_atoms_from_CIF(args.input, float(args.cutoff))

    # Write position and bonding data to a file readable by LAMMPS

    lammps_data = write_LAMMPS_data_file(atoms, bonds, structure_matrix)
    with open(f'data.{args.output}', 'w') as f:
        f.write(lammps_data)
    print(f'\nData written to data.{args.output}')

    if args.verbose:
        print(lammps_data)