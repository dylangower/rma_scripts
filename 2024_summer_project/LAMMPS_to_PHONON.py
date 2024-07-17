# This program takes the output of phana processing of the freqs and eigenvectors at the GAMMA point and converts it to a .phonon file for post processing, e.g. veiwing phonon modes in JMol
# Author Dylan Gower (dxg122@student.bham.ac.uk)

import numpy as np
import argparse

element_masses = {'C':12.0107, 'H':1.00794, 'N':14.0067, 'O':15.9994}
element_masses_inv = {v: k for k, v in element_masses.items()}

def read_LAMMPS_eigenvector_file(filename, lower_bound, upper_bound):
    with open(filename, 'r') as file:
        lines = file.readlines()

    eigenvectors_dict = dict()
    atoms_in_cell = int(lines[0].split()[-1])
    for i, line in enumerate(lines):
        if line[0] == '#' and 'frequency' in line:
            f = float(line.split()[-1]) / 0.03 # Convert to cm^-1
            if f < lower_bound or f > upper_bound:
                continue
            eigenvector = np.array([lines[j].split()[1:6:2] for j in range(i+2, i+2+atoms_in_cell)]).astype(float)
            eigenvectors_dict[f] = eigenvector


    return eigenvectors_dict

def read_LAMMPS_data_file(filename):
    with open(filename, 'r') as file:
        lines_w_blanks = file.readlines()

    lines = list()
    for line in lines_w_blanks:
        if line.strip():
            lines.append(line)

    for i, line in enumerate(lines):
        if 'atoms' in line:
            num_atoms = int(line.split()[0])
        elif 'atom types' in line:
            num_atom_types = int(line.split()[0])
        elif 'xlo' in line:
            lx = float(line.split()[1]) - float(line.split()[0])
        elif 'ylo' in line:
            ly = float(line.split()[1]) - float(line.split()[0])
        elif 'zlo' in line:
            lz = float(line.split()[1]) - float(line.split()[0])
        elif 'xy' in line:
            xy, xz, yz = np.array(line.split()[0:3]).astype(float)
        elif line.strip() == 'Masses':
            mass_index = i+1
        elif line.strip() == 'Atoms':
            pos_index = i+1
    print(pos_index)
    print(num_atoms)

    structure_matrix = np.array([[lx,0,0],[xy, ly, 0], [xz,yz,lz]])
    structure_matrix_inv = np.linalg.inv(structure_matrix)

    
    labels_to_elements = {int(line.split()[0].strip()) : element_masses_inv[float(line.split()[1].strip())] for line in lines[mass_index : mass_index+num_atom_types]}
    
    atom_positions = {int(line.split()[0].strip()) : np.array(line.split()[-3:]).astype(float) @ structure_matrix_inv for line in lines[pos_index:pos_index+num_atoms]}
    atom_elements = {int(line.split()[0].strip()) : labels_to_elements[int(line.split()[2].strip())] for line in lines[pos_index:pos_index+num_atoms]}
    
    return atom_positions, atom_elements, structure_matrix

def make_phonon_file(eigenvector_dict, atom_positions, atom_elements, structure_matrix):
    num_atoms = len(atom_elements.keys())
    #result = [' BEGIN header', f' Number of ions       {num_atoms}', f' Number of branches  234', f' Number of wavevectors  1',
    #        ' Frequencies in   cm-1', ' IR intensities in   (D/A)**2/amu', ' Raman activities in   A**4 amu**(-1)', ' Unit cell vectors (A)']

    result = [''' BEGIN header
 Number of ions         78
 Number of branches     51
 Number of wavevectors  1
 Frequencies in         cm-1
 IR intensities in      (D/A)**2/amu
 Raman activities in    A**4 amu**(-1)
 Unit cell vectors (A)''']
    for i in range(3):
        result.append('   '.join(np.round(structure_matrix[i, :], 8).astype(str)))
    result.append('Fractional Co-ordinates')

    for key, element in atom_elements.items():
        pos = atom_positions[key]
        mass = element_masses[element]
        result.append(f'\t{key}\t{pos[0]:.8f}\t{pos[1]:.10f}\t{pos[2]:.8f}\t{element}\t{mass:.8f}')

    result.append(' END header')
    result.append(f'q-pt=\t1\t{0.0:.8f}\t{0.0:.8f}\t{0.0:.8f}')

    freq = list(eigenvector_dict.keys())

    for i, f in enumerate(freq):
        result.append(f'{i+1}\t{f:.8f}\t{0.0:.8f}')

    result.append('\t\t\tPhonon Eigenvectors')
    result.append('Mode Ion\t\tX\t\tY\t\tZ')

    #print(eigenvector_dict)
    for i,f in enumerate(freq):
        mat = eigenvector_dict[f]
        for j in range(num_atoms):
            vec = mat[j]
            result.append(f'{i+1}\t{j+1}\t{vec[0]:.8f}\t0.0\t{vec[1]:.8f}\t0.0\t{vec[2]:.8f}\t0.0')

    return '\n'.join(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('eigenvector_filename', help='Output file from phana processing.')
    parser.add_argument('data_filename', help='LAMMPS data file used for simulation')
    parser.add_argument('-o', '--output', default='lammps', help='Root of filename to save phonon file to.')
    parser.add_argument('-v', '--verbose', help='Print output to console.', action='store_true')
    parser.add_argument('-min', help='Minimum frequency to include (default=-10000 cm^-1).', default=-10000, type=float)
    parser.add_argument('-max', help='Minimum frequency to include (default=10000 cm^-1).', default=10000, type=float)
    args = parser.parse_args()
    
    eigenvector_dict = read_LAMMPS_eigenvector_file(args.eigenvector_filename, args.min, args.max)
    atom_positions, atom_elements, structure_matrix = read_LAMMPS_data_file(args.data_filename)

    output = make_phonon_file(eigenvector_dict, atom_positions, atom_elements, structure_matrix)
    
    with open(f'{args.output}.phonon', 'w') as file:
        file.write(output)

    if args.verbose:
        print(output)
