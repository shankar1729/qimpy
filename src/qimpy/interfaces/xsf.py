import h5py
import numpy as np
import argparse

def write_xsf(checkpoint, xsf_file, animated=False, dataset_symbol=None):
    h5_file = h5py.File(checkpoint, 'r')
    ions = h5_file['ions']
    types = ions['types'][:]
    symbols = np.repeat(
        np.array(ions.attrs['symbols'].split(",")), 
        np.unique(types, return_counts=True)[1]
    )
    to_ang = 0.529177249  # from qimpy
    lattice = h5_file['lattice']
    history = h5_file['geometry/action/history']
    
    with open(xsf_file, 'w') as f:
        if animated:
            fractional_positions = history['positions'][:]
            animsteps = fractional_positions.shape[0]
            f.write(f'ANIMSTEPS {animsteps}\n')
            f.write('CRYSTAL\n')
            
            if lattice.attrs['movable']:
                lattice_vecs = history['Rbasis'][:] * to_ang
                positions = np.einsum('ijk,ilk->ilj', lattice_vecs, fractional_positions)
                for n, vec_n, pos_n in zip(range(animsteps), lattice_vecs, positions):
                    f.write(f'PRIMVEC {n+1}\n')
                    for vec in vec_n.T:
                        f.write(f'{vec[0]:10.6f} {vec[1]:10.6f} {vec[2]:10.6f}\n')

                    f.write(f'PRIMCOORD {n+1}\n')
                    f.write(f'  {len(pos_n)} 1\n')
                    for i, pos in enumerate(pos_n): 
                        f.write(f'  {symbols[i]} {pos[0]:10.6f} {pos[1]:10.6f} {pos[2]:10.6f}\n')
                
            else:
                lattice_vecs = lattice['Rbasis'][:] * to_ang
                positions = np.einsum('ij,klj->kli', lattice_vecs, fractional_positions)
                f.write('PRIMVEC\n')
                for vec in lattice_vecs.T:
                    f.write(f'{vec[0]:10.6f} {vec[1]:10.6f} {vec[2]:10.6f}\n')

                for n, pos_n in enumerate(positions):
                    f.write(f'PRIMCOORD {n+1}\n')
                    f.write(f'  {len(pos_n)} 1\n')
                    for i, pos in enumerate(pos_n):
                        f.write(f'  {symbols[i]} {pos[0]:10.6f} {pos[1]:10.6f} {pos[2]:10.6f}\n') 
                    
        else:
            lattice_vecs = lattice['Rbasis'][:] * to_ang
            fractional_positions = ions['positions'][:]
            positions = (lattice_vecs @ fractional_positions.T).T
            f.write('CRYSTAL\n')
            f.write('PRIMVEC\n')
            for vec in lattice_vecs.T:
                f.write(f'{vec[0]:10.6f} {vec[1]:10.6f} {vec[2]:10.6f}\n')

            f.write('PRIMCOORD\n')
            f.write(f'  {len(positions)} 1\n')
            for i, pos in enumerate(positions):
                f.write(f'  {symbols[i]} {pos[0]:10.6f} {pos[1]:10.6f} {pos[2]:10.6f}\n')
                
            if dataset_symbol is None:
                return

            dataset = h5_file[f'electrons/{dataset_symbol}'][0]
            f.write('BEGIN_BLOCK_DATAGRID_3D\n')
            f.write(f' {dataset_symbol}\n')
            f.write(f' BEGIN_DATAGRID_3D_{dataset_symbol}\n')
            f.write(f'  {dataset.shape[0]} {dataset.shape[1]} {dataset.shape[2]}\n')
            f.write('    0.000000   0.000000   0.000000\n')
            for vec in lattice_vecs.T:
                f.write(f'  {vec[0]:10.6f} {vec[1]:10.6f} {vec[2]:10.6f}\n')

            for k in range(dataset.shape[2]):
                for j in range(dataset.shape[1]):
                    for i in range(dataset.shape[0]):
                        f.write(f' {dataset[i, j, k]:e}')
                    f.write('\n')

            f.write(' END_DATAGRID_3D\n')
            f.write('END_BLOCK_DATAGRID_3D\n')

    return


def main():
    parser = argparse.ArgumentParser(description='write XSF file from HDF5 checkpoint file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-c','--checkpoint-file', metavar="FILE", 
        help='checkpoint file in HDF5 format')
    parser.add_argument(
        '-x','--xsf-file', default='crystal.xsf', metavar="FILE", 
        help='output file in XSF format (crystal.xsf if unspecified)')
    parser.add_argument(
        '-a','--animated', action="store_true", 
        help='make output an animated XSF file')
    parser.add_argument(
        '-d','--data-symbol', default=None, metavar="SYMBOL", 
        help='add 3d data to XSF file such as electron density (SYMBOL=n)')
    
    args = parser.parse_args()
    write_xsf(args.checkpoint_file, args.xsf_file, args.animated, args.data_symbol)
    
if __name__ == "__main__":
    main()

    
                    
