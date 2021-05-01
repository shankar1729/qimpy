import qimpy as qp
import torch
import pathlib
import re


class Ions:
    """TODO: document class Ions"""

    def __init__(self, *, rc, coordinates, pseudopotentials):
        '''
        Parameters
        ----------
        TODO
        '''
        self.rc = rc
        qp.log.info('\n--- Initializing Ions ---')

        # Read ionic coordinates:
        assert isinstance(coordinates, list)
        self.n_ions = 0      # number of ions
        self.n_types = 0     # number of distinct ion types
        self.symbols = []    # symbol for each ion type
        self.positions = []  # position of each ion
        self.types = []      # type of each ion (index into symbols)
        self.ranges = []     # range / slice to get each ion type
        type_start = 0
        for coord in coordinates:
            assert len(coord) == 4  # TODO: support other attributes
            # Add new symbol or append to existing:
            symbol = str(coord[0])
            if (not self.symbols) or (symbol != self.symbols[-1]):
                self.symbols.append(symbol)
                self.n_types += 1
                if type_start != self.n_ions:
                    self.ranges.append(slice(type_start, self.n_ions))
                    type_start = self.n_ions
            # Add type and position of current ion:
            self.types.append(self.n_types-1)
            self.positions.append([float(x) for x in coord[1:4]])
            self.n_ions += 1

        # Check order:
        if len(set(self.symbols)) < self.n_types:
            raise ValueError(
                'coordinates must group ions of same type together')

        # Report ionic positions:
        qp.log.info('{:d} total ions of {:d} types; positions:'.format(
            self.n_ions, self.n_types))
        for i_ion, position in enumerate(self.positions):
            qp.log.info('- [{:s}, {:11.8f}, {:11.8f}, {:11.8f}]'.format(
                self.symbols[self.types[i_ion]], *tuple(position)))
        self.positions = torch.tensor(self.positions, device=rc.device)
        self.types = torch.tensor(self.types, device=rc.device)

        # Initialize pseudopotentials:
        self.pseudopotentials = []
        if isinstance(pseudopotentials, str):
            pseudopotentials = [pseudopotentials]
        for i_type, symbol in enumerate(self.symbols):
            fname = None  # full filename for this ion type
            symbol_variants = [
                symbol.lower(),
                symbol.upper(),
                symbol.capitalize()]
            # Check each filename provided in order:
            for ps_name in pseudopotentials:
                if ps_name.count('$ID'):
                    # wildcard syntax
                    for symbol_variant in symbol_variants:
                        fname_test = ps_name.replace('$ID', symbol_variant)
                        if pathlib.Path(fname_test).exists():
                            fname = fname_test  # found
                            break
                else:
                    # specific filename
                    basename = pathlib.PurePath(ps_name).stem
                    ps_symbol = re.split(r'[_\-\.]+', basename)[0]
                    if ps_symbol in symbol_variants:
                        fname = ps_name
                        if not pathlib.Path(fname).exists():
                            raise FileNotFoundError(fname)
                        break
                if fname:
                    break
            # Read pseudopotential file:
            if fname:
                self.pseudopotentials.append(
                    qp.ions.Pseudopotential(fname, rc))
            else:
                raise ValueError(
                    'no pseudopotential found for {:s}'.format(symbol))
