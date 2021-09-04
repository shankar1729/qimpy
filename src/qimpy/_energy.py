from typing import List


class Energy(dict):
    """Energy of system with access to components"""

    def __float__(self) -> float:
        """Compute total energy from energy components"""
        return float(sum(self.values()))

    def __repr__(self) -> str:
        terms: List[List[str]] = [[], []]  # collect terms with +/- separately
        for name, value in sorted(self.items()):
            term_index = (1 if (name[0] in '+-') else 0)
            terms[term_index].append(f'{name:>9s} = {value:25.16f}')
        terms[0].extend(terms[1])
        terms[0].append('-' * 37)  # separator
        terms[0].append(f'{self.name:>9s} = {float(self):25.16f}')  # total
        return '\n'.join(terms[0])

    @property
    def name(self) -> str:
        """Appropriate name of (free) energy based on components."""
        if 'Eband' in self:
            return 'Eband'  # Band structure energy
        if '-muN' in self:
            return 'G'  # Grand free energy
        if '-TS' in self:
            return 'F'  # Helmholtz free energy
        return 'E'  # Energy
