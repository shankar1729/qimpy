
class Energy(dict):
    'Energy of system with access to components'

    def __float__(self) -> float:
        'Compute total energy from energy components'
        return float(sum(self.values()))

    def __repr__(self) -> str:
        result = ''
        for name, value in self.items():
            if value:
                result += f'{name:9s} = {value:25.16f}\n'
        result += ('-' * 37)
        result += f'\n{self.name():9s} = {float(self):25.16f}'
        return result

    def name(self) -> str:
        'Appropriate name of (free) energy based on components.'
        if '-muN' in self:
            return 'G'  # Grand free energy
        if '-TS' in self:
            return 'F'  # Helmholtz free energy
        return 'E'  # Energy
