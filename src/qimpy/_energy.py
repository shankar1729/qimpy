import torch


class Energy(dict[str, torch.Tensor]):
    """Energy of system with access to components"""

    name: str  #: standard label for (free) energy type, e.g., E, A, Phi etc.

    def __init__(self, name: str = "E") -> None:
        super().__init__()
        self.name = name

    def __float__(self) -> float:
        """Compute total energy from energy components"""
        return float(self.total.item())

    def __repr__(self) -> str:
        terms: list[list[str]] = [[], []]  # collect terms with +/- separately
        total = 0.0
        for name, value in sorted(self.items()):
            term_index = 1 if (name[0] in "+-") else 0
            cur_value = value.item()
            total += cur_value
            terms[term_index].append(f"{name:>9s} = {cur_value:25.16f}")
        terms[0].extend(terms[1])
        terms[0].append("-" * 37)  # separator
        terms[0].append(f"{self.name:>9s} = {total:25.16f}")
        return "\n".join(terms[0])

    @property
    def total(self) -> torch.Tensor:
        return sum(self.values(), start=torch.tensor(0.0))
