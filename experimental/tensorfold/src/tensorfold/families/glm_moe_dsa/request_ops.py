"""Internal request operations; external commands never supply these objects."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Call:
    operation: str
    args: tuple

    def __post_init__(self):
        if self.operation not in ('target', 'mtp', 'sample') or type(self.args) is not tuple:
            raise ValueError('Invalid suspended request operation')
        if len(self.args) != (4 if self.operation == 'mtp' else 3):
            raise ValueError('Invalid suspended operation arguments')

    @property
    def rows(self):
        return len(self.args[0])
