#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from structure.atoms import Atoms

from ase.build import surface


class Surface:
    def __init__(self, bulk: Atoms, miller_index, layers, vacuum=None, tol=1e-10, periodic=False):
        self._bulk = bulk.ase_converter()
        self.indices = miller_index
        self.layers = layers
        self.vacuum = vacuum
        self.tol = tol
        self.pbc = periodic

    def mk_slab(self):
        """

        Returns:

        """
        return Atoms.atom_converter(
            surface(self._bulk, self.indices, self.layers, self.vacuum, self.tol, self.pbc)
        )


if __name__ == '__main__':
    pass
