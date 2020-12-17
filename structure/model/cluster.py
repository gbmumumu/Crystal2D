#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

from structure.atoms import Atoms, np


class SingleAtomWarn(Exception):
    """
    Raises when structure contains unbonded atom
    """


class Cluster:
    def __init__(self, crystal: Atoms):
        self.crystal = crystal

    @classmethod
    def seek_cluster(cls, matrix, atoms):
        """

        Args:
            matrix:
            atoms:

        Returns:

        """
        if 0 in np.sum(matrix, axis=0):
            raise SingleAtomWarn("This structure has independent unbonded atoms, exclude!")

        clusters, sizes = [], []
        walked = [False for _ in atoms.elements]
        matrix += np.eye(matrix.shape[0], dtype=int)

        def visited(atom_index, atoms_cluster):
            walked[atom_index] = True
            update_cluster = set(np.where(matrix[atom_index] != 0)[0]).union(atoms_cluster)
            atoms_cluster = update_cluster
            for new_atom_index in atoms_cluster:
                if not walked[new_atom_index]:
                    walked[new_atom_index] = True
                    atoms_cluster = visited(new_atom_index, atoms_cluster)
            return atoms_cluster

        for m, _ in enumerate(atoms.elements):
            if not walked[m]:
                _c = set()
                cluster = visited(m, _c)
                clusters.append(cluster)
                sizes.append(len(cluster))

        return len(clusters), clusters, sizes

    def get_clusters_from_bulk(self, adjacency_matrix):
        """

        Args:
            adjacency_matrix:

        Returns:

        """
        return self.seek_cluster(adjacency_matrix, self.crystal)

    @staticmethod
    def get_bulk_type(primitive_cell_max, primitive_cell_min,
                      supercell_max, supercell_min, supercell_atoms_number):
        """

        Args:
            primitive_cell_max:
            primitive_cell_min:
            supercell_max:
            supercell_min:
            supercell_atoms_number:

        Returns:

        """

        def gmt(ratio_number):
            return "2D vdW solid" \
                if ratio_number == 4 else "1D vdW solid" \
                if ratio_number == 2 else "Intercalated 1D/2D"

        if primitive_cell_min == 1:
            return "Exclude! has unbonded atoms!"
        ratio = supercell_max / primitive_cell_max
        if primitive_cell_min == supercell_min:
            return "Exclude! has unbonded molecules!," \
                   "Molecule solid: {}".format(gmt(ratio))
        else:
            if supercell_max == supercell_atoms_number:
                return "Exclude! 3D solid"
            else:
                return gmt(ratio)

    def split(self, clusters_info, index=None):
        """
        split a specific cluster from clusters, index is the cluster index.
        Args:
            clusters_info: calc by "get_clusters", atoms index group set.
            index: the index of cluster, if index is None, set index = 0.

        Returns:

        """
        if index is None:
            index = 0
        try:
            cluster = clusters_info[index]
        except IndexError:
            warnings.warn("cluster index out of range, switch to zero!")
            cluster = clusters_info[0]
        new_positions, new_symbols = [], []
        for n in cluster:
            new_positions.append(self.crystal.cart_coords[n])
            new_symbols.append(self.crystal.elements[n])
        new_positions = np.asarray(new_positions)

        return Atoms(lattice=self.crystal.matrix,
                     elements=new_symbols, coords=new_positions, cartesian=True)

    @classmethod
    def find_stacking_direction(cls, cluster: Atoms):
        """
        find the vacuum direction of a cluster by extend structure along three axis,
        if the cluster number >=2, means this axis is the vacuum direction.
        Returns: axis: 0, 1, 2 if success else None

        """
        try_dir = np.eye(3).astype(int) + np.ones(3)
        direction = []
        for i, d in enumerate(try_dir):
            super_cell = cluster.make_supercell(scale_matrix=d)
            num, *_ = cls.seek_cluster(super_cell.get_bond_adjacency_matrix(), super_cell)
            if num >= 2:
                direction.append(i)
        if len(direction) != 1:
            warnings.warn("direction finding method failed!,"
                          "switch to default direction:[0, 0, 1], "
                          "Failed to center, you can specific crystal "
                          "stacking direction by {stacking_direction}")

            return 2
        return direction[0]

    def restore_mirror_atom(self):
        """
        find the mirror atom and restore the atom coords.
        Returns:

        """
        pass


if __name__ == '__main__':
    pass
