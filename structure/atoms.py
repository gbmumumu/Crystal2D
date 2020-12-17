#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from functools import lru_cache

from ase.io import read as ase_read
from ase.io import write as ase_write
from ase import Atoms as AseAtoms
from ase.build import make_supercell as ase_mks
import spglib

from structure.lattice import Lattice
from structure.settings import SpecieInfo


class AtomInitError(Exception):
    """"""


class Atoms:
    def __init__(self, lattice=None, coords=None, elements=None, numbers=None, cartesian=False):
        self.lattice = Lattice(np.asarray(lattice))
        self.elements = elements
        if elements is None and numbers is not None:
            self.elements = [SpecieInfo['map'].value.get(str(i)) for i in numbers]
        self.coords = np.asarray(coords)
        self.is_cart = cartesian

    def __repr__(self):
        return f"{self.formula}\n{self.matrix}\n{self.cart_coords}"

    @classmethod
    def from_file(cls, file_name, **kwargs):
        """
        read structure using ASE method.
        Args:
            file_name: filepath of cif/poscar/other structure file
            **kwargs: ASE paras, eg: index, format, etc.

        Returns: Atoms object

        """

        return cls.atom_converter(
            ase_read(file_name, **kwargs)
        )

    @classmethod
    def from_dict(cls, **paras):
        info = dict()
        for key, v in paras.items():
            if key in 'lattice' or key in 'matrix':
                info['lattice'] = v
            if key in 'positions' or 'coords' in key:
                if 'cart' in key:
                    is_cart = True
                else:
                    is_cart = False
                info['coords'] = v
                info['cartesian'] = is_cart
            if key in 'elements' or key in 'symbols':
                info['elements'] = v

            if key in 'atomic numbers':
                info['numbers'] = v

        return cls(**info)

    @lru_cache()
    def ase_converter(self, pbc=True):
        """Get ASE representation of the atoms object."""
        try:
            return AseAtoms(
                symbols=self.elements,
                positions=self.cart_coords,
                pbc=pbc,
                cell=self.matrix,
            )
        except Exception as e:
            print("ASE convert failed, plz check!", e)

    @classmethod
    def atom_converter(cls, ase_atoms: AseAtoms):
        try:
            return cls(
                lattice=ase_atoms.get_cell()[:], elements=ase_atoms.get_chemical_symbols(),
                coords=ase_atoms.get_scaled_positions(), cartesian=False
            )
        except Exception as e:
            print("Transform failed: ", e)

    @property
    def matrix(self):
        """

        Returns:

        """
        return self.lattice.lattice()

    @property
    def angles(self):
        """

        Returns:

        """
        return self.lattice.angles

    @property
    def cart_coords(self):
        """

        Returns:

        """
        if self.is_cart:
            return self.coords

        return self.lattice.cart_coords(self.coords)

    @property
    def frac_coords(self):
        """

        Returns:

        """
        if not self.is_cart:
            return self.coords

        return self.lattice.frac_coords(self.coords)

    @property
    def atomic_numbers(self):
        """

        Returns:

        """
        return self.ase_converter().get_atomic_numbers()

    @property
    def formula(self):
        """

        Returns:

        """
        return self.ase_converter().get_chemical_formula()

    @property
    def noa(self):
        """
        number of atoms
        Returns:

        """
        return self.ase_converter().get_global_number_of_atoms()

    def make_primitive(self, symprec=1e-5, **kwargs):
        """

        Args:
            symprec:
            **kwargs:

        Returns:

        """

        return self._make(spglib.find_primitive,
                          self.matrix,
                          self.cart_coords,
                          self.atomic_numbers,
                          symprec=symprec, **kwargs)

    def make_conv(self, **kwargs):
        """

        Args:
            **kwargs:

        Returns:

        """
        return self._make(spglib.refine_cell,
                          self.matrix,
                          self.cart_coords,
                          self.atomic_numbers, **kwargs)

    @staticmethod
    def _make(seek_func, *args, **kwargs):
        latt, f_coords, atomic_num = seek_func(args, **kwargs)
        dat = {
            'lattice': latt,
            'coords': f_coords,
            'numbers': atomic_num,
            'cart': False
        }
        return Atoms.from_dict(**dat)

    def make_supercell(self, scale_matrix=None, **kwargs):
        """

        Args:
            scale_matrix:

        Returns:

        """
        if scale_matrix is None:
            scale_matrix = [2, 2, 2]
        if np.asarray(scale_matrix).shape == (3,):
            scale_matrix = np.eye(3) * scale_matrix

        return self.atom_converter(
            ase_mks(self.ase_converter(), scale_matrix, **kwargs)
        )

    def spacegroup(self, symprec=1e-3):
        """Get spacegroup of the atoms object."""
        sg = spglib.get_spacegroup(
            (self.matrix, self.frac_coords, self.atomic_numbers),
            symprec=symprec,
        )
        return sg

    @staticmethod
    def lattice_points_in_supercell(supercell_matrix):
        """
        Adapted from Pymatgen.

        Returns the list of points on the original lattice contained in the
        supercell in fractional coordinates (with the supercell basis).
        e.g. [[2,0,0],[0,1,0],[0,0,1]] returns [[0,0,0],[0.5,0,0]]

        Args:

            supercell_matrix: 3x3 matrix describing the supercell

        Returns:
            numpy array of the fractional coordinates
        """
        diagonals = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        d_points = np.dot(diagonals, supercell_matrix)

        mins = np.min(d_points, axis=0)
        maxes = np.max(d_points, axis=0) + 1

        ar = (
                np.arange(mins[0], maxes[0])[:, None]
                * np.array([1, 0, 0])[None, :]
        )
        br = (
                np.arange(mins[1], maxes[1])[:, None]
                * np.array([0, 1, 0])[None, :]
        )
        cr = (
                np.arange(mins[2], maxes[2])[:, None]
                * np.array([0, 0, 1])[None, :]
        )

        all_points = ar[:, None, None] + br[None, :, None] + cr[None, None, :]
        all_points = all_points.reshape((-1, 3))

        frac_points = np.dot(all_points, np.linalg.inv(supercell_matrix))

        tvects = frac_points[
            np.all(frac_points < 1 - 1e-10, axis=1)
            & np.all(frac_points >= -1e-10, axis=1)
            ]
        assert len(tvects) == round(abs(np.linalg.det(supercell_matrix)))

        return tvects

    def make_supercell_matrix(self, scaling_matrix):
        """
        Adapted from Pymatgen.

        Makes a supercell. Allowing to have sites outside the unit cell.

        Args:
            scaling_matrix: A scaling matrix for transforming the lattice
            vectors. Has to be all integers. Several options are possible:
            a. A full 3x3 scaling matrix defining the linear combination
             the old lattice vectors. E.g., [[2,1,0],[0,3,0],[0,0,
             1]] generates a new structure with lattice vectors a' =
             2a + b, b' = 3b, c' = c where a, b, and c are the lattice
             vectors of the original structure.
            b. An sequence of three scaling factors. E.g., [2, 1, 1]
             specifies that the supercell should have dimensions 2a x b x
             c.
            c. A number, which simply scales all lattice vectors by the
             same factor.

        Returns:
            Supercell structure. Note that a Structure is always returned,
            even if the input structure is a subclass of Structure. This is
            to avoid different arguments signatures from causing problems. If
            you prefer a subclass to return its own type, you need to override
            this method in the subclass.
        """
        smtx = np.array(scaling_matrix, np.int16)
        if smtx.shape != (3, 3):
            smtx = np.array(smtx * np.eye(3), np.int16)

        new_lattice = Lattice(np.dot(smtx, self.matrix))

        f_lat = self.lattice_points_in_supercell(smtx)
        c_lat = new_lattice.cart_coords(f_lat)

        new_sites = []
        new_elements = []
        for site, el in zip(self.cart_coords, self.elements):
            for v in c_lat:
                new_elements.append(el)
                tmp = site + v
                new_sites.append(tmp)

        return Atoms(lattice=new_lattice.lattice(),
                     elements=new_elements,
                     coords=new_sites, cartesian=True)

    def get_origin(self):
        """Get center of mass of the atoms object."""
        # atomic_mass
        return self.frac_coords.mean(axis=0)

    def center_around_origin(self, new_origin=None):
        """Center around given origin."""
        if new_origin is None:
            new_origin = [0.0, 0.0, 0.5]
        c_o_m = self.get_origin()
        coords = np.zeros((self.noa, 3))
        for i, coord in enumerate(self.frac_coords):
            coords[i] = self.frac_coords[i] - c_o_m + new_origin

        return Atoms(lattice=self.matrix, elements=self.elements, coords=coords, cartesian=False)

    def center(self, axis=2, vacuum=18.0, about=None):
        """
        Center structure with vacuum padding.

        Args:
          vacuum:vacuum size

          axis: direction
          about:
        """
        cell = self.matrix
        p = self.cart_coords

        dirs = np.zeros_like(cell)
        for i in range(3):
            dirs[i] = np.cross(cell[i - 1], cell[i - 2])
            dirs[i] /= np.sqrt(np.dot(dirs[i], dirs[i]))  # normalize
            if np.dot(dirs[i], cell[i]) < 0.0:
                dirs[i] *= -1

        if isinstance(axis, int):
            axes = (axis,)
        else:
            axes = axis

        # if vacuum and any(self.pbc[x] for x in axes):
        #     warnings.warn(
        #         'You are adding vacuum along a periodic direction!')

        # Now, decide how much each basis vector should be made longer
        longer = np.zeros(3)
        shift = np.zeros(3)
        for i in axes:
            p0 = np.dot(p, dirs[i]).min() if len(p) else 0
            p1 = np.dot(p, dirs[i]).max() if len(p) else 0
            height = np.dot(cell[i], dirs[i])
            if vacuum is not None:
                lng = (p1 - p0 + 2 * vacuum) - height
            else:
                lng = 0.0  # Do not change unit cell size!
            top = lng + height - p1
            shf = 0.5 * (top - p0)
            cosphi = np.dot(cell[i], dirs[i]) / np.sqrt(
                np.dot(cell[i], cell[i])
            )
            longer[i] = lng / cosphi
            shift[i] = shf / cosphi

        # Now, do it!
        translation = np.zeros(3)
        for i in axes:
            nowlen = np.sqrt(np.dot(cell[i], cell[i]))
            if vacuum is not None or cell[i].any():
                cell[i] = cell[i] * (1 + longer[i] / nowlen)
                translation += shift[i] * cell[i] / nowlen

        new_coords = p + translation
        if about is not None:
            for vector in cell:
                new_coords -= vector / 2.0
            new_coords += about

        return Atoms(lattice=cell, elements=self.elements, coords=new_coords, cartesian=True)

    def write_file(self, fp, fmt='cif', **kwargs):
        """

        Args:
            fp:
            fmt:
            **kwargs:

        Returns:

        """

        return ase_write(fp, self.ase_converter(), format=fmt, **kwargs)

    def get_distance_matrix(self, pbc=True):
        """
        get distance matrix between all atoms.
        Args:
            pbc:

        Returns:

        """
        return self.ase_converter(pbc).get_all_distances(pbc)

    def get_bond_adjacency_matrix(self, bonded_rule="covalent", threshold=0.45, mic=True):
        """

        Args:
            bonded_rule:
            threshold:
            mic:

        Returns:

        """

        def _add(ra, rb, d, delta):
            return d < ra + rb + delta

        def _minus(ra, rb, d, delta):
            return d < ra + rb - delta

        dm = self.get_distance_matrix(pbc=mic)
        rule = bonded_rule.lower()
        bonded = _add if rule != "vdw" else _minus
        n, _ = dm.shape
        amx = np.zeros((n, n), dtype=int)
        for i in range(n):
            radii_a = SpecieInfo[rule].value.get(self.elements[i])
            for j in range(i + 1, n):
                radii_b = SpecieInfo[rule].value.get(self.elements[j])
                dij = dm[i][j]
                if bonded(radii_a, radii_b, dij, threshold):
                    amx[i][j], amx[j][i] = 1, 1

        return amx


if __name__ == '__main__':
    pass
