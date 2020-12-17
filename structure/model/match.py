#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Design interface using Zur algorithm and Anderson rule. modified from jarvis-tools"""

from itertools import product
import numpy as np
import math


class ZSLGenerator(object):
    """
    Uses Zur algorithm to find best matched interfaces.

    This class is modified from pymatgen.
    """

    def __init__(
            self,
            max_area_ratio_tol=0.09,
            max_area=400,
            max_length_tol=0.03,
            max_angle_tol=0.01,
    ):
        """
        Intialize for a specific film and substrate.

        Parameters for the class.
        Args:
            max_area_ratio_tol(float): Max tolerance on ratio of
                super-lattices to consider equal

            max_area(float): max super lattice area to generate in search

            max_length_tol: maximum length tolerance in checking if two
                vectors are of nearly the same length

            max_angle_tol: maximum angle tolerance in checking of two sets
                of vectors have nearly the same angle between them
        """
        self.max_area_ratio_tol = max_area_ratio_tol
        self.max_area = max_area
        self.max_length_tol = max_length_tol
        self.max_angle_tol = max_angle_tol

    def is_same_vectors(self, vec_set1, vec_set2):
        """
        Check two sets of vectors are the same.

        Args:
            vec_set1(array[array]): an array of two vectors

            vec_set2(array[array]): second array of two vectors
        """
        if (
                np.absolute(rel_strain(vec_set1[0], vec_set2[0]))
                > self.max_length_tol
        ):
            return False
        elif (
                np.absolute(rel_strain(vec_set1[1], vec_set2[1]))
                > self.max_length_tol
        ):
            return False
        elif np.absolute(rel_angle(vec_set1, vec_set2)) > self.max_angle_tol:
            return False
        else:
            return True

    def generate_sl_transformation_sets(self, film_area, substrate_area):
        """Generate transformation sets for film/substrate.

        The transformation sets map the film and substrate unit cells to super
        lattices with a maximum area.

        Args:

            film_area(int): the unit cell area for the film.

            substrate_area(int): the unit cell area for the substrate.

        Returns:
            transformation_sets: a set of transformation_sets defined as:
                1.) the transformation matricies for the film to create a
                super lattice of area i*film area
                2.) the tranformation matricies for the substrate to create
                a super lattice of area j*film area
        """
        transformation_indicies = [
            (i, j)
            for i in range(1, int(self.max_area / film_area))
            for j in range(1, int(self.max_area / substrate_area))
            if np.absolute(film_area / substrate_area - float(j) / i)
               < self.max_area_ratio_tol
        ]

        # Sort sets by the square of the matching area and yield in order
        # from smallest to largest
        for x in sorted(transformation_indicies, key=lambda x: x[0] * x[1]):
            yield (
                gen_sl_transform_matricies(x[0]),
                gen_sl_transform_matricies(x[1]),
            )

    def get_equiv_transformations(
            self, transformation_sets, film_vectors, substrate_vectors
    ):
        """
        Apply the transformation_sets to the film and substrate vectors.

        Generate super-lattices and checks if they matches.
        Returns all matching vectors sets.

        Args:
            transformation_sets(array): an array of transformation sets:
                each transformation set is an array with the (i,j)
                indicating the area multipes of the film and subtrate it
                corresponds to, an array with all possible transformations
                for the film area multiple i and another array for the
                substrate area multiple j.

            film_vectors(array): film vectors to generate super lattices.

            substrate_vectors(array): substrate vectors to generate super
                lattices
        """
        for (
                film_transformations,
                substrate_transformations,
        ) in transformation_sets:
            # Apply transformations and reduce using Zur reduce methodology
            films = [
                reduce_vectors(*np.dot(f, film_vectors))
                for f in film_transformations
            ]

            substrates = [
                reduce_vectors(*np.dot(s, substrate_vectors))
                for s in substrate_transformations
            ]

            # Check if equivalant super lattices
            for (f_trans, s_trans), (f, s) in zip(
                    product(film_transformations, substrate_transformations),
                    product(films, substrates),
            ):
                if self.is_same_vectors(f, s):
                    yield [f, s, f_trans, s_trans]

    def __call__(self, film_vectors, substrate_vectors, lowest=False):
        """Run the ZSL algorithm to generate all possible matching."""
        film_area = vec_area(*film_vectors)
        substrate_area = vec_area(*substrate_vectors)

        # Generate all super lattice comnbinations for a given set of miller
        # indicies
        transformation_sets = self.generate_sl_transformation_sets(
            film_area, substrate_area
        )

        # Check each super-lattice pair to see if they match
        for match in self.get_equiv_transformations(
                transformation_sets, film_vectors, substrate_vectors
        ):
            # Yield the match area, the miller indicies,
            yield self.match_as_dict(
                match[0],
                match[1],
                film_vectors,
                substrate_vectors,
                vec_area(*match[0]),
                match[2],
                match[3],
            )

            if lowest:
                break

    @staticmethod
    def match_as_dict(film_sl_vectors,
                      substrate_sl_vectors,
                      film_vectors,
                      substrate_vectors,
                      match_area,
                      film_transformation,
                      substrate_transformation,
                      ):
        """
        Return dict which contains ZSL match.

        Args:
            film_miller(array)

            substrate_miller(array)
        """
        d = dict()
        d["film_sl_vecs"] = np.asarray(film_sl_vectors)
        d["sub_sl_vecs"] = np.asarray(substrate_sl_vectors)
        d["match_area"] = match_area
        d["film_vecs"] = np.asarray(film_vectors)
        d["sub_vecs"] = np.asarray(substrate_vectors)
        d["film_transformation"] = np.asarray(film_transformation)
        d["substrate_transformation"] = np.asarray(substrate_transformation)

        return d


def gen_sl_transform_matricies(area_multiple):
    """
    Generate the transformation matricies.

    Convert a set of 2D vectors into a super
    lattice of integer area multiple as proven
    in Cassels:
    Cassels, John William Scott. An introduction to the geometry of
    numbers. Springer Science & Business Media, 2012.

    Args:
        area_multiple(int): integer multiple of unit cell area for super
        lattice area.

    Returns:
        matrix_list: transformation matricies to covert unit vectors to
        super lattice vectors.
    """
    return [
        np.array(((i, j), (0, area_multiple / i)))
        for i in get_factors(area_multiple)
        for j in range(area_multiple // i)
    ]


def rel_strain(vec1, vec2):
    """Calculate relative strain between two vectors."""
    return fast_norm(vec2) / fast_norm(vec1) - 1


def rel_angle(vec_set1, vec_set2):
    """
    Calculate the relative angle between two vector sets.

    Args:
        vec_set1(array[array]): an array of two vectors.

        vec_set2(array[array]): second array of two vectors.
    """
    return (
            vec_angle(vec_set2[0], vec_set2[1])
            / vec_angle(vec_set1[0], vec_set1[1])
            - 1
    )


def fast_norm(a):
    """Much faster variant of numpy linalg norm."""
    return np.sqrt(np.dot(a, a))


def vec_angle(a, b, degree=False):
    """Calculate angle between two vectors."""
    cosang = np.dot(a, b)
    sinang = fast_norm(np.cross(a, b))
    if degree:
        return math.degrees(
            np.arctan2(sinang, cosang)
        )
    return np.arctan2(sinang, cosang)



def vec_area(a, b):
    """Area of lattice plane defined by two vectors."""
    return fast_norm(np.cross(a, b))


def reduce_vectors(a, b):
    """Generate independent and unique basis vectors based on Zur et al."""
    if np.dot(a, b) < 0:
        return reduce_vectors(a, -b)
    if fast_norm(a) > fast_norm(b):
        return reduce_vectors(b, a)
    if fast_norm(b) > fast_norm(np.add(b, a)):
        return reduce_vectors(a, np.add(b, a))
    if fast_norm(b) > fast_norm(np.subtract(b, a)):
        return reduce_vectors(a, np.subtract(b, a))
    return [a, b]


def get_factors(n):
    """Generate all factors of n."""
    for x in range(1, n + 1):
        if n % x == 0:
            yield x


def get_hetero_type(A=None, B=None):
    """Provide heterojunction classification using Anderson rule."""
    if B is None:
        B = {}
    if A is None:
        A = {}
    stack = "na"
    int_type = "na"
    try:
        if A["scf_vbm"] - A["avg_max"] < B["scf_vbm"] - B["avg_max"]:
            stack = "BA"
        else:
            C = A
            D = B
            A = D
            B = C
            stack = "AB"
        vbm_a = A["scf_vbm"] - A["avg_max"]
        vbm_b = B["scf_vbm"] - B["avg_max"]
        cbm_a = A["scf_cbm"] - A["avg_max"]
        cbm_b = B["scf_cbm"] - B["avg_max"]
        if vbm_a < vbm_b < cbm_b < cbm_a:
            int_type = "I"
        elif vbm_a < vbm_b < cbm_a < cbm_b:
            int_type = "II"
        elif vbm_a < cbm_a < vbm_b < cbm_b:
            int_type = "III"
    except Exception:
        pass
    return int_type, stack


if __name__ == '__main__':
    pass
