#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict

from structure.atoms import Atoms, np
from structure.lattice import Lattice, lattice_coords_transformer
from structure.model.match import ZSLGenerator, fast_norm, vec_angle, vec_area


def fix_pbc(atoms):
    """Use for making Atoms with vacuum."""
    new_f_coords = []
    for i in atoms.frac_coords:
        if i[2] > 0.5:
            i[2] = i[2] - 1
        if i[2] < -0.5:
            i[2] = i[2] + 1
        new_f_coords.append(i)

    return Atoms(lattice=atoms.matrix, elements=atoms.elements, coords=new_f_coords, cartesian=False)


def add_atoms(top, bottom, distance=None, apply_strain=False):
    """
    Add top and bottom Atoms with a distance array.

    Bottom Atoms lattice-matrix is chosen as final lattice.
    """
    if distance is None:
        distance = [0, 0, 1]
    top = top.center_around_origin([0, 0, 0])
    bottom = bottom.center_around_origin(distance)
    strain_x = (
                       top.matrix[0][0] - bottom.matrix[0][0]
               ) / bottom.matrix[0][0]
    strain_y = (
                       top.matrix[1][1] - bottom.matrix[1][1]
               ) / bottom.matrix[1][1]
    if apply_strain:
        top.apply_strain([strain_x, strain_y, 0])
    #  print("strain_x,strain_y", strain_x, strain_y)
    elements = []
    coords = []
    lattice_mat = bottom.matrix
    for i, j in zip(bottom.elements, bottom.frac_coords):
        elements.append(i)
        coords.append(j)
    top_cart_coords = lattice_coords_transformer(
        new_lattice_mat=top.matrix,
        old_lattice_mat=bottom.matrix,
        cart_coords=top.cart_coords,
    )
    top_frac_coords = bottom.lattice.frac_coords(top_cart_coords)
    for i, j in zip(top.elements, top_frac_coords):
        elements.append(i)
        coords.append(j)

    order = np.argsort(np.array(elements))
    elements = np.array(elements)[order]
    coords = np.array(coords)[order]
    determnt = np.linalg.det(np.array(lattice_mat))
    if determnt < 0.0:
        lattice_mat = -1 * np.array(lattice_mat)
    determnt = np.linalg.det(np.array(lattice_mat))
    if determnt < 0.0:
        print("Serious issue, check lattice vectors.")
        print("Many software follow right hand basis rule only.")
    combined = Atoms(
        lattice=lattice_mat,
        coords=coords,
        elements=elements,
        cartesian=False,
    ).center_around_origin()
    return combined


def make_interface(film: Atoms, subs: Atoms, atol=1, ltol=0.05, max_area=500, max_area_ratio_tol=1.00,
                   seperation=3.0,
                   vacuum=8.0, apply_strain=False):

    z = ZSLGenerator(max_area_ratio_tol=max_area_ratio_tol, max_area=max_area,
                     max_length_tol=ltol, max_angle_tol=atol)

    film = fix_pbc(film)
    subs = fix_pbc(subs)
    matches = list(z(film.matrix[:2], subs.matrix[:2], lowest=True))

    k = {"mismatch_u", "mismatch_v", "mismatch_angle",
         "films_area", "subs_area", "film_sl", "matches",
         "subs_sl", "interface", "formula"}
    hetero_info = defaultdict(str).fromkeys(k)
    hetero_info["matches"] = matches

    uv_substrate = np.asarray(matches[0]["sub_sl_vecs"])
    uv_film = np.asarray(matches[0]["film_sl_vecs"])
    a1, a2, *_ = uv_substrate
    b1, b2, *_ = uv_film
    mismatch_u = fast_norm(b1) / fast_norm(a1) - 1
    mismatch_v = fast_norm(b2) / fast_norm(a2) - 1
    angle1 = vec_angle(a1, a2, True)
    angle2 = vec_angle(b1, b2, True)
    mismatch_angle = abs(angle1 - angle2)
    area1 = vec_area(a1, a2)
    area2 = vec_area(b1, b2)

    substrate_latt = Lattice(
        np.array(
            [uv_substrate[0][:], uv_substrate[1][:], subs.matrix[2, :]]
        )
    )

    *_, scell = subs.lattice.find_matches(
        substrate_latt, ltol=ltol, atol=atol
    )
    film_latt = Lattice(
        np.array([uv_film[0][:], uv_film[1][:], film.matrix[2, :]])
    )
    scell[2] = np.array([0, 0, 1])
    scell_subs = scell
    *_, scell = film.lattice.find_matches(film_latt, ltol=ltol, atol=atol)
    scell[2] = np.array([0, 0, 1])
    scell_film = scell
    film_scell = film.make_supercell_matrix(scell_film)
    subs_scell = subs.make_supercell_matrix(scell_subs)
    hetero_info["mismatch_u"] = mismatch_u
    hetero_info["mismatch_v"] = mismatch_v
    hetero_info["mismatch_angle"] = mismatch_angle
    hetero_info["films_area"] = area1
    hetero_info["subs_area"] = area2
    hetero_info["film_sl"] = film_scell
    hetero_info["subs_sl"] = subs_scell
    substrate_top_z = max(np.array(subs_scell.cart_coords)[:, 2])
    substrate_bot_z = min(np.array(subs_scell.cart_coords)[:, 2])
    film_top_z = max(np.array(film_scell.cart_coords)[:, 2])
    film_bottom_z = min(np.array(film_scell.cart_coords)[:, 2])
    thickness_sub = abs(substrate_top_z - substrate_bot_z)
    thickness_film = abs(film_top_z - film_bottom_z)
    sub_z = (
            (vacuum + substrate_top_z)
            * np.array(subs_scell.matrix[2, :])
            / fast_norm(subs_scell.matrix[2, :])
    )
    shift_normal = (
            sub_z / fast_norm(sub_z) * seperation / fast_norm(sub_z)
    )
    tmp = (
                  thickness_film / 2 + seperation + thickness_sub / 2
          ) / fast_norm(subs_scell.matrix[2, :])
    shift_normal = (
            tmp
            * np.array(subs_scell.matrix[2, :])
            / fast_norm(subs_scell.matrix[2, :])
    )
    interface = add_atoms(
        film_scell, subs_scell, shift_normal, apply_strain=apply_strain
    ).center_around_origin([0, 0, 0.5])
    combined = interface.center(vacuum=vacuum).center_around_origin(
        [0, 0, 0.5]
    )
    hetero_info["interface"] = combined
    hetero_info["formula"] = combined.formula

    return hetero_info


if __name__ == '__main__':
    pass
