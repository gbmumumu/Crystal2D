#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from os.path import *

from .atoms import Atoms, np
from .model.cluster import Cluster
from .model.hetero import make_interface
from .model.surface import Surface


def read_model_from(filepath, new_origin=None):
    """

    Args:
        new_origin:
        *filepath:

    Returns:

    """

    if new_origin is None:
        new_origin = [0, 0, 0]
    return Atoms.from_file(filepath).center_around_origin(new_origin)


def write_model(models, results_dir=None, model_type='hetero', fmt='cif'):
    if results_dir is None:
        results_dir = join(os.getcwd(), 'ModelingResults')
    if not exists(results_dir):
        os.makedirs(results_dir)
    file_name = join(results_dir, models.formula + f"-{model_type}.{fmt}")

    return models.write_file(file_name, fmt=fmt)


def get_hetero_junction_model(substrate, films, save_format="cif", max_area=400,
                              max_area_ratio_tol=0.09,
                              ltol=0.05,
                              atol=0.1, seperation=3.0, apply_strain=False, write=False):
    """

    Args:
        write:
        substrate:
        films:
        save_format:
        max_area:
        max_area_ratio_tol:
        ltol:
        atol:
        seperation:
        apply_strain:

    Returns:

    """
    results = make_interface(films, substrate,
                             max_area=max_area,
                             max_area_ratio_tol=max_area_ratio_tol,
                             ltol=ltol, atol=atol, seperation=seperation,
                             apply_strain=apply_strain)

    comb_model = results["interface"]
    if write:
        return write_model(comb_model, fmt=save_format)

    return results


def classify_bulk_materials(bulk, scale_matrix=None):
    """
    get the bulk material type.
    Args:
        bulk:
        scale_matrix:

    Returns: string.

    """
    if scale_matrix is None:
        scale_matrix = [2, 2, 2]
    super_bulk = bulk.make_supercell(scale_matrix)
    unit_aj_mtx = bulk.get_bond_adjacency_matrix()
    super_aj_mtx = super_bulk.get_bond_adjacency_matrix()
    unit_cluster_num, unit_clusters, noa_in_cluster_unit = Cluster.seek_cluster(unit_aj_mtx, bulk)
    super_cluster_num, super_clusters, noa_in_clusters_super = Cluster.seek_cluster(super_aj_mtx, super_bulk)
    bulk_type = Cluster.get_bulk_type(
        max(noa_in_cluster_unit), min(noa_in_cluster_unit),
        max(noa_in_clusters_super), min(noa_in_clusters_super),
        super_bulk.noa
    )
    return bulk_type


def get_monolayer_model(bulk, clusters_index=0,
                        scale_matrix=None, unitcell=False,
                        save_format='cif', central=False, vacuum=13, write=False):
    """

    Args:
        write:
        vacuum:
        central:
        bulk:
        clusters_index:
        scale_matrix:
        unitcell:
        save_format:

    Returns:

    """
    if unitcell:
        unit_aj_mtx = bulk.get_bond_adjacency_matrix()
        unit_cluster = Cluster(bulk)
        noc, unit_clusters, _ = unit_cluster.get_clusters_from_bulk(unit_aj_mtx)
        cluster = unit_cluster.split(unit_clusters, clusters_index)
    else:
        if scale_matrix is None:
            scale_matrix = [3, 3, 3]
        super_bulk = bulk.make_supercell(scale_matrix)
        super_aj_mtx = super_bulk.get_bond_adjacency_matrix()
        super_cluster = Cluster(super_bulk)
        noc, super_clusters, _ = super_cluster.get_clusters_from_bulk(super_aj_mtx)
        cluster = super_cluster.split(super_clusters, clusters_index)
    poi = Cluster.find_stacking_direction(cluster)
    axis = np.zeros(3)
    axis[poi] = 1
    vcc = cluster.center(axis=poi, vacuum=vacuum)
    name = f'monolayer-vacuum{vacuum}'
    if central:
        vcc = vcc.center_around_origin(axis / 2)
        name = f'monolayer-centered-vacuum{vacuum}'
    if write:
        return write_model(vcc, model_type=name, fmt=save_format)

    return vcc


def cleave_surface_model(bulk, indices, layers,
                         vacuum=13, tol=1e-10,
                         periodic=False, save_format='cif', write=False):
    """

    Args:
        bulk:
        indices:
        layers:
        vacuum:
        tol:
        periodic:
        save_format:
        write:

    Returns:

    """
    surf = Surface(bulk, indices, layers, vacuum, tol, periodic)
    slab = surf.mk_slab()
    name = 'surface-{}'.format(','.join([str(i) for i in indices]))
    if write:
        return write_model(slab, model_type=name, fmt=save_format)
    return slab


if __name__ == '__main__':
    pass
