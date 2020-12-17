#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import *
import argparse

from structure import (
    read_model_from, classify_bulk_materials, cleave_surface_model,
    get_monolayer_model, get_hetero_junction_model
)


def main():
    parser = argparse.ArgumentParser("Crystal2d: An 2D material modeling tool")
    parser.add_argument('-film')
    parser.add_argument('-subs')
    parser.add_argument('-bulk')


if __name__ == '__main__':
    main()
    pth = r"./test/hetero"
    all_path = [join(pth, i) for i in os.listdir(pth)]
    fp1 = join(pth, "05_GaN001_POSCAR")
    fp2 = join(pth, "03_Graphene_POSCAR")

    mat1, mat2 = read_model_from(fp1), read_model_from(fp2)
    hetero_model = get_hetero_junction_model(mat1, mat2, write=True)

    # monolayer
    fp = r"./test/2d/1005003.cif"
    fn, _ = splitext(basename(fp))

    cell = list(read_model_from(fp))[0]

    res = classify_bulk_materials(cell)
    print(res)
    get_monolayer_model(cell, central=True, unitcell=True, vacuum=10, write=True)
    nfp = r"./test/2d/1005003.cif"
    nc = list(read_model_from(nfp))[0]
    cleave_surface_model(nc, [1, 0, 0], layers=3, write=True)
