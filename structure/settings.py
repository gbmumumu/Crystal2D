#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum, unique

from structure.conf import Settings


@unique
class SpecieInfo(Enum):
    covalent = Settings["covalent"]
    vdw = Settings["vdW"]
    jsmol = Settings["jsmol"]
    map = Settings["periodic_table"]
    vesta = Settings["vesta"]


if __name__ == '__main__':
    pass
