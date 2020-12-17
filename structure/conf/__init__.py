#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
from os.path import *

Settings = {}
js = [join(os.path.dirname(__file__), i)
      for i in os.listdir(dirname(__file__))
      if splitext(i)[-1] == '.json']

for file in js:
    with open(file, "r") as f:
        js_obj = json.load(f)
        Settings[basename(file).replace('.json', '')] = js_obj

if __name__ == "__main__":
    pass
