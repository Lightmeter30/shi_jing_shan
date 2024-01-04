# !/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys

image_path = sys.argv[1]
output_path = sys.argv[2]

with open(output_path, 'w') as outfile:
    with open(image_path, 'r') as infile:
        for i, line in enumerate(infile, start=1):
            if line.startswith('#'):
                continue
            if i % 2 == 1:
                outfile.write(line)
