#!/bin/bash

rm Common_cython/*.c
python3 Common_cython/setup2.py build_ext --inplace
python3 Common_cython/setup.py build_ext --inplace

