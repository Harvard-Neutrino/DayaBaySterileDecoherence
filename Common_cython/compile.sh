#!/bin/bash

rm *.c
python3 setup2.py build_ext --inplace
python3 setup.py build_ext --inplace

