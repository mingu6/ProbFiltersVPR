#!/bin/bash 

# Absolute path to this script
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

cd $SCRIPTPATH
echo "Running Monte Carlo localization... (1/4)"
ParticleFilter.py -d DenseVLAD
echo "Running single image matching... (3/4)"
SingleImageMatching.py -d DenseVLAD
