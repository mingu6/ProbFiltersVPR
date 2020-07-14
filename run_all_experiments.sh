#!/bin/bash 

# Absolute path to this script
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

cd $SCRIPTPATH
echo "Running Monte Carlo localization... (1/4)"
ParticleFilter.py
echo "Running topological filter... (2/4)"
TopologicalFilter.py
echo "Running single image matching... (3/4)"
SingleImageMatching.py
echo "Running sequence matching... (4/4)"
SeqMatching.py -q Rain
echo "Generating PR curves and tables..."
results.py 
echo "All experiments complete! All results generated and saved successfully."
