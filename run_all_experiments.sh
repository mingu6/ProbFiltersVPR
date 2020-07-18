#!/bin/bash 

# Absolute path to this script
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

cd $SCRIPTPATH
echo "Running Monte Carlo localization... (1/4)"
python3 src/models/ParticleFilter.py
echo "Running topological filter... (2/4)"
python3 src/models/TopologicalFilter.py
echo "Running single image matching... (3/4)"
python3 src/models/SingleImageMatching.py
echo "Running sequence matching... (4/4)"
python3 src/models/SeqMatching.py
echo "Generating PR curves and tables..."
python3 src/figures/results.py 
echo "All experiments complete! All results generated and saved successfully."
