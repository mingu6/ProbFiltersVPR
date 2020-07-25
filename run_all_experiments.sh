#!/bin/bash 

# Absolute path to this script
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

cd $SCRIPTPATH
echo "Running Monte Carlo localization... (1/5)"
python3 src/models/ParticleFilter.py -q Dusk Night -d Listwise
echo "Running topological filter... (2/5)"
python3 src/models/TopologicalFilter.py -q Dusk Night -d Listwise
echo "Running single image matching... (3/5)"
python3 src/models/SingleImageMatching.py -q Dusk Night -d Listwise
echo "Running sequence matching... (4/5)"
python3 src/models/SeqMatching.py -q Dusk Night -d Listwise
echo "Run graph matching... (5/5)"
python3 src/models/GraphMatching.py -q Dusk Night -d Listwise
echo "Generating PR curves and tables..."
python3 src/figures/results.py
echo "All experiments complete! All results generated and saved successfully."
