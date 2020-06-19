#!/bin/bash 

# Absolute path to this script
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

cd $SCRIPTPATH
echo "Running Monte Carlo localization... (1/4)"
ParticleFilter.py -d NetVLAD
echo "Running topological filter... (2/4)"
TopologicalFilter.py -d NetVLAD
echo "Running single image matching... (3/4)"
SingleImageMatching.py -d NetVLAD
echo "Running SeqSLAM... (4/4)"
SeqSLAM.py -d NetVLAD
echo "All experiments complete!"
