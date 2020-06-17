#!/bin/bash 

# Absolute path to this script
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

cd $SCRIPTPATH
echo "Running Monte Carlo localization... (1/4)"
python3 runParticleFilter.py -A
echo "Running topological filter... (2/4)"
python3 TopologicalFilter.py -A
echo "Running single image matching... (3/4)"
python3 single_frame_baseline.py -A
echo "Running SeqSLAM... (4/4)"
python3 seqSLAM_baseline.py -A
echo "All experiments complete!"