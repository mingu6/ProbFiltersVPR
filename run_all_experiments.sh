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
echo "Running SeqSLAM... (4/4)"
SeqSLAM.py
echo "Generating tables..."
ParticleFilter.py 
tables.py -f tables1.txt
ParticleFilter.py 
tables.py -f tables2.txt
ParticleFilter.py 
tables.py -f tables3.txt
ParticleFilter.py 
tables.py -f tables4.txt
ParticleFilter.py 
tables.py -f tables5.txt
echo "Generate PR curve..."
PR_curves.py -t 3 -R 5
PR_curves.py -t 5 -R 10
echo "All experiments complete! All results generated and saved successfully."
