#!/bin/bash 

# Absolute path to this script
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

cd $SCRIPTPATH
echo "Running SeqSLAM... (4/4)"
SeqSLAM.py -d DenseVLAD
