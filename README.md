# Probabilistic Visual Place Recognition for Hierarchical Localization

This repository contains code to accompany our arXiv preprint *Probabilistic Visual Place Recognition for Hierarchical Localization*. It provides the means to reproduce all of our experiments and general datastructures to apply our proposed methods to other traverse style datasets.

## Instructions

1. Run the makefile inside the base directory to install dependencies and download the RobotCar dataset. 

**NOTE:** You will require an account for permissions to download the RobotCar dataset. See [link](https://robotcar-dataset.robots.ox.ac.uk/) for more information around setting up an account if you do not have one already.

In addition, I would suggest setting up a [conda](https://docs.conda.io/en/latest/miniconda.html) environment with

```
conda create -n ProbFiltersVPR
conda activate ProbFiltersVPR
```

Finally, run 

`make install`

You will be prompted for the base directory for the RobotCar dataset as well as your RobotCar account details to download the relevant traverse data.

2. Run `ready_images.py` (this should be a script so you can run this command directly) inside your terminal to deBayer and undistort the raw image data. These images will be stored in the `ready` folder inside the data directory.

3. Extract features from the readied images. See [NetVLAD](https://github.com/uzh-rpg/netvlad_tf_open) and [DenseVLAD](http://www.ok.ctrl.titech.ac.jp/~torii/project/247/) for respective code to extract features.

You will need to create  an NxD numpy array where N is the number of images and D is the dimensionality of the descriptor (D=4096) by default and save this to for all traverses and descriptor combinations. You may also add your own descriptors to test it out with this model, just use the below format. Ensure the array is ordered by timestamp!!!

`DATA_PATH/processed/{traverse_name}/stereo/left/{descriptor_name.npy}`


4. Download the RTK data from
   [here](https://robotcar-dataset.robots.ox.ac.uk/ground_truth/). Extract the `rtk.csv` files for the relevant traverse names to `{base_data_dir/raw/traverse_name/}`.

5. Run `process_all_rtk.sh` in the base repo directory, which interpolates raw
   RTK data to the image timestamps and creates the reference and query
   traverses.

6. Run `run_all_experiments.sh` to apply our models, baseline and comparisons
   to the generated traverses.
