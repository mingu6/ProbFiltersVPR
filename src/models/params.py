import numpy as np

#################### traverses #######################
queries = {
    'Rain': '2015-10-29-12-18-17',
    'Dusk': '2014-11-21-16-07-03',
    'Night': '2014-12-16-18-44-24'
}

############## Initialization noise ##################

sigma_init = np.array([2., 0.5, 0.5, 0.05, 0.05, 0.1]) 

############## VO error covariance models #############

# zero mean, independent noise
sigma_vo = {
    'Rain': np.array([0.8, 0.3, 0.3, 0.04, 0.04, 0.08]),
    'Dusk': np.array([0.8, 0.3, 0.3, 0.04, 0.04, 0.08]),
    'Night': np.array([0.8, 0.3, 0.3, 0.04, 0.04, 0.08])
} 