import numpy as np

outlier_threshold = 200
#################### traverses #######################
traverses = {
    'Overcast': '2015-03-17-11-08-44',
    'Rain': '2015-10-29-12-18-17',
    'Sun': '2015-03-24-13-47-33',
    'Dusk': '2014-11-21-16-07-03',
    'Night': '2014-12-16-18-44-24'
}

############## Initialization noise ##################

sigma_init = np.array([2., 0.5, 0.5, 0.05, 0.05, 0.1]) 
# sigma_init = np.array([1., 0.25, 0.25, 0.02, 0.02, 0.05]) 

############## VO error covariance models #############

# zero mean, independent noise
sigma_vo = {
    # 'Rain': np.array([0.3, 0.05, 0.05, 0.005, 0.005, 0.01]) ,
    'Rain': np.array([0.8, 0.3, 0.3, 0.04, 0.04, 0.08]),
    'Sun': np.array([0.8, 0.3, 0.3, 0.04, 0.04, 0.08]),
    'Dusk': np.array([0.8, 0.3, 0.3, 0.04, 0.04, 0.08]),
    'Night': np.array([0.8, 0.3, 0.3, 0.04, 0.04, 0.08])
} 