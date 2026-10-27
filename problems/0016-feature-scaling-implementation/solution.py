import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	# standardization
    scaled_mat1 = np.round((data - data.mean(axis=0)) / data.std(axis=0), decimals=4)
    # min-max normalization
    scaled_mat2 = np.round((data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0)), decimals=4)

    return (scaled_mat1, scaled_mat2)