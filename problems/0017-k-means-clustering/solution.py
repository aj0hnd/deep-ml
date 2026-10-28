import numpy as np
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	points = np.array(points, dtype=np.float32)
    last_centroids = np.array(initial_centroids, dtype=np.float32)
    for _ in range(max_iterations):
        euclidean = np.linalg.norm(points[:, None, :] - last_centroids[None, :, :], ord=2, axis=2)
        groups_id = np.argmin(euclidean, axis=1)
        curr_centroids = np.zeros_like(last_centroids)
        for i in range(k):
            curr_centroids[i] = np.mean(points[groups_id == i], axis=0)
        
        if np.array_equal(last_centroids, curr_centroids):
            final_centriods = curr_centroids
            break
        last_centroids = curr_centroids
    
    final_centriods = last_centroids
    return [tuple(elem) for elem in final_centriods.tolist()]