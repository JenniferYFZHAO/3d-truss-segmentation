import numpy as np

def generate_truss_point_cloud(nodes_coords_dict, member_connectivity, points_per_member=100, noise_std=0.05, num_noise_points=50):
    """
    Generates a simulated point cloud for a truss structure with noise.
    
    Args:
        nodes_coords_dict (dict): Dictionary of node coordinates {node_id: [x, y, z]}
        member_connectivity (list of tuples): List of member connections [(start_node_id, end_node_id), ...]
        points_per_member (int): Number of points to generate per member
        noise_std (float): Standard deviation of noise to add to points
        num_noise_points (int): Number of additional noise points to generate
        
    Returns:
        tuple: (point_cloud, ground_truth_membership)
            point_cloud: Generated point cloud array of shape (N, 3)
            ground_truth_membership: Ground truth membership array of shape (N,)
    """
    # Simulate Point Cloud along these members (add some noise)
    simulated_point_cloud_parts = []
    simulated_membership_gt = [] # Ground truth for verification

    for idx, (start_id, end_id) in enumerate(member_connectivity):
        start_coord = nodes_coords_dict[start_id]
        end_coord = nodes_coords_dict[end_id]
        
        # Generate points along the line segment
        t_vals = np.linspace(0, 1, points_per_member)
        line_points = np.array(start_coord)[:, None] + t_vals * (np.array(end_coord) - np.array(start_coord))[:, None]
        line_points = line_points.T # Shape: (num_points, 3)
        
        # Add small random noise
        noise = np.random.normal(scale=noise_std, size=line_points.shape) # Noise with specified std dev
        noisy_line_points = line_points + noise
        
        simulated_point_cloud_parts.append(noisy_line_points)
        
        # Assign ground truth membership
        simulated_membership_gt.extend([idx] * points_per_member)
        
    simulated_point_cloud = np.vstack(simulated_point_cloud_parts)
    ground_truth_membership = np.array(simulated_membership_gt)
    
    # Add some noise points not belonging to any member
    # Determine bounds for noise points
    all_coords = np.array(list(nodes_coords_dict.values()))
    min_coords = np.min(all_coords, axis=0) - 1
    max_coords = np.max(all_coords, axis=0) + 1
    
    noise_points = np.random.uniform(min_coords, max_coords, size=(num_noise_points, 3))
    
    simulated_point_cloud_with_noise = np.vstack([simulated_point_cloud, noise_points])
    # Ground truth for noise points remains -1, extend gt array
    ground_truth_membership_extended = np.hstack([ground_truth_membership, [-1]*num_noise_points])
    
    return simulated_point_cloud_with_noise, ground_truth_membership_extended
