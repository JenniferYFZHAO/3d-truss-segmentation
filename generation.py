import numpy as np

def generate_truss_point_cloud(nodes_coords_dict, member_connectivity, points_per_member=100, 
                               radius=0.1, points_per_circle=20, noise_std=0.05, num_noise_points=50):
    """
    Generates a simulated point cloud for a truss structure with cylindrical members.
    
    Args:
        nodes_coords_dict (dict): Dictionary of node coordinates {node_id: [x, y, z]}
        member_connectivity (list of tuples): List of member connections [(start_node_id, end_node_id), ...]
        points_per_member (int): Number of center points to generate along each member's length
        radius (float): Radius of the cylindrical members
        points_per_circle (int): Number of points to generate around each circle at each center point
        noise_std (float): Standard deviation of noise to add to points
        num_noise_points (int): Number of additional noise points to generate
        
    Returns:
        tuple: (point_cloud, ground_truth_membership)
            point_cloud: Generated point cloud array of shape (N, 3)
            ground_truth_membership: Ground truth membership array of shape (N,)
    """
    # Simulate Point Cloud with cylindrical members
    simulated_point_cloud_parts = []
    simulated_membership_gt = [] # Ground truth for verification

    for idx, (start_id, end_id) in enumerate(member_connectivity):
        start_coord = np.array(nodes_coords_dict[start_id])
        end_coord = np.array(nodes_coords_dict[end_id])
        
        # Calculate member direction vector
        member_vector = end_coord - start_coord
        member_length = np.linalg.norm(member_vector)
        
        # Create local coordinate system for the cylinder
        # First, find two orthogonal vectors perpendicular to member_vector
        if abs(member_vector[0]) > 0.1 or abs(member_vector[1]) > 0.1:
            # Not aligned with z-axis
            v1 = np.array([-member_vector[1], member_vector[0], 0])
        else:
            # Aligned with z-axis, use x-axis
            v1 = np.array([1, 0, 0])
        
        v1 = v1 / np.linalg.norm(v1)  # Normalize
        v2 = np.cross(member_vector, v1)
        v2 = v2 / np.linalg.norm(v2)  # Normalize
        
        # Generate points along the cylinder
        cylinder_points = []
        
        # Generate center points along the member length
        t_vals = np.linspace(0, 1, points_per_member)
        
        for t in t_vals:
            # Center point at this position along the member
            center = start_coord + t * member_vector
            
            # Generate points in a circle around this center
            angles = np.linspace(0, 2 * np.pi, points_per_circle, endpoint=False)
            
            for angle in angles:
                # Calculate point on the circle
                circle_point = center + radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
                cylinder_points.append(circle_point)
        
        cylinder_points = np.array(cylinder_points)
        
        # Add small random noise
        if noise_std > 0:
            noise = np.random.normal(scale=noise_std, size=cylinder_points.shape)
            cylinder_points = cylinder_points + noise
        
        simulated_point_cloud_parts.append(cylinder_points)
        
        # Assign ground truth membership
        total_points_for_member = len(cylinder_points)
        simulated_membership_gt.extend([idx] * total_points_for_member)
        
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
