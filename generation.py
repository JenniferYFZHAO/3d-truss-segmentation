import numpy as np

def generate_truss_point_cloud(nodes_coords_dict, member_connectivity, points_per_member=100, 
                               radius=0.1, points_per_circle=20, noise_std=0.05, num_noise_points=50,
                               node_sphere_radius=0.2, points_per_sphere=50):
    """
    Generates a simulated point cloud for a truss structure with cylindrical members and spherical nodes.
    
    Args:
        nodes_coords_dict (dict): Dictionary of node coordinates {node_id: [x, y, z]}
        member_connectivity (list of tuples): List of member connections [(start_node_id, end_node_id), ...]
        points_per_member (int): Number of center points to generate along each member's length
        radius (float): Radius of the cylindrical members
        points_per_circle (int): Number of points to generate around each circle at each center point
        noise_std (float): Standard deviation of noise to add to points
        num_noise_points (int): Number of additional noise points to generate
        node_sphere_radius (float): Radius of the spherical nodes
        points_per_sphere (int): Number of points to generate on each sphere's surface
        
    Returns:
        tuple: (point_cloud, ground_truth_membership)
            point_cloud: Generated point cloud array of shape (N, 3)
            ground_truth_membership: Ground truth membership array of shape (N,)
    """
    # Simulate Point Cloud with cylindrical members and spherical nodes
    simulated_point_cloud_parts = []
    simulated_membership_gt = [] # Ground truth for verification

    # --- Step 1: Generate cylindrical members ---
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
        
        # Generate center points along the member length (skip the ends to avoid overlap with spheres)
        # Start slightly after start node and end slightly before end node
        t_start = node_sphere_radius / member_length if member_length > 0 else 0
        t_end = 1 - t_start
        t_vals = np.linspace(t_start, t_end, points_per_member)
        
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
        
    # --- Step 2: Generate spherical nodes ---
    # We'll assign sphere points to the first member that connects to the node
    node_to_first_member = {}
    for idx, (start_id, end_id) in enumerate(member_connectivity):
        if start_id not in node_to_first_member:
            node_to_first_member[start_id] = idx
        if end_id not in node_to_first_member:
            node_to_first_member[end_id] = idx
    
    for node_id, node_coord in nodes_coords_dict.items():
        sphere_points = generate_sphere_points(
            center=np.array(node_coord),
            radius=node_sphere_radius,
            num_points=points_per_sphere
        )
        
        # Add small random noise
        if noise_std > 0:
            noise = np.random.normal(scale=noise_std, size=sphere_points.shape)
            sphere_points = sphere_points + noise
        
        simulated_point_cloud_parts.append(sphere_points)
        
        # Assign ground truth membership - assign to the first member that connects to this node
        member_idx = node_to_first_member.get(node_id, -1)
        total_points_for_sphere = len(sphere_points)
        simulated_membership_gt.extend([member_idx] * total_points_for_sphere)
        
    simulated_point_cloud = np.vstack(simulated_point_cloud_parts)
    ground_truth_membership = np.array(simulated_membership_gt)
    
    # --- Step 3: Add some noise points not belonging to any member ---
    # Determine bounds for noise points
    all_coords = np.array(list(nodes_coords_dict.values()))
    min_coords = np.min(all_coords, axis=0) - 1
    max_coords = np.max(all_coords, axis=0) + 1
    
    noise_points = np.random.uniform(min_coords, max_coords, size=(num_noise_points, 3))
    
    simulated_point_cloud_with_noise = np.vstack([simulated_point_cloud, noise_points])
    # Ground truth for noise points remains -1, extend gt array
    ground_truth_membership_extended = np.hstack([ground_truth_membership, [-1]*num_noise_points])
    
    return simulated_point_cloud_with_noise, ground_truth_membership_extended


def generate_sphere_points(center, radius, num_points):
    """
    Generates points on the surface of a sphere using the Fibonacci lattice method.
    
    Args:
        center (numpy.ndarray): Center of the sphere [x, y, z]
        radius (float): Radius of the sphere
        num_points (int): Number of points to generate
        
    Returns:
        numpy.ndarray: Array of points on the sphere surface, shape (num_points, 3)
    """
    points = []
    
    # Use Fibonacci lattice for uniform distribution
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    
    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)  # radius at y
        
        theta = phi * i  # golden angle increment
        
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        
        # Scale to sphere radius and translate to center
        point = center + radius * np.array([x, y, z])
        points.append(point)
    
    return np.array(points)
