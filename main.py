import numpy as np
from scipy.spatial.distance import cdist # For efficient distance calculations

def rough_segmentation(point_cloud, nodes_coords, member_connectivity, max_member_length, tolerance_distance):
    """
    Performs rough point cloud segmentation based on prior knowledge of node coordinates and member connectivity.

    Args:
        point_cloud (numpy.ndarray): The complete point cloud array of shape (N, 3).
        nodes_coords (dict or numpy.ndarray): Node coordinates.
            If dict: {node_id: [x, y, z], ...}
            If array: [[x1, y1, z1], [x2, y2, z2], ...] where index corresponds to node_id.
        member_connectivity (list of tuples): List of member connections [(start_node_id, end_node_id), ...].
        max_member_length (float): The maximum possible length of a member in the structure.
        tolerance_distance (float): Distance threshold to consider a point as belonging to a member's centerline.

    Returns:
        numpy.ndarray: An array of shape (N,) containing member IDs for each point in point_cloud.
                       Unassigned points have an ID of -1.
    """
    print("Starting rough segmentation...")
    
    # --- Preprocessing: Convert inputs to consistent formats ---
    # Assume nodes_coords is an array for simplicity. If it's a dict, convert it first.
    # Let's handle both dict and array inputs for nodes_coords.
    if isinstance(nodes_coords, dict):
        # Sort keys to maintain order, assuming integer IDs starting from 0
        sorted_ids = sorted(nodes_coords.keys())
        nodes_array = np.array([nodes_coords[node_id] for node_id in sorted_ids])
        # Create a mapping from original node_id to array index
        node_id_to_index = {node_id: idx for idx, node_id in enumerate(sorted_ids)}
    else:
        # If nodes_coords is already an array
        nodes_array = np.asarray(nodes_coords)
        node_id_to_index = {idx: idx for idx in range(len(nodes_array))} # Identity mapping

    # Convert member connectivity to use array indices instead of original IDs if needed
    member_indices = []
    for start_id, end_id in member_connectivity:
        start_idx = node_id_to_index.get(start_id)
        end_idx = node_id_to_index.get(end_id)
        if start_idx is not None and end_idx is not None:
            member_indices.append((start_idx, end_idx))
        else:
            print(f"Warning: Member ({start_id}, {end_id}) references unknown node(s). Skipping.")
    member_indices = np.array(member_indices)

    num_points = point_cloud.shape[0]
    membership = np.full(num_points, -1, dtype=int) # Initialize all memberships to -1

    # Precompute member endpoints for faster access
    if len(member_indices) > 0:
        start_nodes = nodes_array[member_indices[:, 0]] # Shape: (num_members, 3)
        end_nodes = nodes_array[member_indices[:, 1]]   # Shape: (num_members, 3)
        all_member_lines = np.stack([start_nodes, end_nodes], axis=1) # Shape: (num_members, 2, 3)
    else:
        print("No valid members found. Returning unassigned memberships.")
        return membership

    # Define search radius (half the side length of the cube)
    search_radius = max_member_length

    # --- Main Loop ---
    # Note: This nested loop approach might be slow for very large datasets.
    # Vectorization is complex here due to the varying number of potential members per point.
    # A KDTree or similar spatial index could significantly speed this up.
    for i, p in enumerate(point_cloud):
        if i % 5000 == 0: # Print progress every 5000 points
            print(f"Processing point {i}/{num_points}...")

        # 1. Determine search domain (cube centered at p)
        # Find points within the search radius in each dimension
        # This is an approximation of the cube search using axis-aligned bounding box check
        # More precise would be to check if distance to cube center < radius in all dims
        # But for efficiency, we'll use a spherical search here, which is simpler and often sufficient.
        # Alternatively, find bounding box and filter nodes/lines.
        
        # Use a spherical search for the "potential nodes"
        node_distances = np.linalg.norm(nodes_array - p, axis=1)
        potential_node_mask = node_distances <= search_radius
        potential_node_indices = np.where(potential_node_mask)[0]

        if len(potential_node_indices) < 2:
            # Not enough potential nodes to define a potential member
            continue 

        # 2. Find "potential members" among those connecting nodes in the potential set
        # Check if both start and end indices of any member are in the potential_node_indices
        # This is done by checking if the start and end indices of each member are subsets of potential_node_indices
        # More efficient way using broadcasting/boolean indexing:
        # Create a boolean mask for potential nodes
        potential_node_set_mask = np.zeros(len(nodes_array), dtype=bool)
        potential_node_set_mask[potential_node_indices] = True
        
        # Check which members have both start and end nodes in the potential set
        starts_in_pot_set = potential_node_set_mask[member_indices[:, 0]]
        ends_in_pot_set = potential_node_set_mask[member_indices[:, 1]]
        potential_member_mask = starts_in_pot_set & ends_in_pot_set
        potential_member_indices = np.where(potential_member_mask)[0]

        if len(potential_member_indices) == 0:
            continue # No potential members found for this point

        # 3. Calculate distance from P_i to each potential member's centerline (line segment)
        # Extract the relevant lines for this point
        pot_lines = all_member_lines[potential_member_indices] # Shape: (num_pot_memb, 2, 3)

        # Calculate distances from p to each potential line segment
        # We need to loop through potential members or vectorize the distance calc
        distances_to_lines = []
        for line in pot_lines:
            dist = point_to_line_segment_distance(p, line[0], line[1]) # p, start_point, end_point
            distances_to_lines.append(dist)
        
        distances_to_lines = np.array(distances_to_lines)

        # 4. Find the closest member within tolerance
        min_dist_idx = np.argmin(distances_to_lines)
        min_dist = distances_to_lines[min_dist_idx]

        if min_dist <= tolerance_distance:
            # Assign the point to the closest member found among potential members
            original_member_idx = potential_member_indices[min_dist_idx]
            membership[i] = original_member_idx

    print("Rough segmentation completed.")
    return membership


def point_to_line_segment_distance(point, seg_start, seg_end):
    """
    Calculates the shortest distance from a point to a 3D line segment.
    Adapted from https://stackoverflow.com/questions/56220596/how-to-find-the-closest-point-on-a-line-segment-to-another-point
    """
    # Vector from segment start to end
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    
    # Handle degenerate case where segment is a point
    if seg_len_sq == 0:
        return np.linalg.norm(point - seg_start)

    # Vector from segment start to the query point
    pt_vec = point - seg_start
    
    # Calculate projection parameter 't'
    t = np.dot(pt_vec, seg_vec) / seg_len_sq
    t = np.clip(t, 0.0, 1.0) # Clamp t to [0, 1] to stay within the segment

    # Calculate the closest point on the segment
    closest_pt = seg_start + t * seg_vec
    
    # Return the distance from the query point to the closest point on the segment
    return np.linalg.norm(point - closest_pt)


# --- Example Usage ---
if __name__ == "__main__":
    # Simulate some data to test the function
    print("--- Running Simulation ---")
    
    # 1. Simulate Nodes (4 nodes forming 3 members)
    simulated_nodes_coords_dict = {
        0: [0.0, 0.0, 0.0],
        1: [5.0, 0.0, 0.0],
        2: [5.0, 5.0, 0.0],
        3: [0.0, 5.0, 0.0]
    }
    # Convert to array format for easier handling in this script
    node_ids_order = sorted(simulated_nodes_coords_dict.keys())
    nodes_array_sim = np.array([simulated_nodes_coords_dict[k] for k in node_ids_order])
    
    # 2. Define Members (Connectivity)
    simulated_member_connectivity = [
        (0, 1), # Member 0: from node 0 to node 1
        (1, 2), # Member 1: from node 1 to node 2
        (2, 3)  # Member 2: from node 2 to node 3
    ]
    
    # 3. Simulate Point Cloud along these members (add some noise)
    simulated_point_cloud_parts = []
    simulated_membership_gt = [] # Ground truth for verification

    for idx, (start_id, end_id) in enumerate(simulated_member_connectivity):
        start_coord = simulated_nodes_coords_dict[start_id]
        end_coord = simulated_nodes_coords_dict[end_id]
        
        # Generate points along the line segment
        num_points_per_member = 100
        t_vals = np.linspace(0, 1, num_points_per_member)
        line_points = np.array(start_coord)[:, None] + t_vals * (np.array(end_coord) - np.array(start_coord))[:, None]
        line_points = line_points.T # Shape: (num_points, 3)
        
        # Add small random noise
        noise = np.random.normal(scale=0.05, size=line_points.shape) # 5cm std dev noise
        noisy_line_points = line_points + noise
        
        simulated_point_cloud_parts.append(noisy_line_points)
        
        # Assign ground truth membership
        simulated_membership_gt.extend([idx] * num_points_per_member)
        
    simulated_point_cloud = np.vstack(simulated_point_cloud_parts)
    ground_truth_membership = np.array(simulated_membership_gt)
    
    # Add some noise points not belonging to any member
    num_noise_points = 50
    noise_x = np.random.uniform(-1, 6, num_noise_points)
    noise_y = np.random.uniform(-1, 6, num_noise_points)
    noise_z = np.random.uniform(-0.5, 0.5, num_noise_points)
    noise_points = np.column_stack((noise_x, noise_y, noise_z))
    
    simulated_point_cloud_with_noise = np.vstack([simulated_point_cloud, noise_points])
    # Ground truth for noise points remains -1, extend gt array
    ground_truth_membership_extended = np.hstack([ground_truth_membership, [-1]*num_noise_points])

    # 4. Run the Segmentation Algorithm
    max_len = 6.0 # Estimated max length between nodes
    tol_dist = 0.1 # Tolerance for assignment (slightly larger than noise std)

    membership_result = rough_segmentation(
        point_cloud=simulated_point_cloud_with_noise,
        nodes_coords=simulated_nodes_coords_dict, # Pass the dictionary
        member_connectivity=simulated_member_connectivity,
        max_member_length=max_len,
        tolerance_distance=tol_dist
    )

    # 5. Compare Results with Ground Truth
    print("\n--- Verification ---")
    print(f"Total points: {len(membership_result)}")
    print(f"Number of assigned points (not -1): {np.sum(membership_result != -1)}")
    print(f"Number of unassigned points (-1): {np.sum(membership_result == -1)}")
    
    # Calculate accuracy for assigned points
    assigned_mask = ground_truth_membership_extended != -1
    if np.any(assigned_mask):
        correct_assignments = np.sum(membership_result[assigned_mask] == ground_truth_membership_extended[assigned_mask])
        total_assigned_gt = np.sum(assigned_mask)
        accuracy_assigned = correct_assignments / total_assigned_gt if total_assigned_gt > 0 else 0
        print(f"Accuracy on assigned points: {accuracy_assigned:.2%} ({correct_assignments}/{total_assigned_gt})")
    else:
        print("No points were part of the ground truth structure.")

    print("\nFirst 10 membership results:", membership_result[:10])
    print("First 10 ground truth memberships:", ground_truth_membership_extended[:10])
    print("Last 10 membership results:", membership_result[-10:])
    print("Last 10 ground truth memberships:", ground_truth_membership_extended[-10:])