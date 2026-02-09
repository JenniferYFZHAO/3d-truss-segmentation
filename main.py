import numpy as np
from segmentation import rough_segmentation
from visualization import visualize_point_cloud

if __name__ == "__main__":
    # --- Running Simulation ---
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

    # 6. Visualize Results
    print("\n--- Visualization ---")
    print("Visualizing pre-segmentation point cloud...")
    visualize_point_cloud(simulated_point_cloud_with_noise, title="Pre-segmentation Point Cloud")
    
    print("Visualizing segmentation results...")
    visualize_point_cloud(simulated_point_cloud_with_noise, membership_result, title="Post-segmentation Point Cloud")
    
    print("Visualizing ground truth...")
    visualize_point_cloud(simulated_point_cloud_with_noise, ground_truth_membership_extended, title="Ground Truth Membership")
