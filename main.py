import numpy as np
from segmentation import rough_segmentation
from visualization import visualize_point_cloud
from generation import generate_truss_point_cloud

if __name__ == "__main__":
    # --- Running Simulation ---
    print("--- Running Simulation ---")
    
    # 1. Define Nodes (4 nodes forming 3 members)
    nodes_coords_dict = {
        0: [0.0, 0.0, 0.0],
        1: [5.0, 0.0, 2.5],
        2: [5.0, 5.0, 5.0],
        3: [0.0, 5.0, 7.5]
    }
    
    # 2. Define Members (Connectivity)
    member_connectivity = [
        (0, 1), # Member 0: from node 0 to node 1
        (1, 2), # Member 1: from node 1 to node 2
        (2, 3)  # Member 2: from node 2 to node 3
    ]
    
    # 3. Generate Point Cloud (cylindrical members with spherical nodes)
    point_cloud, ground_truth_membership = generate_truss_point_cloud(
        nodes_coords_dict=nodes_coords_dict,
        member_connectivity=member_connectivity,
        points_per_member=50,           # 沿杆件长度的点数
        radius=0.2,                     # 圆管半径
        points_per_circle=12,            # 每个圆周上的点数
        noise_std=0.03,
        num_noise_points=50,
        node_sphere_radius=0.35,         # 球节点半径（比杆件半径稍大）
        points_per_sphere=80             # 每个球面上的点数
    )

    # 4. Run the Segmentation Algorithm
    max_len = 6.0 # Estimated max length between nodes
    tol_dist = 0.4 # Tolerance for assignment (needs to be larger than cylinder radius)

    membership_result = rough_segmentation(
        point_cloud=point_cloud,
        nodes_coords=nodes_coords_dict, # Pass the dictionary
        member_connectivity=member_connectivity,
        max_member_length=max_len,
        tolerance_distance=tol_dist
    ) 

    # 5. Compare Results with Ground Truth
    print("\n--- Verification ---")
    print(f"Total points: {len(membership_result)}")
    print(f"Number of assigned points (not -1): {np.sum(membership_result != -1)}")
    print(f"Number of unassigned points (-1): {np.sum(membership_result == -1)}")
    
    # Calculate accuracy for assigned points
    assigned_mask = ground_truth_membership != -1
    if np.any(assigned_mask):
        correct_assignments = np.sum(membership_result[assigned_mask] == ground_truth_membership[assigned_mask])
        total_assigned_gt = np.sum(assigned_mask)
        accuracy_assigned = correct_assignments / total_assigned_gt if total_assigned_gt > 0 else 0
        print(f"Accuracy on assigned points: {accuracy_assigned:.2%} ({correct_assignments}/{total_assigned_gt})")
    else:
        print("No points were part of the ground truth structure.")

    print("\nFirst 10 membership results:", membership_result[:10])
    print("First 10 ground truth memberships:", ground_truth_membership[:10])
    print("Last 10 membership results:", membership_result[-10:])
    print("Last 10 ground truth memberships:", ground_truth_membership[-10:])

    # 6. Visualize Results
    print("\n--- Visualization ---")
    print("Visualizing pre-segmentation point cloud...")
    visualize_point_cloud(point_cloud, title="Pre-segmentation Point Cloud")
    
    print("Visualizing segmentation results...")
    visualize_point_cloud(point_cloud, membership_result, title="Post-segmentation Point Cloud")
    
    print("Visualizing ground truth...")
    visualize_point_cloud(point_cloud, ground_truth_membership, title="Ground Truth Membership")
