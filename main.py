import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist

def build_and_solve_nonrigid_registration(
    source_points_by_member,  # List of (N_k, 3) arrays, each array contains points for member k
    target_points,            # (M, 3) array, the target point cloud to match against
    member_topology,          # Dict: {member_id: [list_of_connected_node_ids]}
    node_coordinates,         # Dict: {node_id: [x, y, z]} coordinates of nodes in the source/template
    correspondences,          # List of tuples (source_point_idx, target_point_idx)
                              # source_point_idx is global index across all source_points_by_member
    stiffness_weight=1.0,     # Weight lambda for the stiffness constraint
    initial_transforms=None   # Optional: List of initial [rx, ry, rz, tx, ty, tz] for each member
):
    """
    Performs non-rigid registration based on topology constraints.

    Args:
        source_points_by_member (list): List of numpy arrays (N_k, 3) for each member.
        target_points (np.ndarray): Target point cloud (M, 3).
        member_topology (dict): Topology information.
        node_coordinates (dict): Coordinates of shared nodes in the template.
        correspondences (list): List of (src_idx, tgt_idx) tuples.
        stiffness_weight (float): Weight for the stiffness term.
        initial_transforms (list, optional): Initial guess for transforms [rx, ry, rz, tx, ty, tz].

    Returns:
        list: Optimized transforms [rx, ry, rz, tx, ty, tz] for each member.
    """
    n_members = len(source_points_by_member)
    if initial_transforms is None:
        initial_transforms = [np.zeros(6) for _ in range(n_members)]

    # Build the sparse system matrix A and vector b
    # Variables are stacked as [T_0, T_1, ..., T_n-1], where T_k = [rx, ry, rz, tx, ty, tz]
    n_vars = n_members * 6
    A = lil_matrix((n_vars, n_vars)) # Use lil_matrix for efficient construction
    b = np.zeros(n_vars)

    # --- 1. Build Data Term Constraints ---
    # For each correspondence (P_i^k, Q_j), we want P_i'^k to be close to Q_j.
    # This creates a residual: r = ||P_i'^k - Q_j||
    # Linearizing the transformation leads to linear terms in A and b.
    # This is the most complex part, requiring careful indexing.
    # We'll simplify by assuming a function that handles one residual.

    def add_data_term(A, b, src_idx_global, tgt_idx, n_members, weight=1.0):
        # Find which member k this source point belongs to
        cumul_sizes = np.cumsum([0] + [len(pts) for pts in source_points_by_member])
        k = np.searchsorted(cumul_sizes[1:], src_idx_global, side='right')
        local_idx = src_idx_global - cumul_sizes[k]

        # Get the source point P_i^k
        P_ik = source_points_by_member[k][local_idx]
        Q_j = target_points[tgt_idx]

        # Derivatives of the transformation w.r.t. T_k = [theta, t]
        # For small angles: P_i'^k = (I + [S(theta)]) * P_ik + t
        # dP_i'^k / dtheta_x = [0, -P_ik.z, P_ik.y]^T (cross product P_ik x [1,0,0])
        # dP_i'^k / dtheta_y = [P_ik.z, 0, -P_ik.x]^T (P_ik x [0,1,0])
        # dP_i'^k / dtheta_z = [-P_ik.y, P_ik.x, 0]^T (P_ik x [0,0,1])
        # dP_i'^k / dt = I (identity matrix 3x3)
        
        # Jacobian matrix for this residual w.r.t. T_k (size 3 x 6)
        J_k = np.zeros((3, 6))
        J_k[:3, :3] = np.array([
            [0, -P_ik[2], P_ik[1]],
            [P_ik[2], 0, -P_ik[0]],
            [-P_ik[1], P_ik[0], 0]
        ])
        J_k[:3, 3:] = np.eye(3) # Translation derivatives

        # Residual vector
        residual = Q_j - P_ik # This is an approximation for the residual (Q_j - P_i'^k) at initial T_k=0

        # Add these 3 equations (for x, y, z) to A and b
        var_offset_k = k * 6
        for dim in range(3):
            A[dim + tgt_idx * 3, var_offset_k:var_offset_k+6] = weight * J_k[dim, :]
            b[dim + tgt_idx * 3] = weight * residual[dim] # Note: This is a simplification

    # Apply data term constraints
    for src_idx, tgt_idx in correspondences:
        add_data_term(A, b, src_idx, tgt_idx, n_members, weight=np.sqrt(stiffness_weight))


    # --- 2. Build Stiffness (Topology) Constraints ---
    # For each pair of connected members (k, l) sharing a node P_node^kl,
    # we enforce (R_k * P_node^kl + t_k) == (R_l * P_node^kl + t_l)
    # This becomes (R_k * P_node^kl + t_k) - (R_l * P_node^kl + t_l) = 0
    # Linearizing gives a constraint equation.
    
    # We need to iterate through all connected member pairs
    # This requires parsing member_topology and node_coordinates
    # Let's assume we have a list of shared nodes between pairs
    # In practice, you'd build this list from member_topology and node_coordinates
    # Example: connected_pairs = [(k, l, node_id), ...]
    # For simplicity, let's create a dummy list or derive it if possible
    # A more robust way would be to find common keys in member_topology values
    # and map them back to member IDs.
    # For now, let's assume a helper function builds this.
    connected_pairs = find_connected_member_pairs(member_topology, node_coordinates)
    
    row_offset = len(correspondences) * 3 # Start after data term equations
    for k, l, node_id in connected_pairs:
        P_node_k = node_coordinates[node_id]
        P_node_l = node_coordinates[node_id] # Should be the same point in template
        if not np.allclose(P_node_k, P_node_l):
             print(f"Warning: Node {node_id} coordinates differ between connected members {k} and {l}")

        # Derivatives for member k: d/dT_k (R_k * P_node + t_k)
        # J_k_stiff = [[P_node x [1,0,0]], [P_node x [0,1,0]], [P_node x [0,0,1]], I_3x3]
        J_k_stiff = np.zeros((3, 6))
        J_k_stiff[:3, :3] = np.array([
            [0, -P_node_k[2], P_node_k[1]],
            [P_node_k[2], 0, -P_node_k[0]],
            [-P_node_k[1], P_node_k[0], 0]
        ])
        J_k_stiff[:3, 3:] = np.eye(3)

        # Derivatives for member l: d/dT_l (-(R_l * P_node + t_l))
        # J_l_stiff = -[[P_node x [1,0,0]], [P_node x [0,1,0]], [P_node x [0,0,1]], I_3x3]
        J_l_stiff = -J_k_stiff

        # Residual for this constraint (should be 0)
        residual_stiff = np.zeros(3) # (R_k * P_node + t_k) - (R_l * P_node + t_l) = 0 initially

        # Add these 3 equations to A and b
        var_offset_k = k * 6
        var_offset_l = l * 6
        for dim in range(3):
            A[row_offset + dim, var_offset_k:var_offset_k+6] = stiffness_weight * J_k_stiff[dim, :]
            A[row_offset + dim, var_offset_l:var_offset_l+6] = stiffness_weight * J_l_stiff[dim, :]
            b[row_offset + dim] = stiffness_weight * residual_stiff[dim]

        row_offset += 3


    # --- 3. Solve the linear system ---
    # Convert A to CSR format for efficient solving
    A_csr = A.tocsr()

    print(f"Solving sparse system: {A_csr.shape[0]} equations, {n_vars} variables.")
    try:
        solution = spsolve(A_csr, b)
    except Exception as e:
        print(f"Error solving system: {e}")
        return initial_transforms

    # --- 4. Extract optimized transforms ---
    optimized_transforms = []
    for i in range(n_members):
        transform = solution[i * 6:(i + 1) * 6]
        optimized_transforms.append(transform)

    print("Non-rigid registration optimization completed.")
    return optimized_transforms

def find_connected_member_pairs(topology_dict, node_coords_dict):
    """
    Helper function to find pairs of members sharing nodes.
    This is a placeholder. You need to implement the logic based on your data structure.
    """
    # Example logic (assuming topology_dict maps member -> [node_ids]):
    # Create a reverse map: node_id -> [member_ids]
    node_to_members = {}
    for mem_id, node_list in topology_dict.items():
        for node_id in node_list:
            if node_id not in node_to_members:
                node_to_members[node_id] = []
            node_to_members[node_id].append(mem_id)

    # Find pairs of members connected via a node
    pairs = []
    for node_id, members in node_to_members.items():
        if len(members) > 1:
            # Add all combinations of pairs for this node
            for i in range(len(members)):
                 for j in range(i + 1, len(members)):
                     pairs.append((members[i], members[j], node_id))
    return pairs

# --- Example Usage (Conceptual) ---
# This part is illustrative and requires actual data
# source_points_by_member = [member_0_points, member_1_points, ...]
# target_points = measured_point_cloud
# member_topology = {0: ['node_A'], 1: ['node_A']} # Member 0 and 1 share node_A
# node_coordinates = {'node_A': [0, 0, 0], ...}
# correspondences = [(global_idx_p0_in_mem0, idx_q0_in_tgt), ...]
# transforms = build_and_solve_nonrigid_registration(
#     source_points_by_member, target_points, member_topology, node_coordinates, correspondences
# )