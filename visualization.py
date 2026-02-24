import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(point_cloud, membership=None, title="Point Cloud Visualization"):
    """
    Visualizes 3D point cloud with optional membership coloring.
    
    Args:
        point_cloud (numpy.ndarray): 3D point cloud array of shape (N, 3).
        membership (numpy.ndarray, optional): Array of shape (N,) containing member IDs for each point.
                                            Unassigned points should have ID -1.
        title (str): Title for the plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if membership is None:
        # Visualize all points with the same color (pre-segmentation)
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=10, c='gray', alpha=0.7)
        ax.set_title(title)
    else:
        # Get unique member IDs
        unique_members = np.unique(membership)
        
        # Generate colormap for members
        if -1 in unique_members:
            # Exclude -1 from color mapping
            colored_members = unique_members[unique_members != -1]
            n_colors = len(colored_members)
            cmap = plt.cm.get_cmap('viridis', n_colors)
            
            # Plot each member with different color
            for i, member_id in enumerate(colored_members):
                mask = membership == member_id
                ax.scatter(point_cloud[mask, 0], point_cloud[mask, 1], point_cloud[mask, 2], 
                          s=10, c=[cmap(i)], label=f'Member {member_id}', alpha=0.7)
            
            # Plot unassigned points (-1) with different color
            unassigned_mask = membership == -1
            if np.any(unassigned_mask):
                ax.scatter(point_cloud[unassigned_mask, 0], point_cloud[unassigned_mask, 1], 
                          point_cloud[unassigned_mask, 2], s=10, c='red', label='Unassigned', alpha=0.7)
        else:
            # All points are assigned, no unassigned points
            n_colors = len(unique_members)
            cmap = plt.cm.get_cmap('viridis', n_colors)
            
            for i, member_id in enumerate(unique_members):
                mask = membership == member_id
                ax.scatter(point_cloud[mask, 0], point_cloud[mask, 1], point_cloud[mask, 2], 
                          s=10, c=[cmap(i)], label=f'Member {member_id}', alpha=0.7)
        
        ax.set_title(title)
        ax.legend()
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

