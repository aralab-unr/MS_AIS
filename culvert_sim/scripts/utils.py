import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial.transform import Rotation as R
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import math
#belief update only points in fov

def rotation_matrix(roll, pitch, yaw):
    # Rotation matrix around the x-axis (roll)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    # Rotation matrix around the y-axis (pitch)
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    # Rotation matrix around the z-axis (yaw)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix (yaw * pitch * roll)
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    return R

# Apply the transformation to a group of 3D points
def transform_points(points, roll, pitch, yaw):
    R = rotation_matrix(roll, pitch, yaw)
    transformed_points = np.dot(points, R.T)  # Apply rotation matrix to each point
    return transformed_points

def generate_uniform_grid(w, l, h, offset_x=0.0, offset_y=0.0, offset_z=0.0, scale=10):
    # Generate uniform grid coordinates on left and right sides
    # num_points_x = max(round(w*scale),10)
    # num_points_y = max(round(l*scale),10)
    # num_points_z = max(round(h*scale),10)

    # y_left, z_left = np.meshgrid(np.linspace(0, l, num_points_y),
    #                              np.linspace(0, h, num_points_z))
    # y_right, z_right = np.meshgrid(np.linspace(0, l, num_points_y),
    #                                np.linspace(0, h, num_points_z))

    # # Generate uniform grid coordinates on top face
    # x_top, y_top = np.meshgrid(np.linspace(0, w, num_points_x),
    #                            np.linspace(0, l, round(l*scale)))

    # left_points = np.column_stack((np.zeros(num_points_y * num_points_z), y_left.flatten(), z_left.flatten()))
    # right_points = np.column_stack((np.ones(num_points_y * num_points_z) * w, y_right.flatten(), z_right.flatten()))
    # top_points = np.column_stack((x_top.flatten(), y_top.flatten(), np.ones(num_points_x * num_points_y) * h))

    # # Combine all points into a single array
    # all_points = np.vstack((left_points, right_points, top_points))

    # all_points[:,0] += offset_x
    # all_points[:,1] += offset_y
    # all_points[:,2] += offset_z
    # # Remove duplicate points
    # unique_points, unique_indices = np.unique(all_points, axis=0, return_index=True)
    # # print(all_points.shape, unique_points.shape)

    # # Extract unique points for each face
    # # num_left = num_points_y * num_points_z
    # # num_right = num_points_y * num_points_z
    # # num_top = num_points_x * num_points_y

    # # unique_left_points = unique_points[:num_left]
    # # unique_right_points = unique_points[num_left:num_left + num_right]
    # # unique_top_points = unique_points[num_left + num_right:num_left + num_right + num_top]

    # # return unique_left_points, unique_right_points, unique_top_points
    # return unique_points

    # Generate uniform grid coordinates for left and right sides (XZ planes)

    # num_points_x = max(round(l * scale), 3)
    # num_points_z = max(round(h * scale), 3)
    # num_points_y = max(round(w * scale), 3)

    # x_left, z_left = np.meshgrid(np.linspace(0, l, num_points_x),
    #                              np.linspace(0, h, num_points_z))
    # x_right, z_right = np.meshgrid(np.linspace(0, l, num_points_x),
    #                                np.linspace(0, h, num_points_z))

    # # Generate uniform grid coordinates on the top face (XY plane)
    # x_top, y_top = np.meshgrid(np.linspace(0, l, num_points_x),
    #                            np.linspace(0, w, num_points_y))

    # left_points = np.column_stack((x_left.flatten(), np.zeros(num_points_x * num_points_z), z_left.flatten()))
    # right_points = np.column_stack((x_right.flatten(), np.ones(num_points_x * num_points_z) * w, z_right.flatten()))
    # top_points = np.column_stack((x_top.flatten(), y_top.flatten(), np.ones(num_points_x * num_points_y) * h))
    
    # # Combine all points into a single array
    # all_points = np.vstack((left_points, right_points, top_points))

    # # Apply offsets
    # all_points[:, 0] += offset_x
    # all_points[:, 1] += offset_y
    # all_points[:, 2] += offset_z

    # # Remove duplicate points
    # unique_points, unique_indices = np.unique(all_points, axis=0, return_index=True)

    # return unique_points

    num_points_x = max(round(l * scale), 3)
    num_points_z = max(round(h * scale), 3)
    num_points_y = max(round(w * scale), 3)

    # Generate meshgrid for each plane
    x_left, z_left = np.meshgrid(np.linspace(0, l, num_points_x), np.linspace(0, h, num_points_z))
    x_right, z_right = np.meshgrid(np.linspace(0, l, num_points_x), np.linspace(0, h, num_points_z))
    x_top, y_top = np.meshgrid(np.linspace(0, l, num_points_x), np.linspace(0, w, num_points_y))

    # Generate 3D points for each plane
    left_points = np.column_stack((x_left.flatten(), np.zeros(num_points_x * num_points_z), z_left.flatten()))
    right_points = np.column_stack((x_right.flatten(), np.ones(num_points_x * num_points_z) * w, z_right.flatten()))
    top_points = np.column_stack((x_top.flatten(), y_top.flatten(), np.ones(num_points_x * num_points_y) * h))

    # Calculate centers for left_points on the XZ plane
    center_x_left = (x_left[:-1, :-1] + x_left[1:, :-1] + x_left[:-1, 1:] + x_left[1:, 1:]) / 4
    center_z_left = (z_left[:-1, :-1] + z_left[1:, :-1] + z_left[:-1, 1:] + z_left[1:, 1:]) / 4
    center_y_left = np.zeros_like(center_x_left)  # Y remains 0 for the left plane
    center_points_left = np.column_stack((center_x_left.flatten(), center_y_left.flatten(), center_z_left.flatten()))

    # Calculate centers for right_points on the XZ plane at Y = w
    center_x_right = (x_right[:-1, :-1] + x_right[1:, :-1] + x_right[:-1, 1:] + x_right[1:, 1:]) / 4
    center_z_right = (z_right[:-1, :-1] + z_right[1:, :-1] + z_right[:-1, 1:] + z_right[1:, 1:]) / 4
    center_y_right = np.ones_like(center_x_right) * w  # Y = w for the right plane
    center_points_right = np.column_stack((center_x_right.flatten(), center_y_right.flatten(), center_z_right.flatten()))

    # Calculate centers for top_points on the XY plane at Z = h
    center_x_top = (x_top[:-1, :-1] + x_top[1:, :-1] + x_top[:-1, 1:] + x_top[1:, 1:]) / 4
    center_y_top = (y_top[:-1, :-1] + y_top[1:, :-1] + y_top[:-1, 1:] + y_top[1:, 1:]) / 4
    center_z_top = np.ones_like(center_x_top) * h  # Z = h for the top plane
    center_points_top = np.column_stack((center_x_top.flatten(), center_y_top.flatten(), center_z_top.flatten()))

    # # Combine all points into a single array
    # all_points = np.vstack((left_points, right_points, top_points))
    # unique_points, unique_indices = np.unique(all_points, axis=0, return_index=True)

    # Combine all CENTER points into a single array
    all_points = np.vstack((center_points_left, center_points_right, center_points_top))

    rotation_top = R.from_euler('y', 90, degrees=True)
    quaternion_top = rotation_top.as_quat()

    # Left points pointing right
    rotation_left = R.from_euler('z', 90, degrees=True)
    quaternion_left = rotation_left.as_quat()

    # Right points pointing left
    rotation_right = R.from_euler('z', -90, degrees=True)
    quaternion_right = rotation_right.as_quat()
    all_orientations = np.vstack((
        np.tile(quaternion_left, (center_points_left.shape[0], 1)),
        np.tile(quaternion_right, (center_points_right.shape[0], 1)),
        np.tile(quaternion_top, (center_points_top.shape[0], 1))
    ))

    all_pose = np.hstack((all_points, all_orientations))
    # Apply offsets
    all_pose[:, 0] += offset_x
    all_pose[:, 1] += offset_y
    all_pose[:, 2] += offset_z

    return all_pose

def generate_uniform_grid_(w, l, h, offset_x=0.0, offset_y=0.0, offset_z=0.0, scale=10, l_scale=5):
    # Generate uniform grid coordinates on left and right sides
    # num_points_x = max(round(w*scale),10)
    # num_points_y = max(round(l*scale),10)
    # num_points_z = max(round(h*scale),10)

    # y_left, z_left = np.meshgrid(np.linspace(0, l, num_points_y),
    #                              np.linspace(0, h, num_points_z))
    # y_right, z_right = np.meshgrid(np.linspace(0, l, num_points_y),
    #                                np.linspace(0, h, num_points_z))

    # # Generate uniform grid coordinates on top face
    # x_top, y_top = np.meshgrid(np.linspace(0, w, num_points_x),
    #                            np.linspace(0, l, round(l*scale)))

    # left_points = np.column_stack((np.zeros(num_points_y * num_points_z), y_left.flatten(), z_left.flatten()))
    # right_points = np.column_stack((np.ones(num_points_y * num_points_z) * w, y_right.flatten(), z_right.flatten()))
    # top_points = np.column_stack((x_top.flatten(), y_top.flatten(), np.ones(num_points_x * num_points_y) * h))

    # # Combine all points into a single array
    # all_points = np.vstack((left_points, right_points, top_points))

    # all_points[:,0] += offset_x
    # all_points[:,1] += offset_y
    # all_points[:,2] += offset_z
    # # Remove duplicate points
    # unique_points, unique_indices = np.unique(all_points, axis=0, return_index=True)
    # # print(all_points.shape, unique_points.shape)

    # # Extract unique points for each face
    # # num_left = num_points_y * num_points_z
    # # num_right = num_points_y * num_points_z
    # # num_top = num_points_x * num_points_y

    # # unique_left_points = unique_points[:num_left]
    # # unique_right_points = unique_points[num_left:num_left + num_right]
    # # unique_top_points = unique_points[num_left + num_right:num_left + num_right + num_top]

    # # return unique_left_points, unique_right_points, unique_top_points
    # return unique_points

    # Generate uniform grid coordinates for left and right sides (XZ planes)

    # num_points_x = max(round(l * scale), 3)
    # num_points_z = max(round(h * scale), 3)
    # num_points_y = max(round(w * scale), 3)

    # x_left, z_left = np.meshgrid(np.linspace(0, l, num_points_x),
    #                              np.linspace(0, h, num_points_z))
    # x_right, z_right = np.meshgrid(np.linspace(0, l, num_points_x),
    #                                np.linspace(0, h, num_points_z))

    # # Generate uniform grid coordinates on the top face (XY plane)
    # x_top, y_top = np.meshgrid(np.linspace(0, l, num_points_x),
    #                            np.linspace(0, w, num_points_y))

    # left_points = np.column_stack((x_left.flatten(), np.zeros(num_points_x * num_points_z), z_left.flatten()))
    # right_points = np.column_stack((x_right.flatten(), np.ones(num_points_x * num_points_z) * w, z_right.flatten()))
    # top_points = np.column_stack((x_top.flatten(), y_top.flatten(), np.ones(num_points_x * num_points_y) * h))
    
    # # Combine all points into a single array
    # all_points = np.vstack((left_points, right_points, top_points))

    # # Apply offsets
    # all_points[:, 0] += offset_x
    # all_points[:, 1] += offset_y
    # all_points[:, 2] += offset_z

    # # Remove duplicate points
    # unique_points, unique_indices = np.unique(all_points, axis=0, return_index=True)

    # return unique_points

    num_points_x = max(round(l * l_scale), 3)
    num_points_z = max(round(h * scale), 3)
    num_points_y = max(round(w * scale), 3)

    # Generate meshgrid for each plane
    x_left, z_left = np.meshgrid(np.linspace(0, l, num_points_x), np.linspace(0, h, num_points_z))
    x_right, z_right = np.meshgrid(np.linspace(0, l, num_points_x), np.linspace(0, h, num_points_z))
    x_top, y_top = np.meshgrid(np.linspace(0, l, num_points_x), np.linspace(0, w, num_points_y))

    # Generate 3D points for each plane
    left_points = np.column_stack((x_left.flatten(), np.zeros(num_points_x * num_points_z), z_left.flatten()))
    right_points = np.column_stack((x_right.flatten(), np.ones(num_points_x * num_points_z) * w, z_right.flatten()))
    top_points = np.column_stack((x_top.flatten(), y_top.flatten(), np.ones(num_points_x * num_points_y) * h))

    # Calculate centers for left_points on the XZ plane
    center_x_left = (x_left[:-1, :-1] + x_left[1:, :-1] + x_left[:-1, 1:] + x_left[1:, 1:]) / 4
    center_z_left = (z_left[:-1, :-1] + z_left[1:, :-1] + z_left[:-1, 1:] + z_left[1:, 1:]) / 4
    center_y_left = np.zeros_like(center_x_left)  # Y remains 0 for the left plane
    center_points_left = np.column_stack((center_x_left.flatten(), center_y_left.flatten(), center_z_left.flatten()))

    # Calculate centers for right_points on the XZ plane at Y = w
    center_x_right = (x_right[:-1, :-1] + x_right[1:, :-1] + x_right[:-1, 1:] + x_right[1:, 1:]) / 4
    center_z_right = (z_right[:-1, :-1] + z_right[1:, :-1] + z_right[:-1, 1:] + z_right[1:, 1:]) / 4
    center_y_right = np.ones_like(center_x_right) * w  # Y = w for the right plane
    center_points_right = np.column_stack((center_x_right.flatten(), center_y_right.flatten(), center_z_right.flatten()))

    # Calculate centers for top_points on the XY plane at Z = h
    center_x_top = (x_top[:-1, :-1] + x_top[1:, :-1] + x_top[:-1, 1:] + x_top[1:, 1:]) / 4
    center_y_top = (y_top[:-1, :-1] + y_top[1:, :-1] + y_top[:-1, 1:] + y_top[1:, 1:]) / 4
    center_z_top = np.ones_like(center_x_top) * h  # Z = h for the top plane
    center_points_top = np.column_stack((center_x_top.flatten(), center_y_top.flatten(), center_z_top.flatten()))

    # # Combine all points into a single array
    # all_points = np.vstack((left_points, right_points, top_points))
    # unique_points, unique_indices = np.unique(all_points, axis=0, return_index=True)

    # Combine all CENTER points into a single array
    all_points = np.vstack((center_points_left, center_points_right))#, center_points_top))

    rotation_top = R.from_euler('y', 90, degrees=True)
    quaternion_top = rotation_top.as_quat()

    # Left points pointing right
    rotation_left = R.from_euler('z', 90, degrees=True)
    quaternion_left = rotation_left.as_quat()

    # Right points pointing left
    rotation_right = R.from_euler('z', -90, degrees=True)
    quaternion_right = rotation_right.as_quat()
    all_orientations = np.vstack((
        np.tile(quaternion_left, (center_points_left.shape[0], 1)),
        np.tile(quaternion_right, (center_points_right.shape[0], 1)),
        #np.tile(quaternion_top, (center_points_top.shape[0], 1))
    ))

    all_pose = np.hstack((all_points, all_orientations))
    # Apply offsets
    all_pose[:, 0] += offset_x
    all_pose[:, 1] += offset_y
    all_pose[:, 2] += offset_z

    return all_pose


def generate_robot_state(w, l, offset_x=0.0, offset_y=0.0, offset_z=0.0, scale_x=10, scale_y=10):
    num_points_x = max(round(l * scale_x), 2)
    num_points_y = max(round(w * scale_y), 2)
    x_top, y_top = np.meshgrid(np.linspace(0, l, num_points_x), np.linspace(0, w, num_points_y))
    center_x_top = (x_top[:-1, :-1] + x_top[1:, :-1] + x_top[:-1, 1:] + x_top[1:, 1:]) / 4
    center_y_top = (y_top[:-1, :-1] + y_top[1:, :-1] + y_top[:-1, 1:] + y_top[1:, 1:]) / 4
    center_points_top = np.column_stack((center_x_top.flatten(), center_y_top.flatten()))
    
    zeros_column = np.zeros(center_points_top.shape[0])
    center_points_top_with_zeros = np.column_stack((center_points_top, zeros_column))
    # tf offset
    center_points_top_with_zeros[:, 0] += offset_x
    center_points_top_with_zeros[:, 1] += offset_y
    center_points_top_with_zeros[:, 1] += offset_z

    y_values = center_points_top_with_zeros[:, 1]  # Extract the y-values (2nd column)
    y_avg = np.mean(y_values)  # Compute the average of the y-values    
    # Step 2: Calculate the absolute difference between each y and the average y
    y_diff = np.abs(y_values - y_avg)    
    # Step 3: Find the minimum difference (i.e., the closest y-values to the average)
    min_diff = np.min(y_diff)    
    # Step 4: Find all points where the y-value is closest to the average y
    center_line = center_points_top_with_zeros[y_diff == min_diff]

    return center_points_top_with_zeros, center_line

def calculate_nearest_distances(center_points):
    distances = cdist(center_points, center_points)  # Compute all pairwise distances
    np.fill_diagonal(distances, np.inf)  # Ignore self-distance by setting the diagonal to infinity
    nearest_distances = distances.min(axis=1)  # Find the minimum distance for each point
    return nearest_distances

def frostum_vertices(near, far, h_fov, v_fov, plot=False):
    """
    Calculate vertices of a frustum pyramid in 3D space based on given parameters.

    Parameters:
    near (float): Distance to the near plane of the frustum.
    far (float): Distance to the far plane of the frustum.
    h_fov (float): Horizontal field of view (in degrees).
    v_fov (float): Vertical field of view (in degrees).
    plot (bool, optional): Whether to plot the vertices in a 3D space. Default is False.

    Returns:
    ndarray: Array of shape (8, 3) containing the vertices of the frustum pyramid in local coordinates.
    """
    # convert from degree to rad
    h_fov_rad = np.radians(h_fov)
    v_fov_rad = np.radians(v_fov)
    
    # Calculate top and bottom near plane dimensions
    x_near = near 
    y_near = near *np.tan(h_fov_rad/2)
    z_near = near * np.tan(v_fov_rad / 2)

    x_far = far 
    y_far = far *np.tan(h_fov_rad/2)
    z_far = far * np.tan(v_fov_rad / 2)
    
    # Define the vertices of the frustum pyramid in local coordinates
    vertices = np.array([
        [x_near,  y_near, -z_near],     # Bottom left corner of near plane
        [x_near, -y_near, -z_near],     # Bottom right corner of near plane
        [x_near, -y_near,  z_near],     # Top right corner of near plane
        [x_near,  y_near,  z_near],     # Top left corner of near plane
        [x_far,  y_far, -z_far],        # Bottom left corner of far plane
        [x_far, -y_far, -z_far],        # Bottom right corner of far plane
        [x_far, -y_far,  z_far],        # Top right corner of far plane
        [x_far,  y_far,  z_far],        # Top left corner of far plane
    ])

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot vertices
        ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='red', s=100)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Plot of Vertices')
        
        plt.show()
    return vertices

def update_points_in_frustum_scipy(points, frustum_vertices):    
    """
    Update the list of points by setting the boolean flag to True if the point is inside the frustum
    using scipy.spatial ConvexHull and Delaunay.
    """
    # Create a convex hull from the frustum vertices
    hull = ConvexHull(frustum_vertices)    
    # Use Delaunay to check if points are inside the convex hull
    delaunay = Delaunay(frustum_vertices)
    # Extract the coordinates from the points array (ignore the boolean column)
    coords = points[:, :3]
    # Use delaunay.find_simplex to check if the points are inside the hull
    inside = delaunay.find_simplex(coords) >= 0
    # Update the boolean flag for points inside the frustum
    points[:, 3] = np.logical_or(points[:, 3], inside)    
    # return points

def point_in_frustum(frustums, point):
    for frustum_vertices in frustums:
        # Create a Delaunay triangulation for the frustum
        delaunay = Delaunay(frustum_vertices)
        
        # Check if the point is inside the convex hull of this frustum
        if delaunay.find_simplex(point) >= 0:
            return True  # Point is inside this frustum    
    return False  # Point is not inside any frustum


def partition_points_by_frustums(frustums, points):
    # Initialize a boolean array to track if each point is inside any frustum
    inside_any_frustum = np.zeros(points.shape[0], dtype=bool)
    
    # Iterate through each frustum
    for frustum_vertices in frustums:
        # Create a convex hull for the frustum
        hull = ConvexHull(frustum_vertices)
        # Create a Delaunay triangulation for efficient point-in-hull tests
        delaunay = Delaunay(frustum_vertices[hull.vertices])
        
        # Check which points are inside the convex hull of this frustum
        inside_frustum = delaunay.find_simplex(points) >= 0
        
        # Update the boolean array if points are inside this frustum
        inside_any_frustum |= inside_frustum
    
    # Separate the points into two arrays based on whether they are inside any frustum
    points_inside = points[inside_any_frustum]
    points_outside = points[~inside_any_frustum]

    return points_inside, points_outside

def all_real(*args):
        """Check if all arguments are real numbers."""
        return all(isRealNumber(arg) for arg in args)

def isRealNumber(number):
        return not np.isnan(number) and not np.isinf(number)

def isDefect(loc, img_w, img_h, varrho=0.01, rho=0.01):
    [x, y, w, h] = loc
    top_left_x = x - w / 2
    top_left_y = y - h / 2
    bottom_right_x = x + w / 2
    bottom_right_y = y + h / 2
    varrho_w = img_w * varrho
    rho_h = img_h * rho
    if top_left_x < varrho_w or top_left_x > img_w - varrho_w or top_left_y < rho_h or top_left_y > img_h - rho_h:
        return False
    return True

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def combine_and_points(arr1, arr2, arr3, threshold=0.2):
    # # Combine the arrays
    # all_points = np.vstack((arr1, arr2, arr3))
    # # Compute the pairwise distances between all points
    # dist_matrix = squareform(pdist(all_points))
    # # Create a boolean mask for distances below the threshold
    # close_points = dist_matrix < threshold
    # # Create a sparse graph where an edge exists between points within the threshold
    # graph = csr_matrix(close_points)
    # # Find connected components (clusters of points within threshold)
    # n_components, labels = connected_components(csgraph=graph, directed=False)
    # # Average points in each cluster to get the new points
    # averaged_points = np.array([all_points[labels == i].mean(axis=0) for i in range(n_components)])
    
    all_points = np.vstack((arr1, arr2, arr3))
    # Extract the x, y, z columns for distance calculations (ignore probability for now)
    xyz_points = all_points[:, :3]
    # Compute the pairwise distances between the x, y, z coordinates
    dist_matrix = squareform(pdist(xyz_points))
    # Create a boolean mask for distances below the threshold
    close_points = dist_matrix < threshold
    # Create a sparse graph where an edge exists between points within the threshold
    graph = csr_matrix(close_points)
    # Find connected components (clusters of points within the threshold)
    n_components, labels = connected_components(csgraph=graph, directed=False)
    # Average points in each cluster (x, y, z and probabilities)
    averaged_points = np.array([all_points[labels == i].mean(axis=0) for i in range(n_components)])

    return averaged_points

def merge_obs(arrays, threshold=0.2):
    """
    Merge points using a vectorized approach without explicit for loops.
    Points within the threshold and having the same class are averaged.
    
    Parameters:
    arrays (list of np.array): List of arrays where each array contains [x, y, z, class, prob]
    threshold (float): Distance threshold for merging points
    
    Returns:
    np.array: Merged array with averaged points
    """

    arrays = [arr for arr in arrays if arr.size > 0]
    
    # If all arrays are empty, return an empty array
    if len(arrays) == 0:
        return np.array([])

    # Merge all arrays into one
    all_points = np.vstack(arrays)    
    # Calculate pairwise distances between all points (ignoring class for now)
    dists = squareform(pdist(all_points[:, :3]))    
    # Create a class mask to only compare points with the same class
    class_mask = (all_points[:, 3][:, None] == all_points[:, 3])    
    # Create adjacency matrix where distances <= threshold and same class
    adjacency_matrix = (dists <= threshold) & class_mask    
    #  Create a sparse CSR matrix from the adjacency matrix
    csr = csr_matrix(adjacency_matrix)    
    # Find connected components
    n_components, labels = connected_components(csgraph=csr, directed=False)    
    #  Merge points in the same connected component by averaging their x, y, z, and prob
    merged_points = np.zeros((n_components, 5))
    
    for i in range(n_components):
        component_points = all_points[labels == i]
        merged_points[i, :3] = np.mean(component_points[:, :3], axis=0)  # Average x, y, z
        merged_points[i, 3] = component_points[0, 3]  # Keep the class (they are all the same)
        merged_points[i, 4] = np.mean(component_points[:, 4])  # Average prob
    
    return merged_points
    
def state(pose, robot_states):
    # Step 1: Check if any element in pose is None
    if None in pose:
        raise ValueError("Error: Pose contains None elements.")    
    # Step 2: Extract the first 3 elements (x, y, z)
    pose_3d = pose[:3]
    # Step 3: Calculate the Euclidean distance between pose_3d and each point in the array
    distances = np.linalg.norm(robot_states - pose_3d, axis=1)
    # Step 4: Find the index of the closest point
    closest_point_idx = np.argmin(distances)
    # Step 5: Return the closest point
    return robot_states[closest_point_idx]

def turning(current_theta, angleToGoal,angular_speed = 0.17):
    cmd_angular_z = 0  # This will hold the angular speed to return

    if angleToGoal >= 0 and current_theta >= 0:  # positive/on the right
        if angleToGoal > current_theta:
            cmd_angular_z = angular_speed
        else:
            cmd_angular_z = -angular_speed

    elif angleToGoal <= 0 and current_theta <= 0:  # negative/on the left
        if angleToGoal > current_theta:
            cmd_angular_z = angular_speed
        else:
            cmd_angular_z = -angular_speed

    else:  # one is positive and one is negative
        if angleToGoal < 0:  # angleToGoal is negative && current is positive
            bot = (angleToGoal - math.pi) + (math.pi - current_theta)
            top = (0 - angleToGoal) + current_theta
            if top > bot:
                cmd_angular_z = -angular_speed
            else:
                cmd_angular_z = angular_speed
        else:  # angleToGoal is + && current is -
            bot = (current_theta - math.pi) + (math.pi - angleToGoal)
            top = (0 - current_theta) + angleToGoal
            if top > bot:
                cmd_angular_z = angular_speed
            else:
                cmd_angular_z = -angular_speed

    return cmd_angular_z

def compute_angle_distance(point, current_pose):
    """
    Computes the angle and distance between two 3D points (ignoring z-axis).
    
    Parameters:
    - point: 1D NumPy array representing [x, y, z] of the target point.
    - current_pose: 1D NumPy array representing [x, y, z] of the current position.
    
    Returns:
    - angleToGoal: Angle (in radians) between the two points, projected onto the x-y plane.
    - distance: Euclidean distance between the two points (ignoring z-axis).
    """
    x_ = point[0] - current_pose[0]
    y_ = point[1] - current_pose[1]    
    angleToGoal = math.atan2(y_, x_)
    distance = math.sqrt(x_ ** 2 + y_ ** 2)    
    return angleToGoal, distance

def shortest_angular_difference(angle1, angle2):
    """
    Calculate the shortest angular difference between two angles in degrees.
    The result is adjusted to be within the range [-180, 180] degrees.
    """
    # Difference between the two angles
    delta_angle = angle1 - angle2
    # Normalize the difference to be within -pi to pi
    error_rad = np.arctan2(np.sin(delta_angle), np.cos(delta_angle))
    return error_rad

def plot_frustum(vertices, point=None):
    """
    Plot the frustum pyramid and optionally a point in 3D space.

    Parameters:
    vertices (ndarray): Array of shape (8, 3) containing the vertices of the frustum pyramid.
    point (ndarray or None, optional): Coordinates of the point to plot. Default is None.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot frustum vertices
    ax.plot([vertices[0, 0], vertices[1, 0], vertices[2, 0], vertices[3, 0], vertices[0, 0]],
            [vertices[0, 1], vertices[1, 1], vertices[2, 1], vertices[3, 1], vertices[0, 1]],
            [vertices[0, 2], vertices[1, 2], vertices[2, 2], vertices[3, 2], vertices[0, 2]], 'b-')
    
    ax.plot([vertices[4, 0], vertices[5, 0], vertices[6, 0], vertices[7, 0], vertices[4, 0]],
            [vertices[4, 1], vertices[5, 1], vertices[6, 1], vertices[7, 1], vertices[4, 1]],
            [vertices[4, 2], vertices[5, 2], vertices[6, 2], vertices[7, 2], vertices[4, 2]], 'b-')
    
    for i in range(4):
        ax.plot([vertices[i, 0], vertices[i + 4, 0]],
                [vertices[i, 1], vertices[i + 4, 1]],
                [vertices[i, 2], vertices[i + 4, 2]], 'b-')
    
    # Plot the point if provided
    if point is not None:
        ax.scatter(point[0], point[1], point[2], color='red', s=10)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Frustum and Point')
    
    # Equal scaling for all axes
    ax.set_box_aspect([1,1,1])
    
    plt.show()

