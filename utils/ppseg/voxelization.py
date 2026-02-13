import numpy as np


def map_xyz_to_voxel(
    vertex_xyz_coords: np.array,
    grid_1d_coord: np.array,
    spacing: float,
    features: np.array = None,
    log_info: bool = False,
    pass_voxel_coord: bool = False,
):
    """Map the coordinates of the atoms to the voxel representation"""
    voxel_length = len(grid_1d_coord)
    if features is None:
        voxel = np.zeros(
            shape=(1, voxel_length, voxel_length, voxel_length), dtype=np.float64
        )
        features = np.ones((vertex_xyz_coords.shape[0], 1))
    else:
        voxel = np.zeros(
            shape=(features.shape[1], voxel_length, voxel_length, voxel_length),
            dtype=np.float64,
        )

    # preparation of grids
    grid_xyz_coords = np.array(
        [(x, y, z) for x in grid_1d_coord for y in grid_1d_coord for z in grid_1d_coord]
    )
    box_1d_size = spacing * voxel_length
    vert2grid_dist = np.ones((grid_xyz_coords.shape[0], 1)) * box_1d_size

    # check each vertex and map to grid
    vertex_to_same_grid = 0
    for idx in range(vertex_xyz_coords.shape[0]):
        vertex_xyz_coord = vertex_xyz_coords[idx]  # [x, y, z] of the vertex
        distances = np.linalg.norm(grid_xyz_coords - vertex_xyz_coord, axis=1)
        min_dist = np.min(distances)

        # matched grid index and the distance between vertex and grid
        match_grid_index = np.where(distances == min_dist)
        assert len(match_grid_index) == 1, "More than one grid point is matched!"
        match_vert2grid_dist = vert2grid_dist[match_grid_index][0][0]

        # [x, y, z] of the matched grid point
        x, y, z = grid_xyz_coords[match_grid_index].flatten()
        idx_x = np.where(grid_1d_coord == x)
        idx_y = np.where(grid_1d_coord == y)
        idx_z = np.where(grid_1d_coord == z)

        # check the distance and assign the data to the voxel
        if min_dist < match_vert2grid_dist:
            # assign data to voxel
            voxel[:, idx_x, idx_y, idx_z] = features[idx, :].reshape(
                features.shape[1], 1, 1
            )

            # print the information
            if vert2grid_dist[match_grid_index] == box_1d_size:
                pass
            else:
                if log_info:
                    print(
                        "A new grid point is assigned by a smaller distance:\n",
                        f"[{vertex_xyz_coord[0]:.3f}, {vertex_xyz_coord[1]:.3f}, "
                        f"{vertex_xyz_coord[2]:.3f}] -> [{x:.3f}, {y:.3f}, {z:.3f}]\n",
                        f"Original distance: w/ dist {match_vert2grid_dist:.3f}\n",
                        f"New distance: w/ dist: {min_dist:.3f}\n",
                    )
                vertex_to_same_grid += 1

            vert2grid_dist[match_grid_index] = min_dist
        else:
            if log_info:
                print(
                    "The grid point has already assigned by a closer vertex:\n",
                    f"[{vertex_xyz_coord[0]:.3f}, {vertex_xyz_coord[1]:.3f}, "
                    f"{vertex_xyz_coord[2]:.3f}] -> [{x:.3f}, {y:.3f}, {z:.3f}]\n",
                    f"New distance: w/ dist: {min_dist:.3f}\n",
                    f"Original distance: w/ dist "
                    f"{vert2grid_dist[match_grid_index][0][0]:.3f}\n",
                )
            vertex_to_same_grid += 1

    if log_info:
        print(f"Number of vertices assigned to the same grid: {vertex_to_same_grid}")
        cnt = sum(voxel[0, :, :, :].flatten() != 0)
        print(f"Number of existing points in the voxel: {cnt}")

    if pass_voxel_coord:
        return voxel, grid_xyz_coords, vert2grid_dist
    else:
        return voxel


def generate_grid_1d_coord(r: float, spacing: float):
    # generate grid 1D coordinates
    actual_grid_length = r * 2
    N = int(np.ceil(actual_grid_length / spacing))
    grid_1d_coord = np.linspace(0, actual_grid_length, N)
    grid_center_1d = np.mean(grid_1d_coord)
    grid_1d_coord -= grid_center_1d
    return grid_1d_coord


def site_voxelization(
    site,
    r: float,
    spacing: float,
    shape: bool,
    pass_voxel_coord: bool = False,
    log_info: bool = False,
):
    """Convert the binding site information to numpy array"""
    site = np.array(site, dtype=np.float64)
    coords = site[:, :3]

    # set grid xyz coordinates and set voxel box size
    grid_1d_coord = generate_grid_1d_coord(r, spacing)

    # set the features
    if not shape:
        if log_info:
            print("Including features in the voxel representation")
        features = site[:, 3:]
        assert features.shape[1] != 0, "The channel of features is 0!"

    else:
        print("Binary occupation only for voxel representation") if log_info else None
        features = None

    # map the coordinates to the voxel
    return map_xyz_to_voxel(
        vertex_xyz_coords=coords,
        grid_1d_coord=grid_1d_coord,
        spacing=spacing,
        features=features,
        pass_voxel_coord=pass_voxel_coord,
        log_info=log_info,
    )


def map_voxel_to_xyz(
    data: np.array,
    r: int,
    spacing: float,
    pocket_center: np.array = None,
    filter_dummy: bool = False,
    mask: np.array = None,
):
    """Data [np.array]: voxel data (C, X, Y, Z)
    r [int]: radius of the voxel
    spacing [float]: spacing between the voxels
    pocket_center [np.array]: pocket center [x, y, z]
    filter_dummy [bool]: filter the dummy voxels
    mask [np.array]: mask for the data
    """
    # initialize the grid 1D coordinates
    grid_1d_coord = generate_grid_1d_coord(r=r, spacing=spacing)
    grid_coord = []
    features = []

    # check the mask
    if mask is not None:
        assert mask.shape == data.shape[1:], "The mask shape is not matched!"

    for x in range(data.shape[1]):
        for y in range(data.shape[2]):
            for z in range(data.shape[3]):
                feature = data[:, x, y, z]

                # using mask to filter the voxels
                if mask is not None and not mask[x, y, z]:
                    continue

                # filter the dummy voxels by feature values
                if filter_dummy and np.all(feature == 0):
                    continue

                # save the grid coordinates and features
                grid_coord.append(
                    [grid_1d_coord[x], grid_1d_coord[y], grid_1d_coord[z]]
                )
                features.append(feature)

    # convert the list to numpy array
    features = np.array(features)
    grid_coord = np.array(grid_coord)
    if pocket_center is not None:
        grid_coord = grid_coord + pocket_center

    data_with_xyz = np.column_stack([grid_coord, features])
    return data_with_xyz
