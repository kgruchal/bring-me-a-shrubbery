import math

import numpy as np


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def dist(hull, points):
    # Construct PyGEL Manifold from the convex hull
    m = gel.Manifold()
    for s in hull.simplices:
        m.add_face(hull.points[s])

    dist = gel.MeshDistance(m)
    res = []
    for p in points:
        # Get the distance to the point
        # But don't trust its sign, because of possible
        # wrong orientation of mesh face
        d = dist.signed_distance(p)

        # Correct the sign with ray inside test
        if dist.ray_inside_test(p):
            if d > 0:
                d *= -1
        else:
            if d < 0:
                d *= -1
        res.append(d)
    return np.array(res)


def get_2d(pcd_numpy, axis_1, axis_2, plot=True):
    slice_1 = pcd_numpy[:, axis_1]
    slice_2 = pcd_numpy[:, axis_2]
    slice_3 = np.zeros((pcd_numpy.shape[0], ))

    slice_2d = np.stack((slice_1, slice_2), axis=1)

    slice_final = np.stack((slice_1, slice_2, slice_3), axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(slice_final)
    if plot:
        o3d.visualization.draw_geometries([pcd])

    return slice_2d


def polar2cart(r, theta, phi):
    return [
        r * math.sin(theta) * math.cos(phi),
        r * math.sin(theta) * math.sin(phi),
        r * math.cos(theta)
    ]


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    #phi, theta, r
    return [az, el, r]


def get_center(np_array):
    center_x = np.average(np_array[:, 0])
    center_y = np.average(np_array[:, 1])
    center_z = np.average(np_array[:, 2])

    return center_x, center_y, center_z


def farthest_point_sampling(points, k):
    print('Performing farthest point sampling....')
    # First, pick a random point to start
    # add it to the solution, remove it from the full list
    solution_set = []
    seed = np.random.randint(0, points.shape[0] - 1)
    solution_set.append(points[seed, :])
    points = np.delete(points, (seed), axis=0)

    # Now, iterate k-1 times
    # Find the farthest point, add to solution, remove from list, repeat
    for i in tqdm(range(k-1)):
        distances = np.full((points.shape[0],), np.inf)
        for j, point in enumerate(points):
            for sol_point in solution_set:
                distances[j] = min(distances[j], distance(point, sol_point))

        picked_index = np.argmax(distances)
        solution_set.append(points[picked_index])
        points = np.delete(points, (picked_index), axis=0)

    return solution_set


def distance(A, B):
    return np.linalg.norm(A-B)


def polar2cart2d(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def pdf(algorithm, kwargs):
    probs = algorithm.pdf(**kwargs)
    return probs


def sample(algorithm, kwargs):
    points = algorithm.rvs(**kwargs)
    return points


def mask(points):
    # Illuminate points based on z-value
    min_z = min(points[:, 2])
    max_z = max(points[:, 2])
    # Determine step based on number of views

    num_views = 10

    step = (max_z - min_z)/num_views

    # Need to round to 1 decimal places
    for z in np.arange(min_z, max_z, step):
        # Select the points with matching z-value
        # print(z)
        temp_points = points[np.round(
            points[:, 2], decimals=1) == np.round(z, decimals=1)]
        scene_2d = get_2d(temp_points, 0, 1)
