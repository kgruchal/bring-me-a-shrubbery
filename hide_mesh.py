from common import *

import argparse
import numpy as np
import open3d as o3d
import os.path
import scipy
import scipy.stats

parser = argparse.ArgumentParser()

parser.add_argument("object", help="Object name to obfuscate")
parser.add_argument("-p", help="Plot points", action='store_true')

args = parser.parse_args()

object_name = args.object

# Radii of spheres
# Increasing order
rs = [0.4, 0.6, 0.8]

known_objects = {
    "bunny": {
        "source": "./data/bunny.ply",
        "cov": (rs[-1]/5.) * np.identity(3),
        "subsamp": (1000, 1.1),
        "factor": 1.0
    },
    "teapot": {
        "source": "./data/teapot.ply",
        "cov": (rs[-1]/2.) * np.identity(3),
        "subsamp": (2500, 1.4),
        "factor": 1/1.5
    },
    "cow": {
        "source": "./data/cow.ply",
        "cov": (rs[-1]/3.) * np.identity(3),
        "subsamp": (2500, 1.4),
        "factor": 1/1.5
    }
}

if object_name not in known_objects:
    print("Unkown object '{}'".format(object))
    exit(1)

object_data = known_objects[object_name]

source_file = object_data['source']
cov = object_data['cov']
density_factor = object_data['factor']
subsamp_factors = object_data['subsamp']

# ----------------PARAMETERS---------------------------------------------------
# Normal parameters:
mean = np.asarray([0, 0, 0])
#cov = (rs[-1]/5.) * np.identity(3)
params = {'mean': mean, 'cov': cov}
distribution = scipy.stats.multivariate_normal


# ------------------------------------Read in data-----------------------------
pcd_bunny = o3d.io.read_point_cloud(source_file)

cloud1 = o3d.io.read_point_cloud("./data/cloud_1_upsampled.pcd")

bunny_np = np.asarray(pcd_bunny.points)

cloud1_np = np.asarray(cloud1.points)

# Need to scale object to be same size as bunny (give them same volume?)
max_side = max((max(bunny_np[:, 0]) - min(bunny_np[:, 0])), (max(bunny_np[:, 1]) -
                                                             min(bunny_np[:, 1])), (max(bunny_np[:, 2]) - min(bunny_np[:, 2])))
# Scale so that max side is inside radii
inside_radius = 0.4
bunny_np = bunny_np*(inside_radius/max_side)
max_side = max((max(bunny_np[:, 0]) - min(bunny_np[:, 0])), (max(bunny_np[:, 1]) -
                                                             min(bunny_np[:, 1])), (max(bunny_np[:, 2]) - min(bunny_np[:, 2])))
print(max_side)

clouds = [cloud1_np]

bunny_subsample_indices = np.random.randint(
    bunny_np.shape[0], size=subsamp_factors[0])
bunny_np = bunny_np[bunny_subsample_indices, :]*subsamp_factors[1]

# Center the bunny at origin:
bunny_np[:, 0] = bunny_np[:, 0] - np.mean(bunny_np[:, 0])
bunny_np[:, 1] = bunny_np[:, 1] - np.mean(bunny_np[:, 1])
bunny_np[:, 2] = bunny_np[:, 2] - np.mean(bunny_np[:, 2])


for i, cloud in enumerate(clouds):
    # Compute hull:
    cloud_hull = scipy.spatial.ConvexHull(cloud)
    cloud_volume = cloud_hull.volume

    # Now, center the cloud
    cloud[:, 0] = cloud[:, 0] - np.mean(cloud[:, 0])
    cloud[:, 1] = cloud[:, 1] - np.mean(cloud[:, 1])
    cloud[:, 2] = cloud[:, 2] - np.mean(cloud[:, 2])

    # Scale cloud to match bunny
    cloud = cloud/(cloud_volume)

    #cloud_indices = np.random.randint(cloud.shape[0], size=400)
    #cloud = cloud[cloud_indices,:]
    clouds[i] = cloud

# -----------------------------SPHERE-----------------------------------------

# First, make sphere
print('Generating spheres.....')

step = 5

phis = np.arange(0, 360, step)
thetas = np.arange(0, 180, step)

spheres = []


for r in rs:
    sphere = []
    for phi in phis:
        for theta in thetas:
            point = polar2cart(r, theta, phi)
            sphere.append(point)
    spheres.append(sphere)

sphere = np.asarray(spheres[0])
spheres = np.asarray(spheres)

# ----------------------------CLOUDS----------------------------------------
# Now, paint clouds
print('Painting clouds.....')
num_clouds = 80


seed_points = []
load_flag = 1

if os.path.isfile('./data/cloud_points.npy') and load_flag:
    seed_points = np.load('./data/cloud_points.npy')
else:
    for sphere_idx in range(spheres.shape[0]):
        seed_points.append(farthest_point_sampling(
            spheres[sphere_idx, :, :], num_clouds))

    seed_points = np.asarray(seed_points)
    print(seed_points.shape)
    np.save('./data/cloud_points.npy', seed_points)


seed_density = (bunny_np.shape[0]/scipy.spatial.ConvexHull(bunny_np).volume)

seed_density *= density_factor

# Now, reduce bunny density based on pdf:
x = np.expand_dims(np.asarray([0, 0, 0]), axis=0)

params['x'] = x
bunny_pdf = pdf(distribution, params)

bunny_density = bunny_pdf*seed_density

bunny_volume = scipy.spatial.ConvexHull(bunny_np).volume

bunny_pts = int(bunny_density*bunny_volume)
print('bunny pts: ', bunny_pts)
bunny_indices = np.random.randint(bunny_np.shape[0], size=int(bunny_pts))
bunny_np = bunny_np[bunny_indices, :]

params['x'] = seed_points
cloud_pdf = pdf(distribution, params)

points = bunny_np

for seed_points_idx in range(seed_points.shape[0]):
    # scale
    cloud = clouds[0]*rs[seed_points_idx]

    for point_idx, seed_point in enumerate(seed_points[seed_points_idx, :, :]):
        rot_cloud = cloud
        cloud_density = cloud_pdf[seed_points_idx, point_idx]*seed_density

        #print('cloud density: ', cloud_density)

        cloud_volume = scipy.spatial.ConvexHull(rot_cloud).volume
        cloud_pts = int(cloud_density*cloud_volume)
        cloud_indices = np.random.randint(cloud.shape[0], size=int(cloud_pts))
        rot_cloud = rot_cloud[cloud_indices, :]

        [phi, theta, r] = cart2sph(seed_point[0], seed_point[1], seed_point[2])
        r = scipy.spatial.transform.Rotation.from_rotvec(
            phi * np.array([0, 1, 0]))
        rot_cloud = r.apply(rot_cloud)

        # center cloud at selected point
        rot_cloud[:, 0] -= np.mean(rot_cloud[:, 0]) - seed_point[0]
        rot_cloud[:, 1] -= np.mean(rot_cloud[:, 1]) - seed_point[1]
        rot_cloud[:, 2] -= np.mean(rot_cloud[:, 2]) - seed_point[2]

        points = np.concatenate((points, rot_cloud), axis=0)

# ----------------------------WRITE TO CSV------------------------------------
print('Writing ' + "{}.csv".format(object_name))
np.savetxt("{}.csv".format(object_name), points, delimiter=',')

# ------------------------------------Plot------------------------------------

if (args.p):
    print('Plotting....')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = [[0, 0, 0] for i in range(points.shape[0])]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

