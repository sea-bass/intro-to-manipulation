""" Utilities for model generation and loading. """

import importlib.resources
import pyroboplan
import pinocchio
import trimesh
import hppfcl
import numpy as np


def get_example_models_folder():
    """
    Returns the full path the example models folder.

    Returns
    -------
        str
            The path to the `pyroboplan` example models folder.
    """
    resource_path = importlib.resources.files(pyroboplan) / "models"
    return resource_path.as_posix()


def create_ellipsoid_mesh(radius_x, radius_y, radius_z, subdivisions=3):
    """
    Creates an ellipsoid mesh with the given radii and subdivisions.

    Parameters
    ----------
        radius_x : float
            The x-axis radius of the ellipsoid.
        radius_y : float
            The y-axis radius of the ellipsoid.
        radius_z : float
            The z-axis radius of the ellipsoid.
        subdivisions : int, optional
            The number of subdivisions to use when creating the sphere mesh.
        
    Returns
    -------
        hppfcl.BVHModelOBB
            The mesh of the ellipsoid.
    """
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    ellipsoid = sphere.copy()
    ellipsoid.apply_scale([radius_x, radius_y, radius_z])
    vertices = ellipsoid.vertices.astype(np.float32)
    faces = ellipsoid.faces.astype(np.int32)
    fcl_mesh = hppfcl.BVHModelOBB()
    fcl_mesh.beginModel(vertices.shape[0], faces.shape[0])
    for vertex in vertices:
        fcl_mesh.addVertex(np.array(vertex, dtype=np.float64))
    for face in faces:
        fcl_mesh.addTriangle(
            np.array(vertices[face[0]], dtype=np.float64),
            np.array(vertices[face[1]], dtype=np.float64),
            np.array(vertices[face[2]], dtype=np.float64)
        )
    fcl_mesh.endModel()
    return fcl_mesh


def add_ellipsoid_model(model, ellipsoid_mesh, position, rotation_matrix, name='ellipsoid', color=np.array([0.5, 0.5, 0.5, 0.5])):
    """
    Add an ellipsoid to the given model at the specified position and rotation.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        ellipsoid_mesh : hppfcl.BVHModelOBB
            The mesh of the ellipsoid to add.
        position : list
            The position of the center of the ellipsoid.
        rotation_matrix : list[array-like]
            The rotation matrix of the ellipsoid.
        name : str, optional
            The name of the ellipsoid to add.
        color : list, optional
            The color of the ellipsoid to add.
    """
    position = np.array(position)
    ellipsoid = pinocchio.GeometryObject(
        name,
        0,
        ellipsoid_mesh,
        pinocchio.SE3(rotation_matrix, position)
    )
    ellipsoid.meshColor = color
    model.addGeometryObject(ellipsoid)
    ellipsoid.collision_model = None
    model.display()