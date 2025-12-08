import numpy as np
from pydrake.all import (
        MultibodyPlant,
        PointCloud,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .meta_controller import MetaController
else:
    MetaController = None

HEIGHT_CUTOFF = 0.02

def _get_indv_cloud(controller: MetaController, port: str) -> PointCloud:
    camera_sys = controller.diagram.GetSubsystemByName(port)
    camera_context = camera_sys.GetMyContextFromRoot(controller.context)
    point_cloud_port = camera_sys.GetOutputPort("point_cloud")
    return point_cloud_port.Eval(camera_context)

def get_point_cloud(controller: MetaController, height: float = HEIGHT_CUTOFF) -> np.ndarray:
    """
    Returns 3xn array of xyz locations of points.
    """
    cloud0 = _get_indv_cloud(controller, "camera0.point_cloud")
    cloud1 = _get_indv_cloud(controller, "camera0.point_cloud")
    cropped0 = cloud0.Crop(np.array([-1e6, -1e6, height]), np.array([1e6, 1e6, 1e6])).xyzs()
    cropped1 = cloud1.Crop(np.array([-1e6, -1e6, height]), np.array([1e6, 1e6, 1e6])).xyzs()
    cloud = np.concatenate([cropped0, cropped1], axis=1)
    return cloud


