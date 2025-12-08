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

def get_point_cloud(controller: MetaController):
    camera_sys = controller.diagram.GetSubsystemByName("camera0.point_cloud")
    camera_context = camera_sys.GetMyContextFromRoot(controller.context)
    point_cloud_port = camera_sys.GetOutputPort("point_cloud")
    cloud: PointCloud = point_cloud_port.Eval(camera_context)
    # only want stuff above the floor
    cropped = cloud.Crop(np.array([-1e6, -1e6, HEIGHT_CUTOFF]), np.array([1e6, 1e6, 1e6]))
    points = cropped.xyzs()
    print(points)
