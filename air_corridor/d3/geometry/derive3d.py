import numpy as np
from scipy import optimize  # type: ignore
import math

from .geom3d import partialCircle3D, directionalPartialTorus
from envs111.uav_corridor_navigation.tools._util import onoff_ramp_points, rotate_to_xy_plane


def partialCircle3D_from_lines(Line1, Line2):
    p1 = Line1.anchor_point
    d1 = Line1.orientation_vec
    p2 = Line2.anchor_point
    d2 = Line2.orientation_vec

    # circle1_points
    circle_1_points, circle_2_points, radius = onoff_ramp_points(p1, d1, p2, d2)

    # begin and end points projected to x-y plane
    bp1, ep1 = rotate_to_xy_plane(circle_1_points)
    circle_1_centric = circle_1_points[0]
    circle_1_direction = np.cross(circle_1_points[1] - circle_1_points[0], circle_1_points[2] - circle_1_points[0])
    circle_1_radius = radius
    circle_1_begin_degree = math.atan2(bp1[1], bp1[0])
    circle_1_end_degree = math.atan2(ep1[1], ep1[0])
    circle_1 = partialCircle3D(circle_1_centric, circle_1_direction, circle_1_radius, circle_1_begin_degree,
                               circle_1_end_degree)

    bp2, ep2 = rotate_to_xy_plane(circle_2_points)
    circle_2_centric = circle_2_points[0]
    circle_2_direction = np.cross(circle_2_points[1] - circle_2_points[0], circle_2_points[2] - circle_2_points[0])
    circle_2_radius = radius
    circle_2_begin_degree = math.atan2(bp2[1], bp2[0])
    circle_2_end_degree = math.atan2(ep2[1], ep2[0])
    circle_2 = partialCircle3D(circle_2_centric, circle_2_direction, circle_2_radius, circle_2_begin_degree,
                               circle_2_end_degree)

    return circle_1, circle_2


def partialTorus_from_lines(Line1, Line2, minor_radius):
    pc1, pc2 = partialCircle3D_from_lines(Line1, Line2)

    #  center, direction, major_radius, minor_radius, begin_degree, end_degree)
    # partialTorus_1 = partialTorus(pc1.center, pc1.direction, pc1.radius, minor_radius, pc1.begin_degree, pc1.end_degree)
    # partialTorus_2 = partialTorus(pc2.center, pc2.direction, pc2.radius, minor_radius, pc2.begin_degree, pc2.end_degree)

    partialTorus_1 = directionalPartialTorus(pc1.center, pc1.orientation_vec, pc1.radius, minor_radius, pc1.begin_degree, pc1.end_degree)
    partialTorus_2 = directionalPartialTorus(pc2.center, pc2.orientation_vec, pc2.radius, minor_radius, pc2.begin_degree, pc2.end_degree)
    #return pc1, pc2
    return partialTorus_1,partialTorus_2
