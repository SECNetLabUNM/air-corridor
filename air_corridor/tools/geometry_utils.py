from envs111.uav_corridor_navigation.d3.geometry.geom3d import *
class GeometryUtils:
    @staticmethod
    def distance_between_point_and_plane(point: Point3D, plane: CirclePlane) -> float:
        point_vector = np.array([point.x, point.y, point.z])
        plane_normal = np.array([plane.a, plane.b, plane.c])
        distance = np.abs(np.dot(plane_normal, point_vector) + plane.d) / np.linalg.norm(plane_normal)
        return distance