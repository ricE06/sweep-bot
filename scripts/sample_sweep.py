import random
import math
import itertools

def dist_to_line(p0: tuple[float, float], p1: tuple[float, float], x: tuple[float, float]):
    a = p0[1] - p1[1]
    b = p1[0] - p0[0]
    c = -(a*p0[0]+b*p0[1])
    norm = math.sqrt(a**2+b**2)
    return abs(a*x[0]+b*x[1]+c) / norm

class SweepGenerator:
    """
    Specifies a rectangle (reachable_min, reachable_max) of two (x, y) pairs
    where the path is allowed to travel in, as well as the allowed
    specified region.
    """
    angle_tolerance = 0.1
    dist_tolerance = 1
    num_segments = 5

    def __init__(self,
                 target: tuple[float, float],
                 reachable_min: tuple[float, float],
                 reachable_max: tuple[float, float],
                 startable_min: tuple[float, float],
                 startable_max: tuple[float, float]):
        self.target = target
        self.reachable_x = (reachable_min[0], reachable_max[0])
        self.reachable_y = (reachable_min[1], reachable_max[1])
        self.startable_x = (startable_min[0], startable_max[0])
        self.startable_y = (startable_min[1], startable_max[1])

    def _sample_in_starting(self):
        x = random.uniform(*self.startable_x)
        y = random.uniform(*self.startable_y)
        return (x, y)

    def _sample_in_reachable(self, start_point, start_angle):
        r = random.uniform(0.1, 0.4)
        angle = start_angle + random.uniform(-0.4, 0.4)
        x = start_point[0] + r * math.cos(angle)
        y = start_point[1] + r * math.sin(angle)
        # clamp inside reachable bounds
        x = max(self.reachable_x[0], min(x, self.reachable_x[1]))
        y = max(self.reachable_y[0], min(y, self.reachable_y[1]))
        return (x,y)


    def _generate_sweep_sampling(self):
        # one segment sweep
        return [self._sample_in_starting(), self.target]
        pass

    def _generate_sweep_manual(self):
        pass

    def _compute_score(self, traj: list[tuple[float, float]],
                       point_cloud: set[tuple[float, float]]):
        total_score = 0
        total_ct = 0
        for t0, t1 in itertools.pairwise(traj):
            seg_score = 0
            for point in point_cloud:
                dist = dist_to_line(t0, t1, point)
                if dist < self.dist_tolerance:
                    seg_score += 1
            total_score += seg_score / len(point_cloud)
            total_ct += 1
        return total_score / total_ct

    def find_sweep(self, point_cloud, num_samples=100):
        best_score = None
        best_traj = None
        for _ in range(100):
            traj = self._generate_sweep_sampling()
            score = self._compute_score(traj, point_cloud)
            if best_score is None or score < best_score:
                best_score = score
                best_traj = traj

        return best_traj
