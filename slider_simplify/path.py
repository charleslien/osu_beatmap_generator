from dataclasses import dataclass
import numpy as np
from typing import List

from slider_simplify import path_approximator, path_type

_PATH_COMPARISON_GRANULARITY = 1000

@dataclass
class Path:
  control_points: List[np.ndarray]
  control_point_types: List[path_type.PathType]
  calculated_path: List[np.ndarray]
  cumulative_length: List[float]

  @property
  def num_path_points(self):
    return len(self.calculated_path)

  @property
  def num_control_points(self):
    return len(self.control_points)

  def get_end_point(self, length=0):
    if length <= 0:
      # invalid length, return full path
      return self.calculated_path[-1]

    for i in range(1, self.num_path_points):
      if self.cumulative_length[i] <= length:
        continue

      distance_to_travel = length - self.cumulative_length[i - 1]

      segment_start = self.calculated_path[i - 1]
      segment = self.calculated_path[i] - segment_start
      segment = segment / np.linalg.norm(segment)
      return segment_start + segment * distance_to_travel

  def get_best_fit(self, path_length, candidates):
    num_candidates = len(candidates)

    scores = [0] * num_candidates
    for i in range(_PATH_COMPARISON_GRANULARITY):
      progress = i / _PATH_COMPARISON_GRANULARITY
      
      original_point = self.get_end_point(
          path_length * progress)
      
      for j, (candidate_path, length) in enumerate(candidates):
        point = candidate_path.get_end_point(length * progress)
        dist = np.linalg.norm(point - original_point)
        scores[j] += dist * dist
    
    return np.argmin(scores)

  def create_path(control_points, types):
    calculated_path = Path._calculate_path(control_points, types)
    cumulative_length = Path._get_cumulative_length(calculated_path)
    
    return Path(control_points[:],
                types[:],
                calculated_path,
                cumulative_length)
  
  def _calculate_path(control_points, types):
    if len(control_points) == 0:
      return []
    
    calculated_path = []
    start = 0

    for i in range(len(control_points)):
      if types[i] == None and i < len(control_points) - 1:
        continue
      
      segment_vertices = control_points[start:i + 1]
      segment_type = types[start] or path.PathType.Linear

      for t in Path._calculate_sub_path(segment_vertices, segment_type):
        if len(calculated_path) == 0 or any(calculated_path[-1] != t):
          calculated_path.append(t)
      
      start = i
    
    return calculated_path

  def _calculate_sub_path(sub_control_points, path_type_instance):
    if path_type_instance == path_type.PathType.LINEAR:
      return [np.copy(point) for point in sub_control_points]
    elif (path_type_instance == path_type.PathType.PERFECT_CURVE and
          len(sub_control_points) == 3):
      subpath = path_approximator.approximate_circular_arc(sub_control_points)
      if len(subpath) > 0:
        return subpath
    elif path_type_instance == path_type.PathType.CATMULL:
      return path_approximator.approximate_catmull(sub_control_points)
    
    return path_approximator.approximate_bezier(sub_control_points)

  def _get_cumulative_length(calculated_path):
    cumulative_length = [0]
    for i in range(1, len(calculated_path)):
      segment_start = calculated_path[i - 1]
      segment = calculated_path[i] - segment_start
      
      segment_length = np.linalg.norm(segment)

      cumulative_length.append(cumulative_length[-1] + segment_length)

    return cumulative_length
