from dataclasses import dataclass
import math
import numpy as np

from slider_simplify import path_type

# This class was transcribed from https://git.mine2.live/DevMiner/osu-api/src/commit/da4f28917209d556d8c4ad1da37f718445373d3b/src/PathApproximator.ts

_BEZIER_TOLERANCE = 0.25
_CIRCULAR_ARC_TOLERANCE = 0.1
_CATMULL_DETAIL = 50

def approximate_bezier(control_points):
    output = []
    count = len(control_points)

    if count == 0:
      return output
    
    subdivision_buffer_1 = {}
    subdivision_buffer_2 = {}

    to_flatten = [control_points[:]]
    free_buffers = []

    left_child = subdivision_buffer_2

    while len(to_flatten) > 0:
      parent = to_flatten.pop()

      if _rate_flatness_of_bezier(parent):
        # If the control points we currently operate on are sufficiently "flat", we use
        # an extension to De Casteljau's algorithm to obtain a piecewise-linear approximation
        # of the bezier curve represented by our control points, consisting of the same amount
        # of points as there are control points.
        _bezier_approximate(parent, output, subdivision_buffer_1, subdivision_buffer_2, count)

        free_buffers.append(parent)
        continue
      
      # If we do not yet have a sufficiently "flat" (in other words, detailed) approximation we keep
      # subdividing the curve we are currently operating on.
      right_child = free_buffers.pop() if len(free_buffers) > 0 else {}

      _subdivide_bezier(parent, left_child, right_child, subdivision_buffer_1, count)

      for i in range(count):
        parent[i] = left_child[i]
      
      to_flatten.append(right_child)
      to_flatten.append(parent)
    
    output.append(control_points[-1])

    return output

def approximate_catmull(control_points):
  result = []
  control_points_length = len(control_points)

  for i in range(control_points_length - 1):
    v1 = control_points[i - 1] if i > 0 else control_points[i]
    v2 = control_points[i]
    v3 = control_points[i + 1] if i < control_points_length - 1 else v2 + v2 - v1
    v4 = control_points[i + 2] if i < control_points_length - 2 else v3 + v3 - v2

    for c in range(_CATMULL_DETAIL + 1):
      result.append(_find_catmull_point(v1, v2, v3, v4, c / _CATMULL_DETAIL))
  
  return result

def approximate_circular_arc(control_points):
  a = control_points[0]
  b = control_points[1]
  c = control_points[2]

  a_sq = np.linalg.norm(b - c) ** 2
  b_sq = np.linalg.norm(a - c) ** 2
  c_sq = np.linalg.norm(a - b) ** 2

  # If we have a degenerate triangle where a side-length is almost zero, then give up and fall
  # back to a more numerically stable method.
  if a_sq < 0.003 or b_sq < 0.003 or c_sq < 0.003:
    return []
  
  s = a_sq * (b_sq + c_sq - a_sq)
  t = b_sq * (a_sq + c_sq - b_sq)
  u = c_sq * (a_sq + b_sq - c_sq)

  total = s + t + u

  # If we have a degenerate triangle with an almost-zero size, then give up and fall
  # back to a more numerically stable method.
  if total < 0.003:
    return []
  
  centre = (a * s + b * t + c * u) / total
  d_a = a - centre
  d_c = c - centre

  r = np.linalg.norm(d_a)

  theta_start = math.atan2(d_a[1], d_a[0])
  theta_end = math.atan2(d_c[1], d_c[0])

  while theta_end < theta_start:
    theta_end += 2*math.pi

  direction = 1
  theta_range = theta_end - theta_start

  # Decide in which direction to draw the circle, depending on which side of
  # AC B lies.
  ortho_a_to_c = c - a
  
  ortho_a_to_c = np.array((ortho_a_to_c[1], -ortho_a_to_c[0]))

  if ortho_a_to_c.dot(b - a) < 0:
    direction = -direction
    theta_range = 2 * math.pi - theta_range

  # We select the amount of points for the approximation by requiring the discrete curvature
  # to be smaller than the provided tolerance. The exact angle required to meet the tolerance
  # is: 2 * Math.Acos(1 - TOLERANCE / r)
  # The special case is required for extremely short sliders where the radius is smaller than
  # the tolerance. This is a pathological rather than a realistic case.
  amount_points = (
      2
      if 2 * r <= _CIRCULAR_ARC_TOLERANCE
      else max(2, math.ceil(theta_range / (2 * math.acos(1 - _CIRCULAR_ARC_TOLERANCE / r))))
  )
  
  output = []
  for i in range(amount_points):
    fract = i / (amount_points - 1)
    theta = theta_start + direction * fract * theta_range

    o = np.array((math.cos(theta), math.sin(theta))) * r

    output.append(centre + o)
  
  return output

def _rate_flatness_of_bezier(control_points):
  length = len(control_points)
  for i in range(1, length - 1):
    scale = control_points[i] * 2
    sub = control_points[i - 1] - scale
    total = sub + control_points[i + 1]

    if np.linalg.norm(total) > _BEZIER_TOLERANCE * 2:
      return False

  return True

def _subdivide_bezier(control_points, l, r, subdivision_buffer, count):
  midpoints = subdivision_buffer

  for i in range(count):
    midpoints[i] = control_points[i]
  
  for i in range(count):
    l[i] = midpoints[0]
    r[count - i - 1] = midpoints[count - i - 1]

    for j in range(count - i - 1):
      midpoints[j] = (midpoints[j] + midpoints[j + 1]) / 2

def _bezier_approximate(control_points, output, subdivision_buffer_1, subdivision_buffer_2, count):
  l = subdivision_buffer_1
  r = subdivision_buffer_2

  _subdivide_bezier(control_points, l, r, subdivision_buffer_1, count)

  for i in range(count - 1):
    l[count + i] = r[i + 1]

  for i in range(1, count - 1):
    index = 2 * i
    p = (l[index - 1] + l[index] * 2 + l[index + 1]) * 0.25

    output.append(p)

def _find_catmull_point(v1, v2, v3, v4, t):
  t2 = t * t
  t3 = t2 * t

  return (0.5 * 
          (2*v2 +
          (-v1 + v3) * t +
          (2*v1 - 5*v2 + 4*v3 - v4) * t2 +
          (-v1 + 3*v2 - 3*v3 + v4) * t3))
