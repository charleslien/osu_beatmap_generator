from dataclasses import dataclass
import numpy as np
from scipy import optimize

from slider_simplify import path, path_type

@dataclass
class ArcPath:
  start: np.ndarray
  end: np.ndarray
  length: float
  angle: float
  midpoint: np.ndarray

  def get_path(self):
    return path.Path.create_path([start, midpoint, end],
                                 [path_type.PathType.PERFECT_CURVE, None, None])

  def get_all(start, end, length):
    """
    return a list of the (up to 2) possible arc paths
    """

    displacement = end - start
    displacement_length = np.linalg.norm(displacement)
    if displacement_length >= length:
      return [ArcPath(np.copy(start),
                      np.copy(end),
                      displacement_length,
                      0,
                      start + displacement / 2)]

    # chord_length / arc_length = 0 -> angle = 2 * pi
    # chord_length / arc_length = 1 -> angle = 0
    # angle approximation = 2 * pi * (1 - chord_length / arc_length)
    angle = optimize.fsolve(ArcPath._length_difference, # equation to find root
                             [2 * np.pi * (1 - displacement_length / length)], # starting guess
                             args=(displacement_length, length),
                             fprime=ArcPath._length_difference_derivative)[0]
    angle = abs(angle)
    while angle > 2 * np.pi:
      angle -= 2 * np.pi

    rotated = np.array((displacement[1], -displacement[0]))
    rotated = rotated * (1 - np.cos(angle / 2)) / (2 * np.sin(angle / 2))

    arc_path_1 = ArcPath(np.copy(start),
                         np.copy(end),
                         length,
                         angle,
                         start + displacement / 2 + rotated)
    arc_path_2 = ArcPath(np.copy(start),
                         np.copy(end),
                         length,
                         -angle,
                         start + displacement / 2 - rotated)

    return [arc_path_1, arc_path_2]

  def _length_difference(angle, chord_length, arc_length):
    """
    returns difference in expected chord length (using law of cosines) and actual chord length
    """
    angle2 = angle * angle
    return (2 * arc_length * arc_length / angle2 * (1 - np.cos(angle)) -
            chord_length * chord_length)

  def _length_difference_derivative(angle, chord_length, arc_length):
    """
    derivative with respect to radius
    """
    angle2 = angle * angle
    angle3 = angle * angle2
    arc_length2 = arc_length * arc_length
    return (-4 * arc_length2 / angle3 * (1 - np.cos(angle)) +
            2 * arc_length2 / angle2 * np.sin(angle))
