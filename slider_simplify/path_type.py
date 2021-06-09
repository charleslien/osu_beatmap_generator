import enum

class PathType(enum.Enum):
  LINEAR = 0
  PERFECT_CURVE = 1
  CATMULL = 2
  BEZIER = 3
