import typing

""" For functions accepting either an int or a list of int as arguments"""
ListOrIndex = typing.Union[int, typing.Sequence[int]]

""" The underlying C++ API does not support any number of extra
  balls per ExtraBallsSet. Here the only accepted values. 
"""
AcceptedNbOfBalls = typing.Literal[3, 10, 20, 50, 100]

""" For 3d position or 3d velocities  """
Point3D = typing.Tuple[float, float, float]

""" Position 3d, Velocity 3d, and contact info
  (True if the ball ever had a contact with the racket, since
  reset was called) """
ExtraBall = typing.Tuple[Point3D, Point3D, bool]
