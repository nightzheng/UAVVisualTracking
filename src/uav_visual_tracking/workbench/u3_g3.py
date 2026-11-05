import numpy as np


def theta(surface,state):
    r=surface.tilt.T@quat(state['rotor_pose'][3:])
    return float(np.rad2deg(np.arctan2(r[0,2],r[2,2])))


from .u3_gazebo import quat
