import numpy as np


def quat(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def camera_transform(pose):
    position=np.asarray(pose[:3]);R=np.array([[0,-1,0],[0,0,-1],[1,0,0]])@quat(pose[3:]).T
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=-R@position
    return position,T
