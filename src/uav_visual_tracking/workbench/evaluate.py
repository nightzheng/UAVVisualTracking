import numpy as np


from .u3_gazebo import quat, camera_transform


from .scene3d import raycast,barycentric3


class IndependentScene:
    def __init__(self,surface,state):
        self.surface=surface;self.hub=np.asarray(state['rotor_pose'][:3]);self.rotor=quat(state['rotor_pose'][3:])
        from .scene3d import ry
        self.rotations=[self.rotor@ry(j*2*np.pi/3) for j in range(3)]
        self.position,self.T=camera_transform(state['camera_pose'])
        self.world=np.concatenate([surface.triangles@r.T+self.hub for r in self.rotations]+[surface.fixed])

    def hit(self,xy,invK):
        ray=self.T[:3,:3].T@invK@np.r_[xy,1.];ray/=np.linalg.norm(ray)
        distance,idx=raycast(self.position,ray,self.world)
        s=self.surface
        if idx is None or idx>=3*len(s.faces):return None
        blade,face=divmod(idx,len(s.faces));tri=s.triangles[face]
        point=(self.position+distance*ray-self.hub)@self.rotations[blade]
        weights=barycentric3(point,tri)
        if weights is None:return None
        uv=np.c_[s.eta[s.faces[face]],s.xi[s.faces[face]]]
        eta,xi=weights@uv
        side='suction_geometric' if face in s.side_ids else 'other_or_edge'
        return dict(blade='ABC'[blade],face_id=int(face),eta=float(eta),xi=float(xi),side=side,
            point=point.tolist(),triangle=tri.tolist(),triangle_uv=uv.tolist())


def align(frames,states,tolerance):
    states=sorted(states,key=lambda s:s['sim_s']);pts=np.array([s['sim_s'] for s in states]);out=[]
    for f in frames:
        j=np.searchsorted(pts,f['pts']+1e-9,side='right')-1
        out.append(states[j] if j>=0 and abs(pts[j]-f['pts'])<=tolerance+1e-10 else None)
    return out
