"""Original metric solid geometry, perspective rasterizer and independent ray GT.

World right-handed X/Y/Z; rotor axis +Y, Z up. Camera x-right/y-down/z-forward.
RGB shading has no GT markers. Depth/instance/primitive buffers are separate.
"""
from dataclasses import dataclass
import numpy as np


def unit(x):
    x=np.asarray(x,float); n=np.linalg.norm(x)
    if n<1e-12: raise ValueError('Degenerate direction')
    return x/n


def ry(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def transform(points,T):
    return np.asarray(points)@T[:3,:3].T+T[:3,3]


def look_at(position,target):
    forward=unit(np.asarray(target)-position)
    right=unit(np.cross(forward,[0,0,1]))
    down=np.cross(forward,right)
    R=np.stack([right,down,forward])
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=-R@position
    return T


def camera_pose(t,cfg,condition):
    if condition not in ('C0','C1'): raise ValueError('Unknown camera condition')
    u=float(t)/((cfg['frames']-1)/cfg['fps']) if condition=='C1' else 0.
    phase=(1-np.cos(2*np.pi*u))/2
    az=np.deg2rad(cfg['camera_azimuth_start_deg']+cfg['camera_azimuth_sweep_deg']*phase)
    radius=cfg['camera_radius_m']+cfg['camera_radius_amplitude_m']*np.sin(2*np.pi*u)
    hub=np.array(cfg['hub_world_m'])
    position=hub+[radius*np.sin(az),-radius*np.cos(az),cfg['camera_height_offset_m']+cfg['camera_height_amplitude_m']*np.sin(2*np.pi*u)]
    target=hub+[cfg['look_offset_x_m']*np.sin(2*np.pi*u),0,cfg['look_offset_z_m']*np.sin(np.pi*u)]
    return position,look_at(position,target),target


def camera_state(timestamps,cfg,condition):
    rows=[];h=1e-4
    for t in timestamps:
        C,T,target=camera_pose(t,cfg,condition)
        cm,tm,_=camera_pose(t-h,cfg,condition);cp,tp,_=camera_pose(t+h,cfg,condition)
        Rwc=T[:3,:3].T
        Rdot=(tp[:3,:3].T-tm[:3,:3].T)/(2*h)
        skew=Rdot@Rwc.T
        rows.append((C,T,np.linalg.inv(T),(cp-cm)/(2*h),np.array([skew[2,1],skew[0,2],skew[1,0]]),target))
    names=['position_world_m','T_camera_from_world','T_world_from_camera','velocity_world_m_s','angular_velocity_world_rad_s','look_target_world_m']
    return {k:np.stack([r[i] for r in rows]) for i,k in enumerate(names)}


def project(camera_xyz,K):
    xyz=np.asarray(camera_xyz,float)
    with np.errstate(divide='ignore',invalid='ignore'):
        return (xyz@np.asarray(K).T)[...,:2]/xyz[...,2,None]


@dataclass
class Mesh:
    name:str
    object_id:int
    vertices:np.ndarray
    faces:np.ndarray
    placement:np.ndarray


def box(name,object_id,lo,hi):
    a,b=np.array(lo),np.array(hi)
    v=np.array([[a[0],a[1],a[2]],[b[0],a[1],a[2]],[b[0],b[1],a[2]],[a[0],b[1],a[2]],
                [a[0],a[1],b[2]],[b[0],a[1],b[2]],[b[0],b[1],b[2]],[a[0],b[1],b[2]]])
    quads=[[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    f=np.array([tri for a,b,c,d in quads for tri in ([a,b,c],[a,c,d])])
    return Mesh(name,object_id,v,f,np.eye(4))


def cylinder(name,object_id,center,r0,r1,length,axis='z',sides=16):
    angles=np.arange(sides)*2*np.pi/sides
    verts=[]
    for end,r in [(0.,r0),(length,r1)]:
        for angle in angles:
            p=[r*np.cos(angle),r*np.sin(angle),end]
            if axis=='y': p=[p[0],p[2],p[1]]
            verts.append(p)
    faces=[]
    for i in range(sides):
        j=(i+1)%sides
        faces += [[i,j,sides+j],[i,sides+j,sides+i]]
    for i in range(1,sides-1): faces += [[0,i+1,i],[sides,sides+i,sides+i+1]]
    T=np.eye(4);T[:3,3]=center
    return Mesh(name,object_id,np.array(verts,float),np.array(faces),T)


def blade_mesh(object_id,cfg):
    radii=np.array([1.2,2.5,5.,9.,13.,15.])*cfg['blade_length_m']/15.
    half=np.array([.45,1.,1.2,.9,.5,.08])
    sweep=np.array([0.,0.,.1,.3,.5,.6])
    h=cfg['blade_thickness_m']/2
    vertices=[]
    for r,c,s in zip(radii,half,sweep):
        vertices.extend([[r,-h,s-c],[r,-h,s+c],[r,h,s+c],[r,h,s-c]])
    faces=[]
    for k in range(len(radii)-1):
        for j in range(4):
            a,b=k*4+j,k*4+(j+1)%4
            c,d=b+4,a+4
            faces.extend([[a,b,c],[a,c,d]])
    faces.extend([[0,2,1],[0,3,2],[20,21,22],[20,22,23]])
    return Mesh('blade_'+chr(64+object_id),object_id,np.array(vertices),np.array(faces),np.eye(4))


def make_meshes(cfg):
    hub=np.array(cfg['hub_world_m']);z=hub[2]
    meshes=[blade_mesh(i,cfg) for i in (1,2,3)]
    meshes += [cylinder('hub',4,hub+[0,-.8,0],1.4,1.4,1.7,'y'),
               cylinder('shaft',5,hub+[0,.9,0],.45,.45,3.5,'y'),
               cylinder('tower',6,[0,2,0],1.3,.8,z,'z'),
               box('external_occluder',7,[5.,-6.5,z-9],[6.2,-5.5,z+9]),
               box('ground',8,[-60,-60,-.2],[60,60,0]),
               box('nacelle',9,[-1.,1.,z-1.],[1.,5.,z+1.])]
    return meshes


def geometry(meshes,theta,cfg):
    world=[];local=[];ids=[];faces_meta=[]
    for mesh in meshes:
        T=mesh.placement.copy()
        if mesh.object_id<=3:
            T[:3,:3]=ry(theta+(mesh.object_id-1)*2*np.pi/3)
            T[:3,3]=cfg['hub_world_m']
        world.append(transform(mesh.vertices,T)[mesh.faces])
        local.append(mesh.vertices[mesh.faces])
        ids.extend([mesh.object_id]*len(mesh.faces))
        faces_meta.extend([(mesh.object_id,i) for i in range(len(mesh.faces))])
    return dict(world=np.concatenate(world),local=np.concatenate(local),object_id=np.array(ids),face_meta=np.array(faces_meta))


def barycentric3(point,triangle):
    a,b,c=triangle
    v0,v1,v2=b-a,c-a,np.asarray(point)-a
    gram=np.array([[v0@v0,v0@v1],[v0@v1,v1@v1]])
    if abs(np.linalg.det(gram))<1e-12: return None
    uv=np.linalg.solve(gram,[v2@v0,v2@v1]);w=np.r_[1-uv.sum(),uv]
    if np.linalg.norm(w@triangle-point)>1e-8 or w.min()<-1e-9: return None
    return w


def material_bindings(meshes,cfg):
    point=np.array(cfg['material_local_xyz_m']);rows=[]
    for mesh in meshes[:3]:
        matches=[(i,barycentric3(point,mesh.vertices[f])) for i,f in enumerate(mesh.faces)]
        face,w=next((i,w) for i,w in matches if w is not None)
        rows.append(dict(material_point_id=mesh.name+':front_mid',blade_id=mesh.name,object_id=mesh.object_id,
            local_xyz_m=point.tolist(),triangle_id=face,barycentric=w.tolist()))
    return rows


def material_world(bindings,theta,cfg):
    return np.stack([ry(theta+(b['object_id']-1)*2*np.pi/3)@np.asarray(b['local_xyz_m'])+cfg['hub_world_m'] for b in bindings])


def clip_polygon(camera,local,near,far):
    pairs=[(a,b) for a,b in zip(camera,local)]
    for plane,sign in [(near,1),(far,-1)]:
        result=[]
        for i,(cur,attr) in enumerate(pairs):
            prev,pa=pairs[i-1]
            ci=(cur[2]-plane)*sign>=0;pi=(prev[2]-plane)*sign>=0
            if ci!=pi:
                u=(plane-prev[2])/(cur[2]-prev[2])
                result.append((prev+u*(cur-prev),pa+u*(attr-pa)))
            if ci: result.append((cur,attr))
        pairs=result
        if not pairs: break
    return pairs


def surface_color(object_id,local,normal):
    palette={1:[211,162,115],2:[113,189,192],3:[194,157,211],4:[139,153,165],5:[84,98,110],
             6:[192,201,207],7:[224,131,49],8:[103,130,110],9:[170,181,193]}
    color=np.broadcast_to(np.array(palette.get(object_id,[180,180,180]),float),(*local.shape[:-1],3)).copy()
    if object_id<=3:
        r,s=local[...,0],local[...,2]
        grain=10*np.sin(r*89.3+s*111.7+object_id*17)*np.sin(r*33.1-s*75.3)
        stripe=(np.mod(r*2.8+s*.9,1)<.09)
        checker=np.mod(np.floor(r*6)+np.floor(s*8),2)
        color+=grain[...,None]+checker[...,None]*6
        color[stripe]*=.6
        # Procedural surface texture only, never per-frame GT overlays.
        distance=np.sqrt((r-10.5)**2+(s-.4)**2)
        color[(distance>.15)&(distance<.23)]=[40,49,55]
        color[distance<.07]=[244,232,178]
    elif object_id==7:
        color[np.mod(local[...,2],1.5)<.25]*=.55
    intensity=.5+.5*abs(unit(normal)@unit([-.25,-.75,.6]))
    return np.clip(color*intensity,0,255).astype(np.uint8)


def rasterize(scene,T,K,size,near=.1,far=200.,rgb=True):
    """Perspective-correct CPU triangle rasterization at integer pixel centers."""
    width,height=size
    depth=np.full((height,width),np.inf)
    instance=np.zeros((height,width),np.int16)
    primitive=np.full((height,width),-1,np.int32)
    image=np.full((height,width,3),[183,207,222],np.uint8) if rgb else None
    camera=transform(scene['world'],T)
    for primitive_id,(triangle,local,world,oid) in enumerate(zip(camera,scene['local'],scene['world'],scene['object_id'])):
        polygon=clip_polygon(triangle,local,near,far)
        if len(polygon)<3: continue
        normal=np.cross(world[1]-world[0],world[2]-world[0])
        if np.linalg.norm(normal)<1e-12: continue
        for j in range(1,len(polygon)-1):
            xyz=np.stack([polygon[k][0] for k in (0,j,j+1)])
            attrs=np.stack([polygon[k][1] for k in (0,j,j+1)])
            uv=project(xyz,K)
            x0=max(0,int(np.ceil(uv[:,0].min())));x1=min(width-1,int(np.floor(uv[:,0].max())))
            y0=max(0,int(np.ceil(uv[:,1].min())));y1=min(height-1,int(np.floor(uv[:,1].max())))
            if x0>x1 or y0>y1: continue
            x,y=np.meshgrid(np.arange(x0,x1+1),np.arange(y0,y1+1))
            a,b,c=uv
            denominator=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(denominator)<1e-12: continue
            w0=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/denominator
            w1=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/denominator
            weights=np.stack([w0,w1,1-w0-w1],axis=-1)
            inside=(weights>=-1e-9).all(-1)
            inv=weights/xyz[:,2]
            invsum=inv.sum(-1)
            z=np.divide(1,invsum,out=np.full_like(invsum,np.inf),where=invsum>0)
            old=depth[y0:y1+1,x0:x1+1]
            update=inside&(z<old)&(z>=near)&(z<=far)
            if not update.any(): continue
            old[update]=z[update]
            instance[y0:y1+1,x0:x1+1][update]=oid
            primitive[y0:y1+1,x0:x1+1][update]=primitive_id
            if rgb:
                attributes=(inv[update]@attrs)*z[update,None]
                image[y0:y1+1,x0:x1+1][update]=surface_color(int(oid),attributes,normal)
    return dict(rgb=image,depth_m=depth.astype(np.float32),instance_id=instance,primitive_id=primitive)


def raycast(origin,direction,triangles,tmin=1e-6,tmax=np.inf):
    """Independent Möller–Trumbore intersection, no projection/raster buffer access."""
    direction=unit(direction);origin=np.asarray(origin)
    a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a
    p=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,p)
    valid=abs(det)>1e-10
    inv=np.divide(1,det,out=np.zeros_like(det),where=valid)
    s=origin-a
    u=np.einsum('ij,ij->i',s,p)*inv
    q=np.cross(s,e1)
    v=(q@direction)*inv
    distance=np.einsum('ij,ij->i',e2,q)*inv
    valid&=(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(distance>=tmin)&(distance<=tmax)
    candidates=np.where(valid,distance,np.inf)
    index=int(np.argmin(candidates))
    return (float(candidates[index]),index) if np.isfinite(candidates[index]) else (None,None)


def visibility(world,object_ids,scene,T,K,size,near,far,tolerance=.003,annotation_known=None):
    xyz=transform(world,T);uv=project(xyz,K)
    C=np.linalg.inv(T)[:3,3];R=T[:3,:3]
    n=len(world);known=np.ones(n,bool) if annotation_known is None else np.asarray(annotation_known,bool)
    states=np.full(n,'unknown',dtype='<U12');reasons=np.full(n,'annotation_unknown',dtype='<U28')
    blocker=np.full(n,0,np.int16);ray_depth=np.full(n,np.nan)
    for p in range(n):
        if not known[p]: continue
        z=xyz[p,2]
        if z<near or z>far or not np.isfinite(uv[p]).all() or (uv[p]<0).any() or (uv[p]>np.array(size)-1).any():
            states[p]='out_of_view'
            reasons[p]='behind_camera' if z<=0 else 'depth_clip' if z<near or z>far else 'outside_image'
            continue
        direction=unit(world[p]-C);forward=(R@direction)[2]
        distance,index=raycast(C,direction,scene['world'],near/forward,far/forward)
        if distance is None:
            reasons[p]='missing_surface_intersection';continue
        ray_depth[p]=distance*forward
        target_distance=np.linalg.norm(world[p]-C)
        blocker[p]=scene['object_id'][index]
        if distance<target_distance-tolerance:
            states[p]='occluded'
            reasons[p]='rotor_self_occlusion' if blocker[p] in (1,2,3) else 'external_occlusion' if blocker[p]==7 else 'structure_occlusion'
        elif abs(distance-target_distance)<=tolerance:
            states[p]='visible';reasons[p]='surface_visible'
        else:
            reasons[p]='surface_geometry_mismatch'
    return dict(world_xyz_m=np.asarray(world),camera_xyz_m=xyz,tracks_view=uv,depth_m=xyz[:,2],
        ray_nearest_depth_m=ray_depth,state=states,reason=reasons,blocker_object_id=blocker,
        visibility=states=='visible',visibility_known=states!='unknown',position_known=known,
        inside_view=(states!='out_of_view')&known)


def validate_depth_samples(scene,T,K,buffers,pixels,near,far):
    """Trace integer-center camera rays and compare independent z and object IDs."""
    C=np.linalg.inv(T)[:3,3];R=T[:3,:3];Ki=np.linalg.inv(K);errors=[]
    for x,y in pixels:
        direction=unit(R.T@(Ki@np.array([x,y,1.])))
        forward=(R@direction)[2]
        d,index=raycast(C,direction,scene['world'],near/forward,far/forward)
        stored=float(buffers['depth_m'][y,x])
        if d is None:
            if np.isfinite(stored): raise ValueError('Raster hit absent from independent ray')
            continue
        z=d*forward
        if abs(stored-z)>max(3e-4,1e-5*z): raise ValueError(f'Independent depth disagreement: {stored} vs {z}')
        if buffers['instance_id'][y,x]!=scene['object_id'][index]: raise ValueError('Independent instance disagreement')
        errors.append(abs(stored-z))
    return max(errors,default=0.)
