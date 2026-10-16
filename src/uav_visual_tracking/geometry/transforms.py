"""Known 2D transform application and gap propagation, not egomotion recovery."""
import numpy as np


def checked_matrix(matrix):
    h=np.asarray(matrix,dtype=float)
    if h.shape!=(3,3) or not np.isfinite(h).all() or np.max(np.abs(h))==0:
        raise ValueError('Transform must be a nonzero finite 3x3 homogeneous matrix')
    return h/np.max(np.abs(h))


def inverse_transform(matrix, condition_limit=1e10):
    h=checked_matrix(matrix)
    if np.linalg.cond(h)>condition_limit:raise ValueError('Singular or numerically ill-conditioned transform')
    return np.linalg.inv(h)


def project_points(points, matrix, valid=None):
    xy=np.asarray(points,dtype=float)
    if xy.ndim<2 or xy.shape[-1]!=2:raise ValueError('Points must have shape (...,2)')
    mask=np.isfinite(xy).all(-1)
    if valid is not None:
        v=np.asarray(valid)
        if v.shape!=xy.shape[:-1] or v.dtype!=np.bool_:raise ValueError('Validity mask must be aligned bool')
        mask &= v
    homogeneous=np.concatenate([xy,np.ones((*xy.shape[:-1],1))],axis=-1)
    with np.errstate(invalid='ignore',divide='ignore'):
        mapped=homogeneous@checked_matrix(matrix).T
        mask &= np.isfinite(mapped).all(-1)&(np.abs(mapped[...,2])>1e-10)
        output=np.full_like(xy,np.nan)
        output[mask]=mapped[mask,:2]/mapped[mask,2,None]
    return output,mask


def compensate_known(points_current, reference_to_current, valid=None):
    return project_points(points_current,inverse_transform(reference_to_current),valid)


def accumulate_reference(relative_previous_to_current, step_valid):
    relative=np.asarray(relative_previous_to_current,dtype=float)
    v=np.asarray(step_valid)
    if relative.ndim!=3 or relative.shape[1:]!=(3,3) or v.shape!=(len(relative),) or v.dtype!=np.bool_:
        raise ValueError('Relative matrices and step-valid mask must align')
    cumulative=np.full((len(relative)+1,3,3),np.nan);cumulative[0]=np.eye(3)
    valid=np.zeros(len(cumulative),bool);valid[0]=True
    for t,h in enumerate(relative,1):
        if not valid[t-1] or not v[t-1]:continue
        try:
            candidate=checked_matrix(h)@cumulative[t-1]
            inverse_transform(candidate)
        except ValueError:continue
        cumulative[t]=checked_matrix(candidate);valid[t]=True
    return cumulative,valid
