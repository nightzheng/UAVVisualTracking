"""Mesh-backed photo footprint evaluation. This module is evaluator-side."""
from pathlib import Path
import json
import sys
import numpy as np

from .common import ARCH as ARCHIVE
REF = ARCHIVE / 'simulator_migration/reference_model'
from .scene3d import look_at, rasterize, ry


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fraction(mask, domain, weights):
    total = weights[domain].sum()
    return float(weights[mask & domain].sum() / total) if total else 0.


def windows(mission):
    count = mission['desired_photos_per_blade']
    lower, upper = mission['allowed_photos_per_blade']
    lo, hi = mission['radial_range']
    overlap = mission['span_overlap_fraction']
    if not (type(count) is int and 10 <= lower <= count <= upper <= 15 and 0 <= lo < hi <= 1 and 0 <= overlap < 1):
        raise ValueError('Invalid photo count, radial range or overlap')
    length = (hi-lo)/(1+(count-1)*(1-overlap))
    return [(lo+i*length*(1-overlap), min(hi, lo+i*length*(1-overlap)+length)) for i in range(count)]


def barycentric_samples(n):
    """Centroids of n² equal-area children of the unit triangle."""
    uv = []
    for i in range(n):
        for j in range(n-i):
            uv.append(((i+1/3)/n, (j+1/3)/n))
            if i+j<n-1:
                uv.append(((i+2/3)/n, (j+2/3)/n))
    uv = np.asarray(uv)
    return np.c_[1-uv.sum(axis=1), uv]


class Surface:
    def __init__(self):
        self.mesh_path = REF/'models/ref_geometry.npz'
        with np.load(self.mesh_path, allow_pickle=False) as data:
            self.vertices, self.faces = data['blade_vertices'], data['blade_faces']
            self.fixed = np.concatenate([data[k][data[k[:-9]+'_faces']] for k in data.files
                                         if k.endswith('_vertices') and not k.startswith('blade')])
        meta = json.loads((REF/'models/ref_metadata.json').read_text('utf8'))
        self.hub = np.array(meta['hub_world'])
        self.tilt = rx(meta['shaft_tilt_rad'])
        cone = rx(np.deg2rad(meta['cone_deg']))
        basis = np.array([[0,1,0],[-1,0,0],[0,0,1]])
        width = meta['section_samples']
        rings = self.vertices.reshape(-1,width,3)
        eta, xi, suction = [], [], []
        for ring, section in zip(rings,meta['checks'],strict=True):
            tangent = section['tangent']
            align = ry(np.arctan2(tangent[0],tangent[2])) @ rx(-np.arcsin(tangent[1]))
            twist = np.deg2rad(section['twist_deg']) + np.arctan2(align[0,1],align[0,0])
            section_rotation = cone @ basis @ align @ rz(twist)
            chord = ring @ section_rotation[:,1]
            xi.extend((chord-chord.min())/np.ptp(chord))
            eta.extend([section['eta']]*width)
            suction.append(section_rotation[:,0])
        self.eta, self.xi = np.asarray(eta), np.asarray(xi)
        self.triangles = self.vertices[self.faces]
        cross = np.cross(self.triangles[:,1]-self.triangles[:,0], self.triangles[:,2]-self.triangles[:,0])
        norm = np.linalg.norm(cross,axis=1)
        # Resolve winding by comparing side faces with the local ring interior.
        ring_ids = self.faces//width
        interior = rings.mean(axis=1)[ring_ids].mean(axis=1)
        exterior = self.triangles.mean(axis=1)-interior
        wall = np.ptp(ring_ids,axis=1)>0
        winding = np.sign(np.median(np.sum(cross[wall]*exterior[wall],axis=1)))
        self.normals = cross*winding/np.maximum(norm[:,None],1e-20)
        self.areas = norm/2
        facing_suction = (self.normals*np.asarray(suction)[ring_ids].mean(axis=1)).sum(axis=1)>0
        self.side_ids = np.flatnonzero(wall & facing_suction & (norm>1e-12))

    def rotation(self, theta_deg, blade):
        return self.tilt @ ry(np.deg2rad(theta_deg+blade*120))

    def sample(self, subdivisions):
        bary = barycentric_samples(subdivisions)
        ids = self.side_ids
        return dict(points=(bary@self.triangles[ids]).reshape(-1,3),
                    eta=(self.eta[self.faces[ids]]@bary.T).ravel(),
                    xi=(self.xi[self.faces[ids]]@bary.T).ravel(),
                    normals=np.repeat(self.normals[ids],len(bary),axis=0),
                    face_ids=np.repeat(ids,len(bary)), barycentric=np.tile(bary,(len(ids),1)),
                    area=np.repeat(self.areas[ids]/len(bary),len(bary)))

    def raster(self, theta, T, K, size):
        blades = [self.triangles@self.rotation(theta,b).T+self.hub for b in range(3)]
        world = np.concatenate([*blades,self.fixed])
        ids = np.repeat([1,2,3,4],[len(self.faces)]*3+[len(self.fixed)])
        return rasterize(dict(world=world,local=world,object_id=ids),T,K,size,far=1000,rgb=False)

    def target(self, samples, mission, window):
        lo, hi = window
        c0, c1 = mission['chord_range']
        return (samples['eta']>=lo)&(samples['eta']<=hi)&(samples['xi']>=c0)&(samples['xi']<=c1)

    def ideal_pose(self,samples,mission,window,theta,blade,distance):
        mask = self.target(samples,mission,window)
        center = np.average(samples['points'][mask],axis=0,weights=samples['area'][mask])
        R = self.rotation(theta,blade)
        target = center@R.T+self.hub
        position = target-distance*self.tilt[:,1]
        return position,look_at(position,target)

    def footprint(self, samples, cfg, window, theta, blade, position, T, K, buffers, exposure_motion=None):
        c = cfg['camera']
        domain = self.target(samples,cfg['mission'],cfg['mission']['radial_range'])
        patch = self.target(samples,cfg['mission'],window)
        R = self.rotation(theta,blade)
        xyz = samples['points']@R.T+self.hub
        cam = xyz@T[:3,:3].T+T[:3,3]
        uv = (cam@K.T)[:,:2]/np.where(np.abs(cam[:,2:])>1e-9,cam[:,2:],1e-9)
        limit = np.array([c['width']-1,c['height']-1])
        margin = np.minimum(uv,limit-uv).min(axis=1)
        in_frame = (cam[:,2]>.1)&(margin>=0)
        ij = np.rint(np.clip(uv,0,limit)).astype(int)
        depth = buffers['depth_m'][ij[:,1],ij[:,0]]
        instance = buffers['instance_id'][ij[:,1],ij[:,0]]
        direction = position-xyz
        ranges = np.linalg.norm(direction,axis=1)
        incidence = ((samples['normals']@R.T)*direction).sum(axis=1)/np.maximum(ranges,1e-9)
        visible = in_frame & (instance==blade+1) & (abs(depth-cam[:,2])<=cfg['evaluation']['depth_tolerance_m'])
        covered = visible & domain & (margin>=c['min_edge_margin_px']) & (incidence>=np.cos(np.deg2rad(c['max_incidence_deg'])))
        weights = samples['area']
        roi = uv[patch]
        bbox = np.r_[roi.min(axis=0),roi.max(axis=0)]
        centroid = np.average(roi,axis=0,weights=weights[patch])
        # Width perpendicular to the projected span in three radial slices.
        eta = samples['eta'][patch]
        edges = np.linspace(*window,4)
        slices = [roi[(eta>=a)&(eta<=b)] for a,b in zip(edges[:-1],edges[1:])]
        span = slices[-1].mean(axis=0)-slices[0].mean(axis=0) if all(len(s) for s in slices) else np.zeros(2)
        tangent = np.array([-span[1],span[0]])/max(np.linalg.norm(span),1e-9)
        # Use chord-stratified, visible samples for width support, not the long bbox dimension.
        widths = []
        for a,b in zip(edges[:-1],edges[1:]):
            valid = patch & visible & (samples['eta']>=a) & (samples['eta']<=b)
            widths.append(float(np.ptp(uv[valid]@tangent)) if np.any(valid) else 0.)
        result = dict(bbox_px=bbox.tolist(), center_error_px=float(np.linalg.norm(centroid-K[:2,2])),
                      in_frame_area_fraction=fraction(in_frame,patch,weights),
                      visible_area_fraction=fraction(visible,patch,weights),
                      patch_coverage=fraction(covered,patch,weights),
                      roi_scale_ratio=float(np.prod(bbox[2:]-bbox[:2])/(c['width']*c['height'])),
                      blade_pixel_ratio=float(np.mean((buffers['instance_id']>0)&(buffers['instance_id']<4))),
                      projected_min_chord_px=min(widths), edge_margin_px=float(margin[patch].min()),
                      incidence_deg=float(np.rad2deg(np.arccos(np.clip(np.average(incidence[patch],weights=weights[patch]),-1,1)))),
                      distance_m=float(np.average(ranges[patch],weights=weights[patch])),
                      exposure_motion_px=exposure_motion, covered_area_m2=float(weights[covered].sum()))
        result['geometric_pass'] = bool(result['in_frame_area_fraction']>=c['min_in_frame_ratio']
            and result['patch_coverage']>=c['min_patch_area_fraction'] and result['center_error_px']<=c['max_center_error_px']
            and result['edge_margin_px']>=c['min_edge_margin_px'] and result['projected_min_chord_px']>=c['min_projected_chord_px']
            and (exposure_motion is None or exposure_motion<=c['max_exposure_motion_px']))
        return result,covered
