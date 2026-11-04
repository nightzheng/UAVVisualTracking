"""Unmarked local RGB contour observations with ephemeral image tracklets."""
import json
import re
import sys
import time
from pathlib import Path
import cv2
import numpy as np


class LocalContour:
    def __init__(self,cfg):
        self.cfg=cfg;self.last_pts=-np.inf;self.prior=None;self.track=0

    def step(self,bgr,pts):
        if pts<=self.last_pts:raise ValueError('Capture PTS must strictly increase')
        if pts-self.last_pts>self.cfg['max_gap_s']:self.prior=None
        self.last_pts=pts
        start=time.perf_counter()
        c=self.cfg
        high=bgr.max(axis=2).astype(np.int16);low=bgr.min(axis=2).astype(np.int16)
        green=bgr[:,:,1].astype(np.int16)-np.maximum(bgr[:,:,0],bgr[:,:,2]).astype(np.int16)
        mask=((high-low<=c['max_channel_range'])&(green<=c['max_green_dominance'])&
              (low>=c['min_intensity'])&(high<=c['max_intensity'])).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        candidates=sorted([i for i in range(1,count) if stats[i,4]>=c['min_component_pixels']],key=lambda i:-stats[i,4])
        out=dict(capture_pts=pts,roi_mask=None,roi_bbox=None,tracked_points=[],image_center_error_px=None,
            roi_scale_ratio=None,blade_pixel_ratio=None,local_direction=None,blade_identity=None,radial_region=None,
            world_phase_deg=None,world_omega_deg_s=None,material_point_id=None,association_status='unknown_physical_identity',
            observation_status='unavailable',tracklet_id=None,quality={})
        selected=np.zeros(mask.shape,bool)
        if candidates and (len(candidates)==1 or stats[candidates[1],4]<c['competing_component_fraction']*stats[candidates[0],4]):
            idx=candidates[0];selected=labels==idx
            x,y,w,h,area=map(int,stats[idx]);yy,xx=np.nonzero(selected);xy=np.c_[xx,yy].astype(float);center=xy.mean(axis=0)
            if self.prior is None or (selected&self.prior).sum()/max((selected|self.prior).sum(),1)<c['min_continuity_iou']:
                self.track+=1
            self.prior=selected
            cov=(xy-center).T@(xy-center)
            axis=np.linalg.eigh(cov)[1][:,-1]
            if axis[0]<0:axis=-axis
            image_center=(np.array([bgr.shape[1],bgr.shape[0]])-1)/2
            out.update(roi_bbox=[x,y,w,h],image_center_error_px=float(np.linalg.norm(center-image_center)),
                roi_scale_ratio=w*h/mask.size,blade_pixel_ratio=area/mask.size,local_direction=axis.tolist(),
                observation_status='relative_roi',tracklet_id=self.track,
                quality=dict(center_px=center.tolist(),component_pixels=area,touches_border=bool(x==0 or y==0 or x+w==bgr.shape[1] or y+h==bgr.shape[0]),
                warning='neutral contour candidate; centroid is not a material point'))
        else:
            self.prior=None
            out['quality']['reason']='no_component' if not candidates else 'competing_components'
        out['compute_ms']=(time.perf_counter()-start)*1000
        return out,selected


def run(spec_path,input_folder,out):
    spec_path,input_folder,out=map(lambda p:Path(p).resolve(),(spec_path,input_folder,out))
    allowed={spec_path,input_folder/'frames.json'}
    repo=Path(__file__).resolve().parents[4]
    current=None
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        import os
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(repo) and path not in allowed and path!=current and not path.is_relative_to(out):
            raise PermissionError('RGB consumer denied input: '+str(path))
    sys.addaudithook(guard)
    cfg=json.loads(spec_path.read_text());rows=json.loads((input_folder/'frames.json').read_text())
    if cfg['allow_future_frames'] or cfg['allow_oracle_identity']:raise ValueError('Forbidden RGB permissions')
    model=LocalContour(cfg);records=[];masks=[];start=time.perf_counter()
    for i,row in enumerate(rows):
        if not re.fullmatch(r'rgb/\d{5}\.png',row['file']):raise ValueError('Invalid current RGB path')
        current=(input_folder/row['file']).resolve()
        if current.parent!=input_folder/'rgb':raise ValueError('Input path escaped RGB directory')
        bgr=cv2.imdecode(np.frombuffer(current.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        current=None
        obs,mask=model.step(bgr,row['pts'])
        obs.update(frame_index=i,available_at=time.perf_counter()-start,availability_clock='offline replay monotonic seconds')
        if obs['roi_bbox'] is not None:obs['roi_mask']=dict(file='roi_masks.npz',index=i,shape=list(mask.shape),encoding='packbits')
        records.append(obs);masks.append(np.packbits(mask))
    (out/'observations.json').write_text(json.dumps(records,indent=2,allow_nan=False),encoding='utf8')
    np.savez_compressed(out/'roi_masks.npz',packed=np.array(masks),shape=mask.shape)


if __name__=='__main__':run(*sys.argv[1:4])
