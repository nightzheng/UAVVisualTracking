"""Oracle 2D transform fixtures: no image-based camera estimation or real GT."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from uav_visual_tracking.common.provenance import sha256,write_json
from uav_visual_tracking.geometry.transforms import compensate_known,project_points


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    image=cv2.imread(str(args.image))
    if image is None:raise ValueError('Source image unreadable')
    height,width=image.shape[:2]
    xx,yy=np.meshgrid(np.linspace(100,width-100,5),np.linspace(80,height-80,4))
    reference=np.column_stack([xx.ravel(),yy.ravel()]);records=[];panels=[]
    for kind in ['translation','partial_affine','homography']:
        out=args.output/kind;out.mkdir();sequence=[];row=[]
        for frame in range(6):
            angle=np.deg2rad(frame*1.2);scale=1+.015*frame
            a=scale*np.cos(angle);b=scale*np.sin(angle)
            h=np.array([[1.,0.,frame*8.],[0.,1.,-frame*3.],[0.,0.,1.]])
            if kind!='translation':
                cx,cy=width/2,height/2
                h=np.array([[a,-b,cx-a*cx+b*cy+frame*3],[b,a,cy-b*cx-a*cy-frame*2],[0.,0.,1.]])
            if kind=='homography':h[2,:2]=[frame*1e-5,-frame*8e-6]
            # Independent OpenCV projection provides mathematical expected points.
            current=cv2.perspectiveTransform(reference[None].astype(np.float64),h)[0]
            ours,finite=project_points(reference,h)
            back,valid=compensate_known(current,h,np.ones(len(reference),bool))
            error=np.linalg.norm(back-reference,axis=1)
            visible=finite&(current[:,0]>=0)&(current[:,0]<=width-1)&(current[:,1]>=0)&(current[:,1]<=height-1)
            transformed=cv2.warpPerspective(image,h,(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            mask=cv2.warpPerspective(np.full((height,width),255,np.uint8),h,(width,height),flags=cv2.INTER_NEAREST)
            cv2.imwrite(str(out/f'image_{frame:03d}.png'),transformed);cv2.imwrite(str(out/f'valid_image_mask_{frame:03d}.png'),mask)
            np.savez_compressed(out/f'calibration_{frame:03d}.npz',reference_xy=reference,current_xy=current,compensated_xy=back,
                numerical_valid=valid,in_image=visible,reference_to_current=h,
                coordinate_semantics='generated calibration points; not blade material GT')
            rec=dict(fixture_frame=frame,timestamp_seconds=frame/30,transform_direction='reference image -> current image',
                reference_to_current=h.tolist(),current_to_reference=np.linalg.inv(h).tolist(),
                calibration_points=len(reference),max_inverse_error_pixels=float(error.max()),
                max_forward_difference_vs_opencv=float(np.max(np.abs(ours-current))),
                numerical_valid_count=int(valid.sum()),in_image_calibration_points=int(visible.sum()))
            sequence.append(rec)
            display=transformed.copy()
            for point in current[visible]:cv2.circle(display,tuple(np.rint(point).astype(int)),5,(0,255,255),1)
            display=cv2.resize(display,(320,180));banner=np.zeros((42,320,3),np.uint8)
            cv2.putText(banner,f'{kind} fixture {frame}',(4,16),0,.42,(255,255,255),1)
            cv2.putText(banner,'Known 2D transform; NOT real camera GT',(4,34),0,.36,(255,255,255),1)
            row.append(np.vstack([banner,display]))
        write_json(out/'transforms.json',dict(kind=kind,records=sequence,transform_source='program_generated_oracle',
            image_estimation_performed=False,camera_3d_pose=None,world_metric_scale=None))
        records.append(dict(kind=kind,frame_count=6,calibration_points_per_frame=20,
            max_inverse_error_pixels=max(r['max_inverse_error_pixels'] for r in sequence),
            max_forward_difference_vs_opencv=max(r['max_forward_difference_vs_opencv'] for r in sequence)))
        panels.append(np.hstack(row))
    cv2.imwrite(str(args.output/'known_transform_contact.jpg'),np.vstack(panels))
    if any(r['max_inverse_error_pixels']>1e-9 or r['max_forward_difference_vs_opencv']>1e-9 for r in records):
        raise AssertionError('Coordinate fixture failed')
    write_json(args.output/'protocol.json',dict(status='mathematical_contract_passed',records=records,
        source_image_sha256=sha256(args.image),source_original_frame_index=0,
        source_image_role='static development illustration; not independent camera dataset',
        generated_transform_sequences=3,independent_real_camera_sequences=0,known_transform_evaluated_point_frames=360,
        real_camera_motion_estimation_validated=False,real_blade_GT_created=False,
        failure_contract_tests='tests/test_weekly_contracts.py: singular, near-singular, horizon, invalid mask, accumulated gap',
        image_sampling='warpPerspective forward matrix, no WARP_INVERSE_MAP; bilinear image / nearest validity mask'))
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
