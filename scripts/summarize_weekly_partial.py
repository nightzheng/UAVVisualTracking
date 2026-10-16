"""Inspect traceable crop runs and create comparable preview cards, no GT."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from uav_visual_tracking.data.results import load_result
from uav_visual_tracking.data.crops import crop_to_source
from uav_visual_tracking.data.video import resize_coordinates
from uav_visual_tracking.common.provenance import sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);args=parser.parse_args()
    output=args.root/'pilot_review';output.mkdir(exist_ok=False)
    protocol=json.loads((args.root/'protocol.json').read_text('utf-8'));rows=[];checks=[]
    for record in protocol['records']:
        directory=args.root/record['run'];data,manifest=load_result(directory)
        if sha256(directory/'manifest.json')!=record['manifest_sha256']:raise ValueError('Manifest changed')
        for name,value in manifest['artifacts_sha256'].items():
            if sha256(directory/name)!=value:raise ValueError('Artifact changed')
        roi=record['roi'];panels=[]
        csv_rows=__import__('csv').DictReader((directory/'tracks.csv').open(encoding='utf-8',newline=''))
        max_original_error=0.
        for t,row in enumerate(csv_rows):
            src=crop_to_source(data['tracks'][t,0],roi)
            expected=resize_coordinates(src,[960,540],[1920,1080])
            stored=np.array([float(row['x_original']),float(row['y_original'])])
            max_original_error=max(max_original_error,float(np.max(np.abs(expected-stored))))
        for frame in [0,28,54,68,119]:
            image=cv2.imread(str(args.root/(record['view']+'_raw')/f'{frame:06d}.png'))
            xy=data['tracks'][frame,0];h,w=image.shape[:2];vis=bool(data['visibility'][frame,0])
            if np.isfinite(xy).all():cv2.circle(image,tuple(np.rint(xy).astype(int)),5,(0,255,255),-1 if vis else 1)
            scale=min(320/w,190/h);resized=cv2.resize(image,(round(w*scale),round(h*scale)))
            card=np.zeros((242,320,3),np.uint8);card[52:52+resized.shape[0],:resized.shape[1]]=resized
            cv2.putText(card,f'{record["run"]} f={frame}',(4,16),0,.4,(255,255,255),1)
            cv2.putText(card,f'p001 pred visible={int(vis)}; NO GT',(4,33),0,.4,(255,255,255),1)
            panels.append(card)
        rows.append(np.hstack(panels))
        checks.append(dict(run=record['run'],frames=len(data['frame_indices']),points=len(data['point_ids']),
            csv_original_coordinate_max_error=max_original_error,artifacts_verified=True))
    cv2.imwrite(str(output/'partial_pilot_contact.jpg'),np.vstack(rows))
    write_json(output/'verification.json',dict(checks=checks,protocol_sha256=sha256(args.root/'protocol.json'),
        display='preview scales differ by view; inference inputs are unresized crops',
        accuracy_claimed=False,independent_source_sequences=1))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
