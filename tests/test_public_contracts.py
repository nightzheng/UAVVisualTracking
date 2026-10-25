"""CPU engineering contracts. Test doubles are not model inference evidence."""
import csv,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from uav_visual_tracking.data.video import VideoClip
from uav_visual_tracking.data.queries import Queries
from uav_visual_tracking.data.results import save_result,load_result
from uav_visual_tracking.tracking.base import Prediction
from uav_visual_tracking.tracking.interface import processing_to_original,prepare_tapir_input,decode_tapir_output,validate_rgb_queries
from uav_visual_tracking.tracking.cotracker3 import capture_offline_scores
from uav_visual_tracking import demo


def fixture(crop=True,t=5,n=1):
    video=dict(original_resolution=[480,360],processing_resolution=[180,260],fps=24,cfr_observed=True,time_basis='controlled_fixture',color_space='RGB')
    if crop:video.update(source_processing_resolution=[480,360],crop_roi_xyxy=[295,65,475,325])
    clip=VideoClip(np.zeros((t,260,180,3),np.uint8),np.arange(10,10+t),np.arange(t)/24+10/24,video)
    q=Queries(np.array([[min(p,t-2),40+p*10,100] for p in range(n)],np.float32),[f'p{p}' for p in range(n)],['manual']*n)
    xy=np.tile(q.array[None,:,1:],(t,1,1));vis=np.ones((t,n),bool)
    scores=dict(visibility_probability=np.full((t,n),.8),confidence_probability=np.full((t,n),.6),
        visibility_score=np.full((t,n),.8),occlusion_logit=np.full((t,n),-2.),expected_dist_logit=np.full((t,n),-2.))
    meta=dict(model=dict(mode='offline_full_clip',commit='controlled_fixture',checkpoint_sha256='controlled_fixture'),generated_at_utc='controlled_fixture',result_available_after_source_frame=int(clip.frame_indices[-1]))
    prediction=Prediction(xy.copy(),vis.copy(),meta,scores)
    data=dict(tracks=xy.copy(),visibility=vis.copy(),queries=q.array,frame_indices=clip.frame_indices,point_ids=np.array(q.point_ids),timestamps=clip.timestamps,
        **{'raw_score__'+k:v for k,v in scores.items()})
    manifest=dict(schema_version=2,video_id='controlled_fixture',video=video,prediction=meta,config_sha256='controlled_fixture',point_ids=q.point_ids,queries=q.array.tolist())
    truth=processing_to_original(xy,video)
    gq=q.array.astype(float).copy();gq[:,1:]=processing_to_original(q.array[:,1:],video)
    gt=dict(video_id=np.array('controlled_fixture'),tracks_original=truth,queries_original_txy=gq,visibility=vis.copy(),visibility_known=vis.copy(),position_known=vis.copy(),frame_indices=clip.frame_indices,point_ids=np.array(q.point_ids),timestamps=clip.timestamps)
    return clip,q,prediction,data,manifest,gt


class Contracts(unittest.TestCase):
    def test_schema2_crop_csv_and_nonzero_multiquery(self):
        c,q,p,d,m,g=fixture(n=2)
        with tempfile.TemporaryDirectory() as td:
            save_result(td,c,q,p,m);saved,sm=load_result(td)
            self.assertEqual(int(saved['schema_version']),2)
            np.testing.assert_allclose(saved['tracks_original'],g['tracks_original'])
            with (Path(td)/'tracks.csv').open(encoding='utf8') as stream:
                rows=list(csv.DictReader(stream))
            for row in rows:
                if row['active']=='1':
                    i=int(row['clip_frame_index']);j=q.point_ids.index(row['point_id'])
                    self.assertEqual(float(row['x_original']),saved['tracks_original'][i,j,0])
            with self.assertRaises(FileExistsError):save_result(td,c,q,p,m)

    def test_schema2_requires_raw_scores(self):
        c,q,p,d,m,g=fixture();p.raw_scores={}
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):save_result(td,c,q,p,m)

    def test_manifest_tamper_detected(self):
        c,q,p,d,m,g=fixture()
        with tempfile.TemporaryDirectory() as td:
            save_result(td,c,q,p,m)
            m['schema_version']=1;(Path(td)/'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):load_result(td)

    def test_tapir_input_layout_range_and_raster(self):
        c,q,p,d,m,g=fixture();frames,prompt=prepare_tapir_input(c,q,(256,264))
        self.assertEqual(frames.shape,(1,5,256,264,3));self.assertEqual(frames.dtype,np.float32)
        self.assertTrue((frames==-1).all())
        np.testing.assert_allclose(prompt[0,0],[0,100*256/260,40*264/180],atol=1e-5)

    def test_tapir_raw_logits_continuous_score_and_no_clipping(self):
        raw=dict(tracks=np.full((1,1,3,2),-10.),occlusion=np.array([[[0.,1000.,-1000.]]]),expected_dist=np.zeros((1,1,3)))
        xy,vis,fields=decode_tapir_output(raw,[180,260],(256,256),3,1)
        np.testing.assert_allclose(fields['visibility_score'][:,0],[.25,0,.5])
        self.assertFalse(vis.any());self.assertTrue((xy<0).all())
        np.testing.assert_array_equal(fields['occlusion_logit'][:,0],raw['occlusion'][0,0])

    def test_cotracker_score_observer_clones_and_restores(self):
        class Tensor:
            def __init__(self,a):self.a=np.asarray(a);self.shape=self.a.shape;self.ndim=self.a.ndim
            def __getitem__(self,k):return Tensor(self.a[k])
            def detach(self):return self
            def clone(self):return Tensor(self.a.copy())
        score=Tensor(np.full((1,5,2),.8))
        core=SimpleNamespace(forward=lambda:('xy',score,score,'other'))
        original=core.forward
        with capture_offline_scores(core,1) as fields:
            returned=core.forward();score.a[:]=0
            self.assertEqual(fields['visibility_probability'].shape,(5,1))
            self.assertTrue((fields['visibility_probability'].a==.8).all())
            self.assertIs(returned[1],score)
            with self.assertRaises(RuntimeError):core.forward()
        self.assertIs(core.forward,original)
        with self.assertRaises(RuntimeError):
            with capture_offline_scores(core,1):raise RuntimeError('controlled failure')
        self.assertIs(core.forward,original)

    def test_both_demo_backends_save_and_reread_schema2(self):
        for backend in ['B0','B1']:
            c,q,p,d,m,g=fixture()
            with tempfile.TemporaryDirectory() as td:
                root=Path(td)
                for name in ['video.bin','weight.bin']: (root/name).write_bytes(b'controlled test asset')
                (root/'query.json').write_text(json.dumps(dict(queries_txy=q.array.tolist(),point_ids=q.point_ids)))
                config=dict(backend=backend,video=str(root/'video.bin'),checkpoint=str(root/'weight.bin'),source=td,queries=str(root/'query.json'),output=str(root/'result'))
                seen=[]
                def factory(model):seen.append(model);return SimpleNamespace(identity=p.metadata['model'],predict=lambda a,b:p)
                with patch.object(demo,'read_clip',return_value=c):demo.run(config,factory)
                saved,_=load_result(root/'result');self.assertEqual(int(saved['schema_version']),2)
                self.assertTrue(seen[0]['export_raw_scores'])
                self.assertEqual(len([k for k in saved if k.startswith('raw_score__')]),5)
                with self.assertRaises(FileExistsError):demo.run(config,factory)

    def test_external_assets_required(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(FileNotFoundError,'external video'):
                demo.run(dict(output=str(Path(td)/'o'),video=str(Path(td)/'absent')))

    def test_overlay_decodes_all_frames_and_rejects_vfr(self):
        import cv2
        c,q,p,d,m,g=fixture()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'overlay.mp4';r=demo.export_overlay(path,c,q,p)
            self.assertEqual(r['frames'],5)
            cap=cv2.VideoCapture(str(path));self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS),24);cap.release()
            c.timestamps[2]+=.0001
            with self.assertRaisesRegex(ValueError,'CFR'):demo.export_overlay(Path(td)/'vfr.mp4',c,q,p)

    def test_original_queries_crop_mapping(self):
        c,q,p,d,m,g=fixture(n=2)
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'q.json';f.write_text(json.dumps(dict(coordinate_space='original',queries_txy=g['queries_original_txy'].tolist(),point_ids=q.point_ids)))
            loaded=demo.load_queries(f,c);np.testing.assert_allclose(loaded.array,q.array,atol=1e-5)

    def test_time_rgb_and_query_contract(self):
        c,q,p,d,m,g=fixture();c.timestamps[2]=c.timestamps[1]
        with self.assertRaises(ValueError):validate_rgb_queries(c,q)
        c,q,p,d,m,g=fixture();c.metadata['color_space']='BGR'
        with self.assertRaises(ValueError):validate_rgb_queries(c,q)

if __name__=='__main__':unittest.main()
