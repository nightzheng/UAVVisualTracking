"""Analytical fixtures validate engineering logic, never model accuracy."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from uav_visual_tracking.data.video import VideoClip, read_clip, resize_coordinates
from uav_visual_tracking.data.queries import make_queries
from uav_visual_tracking.data.results import save_result, load_result
from uav_visual_tracking.tracking.base import Prediction
from uav_visual_tracking.tracking.cotracker3 import CoTracker3
from uav_visual_tracking.evaluation.metrics import diagnostics, evaluate_gt, evaluate_gt_file
from uav_visual_tracking.visualization.drawing import draw_point
from uav_visual_tracking.cli import run


def clip():
    return VideoClip(np.zeros((4, 48, 64, 3), dtype=np.uint8), np.arange(10, 14), np.arange(10, 14)/20.,
                     dict(original_resolution=[128,96], processing_resolution=[64,48], fps=20,
                          time_basis="opencv_decoded_pts_seconds", cfr_observed=True))


class VideoQueries(unittest.TestCase):
    def test_coordinate_roundtrip_nonuniform_scaling(self):
        points = np.array([[0., 0.], [127.,95.], [20.5,10.5]])
        small = resize_coordinates(points, [128,96], [64,32])
        np.testing.assert_allclose(small[0], [-.25, -1/3])
        np.testing.assert_allclose(resize_coordinates(small, [64,32], [128,96]), points, atol=1e-12)

    def test_manual_space_and_nonzero_query(self):
        q = make_queries(dict(frame=2,points=[[20.5,10.5]],coordinate_space="original"), clip())
        np.testing.assert_allclose(q.array, [[2,10,5]])
        self.assertEqual(q.methods,["manual"])

    def test_invalid_queries(self):
        for config in [dict(points=[]), dict(points=[[1,1],[1,1]]),dict(points=[[np.nan,0]]),
                       dict(points=[[64,0]]),dict(points=[[1,1]],frame=4),dict(points=[[1,1]],frame=1.5),
                       dict(points=[[1,1]],point_ids=["a","b"]),dict(strategy="semantic"),dict(strategy="grid",grid_size=1)]:
            with self.subTest(config=config),self.assertRaises(ValueError): make_queries(config,clip())

    def test_grid_explicit_identity(self):
        q=make_queries(dict(strategy="grid",grid_size=3),clip())
        self.assertEqual(q.array.shape,(9,3)); self.assertEqual(len(set(q.point_ids)),9)
        self.assertTrue(all("no_semantics" in x for x in q.methods))

    def test_decode_clip_and_pts(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"sample.mp4"
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),20.,(128,96))
            self.assertTrue(writer.isOpened())
            for i in range(12):writer.write(np.full((96,128,3),i*10,np.uint8))
            writer.release()
            result=read_clip(path,start_sec=.1,duration=.2,max_frames=4,max_width=64)
            np.testing.assert_array_equal(result.frame_indices,[2,3,4,5])
            self.assertEqual(result.frames.shape,(4,48,64,3))
            np.testing.assert_allclose(result.timestamps,[.1,.15,.2,.25],atol=1e-5)
            self.assertEqual(result.metadata["time_basis"],"opencv_decoded_pts_seconds")

    def test_invalid_video_requests(self):
        with self.assertRaises(ValueError):read_clip("missing",start_sec=-1)
        with self.assertRaises(FileNotFoundError):read_clip("missing")

    def test_invalid_decoder_pts_fallback_is_json_safe(self):
        class Decoder:
            index=0
            def isOpened(self):return True
            def get(self,key):
                return {cv2.CAP_PROP_FPS:20.,3:64,4:48,cv2.CAP_PROP_FRAME_COUNT:3,
                        cv2.CAP_PROP_POS_FRAMES:self.index,cv2.CAP_PROP_POS_MSEC:float('nan')}[key]
            def read(self):
                self.index+=1
                return True,np.zeros((48,64,3),np.uint8)
            def release(self):pass
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'video.mp4';path.touch()
            with patch('uav_visual_tracking.data.video.cv2.VideoCapture',return_value=Decoder()):
                result=read_clip(path,max_frames=3)
            np.testing.assert_allclose(result.timestamps,[0,.05,.1])
            self.assertEqual(result.metadata['time_basis'],'nominal_fps_assumption')
            self.assertEqual(result.metadata['raw_decoder_pts_seconds'],[None,None,None])
            json.dumps(result.metadata,allow_nan=False)


class ResultsMetrics(unittest.TestCase):
    def test_result_roundtrip_and_prequery_csv(self):
        c=clip();q=make_queries(dict(points=[[10,5]],frame=2),c)
        tracks=np.array([[[1,2]],[[2,3]],[[10,5]],[[11,6]]],dtype=np.float32)
        visibility=np.array([[False],[False],[True],[False]])
        metadata=dict(model=dict(mode="offline_full_clip",commit="fixture",checkpoint_sha256="fixture"),
                      generated_at_utc="2026-10-13T00:00:00Z",result_available_after_source_frame=13)
        manifest=dict(video_id="fixture",video_sha256="fixture",config_sha256="fixture",prediction=metadata,video=c.metadata)
        with tempfile.TemporaryDirectory() as directory:
            save_result(directory,c,q,Prediction(tracks,visibility,metadata),manifest)
            stored,m=load_result(directory)
            np.testing.assert_array_equal(stored["tracks"],tracks)
            np.testing.assert_array_equal(stored["visibility"],visibility)
            self.assertEqual(stored["point_ids"].tolist(),q.point_ids)
            import csv
            with (Path(directory)/"tracks.csv").open(encoding="utf-8") as f:rows=list(csv.DictReader(f))
            self.assertEqual(rows[0]["active"],"0");self.assertEqual(rows[0]["x_processing"],"")
            self.assertEqual(rows[2]["x_processing"],"10.0");self.assertEqual(rows[2]["available_after_frame_index"],"13")

    def test_predicted_events_are_not_gt(self):
        tracks=np.array([[[1,1]],[[1,1]],[[1,1]],[[100,1]],[[1,1]]],float)
        visible=np.array([[False],[True],[False],[False],[True]])
        d=diagnostics(tracks,visible,[1],[64,48]);p=d["per_point"][0]
        self.assertFalse(d["gt_available"]);self.assertNotIn("accuracy",d)
        self.assertEqual(p["predicted_invisibility_entries"],1);self.assertEqual(p["predicted_visibility_returns"],1)
        self.assertEqual(p["predicted_exit_events"],1);self.assertEqual(p["predicted_reentry_events"],1)
        self.assertEqual(d["predicted_visible_ratio"],.5)

    def test_nonfinite_diagnostics(self):
        d=diagnostics(np.array([[[np.nan,0]]]),np.ones((1,1),bool),[0],[64,48])
        self.assertEqual(d["nonfinite_active_predictions"],1)

    def test_gt_errors_include_predicted_invisible(self):
        tracks=np.array([[[0,0]],[[3,4]],[[6,8]],[[np.nan,np.nan]]])
        gt=np.zeros_like(tracks);pv=np.array([[True],[False],[True],[False]])
        gv=np.array([[False],[True],[True],[True]]);valid=np.ones((4,1),bool);active=valid.copy()
        m=evaluate_gt(tracks,pv,gt,gv,valid,active)
        self.assertEqual(m["mean_endpoint_error"],7.5);self.assertEqual(m["nonfinite_visible_predictions"],1)
        self.assertEqual(m["localization_coverage"],2/3)
        self.assertEqual(m["visibility"],dict(tp=1,fp=1,fn=2,tn=0,accuracy=.25,precision=.5,recall=1/3,f1=.4))

    def test_empty_gt_has_null_metrics(self):
        x=np.zeros((2,1,2)); v=np.zeros((2,1),bool)
        m=evaluate_gt(x,v,x,v,v,~v)
        self.assertIsNone(m["mean_endpoint_error"]);self.assertIsNone(m["visibility"]["f1"])
        json.dumps(m,allow_nan=False)

    def test_gt_nan_visible_rejected(self):
        x=np.zeros((1,1,2));v=np.ones((1,1),bool);gt=x.copy();gt[0]=np.nan
        with self.assertRaises(ValueError):evaluate_gt(x,v,gt,v,v,v)

    def test_gt_alignment_by_id_and_frame(self):
        data=dict(tracks=np.zeros((2,2,2)),visibility=np.ones((2,2),bool),frame_indices=np.array([10,11]),
                  point_ids=np.array(["a","b"]),queries=np.array([[0,0,0],[0,0,0]]))
        manifest=dict(video_sha256="video",video=dict(processing_resolution=[64,48]))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"gt.npz"
            tracks=np.array([[[6,8],[3,4]],[[0,0],[0,0]]],float)
            np.savez(path,tracks=tracks,visibility=np.ones((2,2),bool),valid=np.ones((2,2),bool),
                     video_sha256="video",coordinate_space="processing",resolution=[64,48],
                     frame_indices=[11,10],point_ids=["b","a"])
            self.assertEqual(evaluate_gt_file(data,manifest,path)["mean_endpoint_error"],3.75)
            manifest["video_sha256"]="wrong"
            with self.assertRaises(ValueError):evaluate_gt_file(data,manifest,path)


class Boundaries(unittest.TestCase):
    def test_window_online_explicitly_unimplemented(self):
        with self.assertRaises(NotImplementedError):CoTracker3(dict(mode="online"))

    def test_no_missing_weight_download(self):
        with self.assertRaises(FileNotFoundError):CoTracker3(dict(checkpoint="absent_scaled_offline.pth"))

    def test_prevent_output_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel=Path(directory)/"sentinel";sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):run(dict(output=directory,video="anything"))
            self.assertEqual(sentinel.read_text(),"keep")

    def test_prediction_visibility_type(self):
        with self.assertRaises(ValueError):Prediction(np.zeros((2,1,2)),np.zeros((2,1),float),{}).validate(2,1)

    def test_visibility_drawing_preserves_hidden_center(self):
        solid=np.zeros((40,40,3),np.uint8);hollow=solid.copy()
        draw_point(solid,[20,20],True,(0,0,255));draw_point(hollow,[20,20],False,(0,0,255))
        self.assertEqual(solid[20,20,2],255);self.assertEqual(hollow[20,20,2],0)


if __name__ == "__main__":unittest.main()
