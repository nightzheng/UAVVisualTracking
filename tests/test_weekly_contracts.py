"""Crop and known transform fixtures; none are material/camera measurement GT."""
import unittest
import cv2
import numpy as np
from uav_visual_tracking.data.video import VideoClip,resize_coordinates
from uav_visual_tracking.data.crops import crop_clip,source_to_crop,crop_to_source,validate_roi
from uav_visual_tracking.geometry.transforms import project_points,compensate_known,inverse_transform,accumulate_reference


class CropContracts(unittest.TestCase):
    def test_crop_pixel_identity_and_timestamps(self):
        frames=np.arange(2*6*8*3,dtype=np.uint8).reshape(2,6,8,3)
        clip=VideoClip(frames,np.array([10,11]),np.array([.5,.55]),dict(processing_resolution=[8,6],original_resolution=[16,12]))
        out=crop_clip(clip,[2,1,7,5])
        np.testing.assert_array_equal(out.frames,frames[:,1:5,2:7]);np.testing.assert_array_equal(out.timestamps,clip.timestamps)
        self.assertEqual(out.metadata['processing_resolution'],[5,4]);self.assertFalse(out.metadata['independent_sequence'])

    def test_crop_source_original_roundtrip(self):
        roi=[520,20,760,260];xy=np.array([[591.,75.],[625.2,145.8]])
        local=source_to_crop(xy,roi);np.testing.assert_allclose(local[0],[71,55])
        source=crop_to_source(local,roi);np.testing.assert_allclose(source,xy)
        original=resize_coordinates(source,[960,540],[1920,1080]);np.testing.assert_allclose(original[0],[1182.5,150.5])

    def test_invalid_roi_rejected(self):
        for roi in [[-1,0,20,20],[0,0,961,540],[0,0,1,2],[1.,1,5,5],[True,0,5,5]]:
            with self.subTest(roi=roi),self.assertRaises(ValueError):validate_roi(roi,[960,540])

    def test_crop_edge_bounds_not_rescaled(self):
        p=np.array([[520.,20.],[759.,259.],[760.,260.]])
        np.testing.assert_allclose(source_to_crop(p,[520,20,760,260]),[[0,0],[239,239],[240,240]])


class KnownTransformContracts(unittest.TestCase):
    def test_translation_direction_and_image_warp(self):
        h=np.array([[1.,0.,7.],[0.,1.,4.],[0.,0.,1.]])
        xy=np.array([[10.,12.]])
        current,valid=project_points(xy,h);np.testing.assert_allclose(current,[[17,16]])
        recovered,valid=compensate_known(current,h,valid);np.testing.assert_allclose(recovered,xy)
        image=np.zeros((40,50),np.uint8);image[12,10]=255
        warped=cv2.warpPerspective(image,h,(50,40),flags=cv2.INTER_NEAREST)
        self.assertEqual(warped[16,17],255);self.assertEqual(warped[12,10],0)

    def test_partial_affine_inverse(self):
        a=.3;s=1.05;h=np.array([[s*np.cos(a),-s*np.sin(a),13],[s*np.sin(a),s*np.cos(a),-8],[0,0,1]])
        xy=np.array([[0.,0.],[50,50],[123,20]])
        current,v=project_points(xy,h);back,v=compensate_known(current,h,v)
        np.testing.assert_allclose(back,xy,atol=1e-12);self.assertTrue(v.all())

    def test_projective_inverse_and_homogeneous_scaling(self):
        h=np.array([[1.,.05,8],[.02,1.,-4],[.0002,-.0001,1.]])
        xy=np.array([[10.,12.],[100.,200.]])
        a,v=project_points(xy,h);b,_=project_points(xy,h*1e-12);np.testing.assert_allclose(a,b)
        back,_=compensate_known(a,h,v);np.testing.assert_allclose(back,xy,atol=1e-10)

    def test_mask_and_projection_horizon(self):
        h=np.array([[1.,0,0],[0,1,0],[1,0,-2]])
        xy=np.array([[2.,5.],[4,5],[np.nan,5],[5,6]])
        out,v=project_points(xy,h,np.array([True,True,True,False]))
        np.testing.assert_array_equal(v,[False,True,False,False]);self.assertTrue(np.isnan(out[~v]).all())

    def test_singular_and_ill_conditioned_rejected(self):
        for h in [np.zeros((3,3)),np.diag([1.,1.,0.]),np.diag([1.,1.,1e-12]),np.full((3,3),np.nan)]:
            with self.subTest(h=h),self.assertRaises(ValueError):inverse_transform(h)

    def test_composition_order_and_gap_does_not_auto_recover(self):
        a=np.array([[1.,0.,5],[0,1,0],[0,0,1]]);b=np.array([[2.,0.,0],[0,2,0],[0,0,1]])
        chain,v=accumulate_reference(np.stack([a,b,a]),np.ones(3,bool))
        p,_=project_points(np.array([[1.,0.]]),chain[2]);np.testing.assert_allclose(p,[[12,0]])
        chain,v=accumulate_reference(np.stack([a,b,a]),np.array([True,False,True]))
        np.testing.assert_array_equal(v,[True,True,False,False]);self.assertTrue(np.isnan(chain[2:]).all())


if __name__=='__main__':unittest.main()
