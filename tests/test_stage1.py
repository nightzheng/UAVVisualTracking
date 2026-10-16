"""Invented arithmetic fixtures for software tests, never wind-turbine GT."""
import copy
import unittest
import numpy as np
from uav_visual_tracking.evaluation.annotations import evaluate_annotations
from uav_visual_tracking.evaluation.metrics import evaluate_gt


def fixture():
    data=dict(tracks=np.array([[[0.,0.]],[[3.,4.]],[[6.,8.]],[[0.,0.]]]),
        visibility=np.array([[True],[False],[True],[True]]),queries=np.array([[0.,0.,0.]]),
        frame_indices=np.array([10,11,12,13]),timestamps=np.arange(10,14)/20,point_ids=np.array(['a']))
    manifest=dict(video_sha256='fixture',video=dict(processing_resolution=[64,48]),config_sha256='config-a')
    point=dict(point_id='a',definition='invented test point',definition_reviewed=True,
        definition_reviewer='test-reviewer',query_correspondence_confirmed=True)
    doc=dict(schema_version=1,annotation_kind='human_visual',video_sha256='fixture',
        coordinate_space='processing',resolution=[64,48],points=[point],annotations=[
            dict(frame_index=10,timestamp_seconds=.5,point_id='a',state='visible',x=0,y=0,uncertainty='none',
                source='human_visual',review_status='reviewed',annotator='test',reviewer='test-reviewer',reviewed_at_utc='test'),
            dict(frame_index=12,timestamp_seconds=.6,point_id='a',state='occluded',x=None,y=None,uncertainty='none',
                source='human_visual',review_status='reviewed',annotator='test',reviewer='test-reviewer',reviewed_at_utc='test'),
            dict(frame_index=13,timestamp_seconds=.65,point_id='a',state='visible',x=0,y=0,uncertainty='none',
                source='human_visual',review_status='reviewed',annotator='test',reviewer='test-reviewer',reviewed_at_utc='test')])
    for row in doc['annotations']:
        row.update(occluded=row['state']=='occluded',out_of_frame=False)
    return data,manifest,doc


class Stage1(unittest.TestCase):
    def test_p90_missing_predictions_and_threshold_denominator(self):
        x=np.array([[[3.,4.]],[[6.,8.]],[[np.nan,np.nan]]]);v=np.ones((3,1),bool)
        m=evaluate_gt(x,v,np.zeros_like(x),v,v,v)
        self.assertEqual(m['p90_endpoint_error'],9.5)
        self.assertEqual(m['position_correctness']['5']['denominator'],3)
        self.assertEqual(m['position_correctness']['5']['rate'],1/3)
        self.assertEqual(m['nonfinite_visible_predictions'],1)

    def test_sparse_frames_do_not_become_interpolated_gt(self):
        data,manifest,doc=fixture();m,(_,_,valid)=evaluate_annotations(doc,data,manifest)
        self.assertEqual(m['annotated_point_frames'],3);self.assertFalse(valid[1,0])
        self.assertEqual(m['visibility']['fp'],1)
        self.assertEqual(m['coverage']['annotated_source_frames'],[10,12,13])
        events=m['events'];entry=next(e for e in events if e['kind']=='gt_hidden_entry_bracket')
        self.assertFalse(entry['adjacent_labeled_frames']);self.assertEqual(entry['from_frame'],10)
        returned=next(e for e in events if e['kind']=='gt_visibility_return_bracket')
        self.assertTrue(returned['adjacent_labeled_frames'])
        self.assertEqual(returned['physical_identity_recovery'],'requires_human_case_review')
        self.assertIsNone(m['identity_metrics']['full_period_recovery_rate'])

    def test_review_uncertainty_and_source_gate(self):
        for change in [dict(review_status='draft'),dict(reviewer=''),dict(source='model_prediction'),dict(uncertainty='identity_ambiguous')]:
            data,manifest,doc=fixture()
            for row in doc['annotations']:row.update(change)
            m,_=evaluate_annotations(doc,data,manifest)
            self.assertFalse(m['gt_available']);self.assertIsNone(m['mean_endpoint_error'])

    def test_definition_correspondence_gate(self):
        data,manifest,doc=fixture();doc['points'][0]['query_correspondence_confirmed']=False
        m,_=evaluate_annotations(doc,data,manifest);self.assertEqual(m['annotated_point_frames'],0)

    def test_duplicate_timestamp_video_coordinate_errors(self):
        data,manifest,original=fixture()
        variants=[]
        doc=copy.deepcopy(original);doc['annotations'].append(doc['annotations'][0].copy());variants.append(doc)
        doc=copy.deepcopy(original);doc['annotations'][0]['timestamp_seconds']=.501;variants.append(doc)
        doc=copy.deepcopy(original);doc['video_sha256']='other';variants.append(doc)
        doc=copy.deepcopy(original);doc['resolution']=[128,96];variants.append(doc)
        doc=copy.deepcopy(original);doc['annotations'][0]['x']=float('nan');variants.append(doc)
        doc=copy.deepcopy(original);doc['annotations'][1]['x']=5;variants.append(doc)
        for doc in variants:
            with self.subTest(doc=doc),self.assertRaises(ValueError):evaluate_annotations(doc,data,manifest)

    def test_failure_is_first_observed_not_inferred_onset(self):
        data,manifest,doc=fixture();doc['annotations'][1].update(state='visible',x=0,y=0,occluded=False)
        m,_=evaluate_annotations(doc,data,manifest,loss_threshold=5)
        self.assertEqual(m['per_point_loss'][0]['first_observed_failure_source_frame'],12)
        self.assertEqual(m['per_point_loss'][0]['previous_observed_good_source_frame'],10)

    def test_identity_case_requires_config_and_gt_evidence(self):
        data,manifest,doc=fixture()
        doc['case_reviews']=[dict(kind='physical_identity_recovery',point_id='a',frame_index=13,
            result_config_sha256='other-config',review_status='reviewed',reviewer='test',evidence='test only')]
        m,_=evaluate_annotations(doc,data,manifest);self.assertEqual(m['identity_metrics']['reviewed_identity_events'],0)
        doc['case_reviews'][0]['result_config_sha256']='config-a';doc['case_reviews'][0]['evidence']=''
        with self.assertRaises(ValueError):evaluate_annotations(doc,data,manifest)

    def test_point_ids_and_source_frames_reorder(self):
        data,manifest,doc=fixture();doc['annotations'].reverse()
        m,_=evaluate_annotations(doc,data,manifest);self.assertEqual(m['mean_endpoint_error'],0)


if __name__=='__main__':unittest.main()
