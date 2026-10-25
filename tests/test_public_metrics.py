"""Hand-checkable science contracts; no frozen input assets are redistributed."""
import copy,itertools,json,tempfile,unittest
from pathlib import Path
import numpy as np
from test_public_contracts import fixture
from uav_visual_tracking.data.results import save_result
from uav_visual_tracking.evaluation.tapvid import evaluate_actual_video,official_compute,OFFICIAL_SHA256
from uav_visual_tracking.evaluation.reliability import accept,evaluate,strict_mean
from uav_visual_tracking.evaluation.entry import evaluate_sequence,evaluate_plan


class Metrics(unittest.TestCase):
    def test_c3_actual_180x260_hand_oracle(self):
        c,q,p,d,m,g=fixture();d['tracks'][1:,:,0]+=6
        r=evaluate_actual_video(d,m,g)
        # 6*256/180=8.533: only the 16px threshold passes (1,2,4,8,16).
        self.assertAlmostEqual(r['delta_avg'],.2);self.assertAlmostEqual(r['AJ'],.2)
        self.assertEqual(r['OA'],1);self.assertEqual(r['actual_evaluation_resolution_wh'],[180,260])
        diagnostic=evaluate(d,m,g,accept(d,'B0'))
        self.assertAlmostEqual(diagnostic['epe_mean'],6)

    def test_complete_exact_and_nonzero_multiquery(self):
        c,q,p,d,m,g=fixture(n=2);r=evaluate_actual_video(d,m,g)
        self.assertEqual([r[k] for k in ('AJ','delta_avg','OA')],[1,1,1])
        self.assertEqual(r['evaluation_count'],7)
        self.assertEqual(evaluate_actual_video(d,m,g,'strided')['evaluation_count'],8)

    def test_unknown_is_standard_NA_not_zero_or_known_only_official(self):
        c,q,p,d,m,g=fixture();g['visibility_known'][2]=False
        with self.assertRaisesRegex(ValueError,'complete'):evaluate_actual_video(d,m,g)
        report=evaluate_sequence(d,m,g,'B0')
        self.assertIsNone(report['standard']);self.assertEqual(report['standard_na_reason']['unknown_visibility'],1)
        r=report['reliability']['M0'];self.assertEqual((r['N'],r['U'],r['accepted_unknown']),(3,4,1))
        self.assertEqual(r['accepted_unknown_risk_bounds'],[0,.25])

    def test_position_unknown_and_hidden_position_policy(self):
        c,q,p,d,m,g=fixture();g['position_known'][2]=False
        self.assertIsNone(evaluate_sequence(d,m,g,'B1')['standard'])
        g['visibility'][2]=False;g['tracks_original'][2]=np.nan
        self.assertIsNotNone(evaluate_sequence(d,m,g,'B1')['standard'])

    def test_alignment_mismatches_fail_even_with_unknown(self):
        for key in ['timestamps','frame_indices','point_ids','queries_original_txy','video_id']:
            c,q,p,d,m,g=fixture();g['visibility_known'][2]=False
            if key=='video_id':g[key]=np.array('other')
            elif key=='point_ids':g[key]=np.array(['other'])
            else:g[key]=g[key].copy();g[key].flat[-1]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):evaluate_sequence(d,m,g,'B0')

    def test_invalid_known_gt_and_query_are_errors(self):
        c,q,p,d,m,g=fixture();g['tracks_original'][2]=np.nan
        with self.assertRaises(ValueError):evaluate_sequence(d,m,g,'B0')
        c,q,p,d,m,g=fixture();g['visibility_known'][0]=False
        with self.assertRaisesRegex(ValueError,'Queries'):evaluate_sequence(d,m,g,'B0')

    def test_four_cells_and_denominators(self):
        c,q,p,d,m,g=fixture();d['tracks'][2,0,0]+=30;d['tracks'][4,0,0]+=30
        gate=np.array([[True],[True],[True],[False],[False]])
        r=evaluate(d,m,g,gate)
        self.assertEqual(r['counts'],dict(n_ac=1,n_ab=1,n_rc=1,n_rb=1))
        self.assertEqual((r['R_sel'],r['C_acc'],r['C_good']),(.5,.5,.25))

    def test_score_boundary_nonfinite_and_no_coordinate_edits(self):
        c,q,p,d,m,g=fixture();before=d['tracks'].copy()
        for model,key in [('B0','visibility_probability'),('B1','visibility_score')]:
            d['raw_score__'+key][:,0]=[.7,.70001,.69999,np.nan,1]
            np.testing.assert_array_equal(accept(d,model,'M1')[:,0],[False,True,False,False,True])
        np.testing.assert_array_equal(d['tracks'],before)
        d['tracks'][4]=np.nan
        self.assertFalse(accept(d,'B0','ALL')[4,0])
        r=evaluate(d,m,g,accept(d,'B0'));self.assertEqual(r['counts']['n_rb'],1)
        standard=evaluate_actual_video(d,m,g);self.assertEqual(standard['nonfinite_prediction_count'],1)
        self.assertEqual(standard['delta_avg'],.75)

    def test_no_acceptance_is_NA_and_strict_mean_propagates(self):
        c,q,p,d,m,g=fixture();r=evaluate(d,m,g,np.zeros_like(d['visibility']))
        self.assertIsNone(r['R_sel']);self.assertEqual(r['C_acc'],0);self.assertEqual(r['C_good'],0)
        self.assertIsNone(strict_mean([.1,None]));self.assertIsNone(strict_mean([]))

    def test_epsilon_inclusive_and_all_unknown_denominator(self):
        c,q,p,d,m,g=fixture();d['tracks'][1:,:,0]+=10
        self.assertEqual(evaluate(d,m,g,accept(d,'B1'),10)['counts']['n_ac'],4)
        self.assertEqual(evaluate(d,m,g,accept(d,'B1'),5)['counts']['n_ab'],4)
        g['visibility_known'][1:]=False;r=evaluate(d,m,g,accept(d,'B1'))
        self.assertEqual((r['N'],r['U']),(0,4));self.assertIsNone(r['C_good']);self.assertIsNone(r['R_sel'])

    def test_packaged_official_metric(self):
        from importlib.resources import files
        import hashlib
        body=files('uav_visual_tracking.evaluation._official').joinpath('evaluation_datasets.py').read_bytes()
        self.assertEqual(hashlib.sha256(body).hexdigest(),OFFICIAL_SHA256)
        self.assertTrue(files('uav_visual_tracking.evaluation._official').joinpath('LICENSE').is_file())
        self.assertTrue(callable(official_compute()))

    def make_plan(self,root):
        plan=dict(parents=['parent1','parent2'],conditions=['C0'],queries=['p0','p1'],models=['B0'],slots=[])
        for i,(parent,query) in enumerate(itertools.product(plan['parents'],plan['queries'])):
            c,q,p,d,m,g=fixture(t=5+i)
            q.point_ids=[query];m['point_ids']=[query];g['point_ids']=np.array([query])
            if i==3:p.visibility[:]=False
            result=root/f'r{i}';result.mkdir();save_result(result,c,q,p,m)
            gt=root/f'g{i}.npz';np.savez_compressed(gt,**g)
            plan['slots'].append(dict(parent=parent,condition='C0',query=query,model='B0',result=result.name,gt=gt.name))
        return plan

    def test_complete_parent_macro_is_query_then_parent(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);plan=self.make_plan(root);r=evaluate_plan(plan,root)
            self.assertEqual(r['status'],'complete_plan')
            macro=next(x for x in r['macro'] if x['scope']=='C0' and x['method']=='M0')
            self.assertEqual(macro['C_good'],.75);self.assertIsNone(macro['R_sel'])
            self.assertEqual(macro['undefined_parents']['R_sel'],['parent2'])
            self.assertEqual(sum(x['n_ac'] for x in r['rows'] if x['method']=='M0'),15)

    def test_missing_slot_and_file_refuse_macro(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);plan=self.make_plan(root);plan['slots'].pop()
            r=evaluate_plan(plan,root);self.assertIsNone(r['macro'])
            self.assertEqual(r['missing_slots'][0]['query'],'p1')
            plan['slots'][0]['gt']='absent.npz';r=evaluate_plan(plan,root)
            self.assertEqual(len(r['missing_slots']),2)

    def test_duplicate_extra_and_mislabeled_plan_slots_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);plan=self.make_plan(root);bad=copy.deepcopy(plan);bad['slots'].append(bad['slots'][0])
            with self.assertRaisesRegex(ValueError,'Duplicate'):evaluate_plan(bad,root)
            bad=copy.deepcopy(plan);bad['slots'][0]['parent']='other'
            with self.assertRaisesRegex(ValueError,'Unexpected'):evaluate_plan(bad,root)
            bad=copy.deepcopy(plan);bad['slots'][0]['result']=plan['slots'][1]['result']
            with self.assertRaisesRegex(ValueError,'query'):evaluate_plan(bad,root)

if __name__=='__main__':unittest.main()
