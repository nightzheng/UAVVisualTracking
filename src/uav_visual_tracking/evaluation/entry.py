"""Explicit GT loading, safe NA handling, and complete-plan parent macro evaluation."""
import itertools
import json
from pathlib import Path
import numpy as np
from ..data.results import load_result
from .tapvid import aligned_inputs, evaluate_actual_video
from .reliability import accept, evaluate, strict_mean

RATE_KEYS=('R_sel','C_acc','C_good','C_vis_good','correct_position_retention','known_hidden_false_accept_rate')
EPE_KEYS=('epe_mean','epe_median','epe_P95')
COUNT_KEYS=('n_ac','n_ab','n_rc','n_rb','N','U','accepted_unknown')


def load_gt(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def evaluate_sequence(data,manifest,gt,model,epsilon=10.,query_mode='first'):
    if query_mode not in {'first','strided'}:raise ValueError('Unknown official query mode')
    backend=manifest.get('prediction',{}).get('model',{}).get('backend')
    if backend and backend!={'B0':'cotracker3','B1':'bootstapir'}.get(model):
        raise ValueError('Declared model does not match prediction backend')
    pred,truth,q=aligned_inputs(data,manifest,gt)
    required_known=gt['visibility_known']&(~gt['visibility']|gt['position_known'])
    if not np.isfinite(truth[gt['visibility_known']&gt['visibility']&gt['position_known']]).all():
        raise ValueError('Known visible GT must be finite')
    # Validate query binding even when standard metrics will be NA.
    qi=q[:,0].astype(int);ni=np.arange(len(q))
    if not required_known[qi,ni].all() or not gt['visibility'][qi,ni].all() or not np.allclose(truth[qi,ni],q[:,1:],atol=1e-5,rtol=0):
        raise ValueError('Queries must be known visible GT locations')
    complete=bool(required_known.all())
    standard=evaluate_actual_video(data,manifest,gt,query_mode) if complete else None
    reason=None if complete else dict(code='incomplete_GT',
        unknown_visibility=int((~gt['visibility_known']).sum()),
        unknown_visible_position=int((gt['visibility_known']&gt['visibility']&~gt['position_known']).sum()),
        message='No official full score; known-only G_geo diagnostics are separate')
    return dict(model=model,standard=standard,standard_na_reason=reason,
        reliability={method:evaluate(data,manifest,gt,accept(data,model,method),epsilon) for method in ('M0','M1','ALL')},
        reliability_query_mode='strictly_after_each_query; independent of official query_mode',
        epsilon_original_px=epsilon)


def slot_key(s):return tuple(s[k] for k in ('parent','condition','query','model'))


def evaluate_plan(plan,base=Path('.'),epsilon=10.,query_mode='first'):
    """Query rates -> parent rates -> equal parent means. Never pool as a macro.

    The caller declares the complete Cartesian opportunity set. Query identity is
    checked against each result, not inferred from successful outputs.
    Missing slots/files return a refusal and no macro. Definedness propagates strictly.
    """
    axes=[]
    for key in ('parents','conditions','queries','models'):
        values=plan[key]
        if not isinstance(values,list) or not values or any(not isinstance(x,str) or not x for x in values) or len(set(values))!=len(values):
            raise ValueError('Nonempty unique string plan axis required: '+key)
        axes.append(values)
    if not set(plan['models'])<={'B0','B1'}:raise ValueError('Plan models must be B0/B1')
    expected=set(itertools.product(*axes));bykey={}
    for s in plan['slots']:
        key=slot_key(s)
        if key in bykey:raise ValueError('Duplicate planned slot: '+str(key))
        if key not in expected:raise ValueError('Unexpected planned slot: '+str(key))
        bykey[key]=s
    missing=[]
    for key in sorted(expected):
        s=bykey.get(key)
        absent=['slot'] if s is None else [k for k in ('result','gt') if k not in s or not (base/s[k]).exists()]
        if not absent and not all((base/s['result']/name).is_file() for name in ('tracks.npz','manifest.json')):
            absent=['result/tracks.npz or manifest.json']
        if absent:missing.append(dict(zip(('parent','condition','query','model'),key),missing=absent))
    if missing:return dict(status='refused_incomplete_plan',expected_slots=len(expected),missing_slots=missing,macro=None)
    rows=[]
    for key in sorted(expected):
        s=bykey[key];data,m=load_result(base/s['result']);gt=load_gt(base/s['gt'])
        if data['point_ids'].tolist()!=[s['query']]:raise ValueError('Each plan slot must bind exactly its declared query: '+str(key))
        for field,value in [('parent_id',s['parent']),('condition_id',s['condition'])]:
            if field in m['video'] and m['video'][field]!=value:raise ValueError('Plan identity mismatch: '+field)
        report=evaluate_sequence(data,m,gt,s['model'],epsilon,query_mode)
        standard=report['standard']
        for method,r in report['reliability'].items():
            rows.append(dict(parent=s['parent'],condition=s['condition'],query=s['query'],model=s['model'],method=method,
                **{k:r[k] for k in RATE_KEYS+EPE_KEYS},**r['counts'],**{k:r[k] for k in ('N','U','accepted_unknown')},
                epe_required=r['epe_required'],epe_finite=r['epe_finite'],accepted_unknown_risk_bounds=r['accepted_unknown_risk_bounds'],
                **{k:standard[k] if standard else None for k in ('AJ','delta_avg','OA')},
                standard_na_reason=report['standard_na_reason']))
    metrics=RATE_KEYS+EPE_KEYS+('AJ','delta_avg','OA');parents=[];macros=[]
    for model in plan['models']:
        for method in ('M0','M1','ALL'):
            for scope in ['ALL_CONDITIONS']+plan['conditions']:
                parent_rows=[]
                for parent in plan['parents']:
                    g=[r for r in rows if r['parent']==parent and r['model']==model and r['method']==method and (scope=='ALL_CONDITIONS' or r['condition']==scope)]
                    p=dict(parent=parent,scope=scope,model=model,method=method,planned_queries=len(g),
                        **{k:strict_mean([r[k] for r in g]) for k in metrics},
                        **{k:sum(r[k] for r in g) for k in COUNT_KEYS},
                        undefined={k:[dict(condition=r['condition'],query=r['query']) for r in g if r[k] is None] for k in metrics})
                    parents.append(p);parent_rows.append(p)
                macros.append(dict(scope=scope,model=model,method=method,planned_parents=len(plan['parents']),
                    **{k:strict_mean([r[k] for r in parent_rows]) for k in metrics},
                    undefined_parents={k:[r['parent'] for r in parent_rows if r[k] is None] for k in metrics}))
    return dict(status='complete_plan',expected_slots=len(expected),missing_slots=[],rows=rows,parent_metrics=parents,macro=macros,
        aggregation='equal query opportunity means within each parent, then equal parent means; NA propagates; standard scores use native visibility',
        epsilon_original_px=epsilon)
