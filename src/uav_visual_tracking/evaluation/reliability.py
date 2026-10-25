"""Frozen G_geo diagnostics and fixed M0/M1 gates; no fitting or identity oracle."""
import numpy as np
from .tapvid import aligned_inputs


def ratio(num,den):
    return float(num/den) if den else None


def segments(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def known_only_diagnostics(data,manifest,gt,epsilon=10.,recovery_frames=3):
    """Fixed model-only gate = predicted visible AND finite; synthetic geometric event only.

    Identity confirmation/oracle labels are never used to decide acceptance.
    Unknown breaks continuous runs, and bracketed returns are not declared exact.
    """
    pred,truth,queries=aligned_inputs(data,manifest,gt)
    if epsilon<=0 or type(recovery_frames) is not int or recovery_frames<1:
        raise ValueError('Positive epsilon and recovery window required')
    visible=gt['visibility'];vk=gt['visibility_known'];pk=gt['position_known']
    t,n=visible.shape
    attempt=np.arange(t)[:,None]>queries[:,0]
    finite=np.isfinite(pred).all(axis=-1)
    valid_gt=np.isfinite(truth).all(axis=-1)
    if (vk&visible&pk&~valid_gt).any():
        raise ValueError('Known visible GT must be finite')
    epe=np.linalg.norm(pred-truth,axis=-1)
    known=attempt & vk & (~visible | pk)
    good=visible & finite & (epe<=epsilon)
    accepted=data['visibility'] & finite & attempt
    ac=int((known&accepted&good).sum());ab=int((known&accepted&~good).sum())
    rc=int((known&~accepted&good).sum());rb=int((known&~accepted&~good).sum())
    N=int(known.sum());U=int(attempt.sum());AU=int((accepted&attempt&~known).sum())
    required=attempt&vk&visible&pk
    epe_values=epe[required&finite]
    ts=data['timestamps']
    frame_durations=np.r_[np.diff(ts),np.diff(ts)[-1]]
    runs=[];first_loss=[];returns=[]
    for p in range(n):
        # Visible wrong measurements; rejection does not erase raw failure evidence.
        bad=required[:,p]&data['visibility'][:,p]&(~finite[:,p]|(epe[:,p]>epsilon))
        for a,b in segments(bad):
            runs.append(dict(point_id=str(data['point_ids'][p]),start_source_frame=int(data['frame_indices'][a]),
                             end_source_frame_inclusive=int(data['frame_indices'][b-1]),frames=int(b-a),
                             seconds=float(frame_durations[a:b].sum())))
        loss=required[:,p]&(~finite[:,p]|~data['visibility'][:,p]|(epe[:,p]>epsilon))
        indices=np.flatnonzero(loss)
        first_loss.append(dict(point_id=str(data['point_ids'][p]),
                               source_frame=int(data['frame_indices'][indices[0]]) if len(indices) else None,
                               right_censored=not bool(len(indices))))
        for r in range(1,t):
            if not (attempt[r,p] and vk[r-1,p] and not visible[r-1,p] and vk[r,p] and visible[r,p]):
                continue
            found=None
            for a in range(r,t-recovery_frames+1):
                b=a+recovery_frames
                if (required[a:b,p]&good[a:b,p]&data['visibility'][a:b,p]).all():
                    found=a;break
                # Only this contiguous known-visible episode belongs to this return.
                if not required[a,p]:
                    break
            returns.append(dict(point_id=str(data['point_ids'][p]),return_source_frame=int(data['frame_indices'][r]),
                                return_epe=float(epe[r,p]) if required[r,p] and finite[r,p] else None,
                                recovery_start_frame=int(data['frame_indices'][found]) if found is not None else None,
                                recovery_confirmed_frame=int(data['frame_indices'][found+recovery_frames-1]) if found is not None else None,
                                delay_seconds=float(ts[found]-ts[r]) if found is not None else None,
                                right_censored=bool(found is None and t-r<recovery_frames)))
    return dict(name='known_only_project_diagnostics',event='G_geo; no independent identity correctness claim',
                epsilon_original_px=epsilon,gate='predicted_visibility & finite_position; no GT/oracle input',
                U=U,N=N,accepted_unknown=AU,known_coverage=ratio(N,U),all_attempt_acceptance=ratio(ac+ab+AU,U),
                counts=dict(n_ac=ac,n_ab=ab,n_rc=rc,n_rb=rb),R_sel=ratio(ab,ac+ab),
                C_acc=ratio(ac+ab,N),C_good=ratio(ac,N),
                accepted_risk_unknown_bounds=[ratio(ab,ac+ab+AU),ratio(ab+AU,ac+ab+AU)],
                epe_required=int(required.sum()),epe_finite=int(len(epe_values)),
                epe_mean=float(epe_values.mean()) if len(epe_values) else None,
                large_error_count=int((required&(~finite|(epe>epsilon))).sum()),large_error_runs=runs,
                run_duration_convention='PTS forward intervals; final interval repeats previous; unknown breaks runs',
                first_loss=first_loss,known_adjacent_returns=returns,
                unknown_visibility_count=int((attempt&~vk).sum()),
                unknown_bracket_policy='only adjacent known-hidden to known-visible returns; brackets require separate adjudication')


def accept(data, model, method='M0'):
    """Extracted frozen rules. Strict >0.7, including the score equality boundary."""
    if model not in {'B0','B1'} or method not in {'M0','M1','ALL'}:
        raise ValueError('Expected B0/B1 and M0/M1/ALL')
    finite=np.isfinite(data['tracks']).all(-1)
    if method=='ALL':return finite.copy()
    if method=='M0':return data['visibility'].astype(bool)&finite
    key='raw_score__visibility_probability' if model=='B0' else 'raw_score__visibility_score'
    if key not in data:raise ValueError('M1 requires continuous raw scores: '+key)
    s=np.asarray(data[key])
    if s.shape!=finite.shape:raise ValueError('Raw score must be T,N')
    return finite&np.isfinite(s)&(s>.7)


def evaluate(data, manifest, gt, gate, epsilon=10.):
    """Reuse frozen known-only arithmetic; source pixels define EPE and G_geo.

    All query frames and prequery frames are excluded from reliability metrics.
    Nonfinite predictions are rejected and still count as failures on known GT.
    EPE is conditional on finite predictions; required/finite counts are reported.
    """
    if gate.shape!=data['visibility'].shape or gate.dtype!=np.bool_:
        raise ValueError('Invalid gate')
    finite=np.isfinite(data['tracks']).all(-1)
    if (gate&~finite).any():raise ValueError('Nonfinite position accepted')
    report=known_only_diagnostics(dict(data,visibility=gate),manifest,gt,epsilon)
    pred,truth,q=aligned_inputs(data,manifest,gt)
    errors=np.linalg.norm(pred-truth,axis=-1)
    active=np.arange(len(pred))[:,None]>q[:,0]
    visible=active&gt['visibility_known']&gt['visibility']&gt['position_known']
    hidden=active&gt['visibility_known']&~gt['visibility']
    good=visible&finite&(errors<=epsilon)
    values=errors[visible&finite]
    report.update(C_vis_good=ratio(int((good&gate).sum()),int(visible.sum())),
        correct_position_retention=ratio(int((good&gate).sum()),int(good.sum())),
        known_hidden_false_accept_rate=ratio(int((hidden&gate).sum()),int(hidden.sum())),
        epe_median=float(np.median(values)) if len(values) else None,
        epe_P95=float(np.percentile(values,95)) if len(values) else None,
        visible_required=int(visible.sum()),hidden_required=int(hidden.sum()),
        zero_acceptance=report['counts']['n_ac']+report['counts']['n_ab']==0,
        accepted_unknown_risk_bounds=report['accepted_risk_unknown_bounds'])
    return report


def strict_mean(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None
