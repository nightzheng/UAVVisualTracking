"""Offline, causal as-of presentation. Read-only to the controller and source evidence."""
from bisect import bisect_right
from collections import defaultdict
from .common import read


def eligible(plan,ack,t,w,station):
    return (plan['station']==station and plan['source_capture_pts']<=plan['plan_created_pts']<=t<=plan['valid_until']
        and plan['plan_created_wall_s']<=ack['command_sent_wall']<=ack['applied_wall_s']<=w<=plan['valid_until_wall_s']
        and ack['applied_sim_s']<=t and ack['publish_ok'])


def select(candidates,t,w,station):
    """Newest available decision owns the status. Never revive a superseded plan."""
    pending=None
    for obs,plan,ack in reversed(candidates):
        if obs['decision_sim_s']>t or obs['decision_wall']>w:continue
        if obs['station']!=station:return pending or ('unavailable',None,None,obs)
        if plan is None:return pending or ('rejected',None,None,obs)
        if t>plan['valid_until'] or w>plan['valid_until_wall_s']:return pending or ('expired',None,ack,obs)
        if ack is None or not eligible(plan,ack,t,w,station):
            if pending is None:pending=('pending',None,ack,obs)
            continue
        return 'live',plan,ack,obs
    return pending or ('unavailable',None,None,None)


class Timeline:
    def __init__(self,path,derived=None):
        derived = derived or path
        self.scene=path.parent.name;self.arm=path.name;self.path=path
        r=lambda x:read(self.path/f'{x}.json')
        self.frames={round(x['pts']*20):x for x in r('frames')}
        assert len(self.frames)==len(r('frames')) and list(self.frames)==list(range(max(self.frames)+1))
        self.states=r('states');self.times=[x['sim_s'] for x in self.states]
        self.transitions=r('transitions')
        self.receipts=r('capture_receipts')
        self.quality=read(derived/'IMAGING_EVALUATION.json')['photos']
        self.spec=r('consumer_spec')
        self.cmds={c['sequence']:c for c in r('commands')}
        self.acks={c['sequence']:c for c in read(derived/'action_alignment.json')}
        by_source={c['source_index']:c for c in self.acks.values() if c['source_index'] is not None}
        self.observations=r('observations')
        self.candidates=[(o,o.get('decision',{}).get('joint_plan'),by_source.get(o['index'])) for o in self.observations
                         if 'decision_sim_s' in o]
        self.raw={round(x['pts']*20):x for x in r('display_plans')}
        self.snapshots=defaultdict(list)
        for raw in self.raw.values():
            p=raw['plan']
            if p:self.snapshots[p['source_capture_pts']].append((p,self.frames[round(raw['pts']*20)]['arrived_wall']))
        self.last={}
        self.rows={}
        for tick in self.frames:
            self.rows[tick]=self.sample(tick)

    def sample(self,tick):
        f=self.frames[tick];t=f['pts'];w=f['arrived_wall']
        ix=bisect_right(self.times,t)-1
        st=self.states[ix] if ix>=0 else None
        k=sum(a['sim_s']<=t for a in self.transitions)
        finished=k>=len(self.spec['banks'])
        station=min(k,len(self.spec['banks'])-1)
        command=self.cmds.get(st['executed_sequence']) if st else None
        status,p,ack,obs=select(self.candidates,t,w,station)
        if finished:status,p='complete',None
        hold=False
        if p:
            snapshots=self.snapshots.get(p['source_capture_pts'],[])
            p=next((x for x,wall in reversed(snapshots) if x['display_pts']<=t and wall<=w),p)
            hold=abs(p['display_pts']-t)>1e-8
            self.last[station]=p
        hist=self.last.get(station) if not p and not finished else None
        raw=self.raw.get(tick,{}).get('plan')
        old=bool(raw and raw['station']==station and any(
            a['source_pts']==raw['source_capture_pts'] and a['command_sent_wall']<=w and a['applied_sim_s']<=t
            for a in self.acks.values()))
        if finished:mode='COMPLETE' if all(q['region_complete'] for q in self.quality) and len(self.quality)==len(self.spec['banks']) else 'ENDED'
        elif st is None:mode='UNAVAILABLE'
        elif command and command['source_index'] is None:mode='TRANSIT'
        elif command:
            original=next((o for o in self.observations if o['index']==command['source_index']),{})
            mode=original.get('decision',{}).get('mode','TRACK')
        elif st.get('expired_sequence',-1)>=0:mode='RECOVER'
        else:mode='HOLD'
        reason={'live':'None','pending':'Awaiting application','expired':'Observation lease elapsed',
                'rejected':'Visual binding rejected','unavailable':'No eligible observation','complete':'Station schedule ended'}[status]
        if status=='rejected' and obs:
            reason={'insufficient_current_material_support':'Insufficient material support',
                'no_legal_initial_anchor':'No valid material anchor','geometry_support_or_consensus_failed':'Geometry consensus failed'}.get(obs.get('reason'),reason)
        return dict(t=t,w=w,frame=f,state=st,state_ix=ix,station=station,mode=mode,status=status,obs=obs,
                    plan=p,hold=hold,historical=hist,old=old,command=command,ack=ack,reason=reason,raw=raw)


def historical_prefix(values,times,t):
    return values[:bisect_right(times,t)]
