"""Live Gazebo transport, sealed RGB worker, executed controls and PNG receipts."""
from collections import deque
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Queue
import shutil
import signal
import subprocess
import sys
from threading import Event, Thread, get_ident
import time
import cv2
import numpy as np
from .common import read,save,sha
from .control import Controller,worker,angles
from .u3_gazebo import camera_transform


def png_sink(queue,out,errors):
    while True:
        p=queue.get()
        if p is None:return
        try:
            w,h=p['wh'];raw=np.frombuffer(p['raw'],np.uint8).reshape(h,p['step'])[:,:w*3].reshape(h,w,3)
            name=out/'rgb'/f'{p["index"]:05d}.png';temp=name.with_name(f'.{name.stem}.tmp.png')
            if not cv2.imwrite(str(temp),cv2.cvtColor(raw,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,1]):
                raise IOError('Sensor PNG write failed')
            os.replace(temp,name)
        except Exception as e:errors.append(repr(e))


def run(out, controller=Controller, cancelled=None):
    cancelled = cancelled or Event()
    planner_thread = get_ident()
    planning = False

    def guard(event, args):
        if planning and get_ident() == planner_thread and event in {"open", "os.listdir", "os.scandir"}:
            raise PermissionError("Planner may only use its supplied live inputs")

    def decide(operation, *args):
        nonlocal planning
        planning = True
        try:
            return operation(*args)
        finally:
            planning = False

    sys.addaudithook(guard)
    out=Path(out).resolve();spec=read(out/'consumer_spec.json');src=spec['source'];control=spec['control']
    (out/'rgb').mkdir(exist_ok=False);(out/'captures').mkdir();cv2.setNumThreads(1)
    os.environ['GZ_PARTITION']=f'u3d_{os.getpid()}'
    os.environ.setdefault('GALLIUM_DRIVER','d3d12');os.environ.setdefault('MESA_D3D12_DEFAULT_ADAPTER_NAME','NVIDIA')
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
    from gz.msgs10.camera_info_pb2 import CameraInfo
    from gz.msgs10.stringmsg_pb2 import StringMsg
    context=mp.get_context('spawn');conn,child=context.Pipe();proc=context.Process(target=worker,args=(child,spec,out));proc.start();child.close()
    try:
        if not conn.poll(12) or conn.recv()!='ready':raise RuntimeError('Visual consumer failed startup')
    except BaseException:
        conn.close();proc.terminate();proc.join(5)
        raise
    incoming=deque();states={};info=[];pending=deque();frames=[];observations=[];commands=[];captures=[];transitions=[];drops=[]
    sink=Queue();errors=[];io=Thread(target=png_sink,args=(sink,out,errors))
    planner=decide(controller,spec);node=Node();pub=node.advertise('/u3/command',StringMsg);last=-np.inf;busy=False;seq=0
    display=[]
    def current_motion(st):
        tick=round(st['sim_s']*1000)
        prev=next((states[k] for k in range(tick-1,tick-12,-1) if k in states),None)
        if prev is None:return np.zeros(3),np.zeros(2)
        dt=st['sim_s']-prev['sim_s'];p,T=camera_transform(st['camera_pose']);q,S=camera_transform(prev['camera_pose'])
        return (p-q)/dt,(angles(p,p+T[2,:3])-angles(q,q+S[2,:3]))/dt
    completion=None;start=time.monotonic();failure=None
    def stamp(msg):return msg.header.stamp.sec+msg.header.stamp.nsec*1e-9
    def image(msg):
        if stamp(msg)<=src['duration_s']:
            incoming.append(dict(pts=stamp(msg),arrived_wall=time.monotonic(),raw=msg.data,wh=[msg.width,msg.height],step=msg.step))
    def calibration(msg):
        if not info:info.append(dict(K=list(msg.intrinsics.k),width=msg.width,height=msg.height))
    def state(msg):
        s=json.loads(msg.data);s['telemetry_wall']=time.monotonic();states[round(s['sim_s']*1000)]=s
    for cls,name,cb in ((Image,'/u3/camera/image',image),(CameraInfo,'/u3/camera/camera_info',calibration),(StringMsg,'/u3/state',state)):
        if not node.subscribe(cls,name,cb):
            conn.close();proc.terminate();proc.join(5)
            raise RuntimeError('Transport subscription failed '+name)
    def send(cmd,row,sim,started):
        nonlocal seq
        ref,yp,reason,expiry,expiry_wall=cmd;yp=np.clip(yp,[-75,-65],[75,65]);done=time.monotonic()
        if sim>expiry or done>expiry_wall:return
        ok=pub.publish(StringMsg(data=' '.join(map(str,[seq,*ref,*yp,expiry,expiry_wall]))))
        commands.append(dict(sequence=seq,station=planner.index,source_index=None if row is None else row['index'],
            source_pts=None if row is None else row['capture_pts'],ref_m=ref.tolist(),yaw_pitch_deg=yp.tolist(),reason=reason,
            plan_started_wall=started,plan_done_wall=done,command_sent_wall=time.monotonic(),valid_until_sim=expiry,
            valid_until_wall=expiry_wall,publish_ok=bool(ok)))
        seq+=1
    with (out/'server.log').open('w') as log:
        try:
            server=subprocess.Popen(['gz','sim','-s','-r','-v','4',str(out/'local.sdf')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:
            conn.close();proc.terminate();proc.join(5)
            raise
        io.start()
        try:
            while time.monotonic()-start<src['duration_s']*1.7+15:
                if cancelled.is_set():raise InterruptedError('Experiment cancelled')
                if server.poll() is not None:raise RuntimeError('Gazebo exited during trial')
                if errors:raise IOError(errors[0])
                if not states:time.sleep(.002);continue
                latest=max(states);telemetry=states[latest];sim=telemetry['sim_s']
                if telemetry.get('safety_fault'):raise RuntimeError('Actuator entered hard safety fault')
                if info and (info[0]['width'],info[0]['height'])!=(640,480):raise RuntimeError('Unexpected actual camera dimensions')
                if info and not np.allclose(np.asarray(info[0]['K']).reshape(3,3),spec['K'],atol=.01):raise RuntimeError('Camera calibration mismatch')
                complete=planner.index>=len(spec['banks'])
                if not complete and planner.navigation_pending:send(decide(planner.navigation),None,sim,time.monotonic())
                while incoming and round(incoming[0]['pts']*1000)<=latest:
                    p=incoming.popleft()
                    if p['pts']<=last:continue
                    last=p['pts'];p['index']=len(frames);frames.append(dict(index=p['index'],pts=p['pts'],arrived_wall=p['arrived_wall'],file=f'rgb/{p["index"]:05d}.png'))
                    sink.put(p.copy())
                    tick=round(p['pts']*1000)
                    at=next((states[k] for k in range(tick,tick-5,-1) if k in states and states[k]['sim_s']<=p['pts']+1e-9),None)
                    snapshot=None
                    if at is not None:
                        xyz,cam=camera_transform(at['camera_pose'])
                        snapshot=decide(planner.snapshot,p['pts'],p['arrived_wall'],xyz,cam,*current_motion(at))
                    display.append(dict(index=p['index'],pts=p['pts'],plan=snapshot,
                                        reason=None if snapshot else 'no_fresh_plan_at_capture'))
                    if round(p['pts']*src['fps'])%2 or complete:
                        drops.append(dict(index=p['index'],reason='fixed_10hz_skip_or_finished'));continue
                    ms=round(p['pts']*1000)
                    aligned=next((states[k] for k in range(ms,ms-5,-1) if k in states and states[k]['sim_s']<=p['pts']+1e-9),None)
                    if aligned is None:drops.append(dict(index=p['index'],reason='no_causal_pose'));continue
                    _,T=camera_transform(aligned['camera_pose']);pending.append(dict(**p,camera_T=T.tolist()))
                if conn.poll():
                    row=conn.recv();busy=False
                    if 'error' in row:raise RuntimeError(row['error'])
                    observations.append(row)
                    if not complete:
                        # The planner gets only current camera state and the worker result.
                        position,T=camera_transform(telemetry['camera_pose']);now=time.monotonic()
                        planner.motion=current_motion(telemetry)
                        decision=decide(planner.step,row,sim,now,position,T);cmd=decision.pop('command');row['decision']=decision
                        row.update(decision_sim_s=sim,decision_wall=now)
                        if cmd:send(cmd,row,sim,now)
                        if decision['capture']:
                            frame=frames[row['index']];source=out/frame['file'];shot=out/'captures'/f'A_{planner.index+1:02d}.png'
                            if source.exists() and time.monotonic()<=row['fresh_until_arrival_wall_s']:
                                shutil.copyfile(source,shot)
                                receipt=dict(station=planner.index,frame_index=row['index'],pts=frame['pts'],file=str(shot.relative_to(out)),
                                    actual_rgb_file=frame['file'],sha256=sha(shot),saved_wall=time.monotonic(),phase_epoch=planner.epoch,
                                    observed_eta=row['eta']['value'],observed_xi=row['xi']['value'],surface_side=row['surface_side']['value'],
                                    estimated_quality=row['estimated_quality'],provisional_novel_m2=decision['provisional_novel_m2'])
                                captures.append(receipt);planner.visited.update(planner.bank['sample_ids'])
                                transitions.append(dict(station=planner.index,sim_s=sim,result='png_saved',next_phase_epoch=planner.last_fresh))
                                decide(planner.advance,sim)
                if planner.index<len(spec['banks']) and sim-planner.started_sim>control['transition_timeout_s']:
                    transitions.append(dict(station=planner.index,sim_s=sim,result='timeout_no_valid_photo',next_phase_epoch=planner.last_fresh))
                    decide(planner.advance,sim)
                if planner.index>=len(spec['banks']) and completion is None:completion=sim
                while pending and time.monotonic()-pending[0]['arrived_wall']>control['max_state_age_s']:
                    drops.append(dict(index=pending.popleft()['index'],reason='expired_before_processing'))
                if not busy and pending and planner.index<len(spec['banks']):
                    p=pending.popleft();p.update(station=planner.index,phase_epoch=planner.epoch);conn.send(p);busy=True
                if sim>src['duration_s']+.3 or (completion is not None and sim>completion+.65):break
                time.sleep(.001)
            else:raise TimeoutError('Gazebo did not reach the requested simulation time')
        except Exception as e:failure=repr(e)
        finally:
            if server.poll() is None:os.killpg(server.pid,signal.SIGINT)
            try:server.wait(timeout=6)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGKILL);server.wait()
            conn.close();proc.terminate();proc.join(5);sink.put(None);io.join()
    for key,data in [('frames',frames),('states',sorted(states.values(),key=lambda r:r['sim_s'])),('camera_info',info),('observations',observations),
        ('commands',commands),('display_plans',display),('capture_receipts',captures),('transitions',transitions),('drops',drops),
        ('runtime',dict(failure=failure,io_errors=errors,stations_attempted=planner.index,task_end_sim_s=completion))]:save(out/f'{key}.json',data)
    print(out.name,len(frames),'sensor frames',len(observations),'visual',len(captures),'saved',failure,flush=True)
    if failure or errors:raise RuntimeError(f'{failure}, {errors}')


if __name__=='__main__':run(sys.argv[1])
