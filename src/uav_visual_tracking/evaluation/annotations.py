"""Human review gate, keyed sparse alignment, and evidence-limited events.

No prediction is accepted as annotation. The review declarations are recorded
human attestations, not an automated guarantee that someone actually reviewed.
"""
import math
import numpy as np
from .metrics import evaluate_gt

STATES = {"visible", "occluded", "out_of_frame", "not_visible_unknown", "uncertain", "pending"}


def align_annotations(document, data, manifest):
    if document.get("schema_version") != 1 or document.get("annotation_kind") != "human_visual":
        raise ValueError("Only schema 1 human visual annotation documents are supported")
    if document["video_sha256"] != manifest["video_sha256"]:
        raise ValueError("Annotation video SHA256 mismatch")
    if document["coordinate_space"] != "processing" or document["resolution"] != manifest["video"]["processing_resolution"]:
        raise ValueError("Annotation coordinate space/resolution mismatch")
    points = document["points"]
    if len({p["point_id"] for p in points}) != len(points):
        raise ValueError("Duplicate point definitions")
    catalog = {p["point_id"]:p for p in points}
    fm = {int(v):i for i,v in enumerate(data["frame_indices"])}
    pm = {str(v):i for i,v in enumerate(data["point_ids"])}
    if not set(pm).issubset(catalog):
        raise ValueError("Annotation missing query point definitions")
    shape = data["tracks"].shape[:2]
    gt = np.full((*shape,2), np.nan)
    visible, valid = np.zeros(shape,bool), np.zeros(shape,bool)
    records, seen = {}, set()
    excluded = {"pending_or_unreviewed":0, "uncertain":0, "point_definition_unconfirmed":0, "outside_result":0}
    for row in document["annotations"]:
        if type(row["frame_index"]) is not int or not isinstance(row["point_id"], str):
            raise ValueError("Annotation frame must be an integer and point ID a string")
        key = (row["frame_index"], row["point_id"])
        if key in seen: raise ValueError("Duplicate annotation frame/point")
        seen.add(key)
        if row["state"] not in STATES: raise ValueError("Unknown annotation state")
        if key[0] not in fm or key[1] not in pm:
            excluded["outside_result"] += 1
            continue
        t,p = fm[key[0]],pm[key[1]]
        if not isinstance(row["timestamp_seconds"], (float,int)) or not math.isfinite(row["timestamp_seconds"]) or abs(row["timestamp_seconds"]-data["timestamps"][t]) > 1e-6:
            raise ValueError("Annotation timestamp does not match decoded frame")
        point = catalog[key[1]]
        if not (point.get("definition_reviewed") is True and point.get("definition_reviewer", "").strip()
                and point.get("query_correspondence_confirmed") is True and point.get("definition", "").strip()):
            excluded["point_definition_unconfirmed"] += 1
            continue
        if row["state"] == "uncertain" or row.get("uncertainty", "pending") != "none":
            excluded["uncertain"] += 1
            continue
        if row["state"] == "pending" or not (row.get("review_status") == "reviewed" and
            row.get("source") == "human_visual" and row.get("annotator", "").strip() and
            row.get("reviewer", "").strip() and row.get("reviewed_at_utc", "").strip()):
            excluded["pending_or_unreviewed"] += 1
            continue
        vis = row["state"] == "visible"
        expected_flags = {"visible":(False,False), "occluded":(True,False),
                          "out_of_frame":(False,True), "not_visible_unknown":(None,None)}[row["state"]]
        if (row.get("occluded"),row.get("out_of_frame")) != expected_flags:
            raise ValueError("Annotation visibility state and occlusion/bounds flags conflict")
        if vis:
            xy = np.asarray([row.get("x"),row.get("y")], dtype=float)
            if not np.isfinite(xy).all() or (xy < 0).any() or (xy > np.asarray(document["resolution"])-1).any():
                raise ValueError("Reviewed visible point requires finite in-image coordinates")
            gt[t,p] = xy
        elif row.get("x") is not None or row.get("y") is not None:
            raise ValueError("Invisible GT must not contain guessed coordinates")
        visible[t,p],valid[t,p] = vis,True
        records[(t,p)] = row
    return gt,visible,valid,records,excluded


def evaluate_annotations(document, data, manifest, thresholds=(2,5,10,20), loss_threshold=20):
    gt,gv,valid,records,excluded = align_annotations(document,data,manifest)
    fm = {int(v):i for i,v in enumerate(data["frame_indices"])}
    pm = {str(v):i for i,v in enumerate(data["point_ids"])}
    active = np.arange(len(data["frame_indices"]))[:,None] >= data["queries"][:,0][None]
    evaluated = valid & active
    metrics = evaluate_gt(data["tracks"],data["visibility"],gt,gv,valid,active,thresholds)
    metrics.update(gt_available=bool(evaluated.any()),
        status="evaluated_reviewed_sparse_gt" if evaluated.any() else "awaiting_human_annotation_and_review",
        excluded_annotations=excluded, coverage=dict(total_active_point_frames=int(active.sum()),
            reviewed_active_point_frames=int(evaluated.sum()),
            annotated_source_frames=[int(v) for v in data["frame_indices"][evaluated.any(1)]],
            reviewed_visible_point_frames=int((evaluated & gv).sum()),
            fraction=float(evaluated.sum()/active.sum()) if active.any() else None),
        uncertainty_policy="exclude ambiguous/uncertain/unreviewed entries; no interpolation",
        event_policy="observed labeled frames only; transitions across gaps give brackets, not exact times or full-clip recovery rates")
    events,per_point = [],[]
    errors = np.linalg.norm(data["tracks"]-gt,axis=2)
    finite = np.isfinite(data["tracks"]).all(2)
    for p,pid in enumerate(data["point_ids"]):
        indices = np.flatnonzero(evaluated[:,p])
        failed = evaluated[:,p] & gv[:,p] & (~finite[:,p] | ~data["visibility"][:,p] | (errors[:,p] > loss_threshold))
        failures = np.flatnonzero(failed)
        first = int(failures[0]) if len(failures) else None
        earlier_ok = indices[(indices < first) & gv[indices,p] & ~failed[indices]] if first is not None else []
        per_point.append(dict(point_id=str(pid), loss_threshold_pixels=loss_threshold,
            first_observed_failure_source_frame=int(data["frame_indices"][first]) if first is not None else None,
            first_observed_failure_timestamp_seconds=float(data["timestamps"][first]) if first is not None else None,
            previous_observed_good_source_frame=int(data["frame_indices"][earlier_ok[-1]]) if len(earlier_ok) else None,
            loss_time_meaning="first observed labeled failure; onset unknown before/within sparse bracket"))
        for t in indices:
            row = records[(int(t),p)]
            def emit(kind, **extra):
                events.append(dict(kind=kind,point_id=str(pid),source_frame=int(data["frame_indices"][t]),
                    timestamp_seconds=float(data["timestamps"][t]),**extra))
            if not gv[t,p] and data["visibility"][t,p]: emit("false_visible",gt_state=row["state"])
            if failed[t]: emit("observed_tracking_failure",error_pixels=float(errors[t,p]) if finite[t,p] else None,
                               predicted_visible=bool(data["visibility"][t,p]))
        # Only a human case reviewer can assert identity association. Geometric
        # proximity to another blade is insufficient evidence of a material ID switch.
        previous = None
        for t in indices:
            row = records[(int(t),p)]
            if previous is not None:
                before = records[(int(previous),p)]
                if before["state"] != row["state"]:
                    kind = "gt_hidden_entry_bracket" if gv[previous,p] and not gv[t,p] else "gt_visibility_return_bracket" if not gv[previous,p] and gv[t,p] else "gt_state_change_bracket"
                    event=dict(kind=kind,point_id=str(pid),from_frame=int(data["frame_indices"][previous]),
                        to_frame=int(data["frame_indices"][t]),adjacent_labeled_frames=bool(t-previous==1),
                        from_state=before["state"],to_state=row["state"])
                    if kind == "gt_visibility_return_bracket":
                        event["geometric_return_correct_at_observation"] = bool(finite[t,p] and data["visibility"][t,p] and errors[t,p] <= loss_threshold)
                        event["physical_identity_recovery"] = "requires_human_case_review"
                    events.append(event)
            previous = int(t)
    for case in document.get("case_reviews", []):
        if case.get("review_status") != "reviewed" or not case.get("reviewer", "").strip(): continue
        if case.get("kind") not in {"physical_identity_switch", "physical_identity_recovery", "failed_identity_recovery"}: continue
        if case.get("point_id") not in pm or case.get("frame_index") not in fm: raise ValueError("Case outside aligned result")
        target_run = case.get("result_config_sha256")
        if target_run != manifest["config_sha256"]: continue
        t,p = fm[case["frame_index"]],pm[case["point_id"]]
        if not evaluated[t,p] or not case.get("evidence", "").strip(): raise ValueError("Identity case needs reviewed GT and evidence")
        if case["kind"] == "physical_identity_switch" and case.get("associated_point_id") not in pm:
            raise ValueError("Identity switch needs associated physical point ID")
        events.append(dict(case, source_frame=case["frame_index"]))
    metrics["per_point_loss"] = per_point
    metrics["events"] = events
    metrics["identity_metrics"] = dict(status="human_case_reviews_only",reviewed_identity_events=sum(e["kind"].startswith(("physical_identity", "failed_identity")) for e in events),full_period_recovery_rate=None)
    return metrics, (gt,gv,evaluated)
