def mission_for(cfg):
    m=cfg['mission']
    return dict(blades=3,surface_sides=[m['side']],desired_photos_per_blade=m['photos_per_blade'],
                allowed_photos_per_blade=[10,15],radial_range=m['radial_range'],chord_range=m['chord_range'],
                span_overlap_fraction=m['span_overlap_fraction'],coverage_policy='surface_footprint',allow_retake=True)
