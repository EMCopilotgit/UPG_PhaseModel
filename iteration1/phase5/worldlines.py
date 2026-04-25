# phase5/worldlines.py
import numpy as np
from scipy.spatial import cKDTree

def match_worldlines(defect_positions, defect_charges, defect_types,
                     max_radius=3.0):
    """
    defect_positions[t]: array (Nt, 2) or (Nt,)
    defect_charges[t]:   array (Nt,)
    defect_types[t]:     array (Nt,) or None

    Returns:
        worldlines: list of dicts with keys:
            "id", "times", "positions", "charges", "types"
        events: list of event dicts (see catalogue format below)
    """
    num_steps = len(defect_positions)

    next_id = 0
    active = {}   # id -> last index in worldlines
    worldlines = []

    # Initialize at t=0
    pos0 = defect_positions[0]
    ch0  = defect_charges[0]
    ty0  = defect_types[0] if defect_types is not None else None

    for i in range(len(pos0)):
        wl = {
            "id": next_id,
            "times": [0],
            "positions": [pos0[i]],
            "charges": [ch0[i]],
            "types": [ty0[i] if ty0 is not None else None],
        }
        worldlines.append(wl)
        active[next_id] = len(worldlines) - 1
        next_id += 1

    events = []

    # Step through time
    for t in range(1, num_steps):
        pos_prev = np.array([worldlines[active[i]]["positions"][-1]
                             for i in active])
        ids_prev = list(active.keys())

        pos_curr = defect_positions[t]
        ch_curr  = defect_charges[t]
        ty_curr  = defect_types[t] if defect_types is not None else None

        if len(pos_curr) == 0:
            # all active worldlines die here
            for wid in ids_prev:
                events.append({
                    "type": "death",
                    "time": t,
                    "worldline_ids": [wid],
                })
            active.clear()
            continue

        tree = cKDTree(pos_curr)
        used_curr = set()

        # Match previous → current
        for idx_prev, wid in enumerate(ids_prev):
            p_prev = pos_prev[idx_prev]
            dist, idx = tree.query(p_prev, k=1)
            if dist <= max_radius and idx not in used_curr:
                # charge/type consistency
                if ch_curr[idx] == worldlines[active[wid]]["charges"][-1]:
                    if ty_curr is None or ty_curr[idx] == worldlines[active[wid]]["types"][-1]:
                        # extend worldline
                        wl = worldlines[active[wid]]
                        wl["times"].append(t)
                        wl["positions"].append(pos_curr[idx])
                        wl["charges"].append(ch_curr[idx])
                        wl["types"].append(ty_curr[idx] if ty_curr is not None else None)
                        used_curr.add(idx)
                        continue

            # no match → death
            events.append({
                "type": "death",
                "time": t,
                "worldline_ids": [wid],
            })
            del active[wid]

        # Any unmatched current defects → births
        for i in range(len(pos_curr)):
            if i in used_curr:
                continue
            wl = {
                "id": next_id,
                "times": [t],
                "positions": [pos_curr[i]],
                "charges": [ch_curr[i]],
                "types": [ty_curr[i] if ty_curr is not None else None],
            }
            worldlines.append(wl)
            active[next_id] = len(worldlines) - 1
            events.append({
                "type": "birth",
                "time": t,
                "worldline_ids": [next_id],
            })
            next_id += 1

    return worldlines, events
