"""Dump the VELO (VP) Monte Carlo hits of a Gauss .sim file with full truth.

Run inside the LHCb stack:  lb-run Gauss/v56r9 python dump_sim.py <file.sim> <out_prefix>

Per event, writes <out_prefix>_<n>.json with
  hits:      x y z (Geant4 mid-point of the step in the sensor), sensor, module (= sensor // 4),
             particle key, time [ns], |p| [MeV]
  particles: every MC particle with at least one VP hit, plus its origin vertex, mother,
             and the primary vertex it comes from
  pvs:       every primary vertex (MCVertex::isPrimary) with its position
Only the main bunch crossing (/Event/MC/...) is read; spill-over crossings are not.
"""
import json
import sys

from Configurables import LHCbApp
from GaudiConf import IOHelper
import GaudiPython

LHCbApp(DataType="Upgrade", Simulation=True, DDDBtag="dddb-20240427",
        CondDBtag="sim10-2024.W40.42-v00.00-md100")
IOHelper("ROOT").inputFiles([sys.argv[1]])
# .sim files hold packed MC containers (/Event/pSim/...): unpack them first
from Configurables import ApplicationMgr, UnpackMCParticle, UnpackMCVertex, MCVPHitUnpacker
ApplicationMgr().TopAlg += [UnpackMCParticle("UnpackMCParticles"), UnpackMCVertex("UnpackMCVertices"),
                            MCVPHitUnpacker("UnpackVPHits", InputName="/Event/pSim/VP/Hits",
                                            OutputName="/Event/MC/VP/Hits")]
out = sys.argv[2]
app = GaudiPython.AppMgr()
evt = app.evtsvc()


def vtx(v):
    p = v.position()
    return [p.x(), p.y(), p.z()]


n = 0
while True:
    app.run(1)
    if not evt["/Event"]:
        break
    hits = evt["/Event/MC/VP/Hits"]
    mcps = evt["/Event/MC/Particles"]
    mcvs = evt["/Event/MC/Vertices"]
    H, parts = [], {}
    for h in hits:
        mp = h.midPoint()
        p = h.mcParticle()
        key = p.key() if p else -1
        H.append([mp.x(), mp.y(), mp.z(), h.sensDetID(), h.sensDetID() // 4, key, h.time(), h.p()])
        if p and key not in parts:
            ov = p.originVertex()
            pv = p.primaryVertex()
            mo = p.mother()
            mom = p.momentum()
            parts[key] = dict(
                pid=p.particleID().pid(), p=mom.P(), pt=mom.Pt(), eta=mom.Eta(),
                tx=mom.Px() / mom.Pz() if mom.Pz() else 0.0,
                ty=mom.Py() / mom.Pz() if mom.Pz() else 0.0,
                charge=p.particleID().threeCharge() / 3.0,
                origin=vtx(ov) if ov else None,
                origin_type=int(ov.type()) if ov else -1,
                mother=mo.key() if mo else -1,
                pv=pv.key() if pv else -1)
    pvs = [dict(key=v.key(), xyz=vtx(v), type=int(v.type()))
           for v in mcvs if v.isPrimary()]
    # number of MC particles produced directly at each PV (generator multiplicity)
    nprod = {}
    for p in mcps:
        ov = p.originVertex()
        if ov and ov.isPrimary():
            nprod[ov.key()] = nprod.get(ov.key(), 0) + 1
    for v in pvs:
        v["n_prompt"] = nprod.get(v["key"], 0)
    hdr = evt["/Event/Gen/Header"]
    with open(f"{out}_{n}.json", "w") as f:
        json.dump(dict(hits=H, hit_columns=["x", "y", "z", "sensor", "module", "key", "t", "p"],
                       particles={str(k): v for k, v in parts.items()}, pvs=pvs,
                       n_collisions=len(pvs),
                       event_number=int(hdr.evtNumber()) if hdr else n), f)
    print(f"event {n}: {len(H)} VP hits, {len(parts)} particles, {len(pvs)} PVs", flush=True)
    n += 1
