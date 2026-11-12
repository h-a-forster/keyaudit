"""Stratified sample estimates: keyaudit's stratified_difference_estimate (f=0, floor from pooled variance) and a Jeffreys-posterior interval by stratum simulation (more honest at zero positives)."""
import numpy as np, json, sys
from keyaudit.stats import stratified_difference_estimate, clopper_pearson
def est(NF, nF, kF, NU, nU, kU, seed=20261109):
    y = np.r_[np.ones(kF), np.zeros(nF-kF), np.ones(kU), np.zeros(nU-kU), np.zeros(NF-nF+NU-nU)]
    lab = np.r_[np.ones(nF+nU, bool), np.zeros(NF-nF+NU-nU, bool)]
    st = np.array(["F"]*nF+["U"]*nU+["F"]*(NF-nF)+["U"]*(NU-nU))
    pooled = (kF+kU)/(nF+nU); floor = pooled*(1-pooled)
    d = stratified_difference_estimate(np.zeros(len(y)), lab, y, st, var_floor={"F": floor, "U": floor})
    rng = np.random.default_rng(seed)
    a = rng.beta(kF+.5, nF-kF+.5, 200000); b = rng.beta(kU+.5, nU-kU+.5, 200000)
    # finite-population draw for unlabeled remainder
    r = (kF + rng.binomial(NF-nF, a) + kU + rng.binomial(NU-nU, b)) / (NF+NU)
    return d, np.quantile(r, [.025,.5,.975])
for name,args in json.loads(sys.argv[1]).items():
    d,r=est(*args); print(name,args,"point %.2f keyaudit-normal [%.2f, %.2f] jeffreys [%.2f, %.2f]"%(d.estimate*100,(d.estimate-1.96*d.se)*100,(d.estimate+1.96*d.se)*100,r[0]*100,r[2]*100))
