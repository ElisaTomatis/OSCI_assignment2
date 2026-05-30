from utils import *

media, deviazione, n_scenario_req, lb, ub, phi_list = in_sample_stability(1, 0.01, 0.05, 5, 42)
print(n_scenario_req)
print(deviazione)
print(media)
print(lb)
print(ub)
print(phi_list)



media, deviazione, n_scenario_req, lb, ub, phi_list = out_sample_stability(1, 0.01, 0.05, 10, 42)
print(n_scenario_req)
print(deviazione)
print(media)
print(lb)
print(ub)
print(phi_list)


# punto 2
rng = np.random.default_rng(42)

S = 100
mu = 1
sigma = 0.01

d = rng.lognormal(mean=mu, sigma=sigma, size=(J, S))

VSS, RP_vss, EEV, x_EV = compute_vss(S, d)
print("VSS:", VSS)
print("RP:", RP_vss)
print("EEV:", EEV)
print("x_EV:", x_EV)

EVPI, WS, RP_evpi, ws_values = compute_evpi(S, d)
print("EVPI:", EVPI)
print("WS:", WS)
print("RP:", RP_evpi)