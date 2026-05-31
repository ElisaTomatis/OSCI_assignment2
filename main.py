from utils import *

seed = 358916
# Numero di scenari
num_scenarios = 10 

mu_pizza = np.array([100,65,45,40,35,50])
sigma_pizza = np.array([30,20,15,12,10,15])

rng = np.random.default_rng(seed)

d = sample_d(rng, mu_pizza, sigma_pizza, (len(mu_pizza), num_scenarios))
print(d)

x, y, obj = solve_model(num_scenarios,d)
print(x)
print(y)
print(round(obj,2))

alpha = 0.05
n_sim = 10
media, deviazione, n_scenario_req, lb, ub, phi_list = in_sample_stability(mu_pizza, sigma_pizza, alpha, n_sim, seed)
print(n_scenario_req)
print(deviazione)
print(media)
print(lb)
print(ub)
print(phi_list)



media, deviazione, n_scenario_req, lb, ub, phi_list = out_sample_stability(mu_pizza, sigma_pizza, alpha, n_sim, seed)
print(n_scenario_req)
print(deviazione)
print(media)
print(lb)
print(ub)
print(phi_list)


# punto 2
rng = np.random.default_rng(42)


d = sample_d(rng, mu_pizza, sigma_pizza, (len(mu_pizza), num_scenarios))

VSS, RP_vss, EEV, x_EV = compute_vss(num_scenarios, d)
print("VSS:", VSS)
print("RP:", RP_vss)
print("EEV:", EEV)
print("x_EV:", x_EV)

EVPI, WS, RP_evpi, ws_values = compute_evpi(num_scenarios, d)
print("EVPI:", EVPI)
print("WS:", WS)
print("RP:", RP_evpi)
