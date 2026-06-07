from utils import *

seed = 42
num_scenarios = 15

# PUNTO 1
mu_pizza = np.array([100,65,45,40,35,50])
sigma_pizza = np.array([30,20,15,12,10,15])

rng = np.random.default_rng(seed)

d = sample_d(rng, mu_pizza, sigma_pizza, (len(mu_pizza), num_scenarios))

x, y, obj = solve_model(num_scenarios,d)
print_section("Soluzione del modello stocastico ATO")
print_array("Domanda simulata per prodotto e scenario", d, decimals=0)
print_array("Componenti prodotti nel primo stadio (x)", x)
print_array("Prodotti assemblati nel secondo stadio (y)", y)
print(f"Profitto atteso campionario: {obj:.2f}")

'''
alpha = 0.05
n_sim = 200
media, deviazione, n_scenario_req, lb, ub, phi_list = in_sample_stability(mu_pizza, sigma_pizza, alpha, n_sim, seed)
print_stability_results("Stabilita' in-sample", media, deviazione, n_scenario_req, lb, ub, phi_list)
in_sample_stability_plot(mu_pizza, sigma_pizza, alpha, n_sim, seed)

n_sim = 100
media, deviazione, n_scenario_req, lb, ub, phi_list = out_sample_stability(mu_pizza, sigma_pizza, alpha, n_sim, seed)
print_stability_results("Stabilita' out-of-sample", media, deviazione, n_scenario_req, lb, ub, phi_list)
out_of_sample_stability_plot(mu_pizza, sigma_pizza, alpha, n_sim, seed)
'''
# PUNTO 2
rng = np.random.default_rng(seed)
d = sample_d(rng, mu_pizza, sigma_pizza, (len(mu_pizza), num_scenarios))

VSS, RP_vss, EEV, x_EV = compute_vss(num_scenarios, d)
print_section("Value of the Stochastic Solution (VSS)")
print(f"RP  - valore del problema stocastico: {RP_vss:.2f}")
print(f"EEV - valore della soluzione deterministica valutata sugli scenari: {EEV:.2f}")
print(f"VSS - valore della soluzione stocastica: {VSS:.2f}")
print_array("Soluzione deterministica x_EV", x_EV)

EVPI, WS, RP_evpi, ws_values = compute_evpi(num_scenarios, d)
print_section("Expected Value of Perfect Information (EVPI)")
print(f"WS   - valore wait-and-see medio: {WS:.2f}")
print(f"RP   - valore del problema stocastico: {RP_evpi:.2f}")
print(f"EVPI - valore dell'informazione perfetta: {EVPI:.2f}")
print_array("Valori wait-and-see per scenario", ws_values)


# ANALISI ROBUSTEZZA DELLA SOLUZIONE
variazioni = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
mu_factors = [1 + v for v in variazioni]
sigma_factors = [1 + v for v in variazioni]

matrice_robustezza = compute_robustness_grid(mu_pizza, sigma_pizza, mu_factors, sigma_factors, num_scenarios, seed)

plot_robustness_heatmap(mu_factors, sigma_factors, matrice_robustezza, output_path="robustezza_heatmap.png")
