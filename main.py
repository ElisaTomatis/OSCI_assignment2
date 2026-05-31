from utils import *

seed = 42
# Numero di scenari
num_scenarios = 10


def print_section(title):
    """Stampa un'intestazione leggibile per separare i risultati."""

    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def print_array(label, values, decimals=2):
    """Stampa vettori e matrici numpy con etichetta e arrotondamento."""

    formatted_values = np.array2string(
        np.asarray(values),
        precision=decimals,
        suppress_small=True,
    )
    print(f"{label}:\n{formatted_values}\n")


def print_stability_results(label, media, deviazione, n_scenario_req, lb, ub, phi_list):
    """Stampa in modo compatto i risultati delle analisi di stabilita'."""

    print_section(label)
    print(f"Numero di scenari richiesto: {n_scenario_req}")
    print(f"Media campionaria phi: {media:.2f}")
    print(f"Deviazione standard phi: {deviazione:.2f}")
    print(f"Intervallo di confidenza: [{lb:.2f}, {ub:.2f}]")
    print_array("Valori phi delle repliche", phi_list)

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

alpha = 0.05
n_sim = 10
media, deviazione, n_scenario_req, lb, ub, phi_list = in_sample_stability(mu_pizza, sigma_pizza, alpha, n_sim, seed)
print_stability_results("Stabilita' in-sample", media, deviazione, n_scenario_req, lb, ub, phi_list)

media, deviazione, n_scenario_req, lb, ub, phi_list = out_sample_stability(mu_pizza, sigma_pizza, alpha, n_sim, seed)
print_stability_results("Stabilita' out-of-sample", media, deviazione, n_scenario_req, lb, ub, phi_list)


# punto 2
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

stress_cases = {
    "media_bassa_varianza_bassa": {
        "mu_vera": mu_pizza * 0.85,
        "sigma_vera": sigma_pizza * 0.80,
    },
    "media_alta_varianza_alta": {
        "mu_vera": mu_pizza * 1.15,
        "sigma_vera": sigma_pizza * 1.25,
    },
    "media_bassa_varianza_alta": {
        "mu_vera": mu_pizza * 0.85,
        "sigma_vera": sigma_pizza * 1.25,
    },
    "media_alta_varianza_bassa": {
        "mu_vera": mu_pizza * 1.15,
        "sigma_vera": sigma_pizza * 0.80,
    },
}

for nome_caso, params in stress_cases.items():
    valore_assunto, valore_x_assunta_su_vera, valore_ottimo_vero, perdita, perdita_percentuale, x_assunta, x_vera = robustness_analysis(
        mu_pizza, sigma_pizza, params["mu_vera"], params["sigma_vera"], num_scenarios, seed)

    print_section(f"Robustezza - {nome_caso}")
    print("Valore soluzione assunta:", round(valore_assunto, 2))
    print("Valore x assunta su distribuzione vera:", round(valore_x_assunta_su_vera, 2))
    print("Valore ottimo con distribuzione vera:", round(valore_ottimo_vero, 2))
    print("Perdita:", round(perdita, 2))
    print("Perdita %:", round(perdita_percentuale, 2))
