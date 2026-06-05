from utils import *

seed = 42
num_scenarios = 15


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

a_pizza = np.array([100,65,45,40,35,50])
b_pizza = np.array([30,20,15,12,10,15])

rng = np.random.default_rng(seed)

d = sample_d(rng, a_pizza, b_pizza, (len(a_pizza), num_scenarios))

x, y, obj = solve_model(num_scenarios,d)
print_section("Soluzione del modello stocastico ATO")
print_array("Domanda simulata per prodotto e scenario", d, decimals=0)
print_array("Componenti prodotti nel primo stadio (x)", x)
print_array("Prodotti assemblati nel secondo stadio (y)", y)
print(f"Profitto atteso campionario: {obj:.2f}")

alpha = 0.05
n_sim = 200
media, deviazione, n_scenario_req, lb, ub, phi_list = in_sample_stability(a_pizza, b_pizza, alpha, n_sim, seed)
print_stability_results("Stabilita' in-sample", media, deviazione, n_scenario_req, lb, ub, phi_list)
in_sample_stability_plot(a_pizza, b_pizza, alpha, n_sim, seed)


n_sim = 100
media, deviazione, n_scenario_req, lb, ub, phi_list = out_sample_stability(a_pizza, b_pizza, alpha, n_sim, seed)
print_stability_results("Stabilita' out-of-sample", media, deviazione, n_scenario_req, lb, ub, phi_list)

out_of_sample_stability_plot(a_pizza, b_pizza, alpha, n_sim, seed)


# punto 2
rng = np.random.default_rng(seed)
d = sample_d(rng, a_pizza, b_pizza, (len(a_pizza), num_scenarios))

VSS, RP_vss, EEV, x_EV = compute_vss(num_scenarios, d)
print_section("Value of the Stochastic Solution (VSS)")
print(f"RP  - valore del problema stocastico: {RP_vss:.2f}")
print(f"EEV - valore della soluzione deterministica valutata sugli scenari: {EEV:.2f}")
print(f"VSS - valore della soluzione stocastica: {VSS:.2f}")
print_array("Soluzione deterministica x_EV", x_EV)
plot_value_histogram(
    ["EEV", "RP", "VSS"],
    [EEV, RP_vss, VSS],
    "Confronto EEV, RP e VSS",
    "confronto_eev_rp_vss.png",
)

EVPI, WS, RP_evpi, ws_values = compute_evpi(num_scenarios, d)
print_section("Expected Value of Perfect Information (EVPI)")
print(f"WS   - valore wait-and-see medio: {WS:.2f}")
print(f"RP   - valore del problema stocastico: {RP_evpi:.2f}")
print(f"EVPI - valore dell'informazione perfetta: {EVPI:.2f}")
print_array("Valori wait-and-see per scenario", ws_values)
plot_value_histogram(
    ["RP", "WS", "EVPI"],
    [RP_evpi, WS, EVPI],
    "Confronto RP, WS ed EVPI",
    "confronto_rp_ws_evpi.png",
)


# ANALISI ROBUSTEZZA DELLA SOLUZIONE

stress_cases = {}
for variazione in [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15]:
    percentuale = int(abs(variazione) * 100)
    direzione = "bassi" if variazione < 0 else "alti"
    stress_cases[f"parametri_{direzione}_{percentuale}pct"] = {
        "a_vero": a_pizza * (1 + variazione),
        "b_vero": b_pizza * (1 + variazione),
    }

for nome_caso, params in stress_cases.items():
    valore_assunto, valore_x_assunta_su_vera, valore_ottimo_vero, perdita, perdita_percentuale, x_assunta, x_vera = robustness_analysis(
        a_pizza, b_pizza, params["a_vero"], params["b_vero"], num_scenarios, seed)

    print_section(f"Robustezza - {nome_caso}")
    print("Valore soluzione assunta:", round(valore_assunto, 2))
    print("Valore x assunta su distribuzione vera:", round(valore_x_assunta_su_vera, 2))
    print("Valore ottimo con distribuzione vera:", round(valore_ottimo_vero, 2))
    print("Perdita:", round(perdita, 2))
    print("Perdita %:", round(perdita_percentuale, 2))


# HEATMAP VSS ED EVPI AL VARIARE DEI PARAMETRI a_i E b_i

a_factors = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
b_factors = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])

vss_grid, evpi_grid = compute_vss_evpi_grid(
    a_pizza,
    b_pizza,
    a_factors,
    b_factors,
    num_scenarios,
    seed,
)

plot_vss_evpi_heatmaps(
    a_factors,
    b_factors,
    vss_grid,
    evpi_grid,
    output_path="vss_evpi_heatmaps.png",
)

print_section("Heatmap VSS ed EVPI")
print_array("Griglia VSS", vss_grid)
print_array("Griglia EVPI", evpi_grid)
print("Grafico salvato in: vss_evpi_heatmaps.png")


# STUDIO SULLA ROBUSTEZZA DELLA DOMANDA
nomi_comp = ["Impasto", "Salsa di pomodoro", "Mozzarella", "Salame piccante", "Prosciutto cotto", 
            "Funghi", "Carciofini", "Mix 4 Formaggi", "Verdure grigliate", "Salsiccia"]


# 2. Eseguiamo l'analisi (usando la funzione 'robustness_gamma_analysis' del messaggio precedente)
risultati = robustness_distribution(a_pizza, b_pizza, S_train=50, S_test=190, seed=42, distribution="gamma")

# 3. Generiamo il grafico
robustness_distribution_plot(risultati, component_names=nomi_comp, distribution="gamma")
print(f'{risultati["perdita"]}, {risultati["perdita_perc"]}, {risultati["x_diff"]}')

risultati = robustness_distribution(a_pizza, b_pizza, S_train=50, S_test=190, seed=42, distribution="uniforme")

# 3. Generiamo il grafico
robustness_distribution_plot(risultati, component_names=nomi_comp, distribution="uniforme")
print(f'{risultati["perdita"]}, {risultati["perdita_perc"]}, {risultati["x_diff"]}')
