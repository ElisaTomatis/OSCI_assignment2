import gurobipy as gp
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

"""
Utility per un modello Assembly To Order (ATO) a due stadi.

Il primo stadio sceglie le quantita' di componenti da produrre prima di
conoscere la domanda. Il secondo stadio, per ciascuno scenario di domanda,
sceglie quante unita' di prodotto finito assemblare usando i componenti
disponibili. La domanda dei prodotti finiti viene simulata con una
distribuzione lognormale.
"""

# Dati del problema.
#
# I: numero di componenti.
# J: numero di prodotti finiti.
# M: numero di risorse/macchine con capacita' limitata.
#
# C[i]: costo unitario di produzione del componente i.
# P[j]: ricavo unitario del prodotto finito j.
# L[m]: capacita' disponibile della macchina m.
# T[i, m]: tempo/capacita' della macchina m richiesto per produrre
#          una unita' del componente i.
# G[i, j]: unita' del componente i necessarie per assemblare una unita'
#          del prodotto finito j.

I = 10  
J = 6   
M = 3   # m1: Impasto, m2: Taglio, m3: Cottura

C = np.array([
    [0.50],  # 1. Impasto
    [0.40],  # 2. Salsa di pomodoro
    [0.80],  # 3. Mozzarella
    [1.00],  # 4. Salame piccante
    [0.90],  # 5. Prosciutto cotto
    [0.70],  # 6. Funghi
    [0.80],  # 7. Carciofini
    [1.50],  # 8. Mix 4 Formaggi
    [0.90],  # 9. Verdure grigliate
    [1.10]   # 10. Salsiccia
])
P = np.array([
    [6.50],  # 1. Margherita
    [8.00],  # 2. Diavola
    [9.50],  # 3. Capricciosa
    [9.00],  # 4. Quattro Formaggi
    [8.50],  # 5. Ortolana
    [9.00]   # 6. Boscaiola
])


G = np.array([
    # Mar, Dia, Cap, 4Fo, Ort, Bos
    [1,   1,   1,   1,   1,   1], # 1. Impasto
    [1,   1,   1,   0,   1,   0], # 2. Salsa pomodoro
    [1,   1,   1,   0,   1,   1], # 3. Mozzarella
    [0,   1,   0,   0,   0,   0], # 4. Salame piccante
    [0,   0,   1,   0,   0,   0], # 5. Prosciutto cotto
    [0,   0,   1,   0,   0,   1], # 6. Funghi
    [0,   0,   1,   0,   0,   0], # 7. Carciofini
    [0,   0,   0,   1,   0,   0], # 8. Mix 4 Formaggi
    [0,   0,   0,   0,   1,   0], # 9. Verdure grigliate
    [0,   0,   0,   0,   0,   1]  # 10. Salsiccia
])


T = np.array([
    # m1(Impasto), m2(Taglio), m3(Cottura)
    [0.5,          0,          0   ], # 1. Impasto
    [0,            0,          0.2 ], # 2. Salsa pomodoro
    [0,            0.3,        0   ], # 3. Mozzarella
    [0,            0.15,       0   ], # 4. Salame piccante
    [0,            0.15,       0   ], # 5. Prosciutto cotto
    [0,            0,          0.4 ], # 6. Funghi
    [0,            0,          0.2 ], # 7. Carciofini
    [0,            0.5,        0   ], # 8. Mix 4 Formaggi
    [0,            0,          0.6 ], # 9. Verdure grigliate
    [0,            0,          0.5 ]  # 10. Salsiccia
])

L = np.array([
    [240], # Impasto
    [300], # Taglio
    [360]  # Cottura
])


def sample_d(rng, mu_pizza, sigma_pizza, size):
    """
    Genera scenari di domanda per i prodotti finiti

    La domanda di ciascun prodotto viene simulata con una distribuzione
    lognormale parametrizzata in modo che media e deviazione standard siano
    quelle passate in input. I valori simulati vengono arrotondati per
    rappresentare un numero intero di pizze richieste

    Parametri
    rng : numpy.random.Generator
        Generatore casuale usato per rendere replicabile la simulazione
    mu_pizza : numpy.ndarray, shape (J,)
        Domanda media attesa per ciascun prodotto
    sigma_pizza : numpy.ndarray, shape (J,)
        Deviazione standard della domanda per ciascun prodotto
    size : tuple[int, int]
        Coppia (J, S), dove J e' il numero di prodotti e S il numero di
        scenari da generare

    Ritorna
    d : numpy.ndarray, shape (J, S)
        Matrice delle domande simulate. L'elemento d[j, s] e' la domanda del
        prodotto j nello scenario s.
    """

    sigmas = np.sqrt(np.log(1+sigma_pizza**2/mu_pizza**2))
    mus = np.log(mu_pizza)-sigmas**2/2
    
    J,S = size
    d = np.zeros((J,S))

    for j in range(J):
        d[j, :] = rng.lognormal(mus[j], sigmas[j], S)

    d = np.round(d)
    
    return d


def solve_model(S, d):
    """
    Risolve il problema stocastico ATO a due stadi per un campione di scenari.

    Parametri
    S : int
        Numero di scenari di domanda considerati nella Sample Average
        Approximation (SAA).
    d : numpy.ndarray, shape (J, S)
        Matrice delle domande simulate. L'elemento d[j, s] e' la domanda del
        prodotto j nello scenario s. Viene usata anche come upper bound per
        la variabile y[j, s], perche' non ha senso assemblare piu' della
        domanda osservata.


    Modello

    Variabili:
    - x[i]: quantita' del componente i prodotta nel primo stadio (prima di osservare la domanda)
    - y[j, s]: quantita' del prodotto j assemblata nel secondo stadio nello scenario s

    Obiettivo:
    massimizzare il profitto atteso campionario
        ricavi attesi dai prodotti assemblati - costi dei componenti prodotti
    Gli scenari sono equiprobabili, quindi pi_s = 1 / S

    Vincoli:
    - capacita' produttiva: T^T x <= L
    - disponibilita' componenti: G y_s <= x per ogni scenario s
    - domanda (lognormale): 0 <= y[j, s] <= d[j, s] (non posso vendere/assemblare più unità del prodotto j rispetto alla domanda osservata)


    Ritorna
    x_np : numpy.ndarray, shape (I, 1)
        Soluzione ottima di primo stadio
    y_np : numpy.ndarray, shape (J, S)
        Soluzioni ottime di secondo stadio per tutti gli scenari
    obj_np : float
        Valore ottimo del profitto atteso campionario
    """

    pi = np.ones((S,1))/S # vettori di probabilità degli scenari (equiprobabili)

    # Bounds
    x_lb = np.zeros((I,1)) # non si possono produrre quantità negative
    y_lb = np.zeros((J,S)) # non si possono assemblare quantità negative
    y_ub = d

    # Variabili decisionali
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()

        with gp.Model(env=env) as model:

            # variabili del modello
            x = model.addMVar((I,1), name="x", lb=x_lb)
            y = model.addMVar((J,S), name="y", lb=y_lb, ub=y_ub)

            # funzione obiettivo da massimizzare: resistuisce il prezzo del prodotto per quantità assemblata attesa
            model.setObjective(-C.flatten() @ x + P.flatten() @ (y @ pi.flatten()), gp.GRB.MAXIMIZE)

            # Vincolo di capacita' produttiva delle macchine.
            constr_cap_prod = model.addConstr(np.transpose(T) @ x <= L, name="constr_cap_prod")

            # Vincolo di disponibilita' dei componenti per l'assemblaggio.
            constr_assembl = model.addConstr(G @ y <= gp.hstack([x] * S), name="constr_assembl")

            model.optimize()

            x_np = x.X # estrazione della soluzione ottima di x
            y_np = y.X # estrazione della soluzione ottima di y
            obj_np = model.ObjVal # estrazione del valore ottimo della funzione obiettivo

    return x_np, y_np, obj_np


def solve_model_x_fixed(S, d, x):
    """
    Valuta una soluzione di primo stadio fissata su un insieme di scenari.

    Questa funzione risolve solo il secondo stadio: la quantita' di componenti
    x e' gia' decisa e non puo' essere modificata. Serve quindi per stimare il
    valore out-of-sample di una soluzione ottenuta su un campione piccolo,
    testandola su un campione piu' grande o diverso.

    Parametri
    S : int
        Numero di scenari su cui valutare la soluzione fissata.
    d : numpy.ndarray, shape (J, S)
        Matrice delle domande degli scenari di valutazione.
    x : numpy.ndarray, shape (I, 1)
        Quantita' di componenti gia' prodotte nel primo stadio.

    Ritorna
    y_np : numpy.ndarray, shape (J, S)
        Quantita' ottime da assemblare in ciascuno scenario, dato x.
    obj_np : float
        Profitto atteso campionario ottenuto mantenendo x fissato.
    """

    pi = np.ones((S,1))/S

    # Bounds
    y_lb = np.zeros((J,S))
    y_ub = d

    # Variabili decisionali
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()

        with gp.Model(env=env) as model:

            y = model.addMVar((J,S), name="y", lb=y_lb, ub=y_ub)

            model.setObjective(-C.flatten() @ x + P.flatten() @ (y @ pi.flatten()), gp.GRB.MAXIMIZE)

            # Vincolo di disponibilita' dei componenti per l'assemblaggio.
            constr_assembl = model.addConstr(G @ y <= gp.hstack([x] * S), name="constr_assembl")

            model.optimize()

            y_np = y.X
            obj_np = model.ObjVal

    return y_np, obj_np


def in_sample_stability(mu, sigma, alpha, n_sim, seed):
    """
    Studia la stabilita' in-sample al crescere del numero di scenari S

    Per ogni valore di S vengono generati, per n_sim repliche, due campioni
    indipendenti di domanda della stessa dimensione. Su ciascun campione si
    risolve il problema stocastico e si calcola la differenza tra i due valori
    ottimi:

        phi = z_S^1 - z_S^2

    Se S e' sufficientemente grande, due campioni della stessa distribuzione
    dovrebbero produrre valori ottimi simili. La procedura aumenta S finche'
    l'intervallo di confidenza per E[phi] contiene 0

    Parametri
    mu : float
        Parametro di media della normale sottostante alla lognormale
    sigma : float
        Deviazione standard della normale sottostante alla lognormale
    alpha : float
        Livello di significativita' dell'intervallo di confidenza. Per esempio
        alpha=0.05 produce un intervallo al 95%
    n_sim : int
        Numero di repliche Monte Carlo usate per stimare media e deviazione
        standard di phi per ogni S
    seed : int
        Seed del generatore casuale, utile per rendere replicabili i risultati

    Ritorna
    phi_campionaria : float
        Media campionaria delle differenze phi all'ultimo S testato
    sigma_campionaria : float
        Deviazione standard campionaria delle differenze phi
    n_scenario : int
        Numero di scenari S richiesto per ottenere stabilita' secondo il
        criterio dell'intervallo di confidenza
    lb_conf_int, ub_conf_int : float
        Estremi inferiore e superiore dell'intervallo di confidenza
    phi_list : list[float]
        Differenze osservate nelle n_sim repliche all'ultimo S testato
    """
    
    n_scenario = 1
    rng = np.random.default_rng(seed)

    # quantile della normale standard
    z_alpha = scipy.stats.norm.ppf(1-alpha/2)

    lb_conf_int = np.inf
    ub_conf_int = -np.inf
    profitto_stimato = 1
    
    while (not (lb_conf_int <= 0 <= ub_conf_int)) or (abs(lb_conf_int - ub_conf_int)/profitto_stimato > 0.01):
        # Se lo zero e' dentro l'intervallo, allora la differenza media tra due soluzioni 
        # ottenute con campioni diversi non e' significativamente diversa da zero
        
        profitto_medio = []
        phi_list = []
        n_scenario += 1
        print(n_scenario)
        for _ in range(n_sim):

            d1 = sample_d(rng, mu, sigma, (J, n_scenario))
            d2 = sample_d(rng, mu, sigma, (J, n_scenario))

            _, _, sol1 = solve_model(n_scenario, d1)
            _, _, sol2 = solve_model(n_scenario, d2)
            profitto_medio.append((sol1 + sol2)/2)

            phi_list.append(sol1 - sol2)

        profitto_stimato = np.mean(profitto_medio)
        phi_campionaria = np.mean(phi_list)
        sigma_campionaria = np.std(phi_list, ddof=1)
        # ddof=1 significa che si usa la formula campionaria dividendo per n_sim - 1 

        lb_conf_int = phi_campionaria - z_alpha * sigma_campionaria / np.sqrt(n_sim)
        ub_conf_int = phi_campionaria + z_alpha * sigma_campionaria / np.sqrt(n_sim)

    return phi_campionaria, sigma_campionaria, n_scenario, lb_conf_int, ub_conf_int, phi_list


def in_sample_stability_plot(mu, sigma, alpha, n_sim, seed):
    # Liste per il grafico
    s_values = []
    phi_means = []
    conf_intervals = [] # Per le barre di errore

    n_scenario = 1
    rng = np.random.default_rng(seed)

    # quantile della normale standard
    z_alpha = scipy.stats.norm.ppf(1-alpha/2)

    lb_conf_int = np.inf
    ub_conf_int = -np.inf
    profitto_stimato = 1

    
    while (not (lb_conf_int <= 0 <= ub_conf_int)) or (abs(lb_conf_int - ub_conf_int)/profitto_stimato > 0.005):
        
        profitto_medio = []
        phi_list = []
        n_scenario += 1
        print(n_scenario)
        for _ in range(n_sim):

            d1 = sample_d(rng, mu, sigma, (J, n_scenario))
            d2 = sample_d(rng, mu, sigma, (J, n_scenario))

            _, _, sol1 = solve_model(n_scenario, d1)
            _, _, sol2 = solve_model(n_scenario, d2)
            profitto_medio.append((sol1 + sol2)/2)

            phi_list.append(sol1 - sol2)
        
        profitto_stimato = np.mean(profitto_medio)
        phi_campionaria = np.mean(phi_list)
        sigma_campionaria = np.std(phi_list, ddof=1)

        # Al termine del calcolo per ogni n_scenario, salva i dati:
        s_values.append(n_scenario)
        phi_means.append(phi_campionaria)
        # Salviamo il margine di errore (metà ampiezza IC)
        conf_intervals.append(z_alpha * sigma_campionaria / np.sqrt(n_sim))

        lb_conf_int = phi_campionaria - z_alpha * sigma_campionaria / np.sqrt(n_sim)
        ub_conf_int = phi_campionaria + z_alpha * sigma_campionaria / np.sqrt(n_sim)

    plt.figure(figsize=(10, 6))
    plt.errorbar(s_values, phi_means, yerr=conf_intervals, fmt='-o', capsize=5, ecolor='red', label='Media $\phi$ con IC 95%')
    plt.axhline(y=0, color='black', linestyle='--') # Linea di riferimento per lo zero
    plt.xlabel('Numero di Scenari (S)')
    plt.ylabel('Differenza Valore Ottimo ($\phi$)')
    plt.title('Analisi Stabilità In-Sample: Convergenza della differenza media')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('stabilita_in_sample.png', dpi=300, bbox_inches='tight')
    plt.show()


def out_sample_stability(mu, sigma, alpha, n_sim, seed):
    """
    Studia la stabilita' out-of-sample delle soluzioni ottenute con S scenari

    Per ogni valore di S la funzione:
    1. genera un campione grande D, usato come approssimazione della distribuzione vera della domanda
    2. per ogni replica genera un campione piccolo d con S scenari
    3. risolve il problema su d, ottenendo una soluzione di primo stadio x
    4. valuta quella stessa x sul campione grande D con solve_model_x_fixed
    5. confronta il valore in-sample con il valore out-of-sample
        phi = valore_in_sample - valore_out_of_sample

    La procedura aumenta S finche' l'intervallo di confidenza per E[phi]
    contiene 0, oppure finche' S arriva a 50. Un intervallo che contiene 0
    indica che, con il criterio adottato, non emerge una differenza sistematica
    tra il valore stimato sul campione di ottimizzazione e quello stimato sul
    campione di valutazione

    Parametri
    mu : float
        Parametro di media della normale sottostante alla lognormale
    sigma : float
        Deviazione standard della normale sottostante alla lognormale
    alpha : float
        Livello di significativita' dell'intervallo di confidenza
    n_sim : int
        Numero di repliche Monte Carlo per ogni S
    seed : int
        Seed del generatore casuale

    Ritorna
    phi_campionaria : float
        Media campionaria delle differenze phi all'ultimo S testato
    sigma_campionaria : float
        Deviazione standard campionaria delle differenze phi
    n_scenario : int
        Numero di scenari S raggiunto dalla procedura
    lb_conf_int, ub_conf_int : float
        Estremi dell'intervallo di confidenza per E[phi]
    phi_list : list[float]
        Differenze osservate nelle n_sim repliche all'ultimo S testato
    """
    
    n_scenario = 0
    big_n_scenario = 200
    rng = np.random.default_rng(seed)

    z_alpha = scipy.stats.norm.ppf(1-alpha/2)

    lb_conf_int = np.inf
    ub_conf_int = -np.inf

    while (not (lb_conf_int <= 0 <= ub_conf_int) and (n_scenario < 100)):
        
        phi_list = []
        n_scenario += 10
        print(n_scenario)

        for _ in range(n_sim):

            d = sample_d(rng, mu, sigma, (J, n_scenario))
            x, _, sol = solve_model(n_scenario, d)
            # valore in-sample, stimato sugli stessi scenari usati per ottimizzare

            # Se produco i componenti x trovati prima, quanto guadagno quando la domanda segue il campione grande D?
            D = sample_d(rng, mu, sigma, (J, big_n_scenario))
            _, SOL = solve_model_x_fixed(big_n_scenario, D, x)

            phi_list.append(sol - SOL)

        phi_campionaria = np.mean(phi_list)
        sigma_campionaria = np.std(phi_list, ddof=1)

        lb_conf_int = phi_campionaria - z_alpha * sigma_campionaria / np.sqrt(n_sim)
        ub_conf_int = phi_campionaria + z_alpha * sigma_campionaria / np.sqrt(n_sim)
        print(f"intervallo di confidenza [{lb_conf_int},{ub_conf_int}]")

    return phi_campionaria, sigma_campionaria, n_scenario, lb_conf_int, ub_conf_int, phi_list


def out_of_sample_stability_plot(mu, sigma, alpha, n_sim, seed):
    """
    Esegue l'analisi out-of-sample e genera il grafico a forchetta.
    """
    n_scenario = 0
    max_s = 100
    big_n_scenario = 200  # Campione grande per approssimare la distribuzione vera
    rng = np.random.default_rng(seed)
    z_alpha = scipy.stats.norm.ppf(1-alpha/2)

    # Liste per memorizzare i dati del plot
    s_values = []
    in_sample_means = []
    out_sample_means = []
    
    # Generiamo un campione GRANDE una sola volta per coerenza nel test out-of-sample
    # D_big = sample_d(rng, mu, sigma, (J, big_n_scenario))

    lb_conf_int = np.inf
    ub_conf_int = -np.inf

    print("Inizio analisi Out-of-Sample...")

    # Continua finché non c'è stabilità o raggiungiamo il limite max_s
    while (not (lb_conf_int <= 0 <= ub_conf_int) and (n_scenario < max_s)):

        n_scenario += 5 # Incremento di 2 per rendere il plot più leggibile e veloce
        s_values.append(n_scenario)
        
        tmp_in_sample = []
        tmp_out_sample = []
        phi_list = []

        for _ in range(n_sim):
            # 1. Campione piccolo per ottimizzazione
            d_small = sample_d(rng, mu, sigma, (J, n_scenario))
            
            # 2. Ottimizzazione (In-Sample)
            x_hat, _, sol_in = solve_model(n_scenario, d_small)
            tmp_in_sample.append(sol_in)

            # 3. Valutazione della x_hat sul campione grande (Out-of-Sample)
            D_big = sample_d(rng, mu, sigma, (J, big_n_scenario))
            _, sol_out = solve_model_x_fixed(big_n_scenario, D_big, x_hat)
            tmp_out_sample.append(sol_out)
            
            # Differenza per il criterio di stop
            phi_list.append(sol_in - sol_out)

        # Medie per il plot
        avg_in = np.mean(tmp_in_sample)
        avg_out = np.mean(tmp_out_sample)
        in_sample_means.append(avg_in)
        out_sample_means.append(avg_out)

        # Calcolo intervallo di confidenza per il criterio di stop
        phi_campionaria = np.mean(phi_list)
        sigma_phi = np.std(phi_list, ddof=1)
        lb_conf_int = phi_campionaria - z_alpha * sigma_phi / np.sqrt(n_sim)
        ub_conf_int = phi_campionaria + z_alpha * sigma_phi / np.sqrt(n_sim)
        print(f"intervallo di confidenza [{lb_conf_int},{ub_conf_int}]")
        
        print(f"S={n_scenario} | In-Sample: {avg_in:.2f} | Out-of-Sample: {avg_out:.2f} | Gap: {phi_campionaria:.2f}")

    # --- GENERAZIONE E SALVATAGGIO DEL PLOT ---
    plt.figure(figsize=(10, 6))
    plt.plot(s_values, in_sample_means, 'o-', label='Valore In-Sample ($z_S^*$)', color='blue')
    plt.plot(s_values, out_sample_means, 's-', label='Valore Out-of-Sample (Evaluated $\hat{x}$)', color='green')
    
    plt.title('Stabilità Out-of-Sample: Grafico a Forchetta', fontsize=14)
    plt.xlabel('Numero di Scenari di Ottimizzazione ($S$)', fontsize=12)
    plt.ylabel('Valore della Funzione Obiettivo (Profitto Atteso)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Salvataggio come richiesto per il report
    plt.savefig('stabilita_out_sample_forchetta.pdf', bbox_inches='tight')
    plt.savefig('stabilita_out_sample_forchetta.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_vss(S, d):
    """
    Calcola il Value of the Stochastic Solution (VSS)

    RP e' il valore ottimo del problema stocastico
    EEV e' il valore atteso ottenuto usando la soluzione deterministica
    calcolata sulla domanda media

    Per un problema di massimizzazione:
        VSS = RP - EEV
    """

    # Problema stocastico: ottimizzo usando tutti gli scenari
    _, _, RP = solve_model(S, d)

    # Domanda media campionaria
    d_mean = np.mean(d, axis=1, keepdims=True)

    # Problema deterministico: ottimizzo usando solo la domanda media
    x_EV, _, _ = solve_model(1, d_mean)

    # Valuto la soluzione deterministica x_EV sugli scenari originali
    _, EEV = solve_model_x_fixed(S, d, x_EV)

    VSS = RP - EEV

    return VSS, RP, EEV, x_EV


def compute_evpi(S, d):
    """
    Calcola l'Expected Value of Perfect Information (EVPI)

    RP e' il valore ottimo del problema stocastico, dove x deve essere scelto
    prima di conoscere quale scenario si realizzera'

    WS e' il valore wait-and-see: per ogni scenario si risolve il problema
    come se la domanda fosse nota in anticipo

    Per un problema di massimizzazione:
        EVPI = WS - RP
    """

    # Problema stocastico: x e' unica per tutti gli scenari
    _, _, RP = solve_model(S, d)

    ws_values = []

    for s in range(S):
        # Estraggo lo scenario s come matrice J x 1
        d_s = d[:, [s]]

        # Risolvo il problema sapendo gia' che si realizzera' quello scenario
        _, _, obj_s = solve_model(1, d_s)

        ws_values.append(obj_s)

    # Media dei valori wait-and-see, scenari equiprobabili
    WS = np.mean(ws_values)

    EVPI = WS - RP

    return EVPI, WS, RP, ws_values


def robustness_analysis(mu_assunta, sigma_assunta, mu_vera, sigma_vera, S, seed):
    """
    Valuta la robustezza della soluzione di primo stadio rispetto a errori
    nel modello di domanda

    La funzione ottimizza x usando la distribuzione assunta, poi valuta quella
    stessa x su scenari generati dalla distribuzione vera. Il risultato viene
    confrontato con il valore che si otterrebbe ottimizzando direttamente sulla
    distribuzione vera
    """

    rng = np.random.default_rng(seed)

    d_train = sample_d(rng, mu_assunta, sigma_assunta, (J, S))
    x_assunta, _, valore_assunto = solve_model(S, d_train)

    d_test = sample_d(rng, mu_vera, sigma_vera, (J, S))
    _, valore_robusto = solve_model_x_fixed(S, d_test, x_assunta)

    x_vera, _, valore_ottimo_vero = solve_model(S, d_test)

    perdita = valore_ottimo_vero - valore_robusto
    perdita_percentuale = perdita / valore_ottimo_vero * 100

    return valore_assunto, valore_robusto, valore_ottimo_vero, perdita, perdita_percentuale, x_assunta, x_vera


def compute_vss_evpi_grid(mu_base, sigma_base, mu_factors, sigma_factors, S, seed):
    """
    Calcola VSS ed EVPI per una griglia di valori di media e deviazione standard.

    Ogni cella della griglia corrisponde a una distribuzione di domanda ottenuta
    moltiplicando la media base per un fattore di mu e la deviazione standard
    base per un fattore di sigma.

    Parametri
    mu_base : numpy.ndarray, shape (J,)
        Vettore delle medie di riferimento.
    sigma_base : numpy.ndarray, shape (J,)
        Vettore delle deviazioni standard di riferimento.
    mu_factors : list[float] or numpy.ndarray
        Moltiplicatori applicati a mu_base.
    sigma_factors : list[float] or numpy.ndarray
        Moltiplicatori applicati a sigma_base.
    S : int
        Numero di scenari generati per ogni combinazione.
    seed : int
        Seed del generatore casuale.

    Ritorna
    vss_grid : numpy.ndarray, shape (len(sigma_factors), len(mu_factors))
        Valori di VSS per ogni combinazione.
    evpi_grid : numpy.ndarray, shape (len(sigma_factors), len(mu_factors))
        Valori di EVPI per ogni combinazione.
    """

    rng = np.random.default_rng(seed)
    vss_grid = np.zeros((len(sigma_factors), len(mu_factors)))
    evpi_grid = np.zeros((len(sigma_factors), len(mu_factors)))

    for i, sigma_factor in enumerate(sigma_factors):
        for j, mu_factor in enumerate(mu_factors):
            mu_test = mu_base * mu_factor
            sigma_test = sigma_base * sigma_factor
            d = sample_d(rng, mu_test, sigma_test, (J, S))

            VSS, _, _, _ = compute_vss(S, d)
            EVPI, _, _, _ = compute_evpi(S, d)

            vss_grid[i, j] = VSS
            evpi_grid[i, j] = EVPI

    return vss_grid, evpi_grid


def plot_vss_evpi_heatmaps(mu_factors, sigma_factors, vss_grid, evpi_grid, output_path=None):
    """
    Disegna due heatmap affiancate per visualizzare VSS ed EVPI.

    Sull'asse orizzontale vengono riportati i moltiplicatori della media,
    mentre sull'asse verticale vengono riportati i moltiplicatori della
    deviazione standard. Se output_path e' valorizzato, il grafico viene
    salvato su file; altrimenti viene mostrato a schermo.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    heatmaps = [
        (axes[0], vss_grid, "VSS"),
        (axes[1], evpi_grid, "EVPI"),
    ]

    for ax, grid, title in heatmaps:
        image = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Moltiplicatore media")
        ax.set_ylabel("Moltiplicatore deviazione standard")
        ax.set_xticks(np.arange(len(mu_factors)))
        ax.set_yticks(np.arange(len(sigma_factors)))
        ax.set_xticklabels([f"{factor:.2f}" for factor in mu_factors])
        ax.set_yticklabels([f"{factor:.2f}" for factor in sigma_factors])

        for i in range(len(sigma_factors)):
            for j in range(len(mu_factors)):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="white")

        fig.colorbar(image, ax=ax)

    if output_path is None:
        plt.show()
    else:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
