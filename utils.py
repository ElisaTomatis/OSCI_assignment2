import gurobipy as gp
import numpy as np
import pandas as pd
import scipy

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
    
    n_scenario = 0
    rng = np.random.default_rng(seed)

    # quantile della normale standard
    z_alpha = scipy.stats.norm.ppf(1-alpha/2)

    lb_conf_int = np.inf
    ub_conf_int = -np.inf

    while not (lb_conf_int <= 0 <= ub_conf_int):
        # Se lo zero e' dentro l'intervallo, allora la differenza media tra due soluzioni 
        # ottenute con campioni diversi non e' significativamente diversa da zero
        
        phi_list = []
        n_scenario += 1

        for _ in range(n_sim):

            d1 = sample_d(rng, mu, sigma, (J, n_scenario))
            d2 = sample_d(rng, mu, sigma, (J, n_scenario))

            _, _, sol1 = solve_model(n_scenario, d1)
            _, _, sol2 = solve_model(n_scenario, d2)

            phi_list.append(sol1 - sol2)

        phi_campionaria = np.mean(phi_list)
        sigma_campionaria = np.std(phi_list, ddof=1)
        # ddof=1 significa che si usa la formula campionaria dividendo per n_sim - 1 

        lb_conf_int = phi_campionaria - z_alpha * sigma_campionaria / np.sqrt(n_sim)
        ub_conf_int = phi_campionaria + z_alpha * sigma_campionaria / np.sqrt(n_sim)

    return phi_campionaria, sigma_campionaria, n_scenario, lb_conf_int, ub_conf_int, phi_list



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
    big_n_scenario = 100
    rng = np.random.default_rng(seed)

    z_alpha = scipy.stats.norm.ppf(1-alpha/2)

    lb_conf_int = np.inf
    ub_conf_int = -np.inf

    while (not (lb_conf_int <= 0 <= ub_conf_int) and (n_scenario < 50)):
        
        D = sample_d(rng, mu, sigma, (J, big_n_scenario))
        phi_list = []
        n_scenario += 1

        for _ in range(n_sim):

            d = sample_d(rng, mu, sigma, (J, n_scenario))
            x, _, sol = solve_model(n_scenario, d)
            # valore in-sample, stimato sugli stessi scenari usati per ottimizzare

            # Se produco i componenti x trovati prima, quanto guadagno quando la domanda segue il campione grande D?
            _, SOL = solve_model_x_fixed(big_n_scenario, D, x)

            phi_list.append(sol - SOL)

        phi_campionaria = np.mean(phi_list)
        sigma_campionaria = np.std(phi_list, ddof=1)

        lb_conf_int = phi_campionaria - z_alpha * sigma_campionaria / np.sqrt(n_sim)
        ub_conf_int = phi_campionaria + z_alpha * sigma_campionaria / np.sqrt(n_sim)

    return phi_campionaria, sigma_campionaria, n_scenario, lb_conf_int, ub_conf_int, phi_list


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