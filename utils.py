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

'''I = 3
J = 2
M = 5


C = np.random.random((I,1))
P = np.random.random((J,1))
L = 20*np.random.random((M,1))
T = np.random.random((I,M))
G = 20*np.random.random((I,J))'''

I = 3
J = 2
M = 1

C = np.array([
    [1], [1], [3]
])

P = np.array([
    [6],[8.5]
])

L = np.array([
    [6]
])

T = np.array([
    [0.5],
    [0.25],
    [0.25]
])

G = np.array([
    [1,1],
    [1,1], 
    [0,1]
])



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
    Studia la stabilita' in-sample al crescere del numero di scenari S.

    Per ogni valore di S vengono generati, per n_sim repliche, due campioni
    indipendenti di domanda della stessa dimensione. Su ciascun campione si
    risolve il problema stocastico e si calcola la differenza tra i due valori
    ottimi:

        phi = z_S^1 - z_S^2

    Se S e' sufficientemente grande, due campioni della stessa distribuzione
    dovrebbero produrre valori ottimi simili. La procedura aumenta S finche'
    l'intervallo di confidenza per E[phi] contiene 0.

    Parametri
    mu : float
        Parametro di media della normale sottostante alla lognormale.
    sigma : float
        Deviazione standard della normale sottostante alla lognormale.
    alpha : float
        Livello di significativita' dell'intervallo di confidenza. Per esempio
        alpha=0.05 produce un intervallo al 95%.
    n_sim : int
        Numero di repliche Monte Carlo usate per stimare media e deviazione
        standard di phi per ogni S.
    seed : int
        Seed del generatore casuale, utile per rendere replicabili i risultati.

    Ritorna
    phi_campionaria : float
        Media campionaria delle differenze phi all'ultimo S testato.
    sigma_campionaria : float
        Deviazione standard campionaria delle differenze phi.
    n_scenario : int
        Numero di scenari S richiesto per ottenere stabilita' secondo il
        criterio dell'intervallo di confidenza.
    lb_conf_int, ub_conf_int : float
        Estremi inferiore e superiore dell'intervallo di confidenza.
    phi_list : list[float]
        Differenze osservate nelle n_sim repliche all'ultimo S testato.
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

        for sim in range(n_sim):

            d1 = rng.lognormal(mean=mu, sigma=sigma, size=(J, n_scenario))
            d2 = rng.lognormal(mean=mu, sigma=sigma, size=(J, n_scenario))

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
    Studia la stabilita' out-of-sample delle soluzioni ottenute con S scenari.

    Per ogni valore di S la funzione:
    1. genera un campione grande D, usato come approssimazione della distribuzione vera della domanda;
    2. per ogni replica genera un campione piccolo d con S scenari;
    3. risolve il problema su d, ottenendo una soluzione di primo stadio x;
    4. valuta quella stessa x sul campione grande D con solve_model_x_fixed;
    5. confronta il valore in-sample con il valore out-of-sample:

        phi = valore_in_sample - valore_out_of_sample

    La procedura aumenta S finche' l'intervallo di confidenza per E[phi]
    contiene 0, oppure finche' S arriva a 50. Un intervallo che contiene 0
    indica che, con il criterio adottato, non emerge una differenza sistematica
    tra il valore stimato sul campione di ottimizzazione e quello stimato sul
    campione di valutazione.

    Parametri
    mu : float
        Parametro di media della normale sottostante alla lognormale.
    sigma : float
        Deviazione standard della normale sottostante alla lognormale.
    alpha : float
        Livello di significativita' dell'intervallo di confidenza.
    n_sim : int
        Numero di repliche Monte Carlo per ogni S.
    seed : int
        Seed del generatore casuale.

    Ritorna
    phi_campionaria : float
        Media campionaria delle differenze phi all'ultimo S testato.
    sigma_campionaria : float
        Deviazione standard campionaria delle differenze phi.
    n_scenario : int
        Numero di scenari S raggiunto dalla procedura.
    lb_conf_int, ub_conf_int : float
        Estremi dell'intervallo di confidenza per E[phi].
    phi_list : list[float]
        Differenze osservate nelle n_sim repliche all'ultimo S testato.
    """
    
    n_scenario = 0
    big_n_scenario = 100
    rng = np.random.default_rng(seed)

    z_alpha = scipy.stats.norm.ppf(1-alpha/2)

    lb_conf_int = np.inf
    ub_conf_int = -np.inf

    while (not (lb_conf_int <= 0 <= ub_conf_int) and (n_scenario < 50)):
        
        D = rng.lognormal(mean=mu, sigma=sigma, size=(J, big_n_scenario))
        phi_list = []
        n_scenario += 1

        for sim in range(n_sim):

            d = rng.lognormal(mean=mu, sigma=sigma, size=(J, n_scenario))
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
    Calcola il Value of the Stochastic Solution (VSS).

    RP e' il valore ottimo del problema stocastico.
    EEV e' il valore atteso ottenuto usando la soluzione deterministica
    calcolata sulla domanda media.

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
    Calcola l'Expected Value of Perfect Information (EVPI).

    RP e' il valore ottimo del problema stocastico, dove x deve essere scelto
    prima di conoscere quale scenario si realizzera'.

    WS e' il valore wait-and-see: per ogni scenario si risolve il problema
    come se la domanda fosse nota in anticipo.

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