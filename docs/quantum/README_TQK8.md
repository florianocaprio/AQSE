# AQSE — TQK8: VQC → kernel → loss → QNG

**8 qubit, 8 feature, 16 parametri addestrabili, 7 CZ per preparazione di stato.**
Versione di riferimento per un esperimento controllato di Quantum Machine Learning
su dati classici acquisiti da sensori. Non è un controllo diretto del sensore,
non implementa correzione degli errori e non promette un vantaggio quantistico.

## Stato della verifica

È stata eseguita la pipeline completa con il simulatore NumPy indipendente incluso.
Sei test numerici sono passati; due test Qiskit sono saltati perché Qiskit non è
installato nell'ambiente di generazione e la sua installazione è stata impedita
dall'accesso di rete. Il codice Qiskit è stato verificato sintatticamente e confrontato
con la documentazione ufficiale, ma **non eseguito qui con l'SDK Qiskit né su QPU**.

I test coprono positività/simmetria/diagonale del kernel ideale, dipendenza da tutti
16 i parametri su dati generici, gradiente completo contro differenze finite,
metrica e gradiente ottenuti dalle formule di misura contro quelli da derivate di
stato, riduzione della loss e controllo preventivo del budget.
`verified_numpy_run/` contiene log e risultato della demo sintetica. I numeri non
sono una validazione scientifica sui magnetometri o sugli interferometri atomici.

## 1. Correzione del circuito proposto in precedenza

Un circuito della forma U_theta(x)=W(theta)D(x) NON dà un fidelity kernel trainabile:
la W(theta) comune si cancella in U_theta(z)^dagger U_theta(x).
La presenza di nomi di parametri nel circuito non è sufficiente per addestrare il kernel.

Qui l'ordine fisico, dal primo gate all'ultimo, è:

    RY(x) → RZ(alpha) → CZ su catena → RY(beta) → RZ(x)

Sono 8 angoli di input, riutilizzati due volte; alpha e beta sono due vettori di
8 parametri indipendenti. L'ultima rotazione dipende dal dato, non soltanto da theta.
Questa disposizione modifica effettivamente il kernel al variare di theta.
Non è presentata come ansatz ottimale: è un candidato semplice e verificabile.

Per ogni q in {0,...,7}: RY(x_q), RZ(theta_q), interazioni CZ, RY(theta_(8+q)), RZ(x_q).
CZ, in ordine: (0,1), (2,3), (4,5), (6,7), (1,2), (3,4), (5,6).
I 7 accoppiamenti formano una catena e non richiedono un collegamento ad anello.
Gli indici 0–7 sono LOGICI e devono essere mappati su una catena fisica calibrata.

## 2. Pipeline

    8 feature classiche
        ↓  scaler, fittato solo sul training set
    x: 8 angoli
        ↓  VQC U_theta(x)
    stato |psi_theta(x)>
        ↓  confronti fra campioni
    K_theta(i,j)=|<psi_theta(x_i)|psi_theta(x_j)>|²
        ↓  loss supervisionata di alignment
    L(theta), gradiente e metrica di Fubini–Study media
        ↓  QNG regolarizzato (solve classico)
    delta_theta → nuovo VQC

Alla fine theta viene congelato; K_test,train viene usato da una SVC classica.
Il QNG appartiene al feedback di addestramento: non è un livello di inferenza
attraversato dal dato e non produce direttamente un embedding.
Per un embedding persistente occorre aggiungere una costruzione esplicita, ad
esempio feature rispetto a landmark e trasformazione Nyström. Non è necessario
farlo per validare il ciclo richiesto, che qui usa direttamente un kernel precomputato.

## 3. Loss e geometria

H = I − 11ᵀ/B; Kc = H K H; T = H yyᵀ H.

    L = 1 − <Kc,T>_F / (||Kc||_F ||T||_F)

La demo assume due classi, label −1/+1. Per esempio: una condizione fisica
d'interesse e una condizione di riferimento, definite mediante acquisizioni
calibrate. Non inventare etichette a partire dal kernel stesso.

La loss premia l'allineamento tra similarità e supervisione; non massimizza
automaticamente sensibilità metrologica, pT/sqrt(Hz) o precisione gravimetrica.
Per un task di regressione fisica va definita una loss collegata all'errore della
quantità stimata, alle incertezze e alle condizioni di riferimento.

    g_ab(x) = Re[<∂a psi|∂b psi> − <∂a psi|psi><psi|∂b psi>]
    g_bar = media dei g(x) sul batch
    (g_bar + lambda I) v = grad L
    theta_new = theta − eta v

Convenzione: g è la metrica Fubini–Study; per stati puri F_Q=4g. Non mescolare
questa formula con la stessa learning rate su F_Q. g_bar è una scelta di metrica
sugli stati dell'encoder: non è la matrice K, né l'unica geometria possibile per
un problema kernel. Non è la Fisher rispetto al campo magnetico o all'accelerazione.

La versione esatta usa una line search Armijo e un limite alla norma del passo.
Queste sono scelte numeriche di stabilizzazione, non teoremi di convergenza globale.
Non viene formata esplicitamente l'inversa della metrica: si usa numpy.linalg.solve.

## 4. Gradiente: due dettagli importanti

Ogni theta compare una sola volta in ciascuna preparazione di stato. Per la
DERIVATA DELLO STATO, nel simulatore, vale:

    ∂r |psi> = [|psi(theta+pi/2 e_r)> − |psi(theta−pi/2 e_r)>] / (2 sqrt(2))

Il prefattore non è quello della regola per valori medi/probabilità.
Il simulatore ricava poi dK derivando sia il bra sia il ket, e applica la catena:

    ∂r L = sum_ij (∂L/∂K_ij) (∂r K_ij)

La loss normalizzata è NON lineare: non si applica direttamente la regola
`[L(theta+pi/2)-L(theta-pi/2)]/2` alla loss intera.

Nel circuito di overlap, theta compare in entrambe le metà. Nel modulo di misura
`sampler_qng.py` vengono spostate separatamente le due occorrenze. Definendo
F_ij(a,b)=|<psi_b(x_j)|psi_a(x_i)>|², per ciascun parametro:

    ∂r K_ij = 1/2 [F(theta+s,theta)-F(theta−s,theta)
                    +F(theta,theta+s)-F(theta,theta−s)]

con s=(pi/2)e_r. Le quattro probabilità vengono stimate con il medesimo circuito
compute–uncompute di 8 qubit, senza ancille.

## 5. Installazione ed esecuzione

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tqk8.py --engine qiskit --steps 15
python -m unittest -v test_tqk8
```

Il notebook `TQK8_walkthrough.ipynb` usa gli stessi moduli; aprirlo dalla directory
che contiene `tqk8.py`. Non incorpora una seconda implementazione divergente.

Per ripetere le verifiche indipendenti senza Qiskit (servono NumPy e scikit-learn):

```bash
python tqk8.py --engine numpy --steps 15
```

I risultati sono scritti in `output_tqk8/`: parametri, scaler, dati di riferimento,
matrici kernel, storico della loss e metriche della demo. Per effettuare inferenza
con una SVC bisogna conservare anche il modello classico o ricostruirlo da K_train
e y_train; il solo vettore theta non è un classificatore.

Dati propri: CSV con colonne `f0,f1,f2,f3,f4,f5,f6,f7,label`, label −1/+1.
`example_synthetic.csv` è un esempio di formato, NON una misura sperimentale.

```bash
python tqk8.py --csv example_synthetic.csv --engine qiskit --steps 15
```

Lo split casuale nel comando è didattico. Per serie temporali reali separare
training/test per sessione, tempo e dispositivo; evitare finestre sovrapposte nei
due insiemi. Congelare normalizzazione e iperparametri prima del test. Gestire
esplicitamente le variabili periodiche (fase) e i dati mancanti.

## 6. Misurare un kernel con shot

```python
from qiskit.primitives import StatevectorSampler
from sampler_qng import SamplerOverlap

executor = SamplerOverlap(StatevectorSampler(seed=42), shots=4096)
k_estimated = executor.probabilities([(x_i, x_j, theta, theta)])[0]
```

StatevectorSampler produce campionamento finito da stati IDEALI. Non simula
automaticamente la decoerenza o il 3% di errore delle porte a due qubit.
Una QPU restituisce conteggi, non le 256 ampiezze complesse dello stato.

Un passo completo di QNG diagonale da probabilità:

```python
from sampler_qng import sampled_qng_step, circuit_budget
print(circuit_budget(8, 4, metric="diagonal"))
# X_batch: 8 campioni, ciascuno di 8 feature già scalate; entrambe le classi.
theta_new, diagnostics = sampled_qng_step(
    executor, X_batch, y_batch, theta,
    X_metric=X_batch[:4], metric="diagonal"
)
```

Il budget dell'executor è cumulativo: verificarlo per TUTTE le iterazioni previste.
La variante shot-based non usa la line search deterministica della versione esatta
ed è un aggiornamento stocastico; occorrono esperimenti per fissare shot, eta e lambda.

## 7. Metrica mediante probabilità di overlap

Per f(delta)=|<psi_theta(x)|psi_(theta+delta)(x)>|²:

    g_aa = [1 − f(pi e_a)] / 4
    g_ab = −[f(++ ) − f(+-) − f(-+) + f(--)] / 8, a != b

Nei quattro termini fuori diagonale si applicano shift ±pi/2 sui parametri a,b
soltanto nella seconda preparazione. Sono identità per il circuito ideale puro,
con ogni parametro presente una volta. I test verificano l'uguaglianza con il QGT.

La diagonale del kernel ideale è impostata a uno. I kernel campionati possono
non essere PSD. Il modulo non applica una proiezione PSD al kernel dentro la loss,
per non ignorarne la derivata. Prima di impiegare una SVM con matrici rumorose
serve una strategia coerente di regolarizzazione/approssimazione PSD e di estensione
fuori campione. La demo SVC usa solo il kernel esatto.

Per la metrica rumorosa si proiettano gli autovalori negativi a zero e si aggiunge
damping. La stima rimane un proxy: le identità pure non producono automaticamente
la Fisher quantistica dello stato misto generato dall'hardware rumoroso.

## 8. Esecuzione su QPU e sottoinsieme fisico

Per IBM installare separatamente `qiskit-ibm-runtime`. Credenziali, nome backend e
lista fisica sono intenzionalmente NON prefissati. Un fornitore diverso richiede
il suo adapter SamplerV2/BackendV2 e le sue regole di compilazione.

```python
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from sampler_qng import SamplerOverlap

service = QiskitRuntimeService()  # account già autorizzato e salvato dall'utente
backend = service.backend("NOME_BACKEND_AUTORIZZATO")
# layout = lista di 8 indici fisici scelti con calibrazioni e connettività reali.
assert len(layout) == 8 and len(set(layout)) == 8
pm = generate_preset_pass_manager(
    backend=backend, initial_layout=layout, optimization_level=1)
executor = SamplerOverlap(
    SamplerV2(mode=backend), pass_manager=pm,
    shots=1024, max_circuits=2000, allowed_physical_qubits=layout)
```

La scelta deve preservare la catena (0–1–...–7), non soltanto minimizzare gli errori
single-qubit. Il controllo `allowed_physical_qubits` rifiuta circuiti che il
transpiler abbia spostato fuori dal sottoinsieme, prima dell'invio. Verificare
numero/durata delle porte native e readout. Inizializzare un layout non equivale
in generale a proibire ogni uso di qubit fisici esterni da parte del routing.

## 9. Costo esplicito

Preparazione: 32 rotazioni single-qubit + 7 CZ. Kernel: fino a 64 rotazioni + 14 CZ
prima di compilazione/semplificazione, e 8 misure. Nessuna ancilla.

Per P=16, B campioni, M campioni usati per la metrica:

    coppie = B(B−1)/2
    kernel + gradienti = coppie * (1 + 4P)
    g diagonale = M*P
    g piena = M*[P + 2P(P−1)]

B=8,M=4: 1.884 circuiti per passo con g diagonale; 3.804 con g piena.
A 1.024 shot: rispettivamente 1.929.216 e 3.895.296 esecuzioni del circuito.
Non inclusi validazione, mitigazione, ripetizioni, ricerca degli iperparametri o
ricalibrazioni. Sono costi dell'implementazione esplicita, NON limiti teorici minimi.

Con una fidelità two-qubit riportata del 97%, un circuito poco profondo non è da
solo una garanzia di utilità. Confrontare kernel fisso, TQK con ottimizzatore
ordinario, TQK+QNG e baseline classiche a budget dichiarato. Usare un modello di
rumore calibrato, non il solo numero medio di fidelità.

## 10. Fonti primarie

- Hubregtsen et al., Training quantum embedding kernels on near-term quantum
  computers, Physical Review A 106, 042431 (2022).
  https://arxiv.org/abs/2105.02276
- Stokes, Izaac, Killoran, Carleo, Quantum Natural Gradient, Quantum 4, 269 (2020).
  https://arxiv.org/abs/1909.02108
- Schuld et al., Evaluating analytic gradients on quantum hardware,
  Physical Review A 99, 032331 (2019). https://arxiv.org/abs/1811.11184
- Qiskit Statevector:
  https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.Statevector
- Qiskit StatevectorSampler:
  https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.primitives.StatevectorSampler
- Qiskit ParameterVector:
  https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.ParameterVector
- IBM SamplerV2:
  https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2
- Transpiler/pass manager:
  https://quantum.cloud.ibm.com/docs/en/guides/transpile-with-pass-managers

L'ansatz specifico, il codice e il protocollo qui proposti sono un'implementazione
sperimentale: le fonti non affermano che questo preciso circuito sia ottimale per
la sensoristica, né che offra un vantaggio pratico sul classico.
