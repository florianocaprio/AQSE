# AQSE — Adaptive Quantum Sensor Engine

AQSE è un dimostratore locale di ricerca per una pipeline ibrida
classica/quantistica applicata a reti simulate di magnetometri. Il draft
end-to-end collega osservazioni sensoriali, feature causali, il VQC/TQK
dell'autore, una rappresentazione AFSE a landmark fissi e un piccolo modello
classico. La GUI tecnica è in inglese e mantiene visibili provenienza, qualità,
degradazione e limiti interpretativi.

> **Etichetta scientifica:** `research / not validated for field deployment`.
> AQSE non è uno strumento certificato, non dimostra quantum advantage e non
> fornisce certificati causali universali.

## Percorso end-to-end

```text
ambiente simulato in evoluzione
        ↓
1–8 magnetometri vettoriali continui
        ↓ ObservationFrame, senza simulator truth
riferimento osservato di sessione (8 s)
        ↓
finestre causali State8 (4 s, hop 1 s, 100 Hz)
        ↓
AngleScaler TRAIN-only → 8 angoli
        ↓
VQC a 8 qubit dell'autore → fidelity TQK
        ↓
AFSE Nyström a riferimento congelato
        ↓
MLP classico congelato → score e stato operativo
```

Il simulatore continua indipendentemente dal rendering della GUI e dal calcolo
del modello. Il worker di analisi conserva buffer limitati, elabora la finestra
completa più recente, conta le finestre saltate e si mette in
`PAUSED_FOR_TRAINING` quando il calcolo pesante occupa lo slot condiviso. QNG è
un'operazione di training esplicita: non viene mai eseguito come livello di
inferenza e non parte all'arrivo di una misura.

## Componenti implementati nel draft

- simulatore scalare storico e rete vettoriale continua da 1 a 8 nodi;
- separazione tra osservazioni predittive e generator truth;
- sessioni `start`, `pause`, `resume`, `stop`, `reset`, `replay` e `step`;
- perturbazioni ambientali, strumentali locali/condivise, dropout, clipping e
  lettura stuck, con tempo/frame di applicazione registrato;
- profili `aqse.local-state8.v1` e `aqse.network-state8.v1`;
- encoder reale a otto coordinate, fitted soltanto su TRAIN;
- adapter al VQC/TQK protetto e confronto exact-state locale;
- AFSE `aqse.afse.nystrom-ridge32.v1` a landmark TRAIN-only;
- MLP `aqse.classical.mlp-32x16-tanh-lbfgs.v1`, persistito come dati numerici
  JSON e rieseguito con un evaluator NumPy non eseguibile;
- bundle locali/network versionati, selezione su VALIDATION e applicazione
  atomica esplicita;
- worker continuo observation-only e API per stato, training, cancellazione,
  registry e applicazione;
- otto worksheet React collegate a valori backend reali. Nessuna worksheet
  inventa feature, embedding, score o curve di training.

L'implementazione presente non sostituisce l'evidenza di validazione. Comandi,
conteggi e risultati realmente misurati sono registrati in
[`docs/validation/end-to-end-demo.md`](docs/validation/end-to-end-demo.md); lo
stato di consegna sintetico è in
[`docs/final-delivery.md`](docs/final-delivery.md).

## Profili State8

I due profili usano il Nord world-frame osservato, espresso una sola volta in
nT e relativo alla media osservata congelata nei primi 8 secondi. Non usano
parametri nascosti dell'errore simulato.

| Indice | Entrambi i profili | Unità |
| --- | --- | --- |
| f0 | media dell'anomalia Nord | nT |
| f1 | `1.4826 × MAD` attorno alla mediana | nT |
| f2 | pendenza OLS rispetto al tempo di acquisizione | nT/s |
| f3 | rapporto di potenza `10 log10(lower/upper)` | dB |
| f4 | temperatura media meno riferimento osservato | K |
| f7 | campioni ricevuti / campioni attesi | adimensionale |

La coordinata f3 usa un periodogramma one-sided con finestra Hann dopo detrend
lineare: banda inferiore `[0.25, 2)` Hz, superiore `[2, 20]` Hz e floor
numerico fisso `1e-12 nT²` per banda.

| Profilo | f5 | f6 | Impiego |
| --- | --- | --- | --- |
| `aqse.local-state8.v1` | RMS delle differenze successive, nT | correlazione Pearson lag-one | singolo nodo o fallback locale |
| `aqse.network-state8.v1` | RMS del residuo dal peer median leave-one-out, nT | media aritmetica delle correlazioni Pearson firmate coi peer | contesto valido con almeno 3 nodi |

Finestre incomplete, clipping, pose inaffidabile, riferimento non valido,
feature non finite o correlazioni non definite producono valori nullable,
validità per-feature e astensione. Nessun dato mancante viene imputato,
interpolato o trasformato in zero. Il profilo armonico storico e l'encoding
`phase-direct.v1` restano separati e compatibili soltanto coi propri artefatti.

## Modalità operative e limiti di attribuzione

- **N=1:** bundle locale, `LOCAL_ONLY`; può segnalare cambiamento ma non
  attribuirlo ad ambiente o dispositivo.
- **N=2:** percorso locale con ambiguità esplicita; una discrepanza non
  identifica da sola il lato guasto.
- **N≥3:** bundle network quando riferimento, pose, campioni e peer sono
  compatibili; altrimenti fallback locale dichiarato o astensione.

Le classi network sono `NORMAL`, `ENVIRONMENT_COMPATIBLE`,
`DEVICE_COMPATIBLE` e `MIXED_OR_AMBIGUOUS`. Sono ipotesi di pattern nel dominio
simulato, non prove causali. Il task locale usa soltanto `NORMAL` e
`CHANGE_DETECTED`. Gli score sono indicati come **model score — not
probability-calibrated**; la soglia top-score 0,70 e il margine 0,15 sono gate
ingegneristici congelati, non garanzie statistiche.

## Studio dimostrativo congelato

`aqse-network-demo-v1` prevede 160 episodi indipendenti: quattro scenari ×
conteggi nodo 1–8 × cinque repliche. In ogni cella tre episodi sono TRAIN, uno
VALIDATION e uno TEST, per totali 96/32/32. Ogni episodio dura 28 s a 100 Hz;
il riferimento è `[0,8)` e l'esempio supervisionato del nodo focale
pre-dichiarato usa `[18,22)`.

La preparazione confronta soltanto `theta0` e un candidato con al massimo 10
aggiornamenti QNG protetti, seleziona su VALIDATION e, dopo il freeze, esegue
una sola valutazione del nuovo TEST. Le osservazioni e le etichette sono file
fisicamente separati. Il TEST storico Milestone 1D non viene riaperto o
rigenerato; il suo ledger resta verificato opacamente.

## Prerequisiti

- macOS, incluso Apple Silicon;
- Docker Desktop con Docker Compose v2;
- `make`.

Docker usa l'architettura nativa dell'host e non forza `linux/amd64`. Il flusso
ordinario non richiede Python o Node.js installati direttamente sull'host.

## Preparazione e avvio

```sh
cp .env.example .env
make prepare-demo
make demo
```

`make prepare-demo` è un'operazione scientifica esplicita: genera o riusa lo
studio immutabile, effettua fit TRAIN, selezione VALIDATION, singola valutazione
del nuovo TEST e applica la coppia selezionata. Se gli artefatti identici sono
già presenti li riusa senza riaprire TEST. Non interrompere o cancellare
manualmente gli artefatti durante questa fase.

`make demo` avvia i servizi e carica i bundle salvati; non esegue training,
selezione o TEST automaticamente. Gli artefatti persistono per default nella
directory sorella `../AQSE-artifacts`, configurabile con
`AQSE_ARTIFACT_ROOT`. Non commettere la directory artefatti né `.env`.

| Risorsa | URL |
| --- | --- |
| Frontend | `http://localhost:3000` |
| Backend health leggero | `http://localhost:8000/api/health` |
| Quantum readiness leggera | `http://localhost:8000/api/quantum/health` |
| Diagnostica quantistica esplicita | `POST http://localhost:8000/api/quantum/diagnostics` |
| Demo registry | `http://localhost:8000/api/demo/registry` |
| OpenAPI | `http://localhost:8000/openapi.json` |
| FastAPI docs | `http://localhost:8000/docs` |

Le porte sono vincolate a `127.0.0.1`; `.env.example` permette di cambiarle.
Il probe Docker del backend usa esclusivamente `/api/health` e non esegue
calcoli quantistici.

## Dimostrazione manuale breve

1. Aprire `http://localhost:3000` e attivare **Blind mode** per verificare che
   l'inferenza funzioni senza mostrare la generator truth.
2. In **Sensors**, caricare una configurazione a quattro nodi, creare la
   sessione e premere **Start**.
3. In **Overview**, premere **Start analysis**. Attendere il riferimento
   osservato di 8 s e la prima finestra causale completa.
4. Esaminare valori e validità in **Features**, identità theta e circuito in
   **Quantum Engine**, vettore reale in **Local Embedding / AFSE** e score in
   **Neural Model**.
5. Tornare in **Sensors**, disattivare deliberatamente Blind mode per rendere
   disponibili i controlli di scenario, scegliere un evento e usare **Apply
   perturbation**; riattivare Blind mode e confrontare risposte nodo/peer. Il
   comando non entra nell'input del modello.
6. In **QNG Training** il training parte solo col pulsante dedicato, può essere
   annullato e non promuove automaticamente alcun modello.
7. In **Experiments**, applicare esplicitamente una coppia compatibile e
   consultare registry e metriche congelate senza rieseguire TEST.

La guida completa, inclusi reset/replay e stati di errore, è in
[`docs/user-guide.md`](docs/user-guide.md).

## API end-to-end

| Metodo | Percorso | Funzione |
| --- | --- | --- |
| GET | `/api/demo/registry` | artefatti, bundle, freeze e metriche read-only |
| POST | `/api/demo/analysis/{session_id}/start` | acquisisce riferimento e avvia il worker |
| GET | `/api/demo/analysis/{session_id}` | stato, latenze, qualità e ultimi risultati |
| POST | `/api/demo/analysis/{session_id}/stop` | arresta esplicitamente il worker |
| POST | `/api/demo/training/jobs` | avvia un job limitato e idempotente per intent |
| GET | `/api/demo/training/jobs/{job_id}` | stato e step QNG accettati |
| POST | `/api/demo/training/jobs/{job_id}/cancel` | richiede cancellazione |
| POST | `/api/demo/bundles/apply` | applica atomicamente la coppia del freeze |

Le API di sessione, osservazioni, SSE, eventi, feature armoniche e preview
quantistica storica restano disponibili e documentate nell'OpenAPI locale.

## Comandi di sviluppo

```sh
make build          # costruisce backend e frontend
make prepare-demo   # prepara/reusa studio e bundle end-to-end
make demo           # avvia e attende servizi healthy
make up             # avvio Compose di sviluppo
make down           # arresto non distruttivo
make logs           # log dei servizi
make test           # pytest, Ruff, typecheck, ESLint, Vitest e build
make acceptance     # acceptance API/browser-supporting checks documentati
make soak           # soak continuo parametrizzabile
make clean          # rimuove soltanto cache/output locali previsti
```

Il soak finale richiesto usa otto nodi e 600 s di tempo reale. Ridurre la
durata serve soltanto per prove locali e non equivale all'accettazione finale:

```sh
SOAK_DURATION_SECONDS=60 SOAK_NODE_COUNT=4 make soak
```

## Confini scientifici essenziali

- Il VQC, TQK, loss, derivate, metrica di Fubini–Study e QNG originali sono
  proprietà scientifica dell'autore e non vengono ridisegnati.
- AFSE è una mappa classica regolarizzata costruita dal kernel, non una nuova
  funzione quantistica né la wavefunction fisica del sensore.
- Un vettore AFSE o una matrice di fedeltà non dimostrano sensibilità fisica,
  localizzazione o vantaggio quantistico.
- L'OOD basato sul residuo TRAIN p99 è euristico. Gli score MLP non sono
  probabilità calibrate.
- Simulator truth, seed, scenario, interventi nascosti e campioni futuri non
  entrano nell'inferenza; restano nei canali separati di audit/label.
- Prestazioni, beneficio quantum, latenza e stabilità sono affermazioni valide
  solo se presenti come risultati realmente misurati nel record di
  validazione.

Per i limiti completi vedere
[`docs/architecture/scientific-scope-and-limitations.md`](docs/architecture/scientific-scope-and-limitations.md).

## Struttura principale

```text
AQSE/
├── backend/
│   ├── app/
│   │   ├── api/          # REST/SSE e API demo
│   │   ├── classical/    # MLP fitted e runtime NumPy
│   │   ├── demo/         # protocollo, studio, bundle, worker, registry
│   │   ├── embeddings/   # AFSE Nyström
│   │   ├── features/     # profili armonici e State8
│   │   ├── network/      # simulatore/sessioni/eventi
│   │   ├── quantum/      # adapter e sorgenti scientifici protetti
│   │   └── training/     # archivi e training storici
│   ├── scripts/
│   └── tests/
├── frontend/src/
│   ├── api/
│   ├── components/
│   ├── state/
│   └── worksheets/
├── docs/
├── docker-compose.yml
├── Makefile
└── README.md
```

ZIP in `incoming/`, `.env`, `node_modules`, build, cache, log e artefatti
runtime sono esclusi da Git.
