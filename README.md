# AQSE — Adaptive Quantum Sensor Engine

AQSE è un banco di prova locale per una pipeline ibrida classica/quantistica applicata a sensori quantistici. Il Milestone 1C rende navigabile una catena controllata: campo sintetico ideale → rete di magnetometri → osservazioni degradate → feature interpretabili → codifica angolare → confronto VQC/TQK a parametri fissi.

Il progetto è un dimostratore software, non un modello certificato di uno strumento commerciale e non una validazione sperimentale.

## Perimetro scientifico

Il circuito VQC, il fidelity kernel TQK, la loss di centered kernel alignment, l'ottimizzazione QNG, le derivate di stato, la metrica di Fubini–Study e le formule di campionamento e budget sono codice scientifico controllato dall'autore. I sorgenti originali in `backend/app/quantum/user_pipeline/` e i relativi test non vengono ridisegnati né ottimizzati senza istruzioni esplicite.

Stato corrente:

- il simulatore vettoriale e la rete continua da 1 a 8 nodi sono implementati;
- l'estrazione finestrata delle otto feature è implementata;
- la preview locale usa `AngleScaler`, il VQC fornito e il TQK a **theta fisso**;
- la modalità self-reference è esclusivamente esplorativa, non una valutazione predittiva;
- il QNG scientifico esiste nel codice dell'autore, ma il training sui dati sensoriali non è collegato;
- Milestone 1D.1 implementa esclusivamente dataset offline immutabili, lineage e split per episodio; non implementa encoding di produzione né training;
- AFSE ha soltanto un confine architetturale: la matematica non è implementata;
- rete neurale, output engine e QPU fisica non sono implementati;
- non vengono inventati VQC, kernel, loss, QNG o algoritmi AQSE aggiuntivi.

## Architettura

```text
Synthetic field providers
          │
          ▼
1–8 vector magnetometers ──► observation stream (REST + SSE)
          │                            │
          └──► separate truth API      ▼
                                  causal windows
                                        │
                                        ▼
                              8 observable features
                                        │
                                        ▼
                              reference-fit AngleScaler
                                        │
                                        ▼
                         fixed-theta VQC / TQK preview
```

- `backend/`: Python 3.12, FastAPI, Pydantic, NumPy, Qiskit e test pytest.
- `frontend/`: React, TypeScript, Vite e una GUI tecnica in inglese.
- `backend/app/network/`: fisica sintetica, moto, sensori, sessioni, eventi, buffer, SSE e osservabilità.
- `backend/app/features/`: estrazione finestrata, qualità e provenienza delle feature.
- `backend/app/training/`: generazione offline 1D.1, digest canonici, split per lineage e storage sealed.
- `backend/app/quantum/preview.py`: adapter applicativo limitato per la preview a theta fisso.
- `backend/app/quantum/user_pipeline/`: implementazione scientifica fornita dall'autore.
- `docs/`: contratti, limiti scientifici e piano di validazione.

REST gestisce configurazione, controllo e interrogazione. Lo stream di osservazioni usa Server-Sent Events; il confine applicativo mantiene identificativi, cursori e payload versionati, così da permettere una futura evoluzione verso WebSocket senza accoppiare il simulatore al trasporto.

## Simulazione sensoriale

Il backend supporta sia una simulazione vettoriale finita sia sessioni continue in memoria:

- 1–8 nodi con ruolo `sensor` o `remote_reference`;
- misura vettoriale, monoassiale o total-field;
- frame NED e quaternion `wxyz` world-to-sensor;
- campo uniforme, rumore OU comune, gradiente simmetrico a traccia nulla, dipoli puntiformi, anomalie Gaussiane e campi periodici;
- moto statico, tumble, high-dynamic, lineare, attraversamento anomalia e combinato;
- catena strumentale configurabile con disallineamento, cross-axis, soft-iron, gain, bias, drift, temperatura, banda, rumore bianco e saturazione;
- eventi distinti dal rumore: offset del campo, bias/drift del nodo, noise burst, offset strumentale condiviso, dropout e stuck sensor;
- seed e stream casuali separati per riproducibilità;
- sessioni `start`, `pause`, `resume`, `stop`, `reset`, `replay` e avanzamento deterministico `step`;
- buffer circolare limitato, cursor gap esplicito e SSE con `Last-Event-ID`.

Le unità interne e API sono SI: tesla, metri, secondi, kelvin, A·m² e T/m. La GUI converte in unità più leggibili dove indicato.

### Osservazioni e verità

Le osservazioni contengono soltanto ciò che il sensore renderebbe disponibile: misura, metadati osservabili, validità e flag non rivelatori. La generator truth è esposta da endpoint separati e può essere nascosta dalla modalità blind della GUI. Dropout e valori mancanti sono `null`; non vengono sostituiti con zeri o NaN serializzati.

La quantizzazione ADC e il trasporto di rete fisico non sono modellati in questo milestone. Le sessioni sono locali, in memoria e non persistono al riavvio del backend.

## Feature e preview quantistica

L'estrattore opera su finestre causali complete e restituisce, nello stesso ordine del contratto TQK8:

```text
[amplitude, phase, frequency, variance, drift, snr, spectral_peak, temperature]
```

Il canale può essere X, Y, Z o magnitudine. Il profilo implementato usa di default finestre da 1 s con sovrapposizione del 50%; un profilo continuo 4 s / hop 1 s è documentato come candidato, non come implementazione separata. Le soglie di qualità v1 sono fisse e verificabili: almeno due cicli, prominenza spettrale minima di 6 dB, SNR minimo di 0 dB e nessun campione saturo.

Ogni riga è firmata dal backend con provenienza effimera. La preview rifiuta righe alterate, reference bank degenere, leakage temporale e richieste oltre il limite. Lo scaler è adattato soltanto sul reference set. Sono disponibili:

- `self_reference`: Gram matrix interna, marcata **SELF-REFERENCE — EXPLORATORY**;
- `reference_query`: confronto di finestre query future contro un reference bank causale precedente.

Il calcolo usa statevector esatti locali Qiskit o NumPy. Non usa Aer, non configura credenziali IBM e non contatta una QPU.

## GUI del workbench

La pagina è una prima interfaccia tecnica, non il design finale di AQSE. Le otto worksheet sono:

1. Overview
2. Sensors
3. Features
4. Quantum
5. QNG
6. AFSE
7. Neural
8. Experiments

La GUI consente di scegliere preset, modificare la geometria e i parametri dei sensori, importare/esportare configurazioni JSON, avviare o avanzare una sessione, osservare i segnali, estrarre feature e modificare i 16 parametri theta inviati alla preview. L'import sostituisce soltanto il draft e non crea una sessione automaticamente. Modificare una configurazione rende esplicitamente stale i risultati dipendenti. QNG, AFSE e Neural mostrano soltanto lo stato reale del progetto e non generano metriche simulate.

## Prerequisiti

- macOS, inclusi Mac Apple Silicon;
- Docker Desktop con Docker Compose v2;
- `make`.

Docker usa l'architettura nativa dell'host e non forza `linux/amd64`. Il flusso standard non richiede Python o Node.js installati localmente.

## Avvio locale

```sh
cp .env.example .env
make build
make up
```

I bind mount e i processi di reload automatico rendono disponibili le modifiche locali senza ricostruire l'immagine a ogni salvataggio. Non inserire segreti nel file `.env` e non commetterlo.

| Risorsa | URL |
| --- | --- |
| Frontend | `http://localhost:3000` |
| Backend health | `http://localhost:8000/api/health` |
| Quantum adapter readiness | `http://localhost:8000/api/quantum/health` |
| Quantum infrastructure diagnostics | `POST http://localhost:8000/api/quantum/diagnostics` |
| Workbench capabilities | `http://localhost:8000/api/workbench/capabilities` |
| Network health | `http://localhost:8000/api/network/health` |
| Network presets | `http://localhost:8000/api/network/presets` |
| Field providers | `http://localhost:8000/api/network/field-providers` |
| VQC descriptor | `http://localhost:8000/api/quantum/circuit` |
| OpenAPI | `http://localhost:8000/openapi.json` |
| FastAPI documentation | `http://localhost:8000/docs` |

Le porte e il target del proxy Vite possono essere modificati partendo da `.env.example`.
Le porte sono pubblicate soltanto sull'interfaccia loopback `127.0.0.1`; il banco
non espone servizi alla rete locale per impostazione predefinita.

### Artefatti dataset Milestone 1D.1

I dataset non sono serviti da API e non sono generati all'avvio. Il comando offline
seguente esegue prima il pilot controllato e, solo se il gate di copertura è valido,
crea il dataset di sviluppo:

```sh
docker compose run --rm backend python scripts/generate_training_dataset.py
```

Per impostazione predefinita gli artefatti host sono scritti nella directory sorella
`../AQSE-artifacts`, montata nel container come `/artifacts`. Il percorso è configurabile
con `AQSE_ARTIFACT_ROOT`; `AQSE_ARTIFACT_LIMIT_BYTES` impone il limite operativo
predefinito di 1 GiB includendo file temporanei. Manifest e payload scientifici sono
read-only, mentre il ledger di accesso al test è concatenato e append-only.

Le osservazioni, le otto feature e le etichette sono file distinti. Il test di sviluppo
è serializzato e verificato dall'archiver, ma resta sealed: la normale API di caricamento
lo rifiuta e 1D.1 non ne stampa distribuzioni di qualità né metriche predittive.

### Percorso demo consigliato

1. Aprire la worksheet **Sensors** e caricare il preset `Quantum preview signal`.
2. Creare una sessione e generare un numero di campioni sufficiente per più finestre complete.
3. In **Features**, scegliere il sensore e il canale, quindi estrarre le feature.
4. In **Quantum**, verificare tabella raw/encoded, theta e backend esatto.
5. Eseguire prima la preview self-reference esplorativa, oppure creare una separazione causale reference/query.
6. Salvare o reimportare la configurazione, oppure esportare feature ed esperimenti, dalla worksheet **Experiments**.

Per la simulazione verticale sono disponibili anche `Eight-node causal event demo`,
con sorgente mobile e cause componibili, e `Single-sensor ambiguity control`, con
due fasi di uguale offset ma diversa origine. Le cause appartengono esclusivamente
al canale truth e non sono presentate come diagnosi del modello.

## Comandi di sviluppo

- `make build`: costruisce entrambe le immagini Docker.
- `make up`: avvia backend e frontend in background.
- `make down`: arresta i servizi senza rimuovere dati estranei al progetto.
- `make logs`: segue i log di entrambi i servizi.
- `make test`: esegue pytest, Ruff, typecheck, ESLint, Vitest e build Vite.
- `make soak`: esegue il soak test continuo predefinito da 20 minuti su 8 nodi.
- `make clean`: arresta i servizi e rimuove soltanto cache/output locali generati.

Durata e numero di nodi del soak sono sovrascrivibili:

```sh
SOAK_DURATION_SECONDS=60 SOAK_NODE_COUNT=4 SOAK_SEED=42 make soak
```

## Verifica manuale essenziale

```sh
make test
curl --fail http://localhost:8000/api/health
curl --fail http://localhost:8000/api/quantum/health
curl --fail --request POST http://localhost:8000/api/quantum/diagnostics
curl --fail http://localhost:8000/api/network/health
curl --fail http://localhost:8000/api/network/presets
curl --fail http://localhost:3000
```

`GET /api/health` è il solo probe Docker del backend e non importa né esegue
calcoli quantistici. Docker lo controlla ogni 10 secondi con timeout di 3 secondi,
cinque tentativi e 5 secondi iniziali di tolleranza.

`GET /api/quantum/health` è una readiness leggera: verifica disponibilità
dell'adapter, installazione di Qiskit e metadata statici del backend/circuito,
senza costruire statevector, kernel o confronti numerici. Il frontend può
interrogarla periodicamente.

`POST /api/quantum/diagnostics` è invece lo smoke test deterministico esplicito.
Solo su richiesta costruisce il VQC, esegue gli statevector Qiskit e NumPy e li
confronta numericamente, restituendo anche il tempo di esecuzione. Un guard
impedisce diagnostiche concorrenti e risponde HTTP 429 quando una è già attiva.
Questo percorso non addestra modelli, non invoca QNG e non restituisce lo
statevector.

Il piano riproducibile è in `docs/validation/milestone-1c-validation-plan.md`; i comandi realmente eseguiti, i risultati misurati e i limiti osservati sono registrati in `docs/validation/milestone-1c.md`.

## Struttura principale

```text
AQSE/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── features/
│   │   ├── network/
│   │   ├── preprocessing/
│   │   ├── sensors/
│   │   ├── training/
│   │   └── quantum/
│   │       └── user_pipeline/
│   ├── scripts/
│   └── tests/
├── frontend/
│   └── src/
│       ├── api/
│       ├── app/
│       ├── components/
│       ├── diagrams/
│       ├── state/
│       ├── types/
│       └── worksheets/
├── docs/
│   ├── architecture/
│   ├── features/
│   ├── quantum/
│   ├── sensors/
│   └── validation/
├── docker-compose.yml
├── Makefile
├── .env.example
└── README.md
```

Il file ZIP sorgente resta in `incoming/`, directory esclusa da Git. Sono esclusi anche `.env`, `node_modules`, build, cache, log e artefatti runtime.
