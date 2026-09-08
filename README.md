# AQSE — Adaptive Quantum Sensor Engine

AQSE è un dimostratore locale per una pipeline ibrida di elaborazione classica/quantistica applicata a sensori quantistici. Il Milestone 1A integra e valida l’implementazione TQK8 fornita da Floriano, senza aggiungere simulazione sensori, AFSE, interfaccia finale o accesso a QPU fisiche.

## Responsabilità scientifica

Il circuito VQC, il fidelity kernel TQK, la loss di centered kernel alignment, l’ottimizzazione QNG, le derivate di stato, la metrica di Fubini–Study e le formule di campionamento e budget sono codice scientifico controllato dall’autore. I file in `backend/app/quantum/user_pipeline/` sono conservati senza modifiche algoritmiche e non devono essere ridisegnati senza istruzioni esplicite.

Il contratto corrente usa 8 feature generiche `f0..f7`, 8 qubit logici, 16 parametri addestrabili e 7 porte CZ. Le etichette `-1/+1` appartengono esclusivamente alla demo binaria inclusa e non definiscono i futuri task sensoriali AQSE.

## Architettura

- `frontend`: pagina minima di stato React, TypeScript e Vite.
- `backend`: API REST Python e FastAPI.
- `backend/app/quantum/adapter.py`: adapter applicativo per i motori statevector Qiskit e NumPy.
- `backend/app/quantum/user_pipeline/`: implementazione scientifica TQK8 fornita dall’autore.
- `backend/tests/quantum/`: test numerici originali di Floriano.
- `docs/`: notebook, README scientifico e risultati di validazione ricevuti.

QNG è feedback di training, non uno stadio di inferenza. Il Milestone 1A non introduce un embedding persistente o AFSE.

## Motori locali

L’adapter espone due backend esatti:

- Qiskit `Statevector`, usato come motore integrato;
- simulatore di stato NumPy indipendente, usato come riferimento numerico.

Non è installato `qiskit-ibm-runtime`, non vengono configurate credenziali IBM e nessun endpoint contatta una QPU. Qiskit Aer non è necessario per questa implementazione.

## Prerequisiti

- macOS, inclusi Mac Apple Silicon;
- Docker Desktop con Docker Compose v2;
- `make`.

Docker usa l’architettura nativa dell’host e non forza `linux/amd64`. Node.js e Python locali non sono necessari per il flusso Docker.

## Avvio locale

```sh
cp .env.example .env
make build
make up
```

Servizi disponibili:

| Risorsa | URL |
| --- | --- |
| Frontend | `http://localhost:3000` |
| Backend health | `http://localhost:8000/api/health` |
| Quantum infrastructure health | `http://localhost:8000/api/quantum/health` |
| FastAPI docs | `http://localhost:8000/docs` |

Il frontend inoltra `/api` al backend tramite il proxy Vite. Le porte possono essere cambiate nel file `.env` a partire da `.env.example`.

## Verifica

```sh
make test
curl --fail http://localhost:8000/api/health
curl --fail http://localhost:8000/api/quantum/health
curl --fail http://localhost:3000
```

`make test` esegue l’intera suite backend, inclusi gli otto test numerici originali con i cross-check Qiskit attivi, e la build TypeScript/Vite del frontend. Il quantum health endpoint esegue unicamente un piccolo smoke test deterministico di infrastruttura: costruisce il VQC, verifica i conteggi strutturali e confronta uno stato Qiskit con il riferimento NumPy. Non addestra alcun modello e non restituisce statevector.

## Struttura

```text
AQSE/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── models/
│   │   ├── pipeline/
│   │   ├── sensors/
│   │   └── quantum/
│   │       ├── adapter.py
│   │       ├── engine.py
│   │       └── user_pipeline/
│   │           ├── sampler_qng.py
│   │           └── tqk8.py
│   └── tests/
│       ├── data/
│       └── quantum/
├── frontend/
├── docs/
│   ├── notebooks/
│   ├── quantum/
│   └── validation/tqk8/
├── docker-compose.yml
├── Makefile
└── README.md
```

## Comandi

- `make build`: costruisce entrambe le immagini.
- `make up`: avvia i servizi in background.
- `make down`: arresta i servizi.
- `make logs`: segue i log.
- `make test`: esegue test backend e build frontend.
- `make clean`: arresta i servizi e rimuove soltanto cache e output locali generati.

Il file ZIP sorgente rimane in `incoming/`, directory esclusa da Git, e non deve essere committato.
