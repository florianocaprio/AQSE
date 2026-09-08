# AQSE — Adaptive Quantum Sensor Engine

AQSE è un dimostratore locale per una pipeline ibrida di elaborazione classica/quantistica applicata a sensori quantistici. Questo repository contiene esclusivamente la fondazione applicativa; gli algoritmi Qiskit saranno aggiunti separatamente dal proprietario del progetto.

## Architettura

- `frontend`: dashboard minima React, TypeScript e Vite.
- `backend`: API REST Python e FastAPI.
- `backend/app/sensors`: interfacce delle sorgenti sensore.
- `backend/app/quantum`: adapter vuoto per il futuro codice Qiskit.
- `backend/app/pipeline`: contratti della pipeline AQSE.
- `backend/app/models`: modelli dati condivisi dal backend.
- Docker Compose avvia entrambi i servizi usando l'architettura nativa dell'host.

Il frontend usa `/api` e Vite inoltra le richieste al backend. La separazione tra API, sorgenti, motore quantistico e pipeline permette di aggiungere successivamente un canale WebSocket senza accoppiare i componenti.

## Prerequisiti

- macOS, inclusi Mac Apple Silicon
- Docker Desktop con Docker Compose v2
- `make`

Node.js e Python locali non sono necessari per il flusso Docker.

## Avvio locale

```sh
cp .env.example .env
make build
make up
```

Aprire `http://localhost:3000`. Lo stato del backend è disponibile anche su `http://localhost:8000/api/health`.

| Servizio | Porta predefinita |
| --- | ---: |
| Frontend | 3000 |
| Backend | 8000 |

Le porte possono essere cambiate nel file `.env`. Comandi disponibili: `make build`, `make up`, `make down`, `make logs`, `make test` e `make clean`.

## Struttura

```text
AQSE/
├── frontend/             # React + TypeScript + Vite
├── backend/
│   ├── app/
│   │   ├── api/          # endpoint REST
│   │   ├── sensors/      # contratti delle sorgenti
│   │   ├── quantum/      # adapter per Qiskit, senza algoritmi
│   │   ├── pipeline/     # orchestrazione futura
│   │   └── models/       # modelli applicativi
│   └── tests/
├── docs/
├── docker-compose.yml
├── Makefile
└── README.md
```

## Test

```sh
make test
```

Il comando verifica l'endpoint health del backend ed esegue la build TypeScript/Vite del frontend.

