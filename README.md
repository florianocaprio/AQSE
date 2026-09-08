# AQSE — Adaptive Quantum Sensor Engine

AQSE è un dimostratore locale per una pipeline ibrida di elaborazione classica/quantistica applicata a sensori quantistici. Il Milestone 1A integra e valida l’implementazione TQK8 fornita da Floriano; il Milestone 1B aggiunge un magnetometro quantistico sintetico configurabile e l’estrazione di otto feature interpretabili.

## Responsabilità scientifica

Il circuito VQC, il fidelity kernel TQK, la loss di centered kernel alignment, l’ottimizzazione QNG, le derivate di stato, la metrica di Fubini–Study e le formule di campionamento e budget sono codice scientifico controllato dall’autore. I file in `backend/app/quantum/user_pipeline/` sono conservati senza modifiche algoritmiche e non devono essere ridisegnati senza istruzioni esplicite.

Il contratto corrente usa 8 feature generiche `f0..f7`, 8 qubit logici, 16 parametri addestrabili e 7 porte CZ. Le etichette `-1/+1` appartengono esclusivamente alla demo binaria inclusa e non definiscono i futuri task sensoriali AQSE.

## Architettura

- `frontend`: dashboard tecnica React, TypeScript e Vite per stato ambiente,
  configurazione del simulatore e ispezione dei risultati.
- `backend`: API REST Python e FastAPI.
- `backend/app/quantum/adapter.py`: adapter applicativo per i motori statevector Qiskit e NumPy.
- `backend/app/quantum/user_pipeline/`: implementazione scientifica TQK8 fornita dall’autore.
- `backend/app/sensors/`: contratti generici e modello del magnetometro simulato.
- `backend/app/preprocessing/`: analisi spettrale ed estrazione delle feature.
- `backend/tests/quantum/`: test numerici originali di Floriano.
- `docs/`: notebook, documentazione scientifica, sensori e risultati di validazione.

QNG è feedback di training, non uno stadio di inferenza. Il Milestone 1B non introduce un embedding persistente o AFSE e non collega ancora le feature del sensore al motore quantistico.

## Simulatore magnetometro

Il simulatore genera una serie temporale in nanotesla composta da campo di fondo, sinusoide, drift lineare, rumore gaussiano e un transiente opzionale. I seed rendono la componente stocastica ripetibile. Non riproduce uno strumento commerciale e non dichiara accuratezza sperimentale.

L’estrattore restituisce, senza normalizzazione, il vettore ordinato:

```text
[amplitude, phase, frequency, variance, drift, snr, spectral_peak, temperature]
```

Ampiezza, fase, frequenza, varianza, drift, SNR e picco spettrale sono stimati dal segnale; solo la temperatura proviene dai metadati di acquisizione. Formule, unità, convenzioni numeriche e limiti sono descritti in [`docs/sensors/magnetometer.md`](docs/sensors/magnetometer.md).

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
| Sensor catalog | `http://localhost:8000/api/sensors` |
| Magnetometer defaults | `http://localhost:8000/api/sensors/magnetometer/defaults` |
| FastAPI docs | `http://localhost:8000/docs` |

Il frontend inoltra `/api` al backend tramite il proxy Vite. Le porte possono essere cambiate nel file `.env` a partire da `.env.example`.

## Verifica

```sh
make test
curl --fail http://localhost:8000/api/health
curl --fail http://localhost:8000/api/quantum/health
curl --fail http://localhost:8000/api/sensors
curl --fail http://localhost:8000/api/sensors/magnetometer/defaults
curl --fail http://localhost:3000
```

`make test` esegue l’intera suite backend, inclusi gli otto test numerici originali con i cross-check Qiskit attivi e i test deterministici del sensore, quindi la build TypeScript/Vite del frontend. Il quantum health endpoint esegue unicamente un piccolo smoke test deterministico di infrastruttura: costruisce il VQC, verifica i conteggi strutturali e confronta uno stato Qiskit con il riferimento NumPy. Non addestra alcun modello e non restituisce statevector.

Il report riproducibile del Milestone 1B è in [`docs/validation/milestone-1b.md`](docs/validation/milestone-1b.md).

Esempio di simulazione:

```sh
curl --fail \
  --header 'Content-Type: application/json' \
  --data '{"duration":2,"sampling_rate":200,"frequency":8,"random_seed":42}' \
  http://localhost:8000/api/sensors/magnetometer/simulate
```

La pagina frontend permette di configurare il simulatore, visualizzare segnale e spettro e ispezionare le otto feature. Contiene inoltre una bozza locale modificabile dei 16 parametri `θ0…θ15`: quei valori rimangono nella memoria del browser e non vengono inviati o applicati al circuito in questo milestone.

## Struttura

```text
AQSE/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── models/
│   │   ├── pipeline/
│   │   ├── preprocessing/
│   │   ├── sensors/
│   │   │   ├── magnetometer.py
│   │   │   └── models.py
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
│   ├── sensors/
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
