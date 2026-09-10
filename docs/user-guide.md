# AQSE end-to-end demo — guida operativa

## Prima esecuzione

Requisiti: Docker Desktop, Docker Compose v2 e `make`. Da root repository:

```sh
cp .env.example .env
make prepare-demo
make demo
```

`make prepare-demo` è intenzionalmente separato dall'avvio quotidiano. Genera
o riusa lo studio congelato, esegue il fit su TRAIN, seleziona su VALIDATION,
effettua l'unica valutazione autorizzata del nuovo TEST dopo il freeze e salva
i bundle. Se trova gli stessi artefatti completi, li verifica e li riusa senza
riaprire TEST. Non cancellare né modificare a mano `../AQSE-artifacts` mentre
il comando è in corso.

`make demo` avvia backend e frontend senza training o TEST automatici. Aprire:

- GUI: `http://localhost:3000`;
- documentazione API: `http://localhost:8000/docs`.

La pagina è in inglese; questa guida usa esattamente i nomi dei controlli.

## Percorso guidato di cinque minuti

### 1. Verificare lo stato

Aprire **Overview**. Devono essere visibili:

- `Backend API` e `Sensor network service` disponibili;
- `Qiskit exact adapter` disponibile;
- uno studio `PREPARED` e una coppia bundle attiva;
- simulazione e analisi non ancora avviate.

Una readiness verde indica soltanto disponibilità software. Non è una
validazione scientifica. La diagnostica quantistica completa parte soltanto
da **Run quantum diagnostics**; non viene eseguita dal polling dello stato.

### 2. Creare una rete

Aprire **Sensors**:

1. usare i pulsanti nodo per caricare una configurazione a 4 nodi, oppure
   scegliere un preset e premere **Load**;
2. controllare sampling rate 100 Hz, seed e geometria nel `DRAFT
   CONFIGURATION`;
3. premere **Create session**, poi **Start**;
4. osservare `Backend time`, `Latest frame`, buffer e grafici ricevuti.

Draft ed esecuzione sono distinti. Cambiare un controllo nel draft non modifica
la sessione già in esecuzione e rende stale i risultati che dipendono dalla
configurazione precedente.

### 3. Avviare l'analisi

Tornare in **Overview** e premere **Start analysis**. Il worker:

1. raccoglie i primi 800 frame osservati (8 s a 100 Hz);
2. congela riferimento Nord e temperatura per ogni nodo;
3. attende una finestra completa di 400 frame;
4. elabora al massimo la più recente finestra causale disponibile a hop di
   1 s.

Durante il warm-up lo stato è `AWAITING REFERENCE`; nessuna diagnosi viene
fabbricata. Dopo la prima finestra eleggibile lo stato diventa `RUNNING` e
mostra conteggio finestre, età risultato e p50/p95 della durata di calcolo.

### 4. Seguire la pipeline

Con un nodo selezionato, visitare in ordine:

- **Features**: profilo State8, otto valori reali, unità, finestra,
  riferimento, peer e validità;
- **Quantum Engine**: bundle e theta congelati, metadati del circuito a 8
  qubit e numero di righe del riferimento TQK;
- **Local Embedding / AFSE**: vettore `z(x)` a dimensione fissa, λ, residuo e
  flag OOD euristico;
- **Neural Model**: classi, model score AFSE, baseline sugli stessi State8,
  top score, margine e limite di attribuzione.

Il riquadro legacy dentro **Features** e la preview manuale in **Quantum
Engine** sono flussi separati. I 16 theta modificabili della preview non
sostituiscono il theta del bundle live.

### 5. Applicare una perturbazione reale al simulatore

I controlli evento sono nascosti in Blind mode perché rivelano lo scenario.
Per provarli:

1. in **Sensors**, disattivare deliberatamente **Blind mode**;
2. scegliere nodo, evento, magnitudine e durata;
3. premere **Apply perturbation**;
4. verificare nel log il frame/tempo effettivo;
5. riattivare Blind mode e osservare misure, feature e output successivi.

Provare nell'ordine:

- `Common environmental field step`;
- `Single-node drift` o `Single-node noise burst` sul nodo S3;
- `Shared instrument offset`;
- `Sensor dropout`, `Clipping stress` e `Stuck reading`.

Il nome dell'evento, il seed e la causa nascosta non sono input del predittore.
Dropout resta `null`; stuck conserva il valore ricevuto; clipping e input
incompleti devono produrre qualità degradata o `ABSTAIN`, non zeri imputati.

## Interpretare i risultati

### Contesto per numero di nodi

- **1 nodo:** task locale `NORMAL` / `CHANGE_DETECTED`; nessuna attribuzione
  ambiente-dispositivo.
- **2 nodi:** task locale con `local_two_node_ambiguous`; una differenza non
  identifica automaticamente il nodo guasto.
- **3–8 nodi:** task network se i peer sono comparabili; altrimenti
  `degraded_local` o astensione.

### Score, uncertainty e OOD

Gli score del modello non sono probabilità calibrate. `UNCERTAIN` usa soglie
ingegneristiche congelate: top score inferiore a 0,70 o margine tra i primi due
score inferiore a 0,15. Il flag `HEURISTIC OOD` usa un residuo e una soglia
derivata da TRAIN; non identifica una causa fisica e non certifica che un
campione sia davvero fuori distribuzione.

`ENVIRONMENT_COMPATIBLE` e `DEVICE_COMPATIBLE` significano compatibilità del
pattern col dominio simulato. Un offset strumentale condiviso e un campo comune
possono essere osservazionalmente ambigui. Consultare sempre `Context` e
`Attribution note`.

## Lifecycle della sessione

In **Sensors**:

- **Pause** ferma l'avanzamento del simulatore;
- **Resume** riprende la stessa realizzazione;
- **Step** avanza deterministicamente una sessione non running;
- **Stop** chiude l'esecuzione corrente;
- **Reset** riporta la sessione al frame iniziale e forza un nuovo riferimento;
- **Re-run same realization** usa configurazione eseguita e seed congelati;
- **Generate new realization** cambia soltanto il seed del draft;
- **Delete session** rimuove una sessione terminale dalla memoria backend.

Dopo reset o cambio bundle, attendere il nuovo `worker_epoch` e il nuovo
`reference_id`. Risposte asincrone della vecchia epoca non devono essere
considerate correnti.

## Training esplicito e applicazione

Aprire **QNG Training** soltanto per una run intenzionale:

1. premere **Train bounded candidates**;
2. osservare stage e registro degli step realmente accettati;
3. usare **Cancel** per chiedere l'arresto controllato;
4. a job completato, verificare metriche VALIDATION;
5. premere **Apply compatible bundle pair** solo se si vuole cambiare il
   modello live.

Lo stesso intent persistente rende retry/doppio click idempotenti. È ammesso un
solo job pesante. Durante il training le misure continuano, mentre l'analisi
può indicare `PAUSED_FOR_TRAINING`; non viene recuperato un backlog obsoleto.
Un job completato salva candidati e freeze ma non li promuove automaticamente.

La worksheet **Experiments** permette anche di scegliere una coppia local e
network già salvata. Il backend accetta soltanto la coppia esatta del relativo
selection freeze e la applica atomicamente. Opening/refreshing la worksheet non
esegue training, selezione o TEST.

## Persistenza, restart ed export

Bundle, freeze, metriche e job persistono sotto:

```text
${AQSE_ARTIFACT_ROOT:-../AQSE-artifacts}/network-demo/
```

Per verificare il reload:

```sh
make down
make demo
```

La nuova sessione di simulazione va ricreata perché le sessioni restano in
memoria; il registry e la coppia bundle attiva devono invece essere recuperati
senza training automatico.

In **Experiments** si possono esportare configurazione, feature legacy,
registro della sessione browser e registry. Gli export che possono rivelare lo
scenario sono disabilitati in Blind mode. La Blind mode è una barriera UI
locale, non un controllo di sicurezza dell'API.

## Stati operativi principali

| Stato | Significato | Azione |
| --- | --- | --- |
| `BUNDLE UNAVAILABLE` | nessuna coppia compatibile applicata | eseguire preparazione o applicare un freeze |
| `AWAITING REFERENCE` | meno di 8 s osservabili validi | lasciare avanzare la sessione |
| `RUNNING` | ultima finestra completa elaborata | consultare età e qualità |
| `PAUSED FOR TRAINING` | slot exact-state occupato | attendere o cancellare il training |
| `ABSTAIN` | input non eleggibile | leggere i quality flags; non interpretare score assenti |
| `UNCERTAIN` | score/margine sotto gate | non trattare la classe raw come affidabile |
| `HEURISTIC OOD` | residuo oltre soglia TRAIN | trattare come warning, non causa diagnosticata |
| `FAILED` | worker non può mantenere causalità/compatibilità | leggere `state_detail`, poi reset/nuova sessione |

Un `Observation buffer gap` durante il riferimento richiede reset: il worker
non ricostruisce campioni mancanti con dati futuri.

## Verifica e arresto

```sh
make test
make acceptance
make down
```

Il record dei controlli realmente eseguiti è
[`validation/end-to-end-demo.md`](validation/end-to-end-demo.md). Un check non
registrato lì non va considerato superato.
