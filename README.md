# # PRI representation-probes

Empirisk jämförelse av förbehandling/representation för PRI-sekvenser
(stagger, dwell-and-switch) inför en transformer-baserad seq2seq-modell.

Idén: representationen är den enda variabeln. Allt annat (probe-backbone,
optimizer, seeds) hålls fixt. På ren data fungerar allt — det intressanta är
hur **graciöst varje representation degraderar** under jitter och tappade pulser.
Så vi mäter degraderingskurvor, inte punktvärden.

## Filer

| Fil | Innehåll |
|-----|----------|
| `frontends.py` | Utbytbara representationer: `raw`, `delta`, `hardbin`, `softbin`, `logbin`, `fourier_val`, `fourier_time` |
| `probes.py` | Liten fast backbone + tre probe-uppgifter (level / boundary / period) |
| `separability.py` | Träningsfritt galler (silhouette) — kör först, sållar dåliga FE på sekunder |
| `generator_contract.py` | Datakontrakt + syntetisk fallback-generator för smoke-test |
| `sweep.py` | Svep-drivrutin: separabilitet → full probe-svep → `results.csv` |

## Beroenden

```
pip install torch scikit-learn
```

## Kör smoke-test (syntetisk data)

```
python sweep.py
```

Detta bevisar att hela pipelinen kör end-to-end på inbyggd syntetisk data och
skriver `results.csv`.

## Koppla in din egen generator

I `generator_contract.py`:

1. Implementera `your_generator(...)` så att den anropar ditt script och
   returnerar dicten i kontraktet:
   - `pri` `[n,T]` float — observerad sekvens (input)
   - `level_id` `[n,T]` long — nivå per puls
   - `boundary` `[n,T]` long — 1 där en run byter
   - `period` `[n]` float — sann period
   - `mask` `[n,T]` bool — giltiga positioner
2. Sätt `USE_SYNTH = False`.

`level_id`, `boundary` och `period` känner din generator redan till som ground
truth — det är den enda anpassning som krävs.

## Läsa resultaten

- **Fas 1 (SEP):** hög silhouette = nivåerna separerar bra i det kodade rummet.
  Titta på hur den faller när `jitter` ökar.
- **Fas 2 (PROBE):** plotta `metric` mot `noise` per `fe` och `task`.
  Lutningen på degraderingen är hela poängen. Alla tre metriker är "högre = bättre".

## Confounders att hålla koll på

- Matcha `d_model` över alla FE; bin-embeddings lägger till parametrar — notera det.
- Samma optimizer/schema/seeds överallt (redan fixt i `sweep.py`).
- `norm`-läget (`per_seq` vs `global`) är en tyst confounder — det är en explicit flagga; logga vilket du kör.

## Nästa steg (bortom detta galler)

- Byt `train_noise` till en bredare blandning (inkl. `drop_rate`) för realistiska kurvor.
- Lägg till falska pulser (spurious) i din riktiga generator — där skiljer sig representationerna mest.
- Lägg klassiska PRI-baselines (histogram, CDIF/SDIF, autokorrelation) bredvid som referens.

