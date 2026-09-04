# _Mutatis Mutandis_

    git@github.com:leegee/mutatis-mutandis.git
    https://github.com/leegee/mutatis-mutandis.git

        We all declare for liberty but in using the same word we do not all mean the same thing
        -- Abraham Lincoln, Address at the Sanitary Fair in Baltimore, Maryland, on April 18, 186

## Code Synopsis

    cd $PROJECT_ROOT/gui/research && bun dev    #  Research graph

    cd $PROJECT_ROOT/python
    source .venv/Scripts/activate               # Load environment

    cd $PROJECT_ROOT
    python src/tier.... in order

## Conceptual Synopsis

> Who today is using the concept of liberty as it was used by Milton, or Hobbes, or Locke?
>
> Who during the 1600s was expressing in their own language concepts we differntly express today?
>
> How was our contemporary concept of privacy referenced in the past? Cf Entick v Carrington (1765)
>
> Who were considered terrorists in the 17th century? (Fanatics, Sectaries, Enthusiasts, Levellers, Diggers, Muggltonians, Anabaptists, Jesuits...)
>
> How in the past was the concept we term X referenced  if at all?
>
> Specifically: how has the typological symbol of the Son Of Man been transmitted and received through the millennia and how has it interacted
> with surrounding pysiognomic imagery?
>
> What do the contextual embeddings of 'albino' and 'albinism' retrieve through past centuries and in European traditions for which we have corpus?
>
> Can we recursively reverse search over diachronic ranges, taking top results for each period as bridge terms to search with in the earlier  date range?

## Progress

Ideally this project would build a complete Ontological Topology of a corpus, a gigantic semantic space as a structured geometric object, where meaning is illustrated by relative positions, continuity and deformation of distributions across time, rather than through dictionaries. Nice idea but requires 2-5 days GPU or about 6 weeks of CPU...

We tried avoiding corpus-wide embedding, we recursively probe system where semantic topology is reconstructed through anchored neighbourhood expansion rather than exhaustive representation, and performed recursively in reverse chronological order:

1. Search 2026-1926: search 'privacy' - store semantic neighbours
1. Search 1826-2026: search above neighbours, store and repeat for previous century, etc

When we added the EEBO Bibles, FAISS failed us. Now we have added selected texts from ECCO and CLMET, we have had to switch to a disk-based ANN system.

Microsoft's old DiskANNpy library failed at scale, and although the new version in Rust looks promising, there is no Python interface and
not sure we yet have time to fast-itterate in Rust.

Currently trying embeddings over the full corpus with the open source Lance for our 16,000,000 embeddings.

So should we, instead of treating the corpus as an exhaustively represented semantic space to be queried by fixed lexical anchors, instead recursively probe it through anchored neighbourhood expansion, reconstructing semantic topology from contextual observations while traversing the corpus in reverse chronological order.


```
        XML Corpus
            |
    Postgres (text + meta)
            |
    Parquet or Zarr (event log: contextual embeddings)
            |
    LanceDB or FAISS (approximate semantic geometry index)     TODO Check: Milvus
            |
    Query layer (token/window/hybrid encoding)
            |
    Analysis (drift, clustering, interpretation)
            |
    GUI (Solid, d3, CosmosGL, DeckGL)
```

Currently experimenting with ensemble embeddings. Ideally would process clauses, sentances and paragraphs, but MacBERTh is somewhat restricted and EEBO somewhat noisy, so that is not trivial to use or create a sentence transformer.

## Dependencies

Managed by UV (`uv sync`) and bun (`bun install`)

## Colab Notebooks

Update `./macberth_pg_secrets.json` on Google Drive's root dir with the host/port output from `ngrok tcp 5432`.

Don't forget to restart the Colab session when the IP changes.

(Colab workbooks are well out of date)

## Sources

* [The Corpus of Late Modern English Texts, version 3.1](https://fedora.clarin-d.uni-saarland.de/clmet/clmet.html)
* [Eighteenth Century Collection Online TCP](https://github.com/Text-Creation-Partnership/ECCO-TCP)
* [Early English Books Online TCP](https://github.com/Text-Creation-Partnership/EEBO-TCP-Collections-Navigations)

## Bibliography

See [Bibliography](./BIBLIOGRAPHY.md)

## CPU-Bound

For now the methodology is focuosed on my ancient CPU-only (Radeon...), 64 GB setup so fastText over MacBERTh. DirectML sometimes
works, sometimes dies horribly.

## APIs

Every tier has the folllowing:

        main()    = argument parsing, constructs expensive resources
        core()    = the implementation, uses expensive resources
        run()     = one-shot CLI/notebook convenience wrapper
        service() = reusable programmatic API, requires expensive resources

## In Progress

1. extending API to tiers 3+
1. enlgarging the corpus (streaming)
1. parquet

        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export ORT_NUM_THREADS=1
        export OMP_WAIT_POLICY=PASSIVE
        python src/tier1/tier1_0_corpus2zarr.py \
        --store-backend parquet \
        --report-every 1
        --batch-size 32 \
        --store g:/corpus-out/parquet/



## To Do

This work was originally based on EEBO but needs to include other datasources including ECCO which will require
a new primary key to avoid collision on `doc_id`. I have tried adding a joint key with a new field, `corpus`,
so will leave that for the next total itteration.

Ingestion is batched and sharded by year across tiers 0 - 1 (XML to Postgres, Postgres to Zarr FS, Zarr FS to FAISS)

1. T2 - resolve once
1. Stage materialisation in PG then dump to SQLite
1. migrate the materialisation SQLite to PG
1. metrics
1. update ZarrEventLookup to stream/yield
1. expose tier 2's analytics
1. run tier 1's 'masked' path in the Cloud
1. Finish remote job execution
1. api result paging
1. tier 1 should adopt fsspec

### EEBO-TCP Language Composition

Within the activated corpus bounds prior to `langdetect`:

```
eebo=# SELECT lang, COUNT(*) AS count
eebo-# FROM documents
eebo-# GROUP BY lang
eebo-# ORDER BY count DESC;
 lang | count
------+-------
 eng  | 39623
 lat  |   255
 wel  |    83
 fre  |    48
 frm  |     8
 dut  |     7
 mul  |     2
 sco  |     2
 spa  |     2
 ger  |     2
 grc  |     1
 gla  |     1
 new  |     1
 por  |     1
 ```

## Screenshots

![](./docs/screen-202605/deck.png)
![](./docs/screen-202605/1.png)
![](./docs/screen-202605/1-e.png)
![](./docs/screen-202605/2.png)
![](./docs/screen-202605/2-e.png)
![](./docs/screen-202605/3.png)
![](./docs/screen-202605/4.png)
![](./docs/screen-202605/4-s.png)
![](./docs/screen-202605/geo.png)

## Notes

A91273 = ID 99863437 = _Salus populi solus rex_ (London, October 17, 1648) = Thomason Tracts collection E.467 = _Salus populi solus rex = The peoples safety is the sole soveraignty, or The royalist out-reasoned [electronic resource] : calculated for the hopefull recovery of the considerate royalist, from the dangerous infection of the slie sophistry of Iudge Ienkings: in his late legend, published to perswade the people into a voluntary slavery, and obliged servitude to the Kings pleasure: most irrationally asserting, that the King is principium, caput, & finis Parliamenti. That the Parliament hath a power over our lives, liberties, laws, and goods, according to the known laws of the land._ cf https://catalog.folger.edu/record/501701


