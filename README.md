# _Mutatis Mutandis_

# Abstract: About This Project

* [Abstract](./ABSTRACT.md)
* [Concise abstract](./ABSTRACT_concise.md)

## Bibliography

See [Bibliography](./BIBLIOGRAPHY.md)

## People and Projects

* [Manuscript Pamphleteering in Early Stuart England](https://tei-c.org/activities/projects/manuscript-pamphleteering-in-early-stuart-england/)
* [Heuser, Ryan](https://www.english.cam.ac.uk/people/Ryan.Heuser)
* [MacBERTHh](https://huggingface.co/emanjavacas/MacBERTh)
* [Bodleian Repo](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/A50955)
* [Early Modern Manuscripts Online (EMMO)](https://folgerpedia.folger.edu/Early_Modern_Manuscripts_Online_%28EMMO%29?utm_source=chatgpt.com)
* [Early English Books Online Text Creation Partnership (EEBO TCP), Bodleian Digital Library Systems & Services](https://digital.humanities.ox.ac.uk/project/early-english-books-online-text-creation-partnership-eebo-tcp)


## Restoring the Database

Make sure the table space is on an SSD:

```sql
    CREATE TABLESPACE eebo_space LOCATION 'D:/postgres-data-2/eebo';
    Create a temp tablespace if not already and use it for sorting/indexing:

    CREATE TABLESPACE temp_space LOCATION 'D:/postgres-data-2/temp';
    ALTER SYSTEM SET temp_tablespaces = 'temp_space';
    Increase memory for faster index creation

    ALTER SYSTEM SET maintenance_work_mem = '16GB';  -- big enough for token indexes
    ALTER SYSTEM SET work_mem = '256MB';             -- per sort operation
    SELECT pg_reload_conf();
    Kill all connections:

    SELECT pg_terminate_backend(pid)
    FROM pg_stat_activity
    WHERE datname='eebo';
    Restore with 4 worker jobs:

    pg_restore -v -d eebo -j 4 "./db-backup/eebo_backup.dump"
    Monitor:

    -- Active queries (shows index creation)
    SELECT pid, now() - query_start AS duration, state, query
    FROM pg_stat_activity
    WHERE state <> 'idle';

    -- Size of largest tables and indexes
    SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
    FROM pg_stat_user_tables
    ORDER BY pg_total_relation_size(relid) DESC;
    Clean up:

    ALTER SYSTEM RESET maintenance_work_mem;
    ALTER SYSTEM RESET work_mem;
    ALTER SYSTEM RESET temp_tablespaces;
    SELECT pg_reload_conf();
```

## CPU-Bound

For now the methodology is focuosed on my ancient CPU-only (Radeon...), 64 GB setup so fastText over MacBERTh.
