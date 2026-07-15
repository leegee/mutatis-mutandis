# Restoring the Database

Make sure the table space is on an SSD:

```sql
    CREATE TABLESPACE eebo_space LOCATION 'D:/postgres-data-2/eebo';
```

Create a temp tablespace if not already and use it for sorting/indexing:

```sql
    CREATE TABLESPACE temp_space LOCATION 'D:/postgres-data-2/temp';
```

Increase memory for faster index creation

```sql
    ALTER SYSTEM SET temp_tablespaces = 'temp_space';
    ALTER SYSTEM SET maintenance_work_mem = '16GB';  -- big enough for token indexes
    ALTER SYSTEM SET work_mem = '256MB';             -- per sort operation
    SELECT pg_reload_conf();
```

Kill all connections:

```sql
    SELECT pg_terminate_backend(pid)
    FROM pg_stat_activity
    WHERE datname='eebo';
```

Restore with 4 workers:

```bash
    pg_restore -v -d eebo -j 4 "./db-backup/eebo_backup.dump"
```

Monitor:

```sql
    -- Active queries (shows index creation)
    SELECT pid, now() - query_start AS duration, state, query
    FROM pg_stat_activity
    WHERE state <> 'idle';

    -- Size of largest tables and indexes
    SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
    FROM pg_stat_user_tables
    ORDER BY pg_total_relation_size(relid) DESC;
```

Clean up:

```sql
    ALTER SYSTEM RESET maintenance_work_mem;
    ALTER SYSTEM RESET work_mem;
    ALTER SYSTEM RESET temp_tablespaces;
    SELECT pg_reload_conf();
```
