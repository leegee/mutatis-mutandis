# tests/test_postgres_analytical_backend.py

from lib.postgres_analytical_backend import PostgresAnalyticalBackend


def main() -> None:
    with PostgresAnalyticalBackend() as db:
        concepts = db.concepts()
        assert concepts == ["WHITE"]

        concept = db.concept("WHITE")
        assert concept is not None
        assert concept["n_events"] == 10669

        field = db.field_events("WHITE")
        assert len(field) == 82018

        runs = db.retrieval_runs("WHITE")
        assert len(runs) == 45

        geometry = db.geometry("WHITE")
        assert len(geometry) == 82018

        clusters = db.clusters("WHITE")
        assert len(clusters) == 64

        print("PostgreSQL analytical backend: OK")
        print(f"concepts: {len(concepts)}")
        print(f"field events: {len(field)}")
        print(f"retrieval runs: {len(runs)}")
        print(f"geometry: {len(geometry)}")
        print(f"clusters: {len(clusters)}")


if __name__ == "__main__":
    main()
