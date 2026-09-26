"""
Fix REPLICA IDENTITY for all template and pool schemas to enable logical replication.
Run this after upgrading to fix existing schemas.
"""

import logging
from os import environ
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_replica_identity_for_schema(conn, schema: str) -> None:
    """Set REPLICA IDENTITY FULL for all tables in a schema."""
    logger.info(f"Fixing replica identity for schema: {schema}")

    # Get all tables in the schema
    rows = conn.execute(
        text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = :schema AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """),
        {"schema": schema},
    ).fetchall()

    tables = [r[0] for r in rows]
    logger.info(f"Found {len(tables)} tables in {schema}")

    for table in tables:
        try:
            conn.execute(
                text(f'ALTER TABLE "{schema}"."{table}" REPLICA IDENTITY FULL')
            )
            logger.info(f"✓ Set REPLICA IDENTITY FULL for {schema}.{table}")
        except Exception as e:
            logger.warning(
                f"✗ Failed to set replica identity for {schema}.{table}: {e}"
            )


def main():
    db_url = environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")

    engine = create_engine(db_url)

    with engine.begin() as conn:
        # Get all template schemas
        template_rows = conn.execute(
            text("""
                SELECT DISTINCT location 
                FROM public.environments 
                WHERE kind = 'schema'
            """)
        ).fetchall()

        template_schemas = [r[0] for r in template_rows]
        logger.info(f"Found {len(template_schemas)} template schemas")

        for schema in template_schemas:
            fix_replica_identity_for_schema(conn, schema)

        # Get all pool schemas
        pool_rows = conn.execute(
            text("""
                SELECT DISTINCT schema_name 
                FROM public.environment_pool_entries
            """)
        ).fetchall()

        pool_schemas = [r[0] for r in pool_rows]
        logger.info(f"Found {len(pool_schemas)} pool schemas")

        for schema in pool_schemas:
            fix_replica_identity_for_schema(conn, schema)

        # Get all runtime environment schemas
        rte_rows = conn.execute(
            text("""
                SELECT DISTINCT schema 
                FROM public.run_time_environments
                WHERE status != 'deleted'
            """)
        ).fetchall()

        rte_schemas = [r[0] for r in rte_rows]
        logger.info(f"Found {len(rte_schemas)} runtime environment schemas")

        for schema in rte_schemas:
            fix_replica_identity_for_schema(conn, schema)

    logger.info("✓ Finished fixing replica identity for all schemas")


if __name__ == "__main__":
    main()
