-- Fix REPLICA IDENTITY for all template and pool schemas
-- This enables logical replication to capture changes

-- Function to set replica identity for all tables in a schema
CREATE OR REPLACE FUNCTION set_replica_identity_for_schema(schema_name TEXT) 
RETURNS void AS $$
DECLARE
    table_rec RECORD;
BEGIN
    FOR table_rec IN 
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = schema_name 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
    LOOP
        BEGIN
            EXECUTE format('ALTER TABLE %I.%I REPLICA IDENTITY FULL', schema_name, table_rec.table_name);
            RAISE NOTICE 'Set REPLICA IDENTITY FULL for %.%', schema_name, table_rec.table_name;
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'Failed to set replica identity for %.%: %', schema_name, table_rec.table_name, SQLERRM;
        END;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Fix all template schemas
DO $$
DECLARE
    schema_rec RECORD;
BEGIN
    -- Template schemas
    FOR schema_rec IN 
        SELECT DISTINCT location 
        FROM public.environments 
        WHERE kind = 'schema'
    LOOP
        RAISE NOTICE 'Fixing template schema: %', schema_rec.location;
        PERFORM set_replica_identity_for_schema(schema_rec.location);
    END LOOP;
    
    -- Pool schemas
    FOR schema_rec IN 
        SELECT DISTINCT schema_name 
        FROM public.environment_pool_entries
    LOOP
        RAISE NOTICE 'Fixing pool schema: %', schema_rec.schema_name;
        PERFORM set_replica_identity_for_schema(schema_rec.schema_name);
    END LOOP;
    
    -- Runtime environment schemas
    FOR schema_rec IN 
        SELECT DISTINCT schema 
        FROM public.run_time_environments
        WHERE status != 'deleted'
    LOOP
        RAISE NOTICE 'Fixing runtime schema: %', schema_rec.schema;
        PERFORM set_replica_identity_for_schema(schema_rec.schema);
    END LOOP;
END;
$$;

-- Clean up the function
DROP FUNCTION set_replica_identity_for_schema(TEXT);

RAISE NOTICE '✓ Finished fixing replica identity for all schemas';


