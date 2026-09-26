import threading
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.exc import DBAPIError

from src.platform.api import routes
from src.platform.isolationEngine.environment import (
    EnvironmentHandler,
    SchemaDropTimeoutError,
)


class _RecordingConnection:
    def __init__(self, *, drop_error=None):
        self.calls = []
        self.drop_error = drop_error

    def execute(self, statement, parameters=None):
        sql = str(statement)
        self.calls.append((sql, parameters))
        if sql.startswith("DROP SCHEMA") and self.drop_error is not None:
            raise self.drop_error


class _RecordingEngine:
    def __init__(self, connection):
        self.connection = connection

    @contextmanager
    def begin(self):
        yield self.connection


def test_drop_schema_applies_transaction_local_timeouts():
    connection = _RecordingConnection()
    manager = SimpleNamespace(base_engine=_RecordingEngine(connection))
    handler = EnvironmentHandler(
        manager,
        schema_drop_lock_timeout_ms=1234,
        schema_drop_statement_timeout_ms=5678,
    )

    handler.drop_schema("state_test")

    assert connection.calls == [
        (
            "SELECT set_config('lock_timeout', :value, true)",
            {"value": "1234ms"},
        ),
        (
            "SELECT set_config('statement_timeout', :value, true)",
            {"value": "5678ms"},
        ),
        ('DROP SCHEMA IF EXISTS "state_test" CASCADE', None),
    ]


def test_drop_schema_converts_postgres_lock_timeout():
    class LockTimeout(Exception):
        pgcode = "55P03"

    db_error = DBAPIError("DROP SCHEMA", None, LockTimeout("lock timeout"), False)
    connection = _RecordingConnection(drop_error=db_error)
    manager = SimpleNamespace(base_engine=_RecordingEngine(connection))
    handler = EnvironmentHandler(manager)

    with pytest.raises(SchemaDropTimeoutError, match="is busy"):
        handler.drop_schema("state_busy")


@pytest.mark.asyncio
async def test_delete_environment_offloads_schema_ddl(monkeypatch):
    event_loop_thread = threading.get_ident()
    worker_threads = []

    class Handler:
        def drop_schema(self, schema):
            worker_threads.append(threading.get_ident())

        def mark_environment_status(self, environment_id, status):
            worker_threads.append(threading.get_ident())

    class PoolManager:
        def __init__(self):
            self.released = []

        def release_in_use(self, schema, *, recycle):
            self.released.append((schema, recycle))

    pool_manager = PoolManager()

    environment_id = str(uuid4())
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                maintenance_service=None,
                coreIsolationEngine=SimpleNamespace(environment_handler=Handler()),
                pool_manager=pool_manager,
            )
        ),
        state=SimpleNamespace(db_session=object(), principal_id="test-user"),
        path_params={"env_id": environment_id},
    )
    monkeypatch.setattr(
        routes,
        "require_environment_access",
        lambda session, principal_id, env_id: SimpleNamespace(schema="state_test"),
    )

    response = await routes.delete_environment(request)

    assert response.status_code == 200
    assert len(worker_threads) == 2
    assert all(thread_id != event_loop_thread for thread_id in worker_threads)
    assert pool_manager.released == [("state_test", True)]
