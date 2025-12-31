import asyncio
from asyncio import Queue
from typing import Dict, List, Callable

from app.config import load_config
from app.helpers import build_all_queries
from app.types import BenchmarkQuery, ReadyQuery
from app.experiment_writer import write_experiment

# from app.db_client.base_db_client import BaseDBClient

from app.mysql_client.mysql_client import MysqlClient
from app.mysql_client.async_mysql_client import AsyncMysqlClient
from app.mysql_client.create_pool import create_mysql_pool

from app.postgres_client.postgres_client import PostgresClient
from app.postgres_client.async_postgres_client import AsyncPostgresClient
from app.postgres_client.create_pool import create_postgres_pool

from app.duckdb_client.duckdb_client import DuckDbClient

from app.analyze_parsers import (
    parse_analyze_mysql,
    extract_total_runtime,
    extract_runtime_and_filter_scans_postgres,
    extract_runtime_and_filter_scans_duckdb,
)

config = load_config()


# -----------------------------------------------------------
# Database client bootstrap (SYNC)
# -----------------------------------------------------------

def start_db_connections() -> Dict[str, object]:
    clients: Dict[str, object] = {}

    if config.database.mysql.enabled:
        clients["mysql"] = MysqlClient()

    if config.database.postgres.enabled:
        clients["postgres"] = PostgresClient()

    if config.database.duckdb.enabled:
        clients["duckdb"] = DuckDbClient()

    return clients




def get_min_max_of_column(client, column: str):
    """
    Find the min and max value of a column across all tables.
    Used for adaptive sampling / range generation.
    """
    column_lower = column.lower()

    tables = client.get_table_list()
    for table in tables:
        column_list = client.get_column_list_of_table(table)
        normalized_columns = [col.lower() for col in column_list]

        if column_lower in normalized_columns:
            actual_column = column_list[normalized_columns.index(column_lower)]
            return client.get_min_max_of_column(table, actual_column)

    return None, None




# -----------------------------------------------------------
# Queue Worker
# -----------------------------------------------------------

class DatabaseQueueWorker:
    """
    Asynchronous worker that executes benchmark queries
    against enabled databases only.
    """

    def __init__(self, callback: Callable, num_workers: int = 5):
        self.callback = callback
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)

        self.mysql_queue = Queue() if config.database.mysql.enabled else None
        self.postgres_queue = Queue() if config.database.postgres.enabled else None
        self.duckdb_queue = Queue() if config.database.duckdb.enabled else None

        self.mysql_pool = None
        self.postgres_pool = None

        self.postgres_in_progress = False
        self.duckdb_in_progress = False

        asyncio.create_task(self.dispatch_loop())

    async def init(self):
        if config.database.mysql.enabled:
            self.mysql_pool = await create_mysql_pool()

        if config.database.postgres.enabled:
            self.postgres_pool = create_postgres_pool()

        return self

    async def dispatch_loop(self):
        while True:
            if self.mysql_queue and not self.mysql_queue.empty():
                queries, benchmark_query = await self.mysql_queue.get()
                asyncio.create_task(self.run_mysql_task(queries, benchmark_query))

            if (
                self.postgres_queue
                and not self.postgres_queue.empty()
                and not self.postgres_in_progress
            ):
                self.postgres_in_progress = True
                queries, benchmark_query = await self.postgres_queue.get()
                asyncio.create_task(self.run_postgres_task(queries, benchmark_query))

            if (
                self.duckdb_queue
                and not self.duckdb_queue.empty()
                and not self.duckdb_in_progress
            ):
                self.duckdb_in_progress = True
                queries, benchmark_query = await self.duckdb_queue.get()
                asyncio.create_task(self.run_duckdb_task(queries, benchmark_query))

            await asyncio.sleep(0.05)

    async def run_mysql_task(self, queries, benchmark_query):
        async with self.semaphore:
            try:
                async with self.mysql_pool.acquire() as conn:
                    client = await AsyncMysqlClient.create(conn)
                    await self.callback(queries, benchmark_query, client)
            except Exception as e:
                print("[MySQL ERROR]", e)

    async def run_postgres_task(self, queries, benchmark_query):
        async with self.semaphore:
            try:
                async with self.postgres_pool.connection() as conn:
                    client = AsyncPostgresClient(conn)
                    await self.callback(queries, benchmark_query, client)
            except Exception as e:
                print("[Postgres ERROR]", e)
            finally:
                self.postgres_in_progress = False

    async def run_duckdb_task(self, queries, benchmark_query):
        async with self.semaphore:
            try:
                client = DuckDbClient()
                await self.callback(queries, benchmark_query, client)
            except Exception as e:
                print("[DuckDB ERROR]", e)
            finally:
                self.duckdb_in_progress = False

    def schedule_callback(self, queries, benchmark_query: BenchmarkQuery):
        db = benchmark_query.database.lower()

        if db == "mysql" and self.mysql_queue:
            self.mysql_queue.put_nowait((queries, benchmark_query))
        elif db == "postgres" and self.postgres_queue:
            self.postgres_queue.put_nowait((queries, benchmark_query))
        elif db == "duckdb" and self.duckdb_queue:
            self.duckdb_queue.put_nowait((queries, benchmark_query))
        else:
            print(f"[WARN] Database disabled or unknown: {benchmark_query.database}")


# -----------------------------------------------------------
# Result Storage
# -----------------------------------------------------------

class ResultStorage:
    def __init__(self):
        self.raw_result_list = []
        self.parsed_result_list = []
        self.lock = asyncio.Lock()


# -----------------------------------------------------------
# Backend Service
# -----------------------------------------------------------

class BackendService:
    def __init__(self, callback_table_update=None):
        self.clients = start_db_connections()
        self.queue_worker: DatabaseQueueWorker | None = None
        self.result_storage = ResultStorage()
        self.callback_table_update = callback_table_update

    async def initialize_queue_worker(self):
        if not (
            config.database.mysql.enabled
            or config.database.postgres.enabled
            or config.database.duckdb.enabled
        ):
            raise RuntimeError("No database enabled in config")

        self.queue_worker = DatabaseQueueWorker(self.execute_query_batch)
        await self.queue_worker.init()

    def set_table_update_callback(self, callback):
        self.callback_table_update = callback

    async def schedule_query_exectution(self, benchmark_query: BenchmarkQuery, range_values):
        queries = build_all_queries(benchmark_query.query, range_values)
        self.queue_worker.schedule_callback(queries, benchmark_query)
        print(f"Scheduled Query: {benchmark_query.name}")

    async def execute_query_batch(
        self,
        queries: List[ReadyQuery],
        benchmark_query: BenchmarkQuery,
        client,
    ):
        print(f"Starting batch for {benchmark_query.database}")

        raw_results = []
        parsed_results = []

        for i, ready_query in enumerate(queries, start=1):
            try:
                result = await client.analyze_query(ready_query.query)
                formatted = await self._process_result(
                    result, ready_query, benchmark_query
                )
                raw_results.append(result)
                parsed_results.append(formatted)
                print(f"{benchmark_query.database} Query {i}/{len(queries)} done")
            except Exception as e:
                print(f"[ERROR] Query {i}: {e}")

        async with self.result_storage.lock:
            self.result_storage.raw_result_list.append(raw_results)
            self.result_storage.parsed_result_list.append(parsed_results)

            if self.callback_table_update:
                self.callback_table_update(benchmark_query)

    async def _process_result(self, result, ready_query, benchmark_query):
        var_data = ready_query.variables
        var_names = [v["name"] for v in var_data]

        db = benchmark_query.database.lower()

        if db == "postgres":
            parsed_pg = extract_runtime_and_filter_scans_postgres(result, var_names)
            parsed = parsed_pg["filters"]
            total_runtime = parsed_pg["total_runtime"]

        elif db == "mysql":
            parsed = parse_analyze_mysql(result, var_names)
            total_runtime = extract_total_runtime(result)

        elif db == "duckdb":
            parsed_dd = extract_runtime_and_filter_scans_duckdb(result, var_names)
            parsed = parsed_dd["filters"]
            total_runtime = parsed_dd["total_runtime"]

        else:
            parsed = []
            total_runtime = 0

        write_experiment(
            query_id=benchmark_query.name,
            database=benchmark_query.database.lower(),
            parameters={v["name"]: v["value"] for v in var_data},
            runtime_ms=total_runtime,
            filters={
                f["variable"]: {
                    "total_rows": f["total_rows"],
                    "rows_removed": f.get("rows_removed", 0),
                }
                for f in parsed
            },
            plan_json=result,
        )

        return {
            "server": benchmark_query.database,
            "query": benchmark_query.name,
            "runtime": total_runtime,
            "filters": parsed,
            "variables": var_data,
        }