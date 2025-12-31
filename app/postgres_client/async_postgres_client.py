import time

from psycopg import AsyncConnection
from psycopg.rows import dict_row


class AsyncPostgresClient:
    """
    Asynchronous Postgres client using a provided async connection.
    """
    def __init__(self, conn: AsyncConnection):
        self.conn = conn

    async def execute_query(self, query: str):
        """
        Execute the given SQL query asynchronously.
        :param query: SQL query to execute.
        :return: (results, time_taken)
        """
        try:
            async with self.conn.cursor(row_factory=dict_row) as cur:
                start_time = time.time()
                await cur.execute(query)
                results = await cur.fetchall()
                end_time = time.time()
                return results, end_time - start_time
        except Exception as e:
            print("Query failed:", e)
            await self.conn.rollback()
            return None, 0

    async def analyze_query(self, query: str):
        """
        Execute EXPLAIN ANALYZE on the given SQL query asynchronously.
        :param query: SQL query to execute.
        :return: JSON plan object from EXPLAIN (FORMAT JSON)
        """
        query = f"EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) {query}"
        try:
            async with self.conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query)
                result = await cur.fetchall()
                # FORMAT JSON returns a single row with 'QUERY PLAN' containing the JSON
                plan_json = result[0]['QUERY PLAN'][0]
                return plan_json
        except Exception as e:
            print("Query failed:", e)
            await self.conn.rollback()
            return None

    async def get_size_of_database(self, database: str):
        """
        Get the size (in MB) of the given PostgreSQL database.
        """
        query = "SELECT ROUND(pg_database_size(%s) / 1024 / 1024, 2) AS size_mb;"
        async with self.conn.cursor() as cur:
            await cur.execute(query, (database,))
            result = await cur.fetchone()
            return result[0] if result else 0.0

    async def get_databases(self):
        """
        Return all user-created databases (excluding system ones).
        """
        system_dbs = {'template0', 'template1', 'postgres'}
        query = "SELECT datname FROM pg_database WHERE datistemplate = false;"
        async with self.conn.cursor() as cur:
            await cur.execute(query)
            return [row[0] for row in await cur.fetchall() if row[0] not in system_dbs]

    async def set_database(self, database_name: str):
        """
        Not supported with a fixed async connection.
        """
        raise NotImplementedError("Changing databases is not supported with a fixed connection.")