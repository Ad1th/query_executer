import re
import time

import aiomysql
import mysql.connector

from app.config import load_config

config = load_config()


class AsyncMysqlClient:
    def __init__(self, conn: aiomysql.Connection, cursor):
        self.conn = conn
        self.cursor = cursor

    @classmethod
    async def create(cls, conn):
        cursor = await conn.cursor(aiomysql.DictCursor)
        return cls(conn, cursor)

    async def execute_query(self, query: str):
        """
        Execute given query
        :param query:
        :return:
        """
        try:
            start_time = time.time()
            await self.cursor.execute(query)
            end_time = time.time()
            results = await self.cursor.fetchall()
            return results, (end_time - start_time)
        except Exception as e:
            print("Query Failed: ", e)

    async def analyze_query(self, query: str):
        try:
            await self.cursor.execute(f"EXPLAIN ANALYZE {query}")
            results = await self.cursor.fetchone()
            results = results["EXPLAIN"]
            return results
        except Exception as e:
            print("Query Analyze Failed: ", e)

    async def set_database(self, database_name: str):
        """
        Set database
        :param database_name:
        :return:
        """
        await self.cursor.execute(f"USE {database_name};")

    async def get_size_of_database(self, database: str):
        """
        Get size of current selected database
        """
        query = f"""
            SELECT 
                ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS size_mb
            FROM 
                information_schema.tables
            WHERE 
                table_schema = "{database}"
            GROUP BY 
                table_schema;
        """
        await self.cursor.execute(query)
        return await self.cursor.fetchone()[0]

    async def get_databases(self):
        """
        Returns all databases from a MySQL connection
        """
        system_dbs = {'information_schema', 'mysql', 'performance_schema', 'sys'}  # Filter out internal MYSQL databases
        query = "SHOW DATABASES;"
        await self.cursor.execute(query)
        return [row[0] for row in await self.cursor.fetchall() if row[0] not in system_dbs]
