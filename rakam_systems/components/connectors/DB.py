import os
import sqlite3
from typing import Any, List, Optional, Tuple, Union, Dict

from rakam_systems.system_manager import SystemManager
from rakam_systems.components.component import Component

class SQLDB(Component):
    def __init__(self, system_manager: SystemManager, db_path: str = "database.db") -> None:
        self.db_path = db_path
        self.system_manager = system_manager
        self.connection = self._connect_to_db()

    def _connect_to_db(self) -> sqlite3.Connection:
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)  # Create the directory if it doesn't exist
        
        return sqlite3.connect(self.db_path)

    def execute_query(self, query: str, data: Tuple = ()) -> None:
        # Open a new connection each time
        try:
            with sqlite3.connect(self.db_path) as connection:
                cursor = connection.cursor()
                cursor.execute(query, data)
                connection.commit()
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")  # Log the error

    def create_table(self, table: str, columns: str) -> None:
        """
        Creates a table if it does not already exist.
        
        Parameters:
        - table (str): The name of the table to create.
        - columns (str): A string where each column definition is separated by commas and follows
                        the format "column_name data_type".
                        Example: "id INTEGER PRIMARY KEY, name TEXT, age INTEGER"
        """
        # Use the columns string directly in the SQL query
        query = f"CREATE TABLE IF NOT EXISTS {table} ({columns})"
        self.execute_query(query)

    def insert_data(self, table: str, data: dict) -> None:
        """
        Inserts a row of data into the specified table.
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        self.execute_query(query, tuple(data.values()))

    def update_data(self, table: str, data: dict, condition: str, condition_params: Tuple) -> None:
        """
        Updates data in the specified table based on the given condition.
        """
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {condition}"
        params = tuple(data.values()) + condition_params
        self.execute_query(query, params)

    def delete_data(self, table: str, condition: str, condition_params: Tuple) -> None:
        """
        Deletes data from the specified table based on the given condition.
        """
        query = f"DELETE FROM {table} WHERE {condition}"
        self.execute_query(query, condition_params)

    def show_tables_with_content(self) -> Optional[Dict[str, List[Tuple]]]:
        """
        Retrieves all table names and their contents from the database.

        Returns:
        - A dictionary where each key is a table name and the value is a list of rows in that table,
        or None if an error occurs.
        """
        try:
            with sqlite3.connect(self.db_path) as connection:
                cursor = connection.cursor()
                
                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Retrieve content from each table
                tables_content = {}
                for table in tables:
                    cursor.execute(f"SELECT * FROM {table}")
                    rows = cursor.fetchall()
                    tables_content[table] = rows
                
                return tables_content
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return None

    def call_main(self, **kwargs) -> dict:
        return super().call_main(**kwargs)

    def test(self, **kwargs) -> bool:
        """
        Tests the connection to the database.
        """
        try:
            self.connection.cursor().execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False
