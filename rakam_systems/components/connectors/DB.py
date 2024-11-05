import os
import sqlite3
from typing import Any, List, Optional, Tuple, Union

from rakam_systems.system_manager import SystemManager
from rakam_systems.components.component import Component

class SQLDB(Component):
    def __init__(self, system_manager: SystemManager, db_path: str = "database.db") -> None:
        self.db_path = db_path
        self.system_manager = system_manager
        self.connection = self._connect_to_db()

    def _connect_to_db(self) -> sqlite3.Connection:
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)  # Create the directory if it doesn't exist
        
        return sqlite3.connect(self.db_path)

    def execute_query(self, query: str, params: Optional[Union[Tuple, List]] = None) -> List[Tuple[Any]]:
        """
        Executes a query and returns the result as a list of tuples.
        """
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchall()
            self.connection.commit()
            return result
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            self.connection.rollback()
            return []
        finally:
            cursor.close()

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
