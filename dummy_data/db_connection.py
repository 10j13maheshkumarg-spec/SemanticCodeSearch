import sqlite3
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path="app.db"):
        self.db_path = db_path
        self.connection = None

    def connect(self):
        """Establishes a connection to the SQLite database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            logger.info("Successfully connected to the database.")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def execute_query(self, query, parameters=None):
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        if parameters:
            cursor.execute(query, parameters)
        else:
            cursor.execute(query)
            
        self.connection.commit()
        return cursor.fetchall()
        
    def close(self):
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed.")
