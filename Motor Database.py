import sqlite3

class MotorDatabase:
    """Sqlite database for motor data"""
    def __init__(self, db_file: str = "motor_database.db"):
        """
            Initializes the database connection
        """
        
