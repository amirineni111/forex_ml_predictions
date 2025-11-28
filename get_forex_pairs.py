#!/usr/bin/env python3
"""
Get all available forex pairs from the database
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.connection import ForexSQLServerConnection

def get_all_forex_pairs():
    """Get all forex pairs from database"""
    try:
        conn = ForexSQLServerConnection()
        if not conn.test_connection():
            print("❌ Database connection failed")
            return []
        
        pairs = conn.get_forex_pairs()
        print(f"Found {len(pairs)} forex pairs:")
        for i, pair in enumerate(pairs, 1):
            print(f"{i:2}. {pair}")
        
        conn.close()
        return pairs
        
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    pairs = get_all_forex_pairs()