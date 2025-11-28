"""
Simple database connection test for forex SQL Server setup.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def test_connection():
    """Test the database connection with current settings."""
    
    load_dotenv()
    
    print("🔍 Testing SQL Server Connection...")
    print("=" * 50)
    
    # Show current configuration (without password)
    server = os.getenv('SQL_SERVER', 'Not configured')
    database = os.getenv('SQL_DATABASE', 'Not configured')
    username = os.getenv('SQL_USERNAME', 'Not configured')
    trusted = os.getenv('SQL_TRUSTED_CONNECTION', 'no')
    driver = os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server')
    
    print(f"📊 Current Configuration:")
    print(f"   Server: {server}")
    print(f"   Database: {database}")
    print(f"   Username: {username}")
    print(f"   Windows Auth: {trusted}")
    print(f"   Driver: {driver}")
    print()
    
    if server == 'your-sql-server-name-or-ip':
        print("❌ Please configure your .env file with actual SQL Server details")
        print("📝 Update the following in .env:")
        print("   SQL_SERVER=your-actual-server-name")
        print("   SQL_DATABASE=your-forex-database-name")
        print("   SQL_USERNAME=your-username (if not using Windows auth)")
        print("   SQL_PASSWORD=your-password (if not using Windows auth)")
        return False
    
    try:
        from database.connection import ForexSQLServerConnection
        
        db = ForexSQLServerConnection()
        
        print("🔄 Attempting connection...")
        if db.test_connection():
            print("✅ Database connection successful!")
            
            # Try to get table info
            print("\n📊 Checking forex tables...")
            table_info = db.get_forex_table_info()
            
            if table_info['tables']:
                print(f"✅ Found {len(table_info['tables'])} forex-related tables:")
                for table in table_info['tables']:
                    print(f"   - {table['TABLE_NAME']} ({table['TABLE_TYPE']})")
            else:
                print("⚠️  No forex tables found - you may need to create them")
            
            # Try to get forex pairs
            pairs = db.get_forex_pairs()
            if pairs:
                print(f"\n💱 Found {len(pairs)} forex pairs:")
                print(f"   {', '.join(pairs[:10])}")  # Show first 10
                if len(pairs) > 10:
                    print(f"   ... and {len(pairs) - 10} more")
            else:
                print("\n⚠️  No forex pairs found in database")
                print("   Make sure your forex data is properly loaded")
            
            return True
        else:
            print("❌ Database connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    test_connection()