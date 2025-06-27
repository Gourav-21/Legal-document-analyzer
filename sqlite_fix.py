# SQLite3 compatibility fix for ChromaDB on Streamlit Cloud
"""
This module handles SQLite3 compatibility issues with ChromaDB on Streamlit Cloud.
Import this module at the very beginning of your main application file.
"""

import sys

def fix_sqlite3():
    """
    Replace the system sqlite3 with pysqlite3-binary for ChromaDB compatibility.
    This is required because Streamlit Cloud has an older version of SQLite3.
    """
    try:
        # Try to import pysqlite3 and replace the system sqlite3
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
        print("✅ SQLite3 compatibility fix applied successfully")
        return True
    except ImportError as e:
        print(f"⚠️ Could not apply SQLite3 fix: {e}")
        print("📝 Make sure pysqlite3-binary is installed: pip install pysqlite3-binary")
        return False

# Automatically apply the fix when this module is imported
fix_sqlite3()
