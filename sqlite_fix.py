# SQLite3 compatibility fix for ChromaDB on Streamlit Cloud
"""
This module handles SQLite3 compatibility issues with ChromaDB on Streamlit Cloud.
Import this module at the very beginning of your main application file.
"""

import sys

def fix_sqlite3():
    """
    Replace the system sqlite3 with pysqlite3-binary for ChromaDB compatibility.
    This is required when running on platforms with older SQLite3 versions (< 3.35).
    """
    try:
        import sqlite3
        
        # Check if current SQLite version is compatible
        current_version = sqlite3.sqlite_version
        version_parts = [int(x) for x in current_version.split('.')]
        
        # If SQLite version is 3.35 or higher, no fix is needed
        if version_parts[0] > 3 or (version_parts[0] == 3 and version_parts[1] >= 35):
            print(f"✅ SQLite3 version {current_version} is compatible with ChromaDB - no fix needed")
            return True
    except Exception as e:
        print(f"⚠️ Could not check SQLite version: {e}")

    try:
        # Try to import pysqlite3 and replace the system sqlite3
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
        print("✅ SQLite3 compatibility fix applied successfully")
        return True
    except ImportError as e:
        print(f"⚠️ Could not apply SQLite3 fix: {e}")
        print("📝 For older SQLite versions, install pysqlite3-binary or upgrade your system SQLite")
        print(f"📊 Current SQLite version: {current_version} (ChromaDB requires 3.35+)")
        return False

# Automatically apply the fix when this module is imported
fix_sqlite3()
