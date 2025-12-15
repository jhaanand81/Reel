"""Check credits for prem22@gmail.com"""
import sqlite3
from pathlib import Path

db_path = Path('data/reelsense.db')
print(f"Database: {db_path} (exists: {db_path.exists()})")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Find user
cursor.execute("SELECT id, email, name, credits FROM users WHERE email = 'prem22@gmail.com'")
user = cursor.fetchone()

if user:
    print(f"\nUser found:")
    print(f"  ID: {user[0]}")
    print(f"  Email: {user[1]}")
    print(f"  Name: {user[2]}")
    print(f"  Credits: {user[3]}")
else:
    print("\nUser not found!")
    # List all users
    cursor.execute("SELECT email, credits FROM users")
    users = cursor.fetchall()
    print(f"\nAll users:")
    for u in users:
        print(f"  {u[0]}: {u[1]} credits")

conn.close()
