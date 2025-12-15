import sqlite3

conn = sqlite3.connect('data/reelsense.db')
cursor = conn.cursor()

# Show all users
cursor.execute('SELECT email, name, role FROM users')
users = cursor.fetchall()
print('Current users:', users)

# Update ALL users to admin for now (you can be more selective later)
cursor.execute("UPDATE users SET role='admin'")
print(f'Updated {cursor.rowcount} users to admin role')

conn.commit()

# Verify
cursor.execute('SELECT email, name, role FROM users')
admins = cursor.fetchall()
print('Users after update:', admins)

conn.close()
print('\nDone! Now restart the server and log back in.')
