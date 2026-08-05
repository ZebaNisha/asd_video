from db.db_connection import get_connection

conn = get_connection()
if conn:
    print("DB connection working")
    conn.close()
else:
    print("DB connection failed")

