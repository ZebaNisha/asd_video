from db.db_connection import get_connection

conn = get_connection()
if conn:
    print("DB connection working")
    conn.close()
else:
    print("DB connection failed")

  //  python test_db.py

  https://chatgpt.com/share/6a39f588-5ff8-83ee-bf58-f7a33ef44b70