from db.db_connection import get_connection

def insert_video_analysis(video_name, feature_file, prediction, confidence):
    connection = get_connection()
    if connection is None:
        print("Database connection failed")
        return

    try:
        cursor = connection.cursor()

        query = """
        INSERT INTO video_analysis (video_name, feature_file, prediction, confidence)
        VALUES (%s, %s, %s, %s)
        """

        values = (video_name, feature_file, prediction, confidence)
        cursor.execute(query, values)
        connection.commit()

        print("Record inserted successfully")

    except Exception as e:
        print("Error inserting record:", e)

    finally:
        cursor.close()
        connection.close()