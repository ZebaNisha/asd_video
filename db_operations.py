from db.db_connection import get_connection

def save_result(video_name, prediction, confidence):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO video_analysis (video_name, prediction, confidence)
    VALUES (%s, %s, %s)
    """
    cursor.execute(query, (video_name, prediction, confidence))
    conn.commit()

    cursor.close()
    conn.close()