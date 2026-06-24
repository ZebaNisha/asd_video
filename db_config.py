DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "your_mysql_password",
    "database": "autism_video_db"
}
CREATE DATABASE autism_video_db;
USE autism_video_db;

CREATE TABLE video_analysis (
    id INT PRIMARY KEY AUTO_INCREMENT,
    video_name VARCHAR(255),
    feature_file VARCHAR(255),
    prediction VARCHAR(100),
    confidence DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);