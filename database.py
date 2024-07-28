import psycopg2
import csv

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="candidates_db",
    user="postgres",
    password="aryan1521",
    host="localhost"
)
cur = conn.cursor()

# Read data from CSV and insert into the table
with open('dataset.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row if there is one
    for row in reader:
        cur.execute(
            "INSERT INTO candidates (name, contact_details, location, job_skills, experience, projects, comments) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            row
        )


conn.commit()
cur.close()
conn.close()
