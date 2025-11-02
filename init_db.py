import sqlite3
import os
from werkzeug.security import generate_password_hash

DB_PATH = 'mood_ring.db'

def init_db(create_admin=False, admin_username='admin', admin_password=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # Detections table
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  name TEXT NOT NULL,
                  image_data BLOB,
                  emotion TEXT NOT NULL,
                  confidence REAL,
                  detection_type TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')

    conn.commit()

    if create_admin:
        if not admin_password:
            admin_password = os.environ.get('MOOD_ADMIN_PASSWORD', 'admin')
        password_hash = generate_password_hash(admin_password)
        try:
            c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                      (admin_username, password_hash))
            conn.commit()
            print(f"Admin user '{admin_username}' created.")
        except sqlite3.IntegrityError:
            print(f"Admin user '{admin_username}' already exists.")

    conn.close()
    print(f"Database initialized at {DB_PATH}")

if __name__ == '__main__':
    # By default do not create admin; pass CREATE_ADMIN=1 env var to create
    create_admin = os.environ.get('CREATE_ADMIN', '0') == '1'
    admin_user = os.environ.get('CREATE_ADMIN_USER', 'admin')
    admin_pass = os.environ.get('CREATE_ADMIN_PASS')
    init_db(create_admin=create_admin, admin_username=admin_user, admin_password=admin_pass)
