# data/db.py
import sqlite3, os, datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "scamguard.db")
os.makedirs(DATA_DIR, exist_ok=True)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _ensure_column(conn, name: str, type_sql: str):
    """Best-effort: add column if it doesn't already exist."""
    try:
        conn.execute(f"ALTER TABLE prediction_logs ADD COLUMN {name} {type_sql}")
    except sqlite3.OperationalError:
        # column already exists or table missing (created below)
        pass

def init_db():
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            text_excerpt TEXT NOT NULL,
            prediction_label TEXT NOT NULL,
            confidence REAL,
            was_needs_review INTEGER DEFAULT 0,
            review_source TEXT,
            review_pay_fee TEXT,
            review_personal_info TEXT,
            review_notes TEXT,
            created_at TEXT NOT NULL
        )
        """)
        # in case table existed from older version → make sure new columns exist
        _ensure_column(conn, "was_needs_review", "INTEGER DEFAULT 0")
        _ensure_column(conn, "review_source", "TEXT")
        _ensure_column(conn, "review_pay_fee", "TEXT")
        _ensure_column(conn, "review_personal_info", "TEXT")
        _ensure_column(conn, "review_notes", "TEXT")
        conn.commit()

MAX_EXCERPT_LEN = None  # now we want full text; set a number if you later want to cap

def insert_log(
    user_id: str,
    text_excerpt: str,
    prediction_label: str,
    confidence: float | None,
    was_needs_review: int = 0
) -> int:
    """Insert a prediction and return its log_id."""
    if not text_excerpt:
        text_excerpt = "(empty)"
    text_excerpt = text_excerpt.strip().replace("\x00", "")

    if MAX_EXCERPT_LEN:
        text_excerpt = text_excerpt[:MAX_EXCERPT_LEN]

    ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO prediction_logs
                (user_id, text_excerpt, prediction_label, confidence,
                 was_needs_review, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(user_id),
            text_excerpt,
            prediction_label,
            confidence,
            int(was_needs_review),
            ts
        ))
        conn.commit()
        return cur.lastrowid

def save_review_context(
    log_id: int,
    source: str,
    pay_fee: str,
    personal_info: str,
    notes: str
):
    with get_db() as conn:
        conn.execute("""
            UPDATE prediction_logs
            SET review_source = ?,
                review_pay_fee = ?,
                review_personal_info = ?,
                review_notes = ?
            WHERE id = ?
        """, (source, pay_fee, personal_info, notes, int(log_id)))
        conn.commit()

def fetch_logs(user_id: str, limit: int = 200):
    with get_db() as conn:
        cur = conn.execute("""
            SELECT id,
                   text_excerpt,
                   prediction_label,
                   confidence,
                   was_needs_review,
                   created_at
            FROM prediction_logs
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (str(user_id), int(limit)))
        return cur.fetchall()

def delete_log(user_id: str, log_id: int) -> int:
    """Delete a single log that belongs to this user. Returns rows affected (0 or 1)."""
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM prediction_logs WHERE id = ? AND user_id = ?",
            (int(log_id), str(user_id)),
        )
        conn.commit()
        return cur.rowcount

def fetch_review_logs(limit: int = 300):
    """Return logs marked as 'Needs Review' across ALL users."""
    with get_db() as conn:
        cur = conn.execute("""
            SELECT id, user_id, text_excerpt, prediction_label, confidence, created_at
            FROM prediction_logs
            WHERE LOWER(prediction_label) = 'needs review'
            ORDER BY created_at DESC
            LIMIT ?
        """, (int(limit),))
        return cur.fetchall()

def fetch_all_logs(limit: int = 500):
    """Return logs for admin view across ALL users."""
    with get_db() as conn:
        cur = conn.execute("""
            SELECT id, user_id, text_excerpt, prediction_label, confidence, created_at
            FROM prediction_logs
            ORDER BY created_at DESC
            LIMIT ?
        """, (int(limit),))
        return cur.fetchall()

def update_log_label(log_id: int, new_label: str):
    """Allow admin to correct a label."""
    new_label = new_label.strip().title()
    with get_db() as conn:
        conn.execute("""
            UPDATE prediction_logs
            SET prediction_label = ?
            WHERE id = ?
        """, (new_label, int(log_id)))
        conn.commit()

def fetch_logs_by_label(label: str, limit: int = 300):
    """Return logs filtered by Real/Fake/Needs Review."""
    label = label.strip().lower()
    with get_db() as conn:
        cur = conn.execute("""
            SELECT id, user_id, text_excerpt, prediction_label, confidence, created_at
            FROM prediction_logs
            WHERE LOWER(prediction_label) = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (label, int(limit)))
        return cur.fetchall()

def count_needs_review(user_id: str) -> int:
    with get_db() as conn:
        cur = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_logs
            WHERE user_id = ? AND prediction_label = 'Needs Review'
        """, (str(user_id),))
        row = cur.fetchone()
        return int(row[0] if row else 0)

def mark_review_decision(log_id: int, final_label: str) -> None:
    """
    Admin decided on a Needs Review item.
    We set prediction_label to 'Real'/'Fake' BUT keep was_needs_review = 1.
    """
    with get_db() as conn:
        conn.execute("""
            UPDATE prediction_logs
            SET prediction_label = ?, was_needs_review = 1
            WHERE id = ?
        """, (final_label, int(log_id)))
        conn.commit()

def get_admin_stats():
    """Aggregate stats for admin dashboard."""
    with get_db() as conn:
        # Total + label breakdown
        cur = conn.execute("""
            SELECT
                COUNT(*) AS total_scans,
                SUM(CASE WHEN prediction_label = 'Real' THEN 1 ELSE 0 END) AS real_count,
                SUM(CASE WHEN prediction_label = 'Fake' THEN 1 ELSE 0 END) AS fake_count,
                SUM(CASE WHEN prediction_label = 'Needs Review' THEN 1 ELSE 0 END) AS review_count
            FROM prediction_logs
        """)
        row = cur.fetchone()
        total_scans   = row["total_scans"] or 0
        real_count    = row["real_count"] or 0
        fake_count    = row["fake_count"] or 0
        review_count  = row["review_count"] or 0

        # Last 7 days activity
        seven_days_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat(timespec="seconds") + "Z"
        cur2 = conn.execute("""
            SELECT COUNT(*) AS cnt_7d
            FROM prediction_logs
            WHERE created_at >= ?
        """, (seven_days_ago,))
        row2 = cur2.fetchone()
        last_7_days = row2["cnt_7d"] or 0

        # Average confidence (proxy for model certainty)
        cur3 = conn.execute("""
            SELECT AVG(confidence) AS avg_conf
            FROM prediction_logs
            WHERE confidence IS NOT NULL
        """)
        row3 = cur3.fetchone()
        avg_conf = row3["avg_conf"] if row3 and row3["avg_conf"] is not None else None

        # Top active users (by count of scans)
        cur4 = conn.execute("""
            SELECT
                user_id,
                COUNT(*) AS scan_count,
                MIN(created_at) AS first_scan,
                MAX(created_at) AS last_scan
            FROM prediction_logs
            GROUP BY user_id
            ORDER BY scan_count DESC
            LIMIT 10
        """)
        per_user = cur4.fetchall()

    return {
        "total_scans": total_scans,
        "real_count": real_count,
        "fake_count": fake_count,
        "review_count": review_count,
        "last_7_days": last_7_days,
        "avg_confidence": avg_conf,
        "per_user": per_user,
    }