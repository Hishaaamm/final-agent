import sqlite3
from typing import List, Dict, Optional
from power_automate import send_leave_email, MANAGER_EMAIL
DB_NAME = "enterprise_assistant.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        emp_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT,
        role TEXT DEFAULT 'employee',
        manager_name TEXT DEFAULT 'Manager'
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS leave_balances (
        emp_id TEXT PRIMARY KEY,
        casual_total INTEGER DEFAULT 12,
        casual_used INTEGER DEFAULT 0,
        sick_total INTEGER DEFAULT 6,
        sick_used INTEGER DEFAULT 0,
        FOREIGN KEY(emp_id) REFERENCES employees(emp_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS leave_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        emp_id TEXT NOT NULL,
        employee_name TEXT NOT NULL,
        leave_type TEXT NOT NULL,
        date TEXT NOT NULL,
        reason TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        FOREIGN KEY(emp_id) REFERENCES employees(emp_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS it_tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        emp_id TEXT NOT NULL,
        employee_name TEXT NOT NULL,
        issue_type TEXT NOT NULL,
        priority TEXT NOT NULL,
        reason TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'open',
        assigned_engineer TEXT DEFAULT 'Unassigned'
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS known_outages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_type TEXT NOT NULL,
        description TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active'
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS maintenance_schedule (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        system_name TEXT NOT NULL,
        description TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'planned'
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS asset_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        emp_id TEXT NOT NULL,
        employee_name TEXT NOT NULL,
        asset_type TEXT NOT NULL,
        reason TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'manager_approval_pending',
        manager_approval TEXT DEFAULT 'pending',
        it_approval TEXT DEFAULT 'pending',
        inventory_status TEXT DEFAULT 'pending'
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inventory (
        asset_type TEXT PRIMARY KEY,
        available_quantity INTEGER NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS activity_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        role TEXT,
        intent TEXT,
        user_message TEXT,
        assistant_response TEXT
    )
    """)

    seed_employees(cursor)
    seed_default_data(cursor)

    conn.commit()
    conn.close()


def seed_employees(cursor):
    employees = [
        ("EMP001", "Virat","hishammohd313@gmail.com",  "employee", "Manager"),
        ("EMP002", "Hisham", "hishammohd313@gmail.com", "employee", "Manager"),
        ("EMP003", "Rohit", "hishammohd313@gmail.com", "employee", "Manager"),
        ("EMP004", "Rahul", "hishammohd313@gmail.com", "employee", "Manager"),
        ("EMP005", "Dhoni", "hishammohd313@gmail.com", "employee", "Manager"),
        ("EMP006", "Faizan", "hishammohd313@gmail.com", "employee", "Manager"),
        ("EMP007", "Dev", "hishammohd313@gmail.com", "employee", "Manager"),
        ("EMP008", "Kiran", "hishammohd313@gmail.com", "employee", "Manager"),
        ("EMP009", "Manoj", "hishammohd313@gmail.com", "employee", "Manager"),
        ("EMP010", "Arjun", "hishammohd313@gmail.com", "employee", "Manager"),
    ]

    for emp in employees:
        # Insert employee
        cursor.execute("""
        INSERT OR IGNORE INTO employees (emp_id, name, email, role, manager_name)
        VALUES (?, ?, ?, ?, ?)
        """, emp)

        # Insert leave balance
        cursor.execute("""
        INSERT OR IGNORE INTO leave_balances
        (emp_id, casual_total, casual_used, sick_total, sick_used)
        VALUES (?, 12, 0, 6, 0)
        """, (emp[0],))


def seed_default_data(cursor):
    cursor.execute("""
    INSERT OR IGNORE INTO known_outages (id, issue_type, description, status)
    VALUES
    (1, 'vpn', 'VPN is slow for some users due to gateway maintenance.', 'active'),
    (2, 'outlook', 'Some users may face delay in email sync.', 'active')
    """)

    cursor.execute("""
    INSERT OR IGNORE INTO maintenance_schedule (id, system_name, description, status)
    VALUES
    (1, 'network', 'Network maintenance planned this weekend.', 'planned'),
    (2, 'printer', 'Printer server update is scheduled.', 'planned')
    """)

    cursor.execute("""
    INSERT OR IGNORE INTO inventory (asset_type, available_quantity)
    VALUES
    ('laptop', 3),
    ('monitor', 5),
    ('keyboard', 10),
    ('mouse', 10),
    ('vpn token', 2),
    ('software license', 4)
    """)


# ---------------- EMPLOYEE / LEAVE ----------------

def get_employee(emp_id: str) -> Optional[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM employees WHERE emp_id = ?", (emp_id.upper().strip(),))
    row = cursor.fetchone() #fetch one row

    conn.close()
    return dict(row) if row else None

def get_employee_details(emp_id: str):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM employees WHERE emp_id = ?", (emp_id.upper(),))
    row = cursor.fetchone()

    conn.close()

    return dict(row) if row else None

def insert_leave_request(emp_id: str, leave_type: str, date: str, reason: str) -> Dict:
    emp_id = emp_id.upper().strip()
    leave_type = leave_type.lower().strip()

    employee = get_employee(emp_id)

    if not employee:
        return {"success": False, "message": f"No employee found with ID {emp_id}."}

    if leave_type not in ["sick", "casual"]:
        return {"success": False, "message": "Leave type must be sick or casual."}

    balance = check_leave_balance(emp_id)

    if not balance.get("success"):
        return balance

    if leave_type == "sick" and balance["sick_remaining"] <= 0:
        return {"success": False, "message": "No sick leave balance available."}

    if leave_type == "casual" and balance["casual_remaining"] <= 0:
        return {"success": False, "message": "No casual leave balance available."}

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO leave_requests
    (emp_id, employee_name, leave_type, date, reason, status)
    VALUES (?, ?, ?, ?, ?, 'pending')
    """, (emp_id, employee["name"], leave_type, date, reason))

    conn.commit()

    request_id = cursor.lastrowid   

    

    send_leave_email({
        "event_type": "new_leave_request",
        "manager_email": MANAGER_EMAIL,
        "employee_name": employee["name"],
        "emp_id": emp_id,
        "leave_type": leave_type,
        "date": date,
        "reason": reason,
        "status": "pending",
        "request_id": request_id
    })

    conn.close()

    return {
        "success": True,
        "request_id": request_id,
        "message": (
            f"{leave_type.capitalize()} leave request submitted for "
            f"{employee['name']} ({emp_id}). Request ID: {request_id}. "
            f"Waiting for manager approval."
        )
    }


def check_leave_balance(emp_id: str) -> Dict:
    emp_id = emp_id.upper().strip()

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
    SELECT e.emp_id, e.name, b.casual_total, b.casual_used, b.sick_total, b.sick_used
    FROM employees e
    JOIN leave_balances b ON e.emp_id = b.emp_id
    WHERE e.emp_id = ?
    """, (emp_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return {"success": False, "message": f"No employee found with ID {emp_id}."}

    return {
        "success": True,
        "emp_id": row["emp_id"],
        "employee_name": row["name"],
        "casual_total": row["casual_total"],
        "casual_used": row["casual_used"],
        "casual_remaining": row["casual_total"] - row["casual_used"],
        "sick_total": row["sick_total"],
        "sick_used": row["sick_used"],
        "sick_remaining": row["sick_total"] - row["sick_used"],
    }


def get_leave_history(emp_id: str) -> List[Dict]:
    emp_id = emp_id.upper().strip()

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM leave_requests
    WHERE emp_id = ?
    ORDER BY id DESC
    """, (emp_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]

def get_all_employees() -> List[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM employees
    ORDER BY emp_id
    """)

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]

def get_all_leave_requests() -> List[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM leave_requests ORDER BY id DESC")

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_pending_leave_requests() -> List[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM leave_requests
    WHERE status = 'pending'
    ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def approve_leave_request(request_id: int, role: str) -> Dict:
    if role not in ["manager", "admin"]:
        return {
            "success": False,
            "message": "Only Manager or Admin can approve leave requests."
        }

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM leave_requests WHERE id = ?", (request_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return {"success": False, "message": "Leave request not found."}

    if row["status"] != "pending":
        conn.close()
        return {"success": False, "message": "Only pending requests can be approved."}

    emp_id = row["emp_id"]
    leave_type = row["leave_type"]

    cursor.execute("""
    UPDATE leave_requests
    SET status = 'approved'
    WHERE id = ?
    """, (request_id,))

    if leave_type == "sick":
        cursor.execute("""
        UPDATE leave_balances
        SET sick_used = sick_used + 1
        WHERE emp_id = ?
        """, (emp_id,))
    else:
        cursor.execute("""
        UPDATE leave_balances
        SET casual_used = casual_used + 1
        WHERE emp_id = ?
        """, (emp_id,))

    # Get employee email details before closing DB
    cursor.execute("""
    SELECT e.email, e.name, l.emp_id, l.leave_type, l.date, l.reason
    FROM leave_requests l
    JOIN employees e ON l.emp_id = e.emp_id
    WHERE l.id = ?
    """, (request_id,))

    email_row = cursor.fetchone()

    conn.commit()
    conn.close()

    # Send approval email
    if email_row:
        try:
            from power_automate import send_leave_email

            send_leave_email({
                "event_type": "leave_approved",
                "employee_email": email_row["email"],
                "employee_name": email_row["name"],
                "emp_id": email_row["emp_id"],
                "leave_type": email_row["leave_type"],
                "date": email_row["date"],
                "reason": email_row["reason"],
                "status": "approved",
                "request_id": request_id
            })
        except Exception as e:
            print("Approval email error:", e)

    return {
        "success": True,
        "message": f"Leave request {request_id} approved successfully."
    }

def reject_leave_request(request_id: int, role: str) -> Dict:
    if role not in ["manager", "admin"]:
        return {
            "success": False,
            "message": "Only Manager, HR, or Admin can reject leave requests."
        }

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE leave_requests
    SET status = 'rejected'
    WHERE id = ? AND status = 'pending'
    """, (request_id,))

    updated = cursor.rowcount > 0

    conn.commit()
    cursor.execute("""
    SELECT e.email, e.name, l.emp_id, l.leave_type, l.date, l.reason
    FROM leave_requests l
    JOIN employees e ON l.emp_id = e.emp_id
    WHERE l.id = ?
    """, (request_id,))

    email_row = cursor.fetchone()

    if email_row:
        send_leave_email({
            "event_type": "leave_rejected",
            "employee_email": email_row["email"],
            "employee_name": email_row["name"],
            "emp_id": email_row["emp_id"],
            "leave_type": email_row["leave_type"],
            "date": email_row["date"],
            "reason": email_row["reason"],
            "status": "rejected",
            "request_id": request_id
        })
    conn.close()

    if updated:
        return {"success": True, "message": f"Leave request {request_id} rejected."}

    return {"success": False, "message": "Leave request not found or already processed."}


def cancel_leave_request(request_id: int, emp_id: str, role: str) -> Dict:
    conn = get_connection()
    cursor = conn.cursor()

    if role in ["manager", "hr", "admin"]:
        cursor.execute("""
        UPDATE leave_requests
        SET status = 'cancelled'
        WHERE id = ? AND status = 'pending'
        """, (request_id,))
    else:
        cursor.execute("""
        UPDATE leave_requests
        SET status = 'cancelled'
        WHERE id = ? AND emp_id = ? AND status = 'pending'
        """, (request_id, emp_id.upper().strip()))

    updated = cursor.rowcount > 0

    conn.commit()
    conn.close()

    if updated:
        return {"success": True, "message": f"Leave request {request_id} cancelled successfully."}

    return {
        "success": False,
        "message": "Leave request not found, already processed, or you do not have access."
    }


def check_leave_status(request_id: int, emp_id: str, role: str) -> Optional[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if role in ["manager", "hr", "admin"]:
        cursor.execute("SELECT * FROM leave_requests WHERE id = ?", (request_id,))
    else:
        cursor.execute("""
        SELECT * FROM leave_requests
        WHERE id = ? AND emp_id = ?
        """, (request_id, emp_id.upper().strip()))

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


# ---------------- IT ----------------

def check_known_outage(issue_type: str) -> Optional[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM known_outages
    WHERE lower(issue_type) LIKE ? AND status = 'active'
    """, (f"%{issue_type.lower()}%",))

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def check_maintenance(issue_type: str) -> Optional[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM maintenance_schedule
    WHERE lower(system_name) LIKE ? AND status = 'planned'
    """, (f"%{issue_type.lower()}%",))

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def check_duplicate_ticket(emp_id: str, issue_type: str = "") -> Optional[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM it_tickets
    WHERE emp_id = ?
    AND status IN ('open', 'in_progress')
    ORDER BY id DESC
    LIMIT 1
    """, (emp_id.upper().strip(),))

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def insert_it_ticket(emp_id: str, issue_type: str, priority: str, reason: str):
    conn = get_connection()
    cursor = conn.cursor()

    employee = get_employee(emp_id)

    cursor.execute("""
    INSERT INTO it_tickets
    (emp_id, employee_name, issue_type, priority, reason, status, assigned_engineer)
    VALUES (?, ?, ?, ?, ?, 'open', 'Unassigned')
    """, (emp_id, employee["name"], issue_type, priority, reason))

    conn.commit()

    ticket_id = cursor.lastrowid  # 🔥 AFTER COMMIT

    conn.close()

    return ticket_id



def get_it_tickets(emp_id: str, role: str) -> List[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if role in ["it", "admin"]:
        cursor.execute("""
        SELECT * FROM it_tickets
        ORDER BY id DESC
        """)
    else:
        cursor.execute("""
        SELECT * FROM it_tickets
        WHERE emp_id = ?
        ORDER BY id DESC
        """, (emp_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]

def get_it_ticket_status(ticket_id: int, emp_id: str, role: str) -> Optional[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if role in ["it", "admin"]:
        cursor.execute("""
        SELECT * FROM it_tickets
        WHERE id = ?
        """, (ticket_id,))
    else:
        cursor.execute("""
        SELECT * FROM it_tickets
        WHERE id = ? AND emp_id = ?
        """, (ticket_id, emp_id))

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def assign_it_ticket(ticket_id: int, engineer_name: str, role: str) -> Dict:
    if role not in ["it", "admin"]:
        return {
            "success": False,
            "message": "Access denied. Only IT team or Admin can assign tickets."
        }

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE it_tickets
    SET assigned_engineer = ?, status = 'in_progress'
    WHERE id = ? AND status IN ('open', 'in_progress')
    """, (engineer_name, ticket_id))

    updated = cursor.rowcount > 0

    conn.commit()
    conn.close()

    if updated:
        return {
            "success": True,
            "message": f"Ticket {ticket_id} assigned to {engineer_name}."
        }

    return {
        "success": False,
        "message": "Ticket not found or already resolved."
    }


def resolve_it_ticket(ticket_id: int, role: str) -> Dict:
    if role not in ["it", "admin"]:
        return {
            "success": False,
            "message": "Access denied. Only IT team or Admin can resolve tickets."
        }

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE it_tickets
    SET status = 'resolved'
    WHERE id = ? AND status != 'resolved'
    """, (ticket_id,))

    updated = cursor.rowcount > 0

    conn.commit()
    conn.close()

    if updated:
        return {
            "success": True,
            "message": f"Ticket {ticket_id} resolved successfully."
        }

    return {
        "success": False,
        "message": "Ticket not found or already resolved."
    }


# ---------------- ASSET ----------------

def check_inventory(asset_type: str) -> Optional[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM inventory
    WHERE lower(asset_type) = ?
    """, (asset_type.lower(),))

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def insert_asset_request(emp_id: str, asset_type: str, reason: str) -> int:
    emp_id = emp_id.upper().strip()

    employee = get_employee(emp_id)

    if not employee:
        return None

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO asset_requests
    (emp_id, employee_name, asset_type, reason, status, manager_approval, it_approval, inventory_status)
    VALUES (?, ?, ?, ?, 'manager_approval_pending', 'pending', 'pending', 'pending')
    """, (emp_id, employee["name"], asset_type, reason))

    request_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return request_id

def get_asset_request_status(request_id: int, employee_name: str, role: str) -> Optional[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if role in ["it", "admin", "hr"]:
        cursor.execute("SELECT * FROM asset_requests WHERE id = ?", (request_id,))
    else:
        cursor.execute("""
        SELECT * FROM asset_requests
        WHERE id = ? AND employee_name = ?
        """, (request_id, employee_name))

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None

def check_active_asset_request(emp_id: str) -> Optional[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM asset_requests
    WHERE emp_id = ?
    AND status IN (
        'manager_approval_pending',
        'it_approval_pending',
        'inventory_pending',
        'pending'
    )
    ORDER BY id DESC
    LIMIT 1
    """, (emp_id.upper().strip(),))

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None

def approve_asset_request(request_id: int, role: str) -> Dict:
    if role not in ["manager","it", "admin"]:
        return {"success": False, "message": "Only IT team or Admin can approve asset requests."}

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM asset_requests WHERE id = ?", (request_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return {"success": False, "message": "Asset request not found."}

    asset_type = row["asset_type"]

    cursor.execute("""
    SELECT available_quantity FROM inventory
    WHERE lower(asset_type) = ?
    """, (asset_type.lower(),))

    inventory = cursor.fetchone()

    if not inventory or inventory["available_quantity"] <= 0:
        conn.close()
        return {"success": False, "message": f"{asset_type} is out of stock."}

    cursor.execute("""
    UPDATE asset_requests
    SET status = 'fulfilled',
        manager_approval = 'approved',
        it_approval = 'approved',
        inventory_status = 'available'
    WHERE id = ?
    """, (request_id,))

    cursor.execute("""
    UPDATE inventory
    SET available_quantity = available_quantity - 1
    WHERE lower(asset_type) = ?
    """, (asset_type.lower(),))

    conn.commit()
    conn.close()

    return {"success": True, "message": f"Asset request {request_id} approved and fulfilled."}


def reject_asset_request(request_id: int, role: str) -> Dict:
    if role not in ["manager","it", "admin"]:
        return {"success": False, "message": "Only IT team or Admin can reject asset requests."}

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE asset_requests
    SET status = 'rejected',
        manager_approval = 'rejected',
        it_approval = 'rejected'
    WHERE id = ?
    """, (request_id,))

    updated = cursor.rowcount > 0

    conn.commit()
    conn.close()

    if updated:
        return {"success": True, "message": f"Asset request {request_id} rejected."}

    return {"success": False, "message": "Asset request not found."}

def get_asset_requests_by_emp_id(emp_id: str, role: str) -> List[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if role in ["manager", "it", "hr", "admin"]:
        cursor.execute("""
        SELECT * FROM asset_requests
        ORDER BY id DESC
        """)
    else:
        cursor.execute("""
        SELECT * FROM asset_requests
        WHERE emp_id = ?
        ORDER BY id DESC
        """, (emp_id.upper().strip(),))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]

# ---------------- CHAT MEMORY ----------------

def save_message(session_id: str, role: str, content: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO chat_memory (session_id, role, content)
    VALUES (?, ?, ?)
    """, (session_id, role, content))

    conn.commit()
    conn.close()


def load_memory(session_id: str, limit: int = 10) -> List[Dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
    SELECT role, content
    FROM chat_memory
    WHERE session_id = ?
    ORDER BY id DESC
    LIMIT ?
    """, (session_id, limit))

    rows = cursor.fetchall()
    conn.close()

    messages = [dict(row) for row in rows]
    messages.reverse()

    return messages

def add_employee(emp_id: str, name: str, email: str, role: str = "employee", manager_name: str = "Manager") -> Dict:
    emp_id = emp_id.upper().strip()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT emp_id FROM employees WHERE emp_id = ?", (emp_id,))
    if cursor.fetchone():
        conn.close()
        return {"success": False, "message": f"Employee {emp_id} already exists."}

    cursor.execute("""
    INSERT INTO employees (emp_id, name, email, role, manager_name)
    VALUES (?, ?, ?, ?, ?)
    """, (emp_id, name, email, role, manager_name))

    cursor.execute("""
    INSERT OR IGNORE INTO leave_balances
    (emp_id, casual_total, casual_used, sick_total, sick_used)
    VALUES (?, 12, 0, 6, 0)
    """, (emp_id,))

    conn.commit()
    conn.close()

    return {"success": True, "message": f"Employee {name} ({emp_id}) added successfully."}


def delete_employee(emp_id: str) -> Dict:
    emp_id = emp_id.upper().strip()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM employees WHERE emp_id = ?", (emp_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return {"success": False, "message": f"No employee found with ID {emp_id}."}

    cursor.execute("DELETE FROM leave_balances WHERE emp_id = ?", (emp_id,))
    cursor.execute("DELETE FROM leave_requests WHERE emp_id = ?", (emp_id,))
    cursor.execute("DELETE FROM it_tickets WHERE emp_id = ?", (emp_id,))
    cursor.execute("DELETE FROM employees WHERE emp_id = ?", (emp_id,))

    conn.commit()
    conn.close()

    return {"success": True, "message": f"Employee {emp_id} deleted successfully."}

def save_activity_log(
    role: str,
    intent: str,
    user_message: str,
    assistant_response: str
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO activity_logs
    (role, intent, user_message, assistant_response)
    VALUES (?, ?, ?, ?)
    """, (
        role,
        intent,
        user_message,
        assistant_response
    ))

    conn.commit()
    conn.close()

def get_activity_logs(limit: int = 50):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM activity_logs
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]