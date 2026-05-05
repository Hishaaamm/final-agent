from langchain_core.tools import tool

from models import (
    ApplyLeaveInput,
    EmployeeIdInput,
    RaiseITTicketInput,
    AssetRequestInput
)

from database import (
    insert_leave_request,
    get_leave_history,
    get_pending_leave_requests,
    get_all_leave_requests,
    cancel_leave_request,
    check_leave_status,
    check_leave_balance,

    check_known_outage,
    check_maintenance,
    check_duplicate_ticket,
    insert_it_ticket,
    get_it_tickets,
    get_it_ticket_status,

    check_inventory,
    insert_asset_request,
    get_asset_request_status,

    approve_leave_request,
    reject_leave_request,
    assign_it_ticket,
    resolve_it_ticket,
    approve_asset_request,
    reject_asset_request
)


# ---------------- LEAVE ----------------

@tool(args_schema=ApplyLeaveInput)
def apply_leave_tool(emp_id: str, leave_type: str, date: str, reason: str) -> dict:
    """Apply sick or casual leave using employee ID."""
    return insert_leave_request(emp_id, leave_type, date, reason)


def view_leave_history_with_role(emp_id: str, role: str) -> dict:
    if role in ["manager", "hr", "admin"]:
        return {"success": True, "history": get_all_leave_requests()}

    return {"success": True, "history": get_leave_history(emp_id)}


def get_pending_leaves_tool() -> dict:
    return {"success": True, "requests": get_pending_leave_requests()}


def cancel_leave_with_role(request_id: int, emp_id: str, role: str) -> dict:
    return cancel_leave_request(request_id, emp_id, role)


def check_leave_status_with_role(request_id: int, emp_id: str, role: str) -> dict:
    result = check_leave_status(request_id, emp_id, role)

    if not result:
        return {
            "success": False,
            "message": "No leave request found or you do not have access."
        }

    return {"success": True, "request": result}


@tool(args_schema=EmployeeIdInput)
def check_balance_tool(emp_id: str) -> dict:
    """Check leave balance using employee ID."""
    return {
        "success": True,
        "balance": check_leave_balance(emp_id)
    }



def approve_leave_with_role(request_id: int, role: str) -> dict:
    return approve_leave_request(request_id, role)


def reject_leave_with_role(request_id: int, role: str) -> dict:
    return reject_leave_request(request_id, role)


# ---------------- IT ----------------

@tool(args_schema=RaiseITTicketInput)
def raise_it_ticket_tool(emp_id: str, issue_type: str, priority: str, reason: str) -> dict:
    """Raise an IT support ticket after checking outages, maintenance, and active tickets."""

    outage = check_known_outage(issue_type)
    if outage:
        return {
            "success": False,
            "message": f"Known outage: {outage['description']}"
        }

    maintenance = check_maintenance(issue_type)
    if maintenance:
        return {
            "success": False,
            "message": f"Planned maintenance: {maintenance['description']}"
        }

    duplicate = check_duplicate_ticket(emp_id, issue_type)
    if duplicate:
        return {
            "success": False,
            "message": (
                f"You already have an active IT ticket.\n"
                f"Ticket ID: {duplicate['id']}\n"
                f"Issue: {duplicate['issue_type']}\n"
                f"Status: {duplicate['status']}\n\n"
                f"Please wait until this ticket is resolved before raising a new one."
            )
        }

    ticket_id = insert_it_ticket(emp_id, issue_type, priority, reason)

    return {
        "success": True,
        "message": f"IT ticket created. ID: {ticket_id}"
    }

def view_it_tickets_with_role(emp_id: str, role: str) -> dict:
    return {
        "success": True,
        "tickets": get_it_tickets(emp_id, role)
    }


def check_it_ticket_status_with_role(ticket_id: int, emp_id: str, role: str) -> dict:
    ticket = get_it_ticket_status(ticket_id, emp_id, role)

    if not ticket:
        return {
            "success": False,
            "message": "Ticket not found or access denied."
        }

    return {
        "success": True,
        "ticket": ticket
    }


def assign_it_ticket_with_role(ticket_id: int, engineer_name: str, role: str) -> dict:
    return assign_it_ticket(ticket_id, engineer_name, role)


def resolve_it_ticket_with_role(ticket_id: int, role: str) -> dict:
    return resolve_it_ticket(ticket_id, role)

# ---------------- ASSET ----------------

@tool(args_schema=AssetRequestInput)
def request_asset_tool(employee_name: str, asset_type: str, reason: str) -> dict:
    """Request an IT asset after checking inventory."""
    inventory = check_inventory(asset_type)

    if not inventory:
        return {"success": False, "message": "Asset not found"}

    if inventory["available_quantity"] <= 0:
        return {"success": False, "message": "Out of stock"}

    request_id = insert_asset_request(employee_name, asset_type, reason)

    return {
        "success": True,
        "message": f"Asset request submitted. ID: {request_id}"
    }


def check_asset_status_with_role(request_id: int, employee_name: str, role: str) -> dict:
    asset = get_asset_request_status(request_id, employee_name, role)

    if not asset:
        return {
            "success": False,
            "message": "Asset request not found"
        }

    return {"success": True, "asset": asset}


def approve_asset_with_role(request_id: int, role: str) -> dict:
    return approve_asset_request(request_id, role)


def reject_asset_with_role(request_id: int, role: str) -> dict:
    return reject_asset_request(request_id, role)