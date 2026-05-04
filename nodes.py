from unittest import result

from dotenv import load_dotenv
import re

load_dotenv()

from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from models import AgentState, RouterOutput, DetailExtraction,LeaveTypeValidation
from rag import answer_policy_question

from tools import (
    apply_leave_tool,
    view_leave_history_with_role,
    cancel_leave_with_role,
    check_leave_status_with_role,
    check_balance_tool,
    get_pending_leaves_tool,

    raise_it_ticket_tool,
    view_it_tickets_with_role,
    check_it_ticket_status_with_role,

    request_asset_tool,
    check_asset_status_with_role,

    approve_leave_with_role,
    reject_leave_with_role,
    assign_it_ticket_with_role,
    resolve_it_ticket_with_role,
    approve_asset_with_role,
    reject_asset_with_role,
)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def active_leave_history_to_text(chat_history):
    """
    Only keep messages from the latest unfinished leave flow.
    This prevents old leave type/date/reason from being reused.
    """
    if not chat_history:
        return ""

    start_index = None

    for i in range(len(chat_history) - 1, -1, -1):
        msg = chat_history[i]
        content = msg.get("content", "").lower()
        role = msg.get("role", "")

        if "leave_flow_completed" in content:
            break

        if role == "user" and (
            "apply leave" in content
            or "apply for leave" in content
            or "take leave" in content
            or "want leave" in content
            or "leave request" in content
        ):
            start_index = i
            break

    if start_index is None:
        return ""

    return "\n".join(
        f"{m['role']}: {m['content']}"
        for m in chat_history[start_index:]
    )


# ---------------- ROUTER ----------------

def router_node(state: AgentState) -> dict:
    import re

    current_text = state.user_input.strip().lower()

    full_history = "\n".join(
        [m.get("content", "").lower() for m in state.chat_history[-8:]]
    )

    recent = full_history[-300:]  # shorter = better

    # ---------------- SMALL TALK ----------------
    if current_text in ["hi", "hello", "hey", "hai"]:
        return {"intent": "small_talk"}

    if current_text in ["thanks", "thank you", "ok", "okay", "bye"]:
        return {"intent": "small_talk"}

    # ---------------- FAST INTENTS (CURRENT MESSAGE FIRST) ----------------
    # ORDER MATTERS
    if "pending leave" in current_text or "pending requests" in current_text or "show all leave" in current_text or "all leave request" in current_text:
        return {"intent": "pending_leaves"}

    if "approve" in current_text and "leave" in current_text:
        return {"intent": "approve_leave"}

    if "reject" in current_text and "leave" in current_text:
        return {"intent": "reject_leave"}
    
    if "leave summary" in current_text or "team leave summary" in current_text or "leave report" in current_text:
        return {"intent": "leave_summary"}

    # APPLY LEAVE MUST COME FIRST
    if "apply leave" in current_text or "apply for leave" in current_text or "take leave" in current_text or ("apply" in current_text and "leave" in current_text):
        return {"intent": "apply_leave"}

    if "leave balance" in current_text or "balance" in current_text:
        return {"intent": "leave_balance"}

    if "leave request status" in current_text or "leave status" in current_text or "request status" in current_text or "check status" in current_text:
        return {"intent": "leave_status"}

    if "leave history" in current_text or "applied leaves" in current_text or "view leave" in current_text:
        return {"intent": "view_leave"}

    if "pending leave" in current_text or "pending approvals" in current_text:
        return {"intent": "pending_leaves"}

    # ---------------- EMP ID FOLLOW-UP ----------------
    if re.search(r"\bEMP\d{3}\b", current_text, re.IGNORECASE):
        last_message = state.chat_history[-1]["content"].lower() if state.chat_history else ""

    if "balance" in last_message:
        return {"intent": "leave_balance"}

    if "status" in last_message:
        return {"intent": "leave_status"}

    if "history" in last_message or "applied leaves" in last_message:
        return {"intent": "view_leave"}

    return {"intent": "apply_leave"}

    # ---------------- LLM FALLBACK ----------------
    prompt = ChatPromptTemplate.from_template("""
                                              
You are the router for an Enterprise HR + IT Assistant.

Classify the user's message into exactly ONE intent.

Available intents:
small_talk, rag, apply_leave, view_leave, cancel_leave, leave_status,
leave_balance, approve_leave, reject_leave, pending_leaves, employee_details,
raise_it_ticket, view_it_tickets, it_ticket_status, assign_it_ticket,
resolve_it_ticket, request_asset, asset_status, approve_asset, reject_asset, unknown.

Recent conversation:
{history}

Current message:
{user_input}
""")

    structured_llm = llm.with_structured_output(RouterOutput)
    chain = prompt | structured_llm

    result = chain.invoke({
        "history": full_history,
        "user_input": state.user_input
    })

    return {"intent": result.intent}

# ---------------- DETAIL EXTRACTION ----------------

def extract_details_node(state: AgentState) -> dict:
    prompt = ChatPromptTemplate.from_template("""
You extract structured details for an Enterprise HR + IT Assistant.

Use reasoning, not keyword matching.

GENERAL RULES:
- Current user message has highest priority.
- Use recent conversation only for unfinished follow-up tasks.
- If user corrects a previous value, use the correction.
- Do not reuse old completed request details.
- For leave actions, employee ID must come from the current message only.
- If current message does not contain employee ID like EMP001, emp_id must be null.

LEAVE RULES:
- Extract emp_id, leave_type, date, reason.
- leave_type must be either sick, casual, or null.
- If user explicitly corrects leave type, use the corrected leave type.
- If user asks for sick leave but the reason is clearly non-medical, set leave_type = null.
- If user asks for casual leave but the reason is clearly medical, set leave_type = null.
- If leave type is unclear or conflicting, set leave_type = null.
- Keep the reason as the user’s actual reason.

-If recent conversation contains LEAVE_FLOW_COMPLETED, do not reuse any old leave details from before that marker.
A new leave request must start fresh.
                                              
Examples:
User: "I want sick leave tomorrow because I have fever"
leave_type = sick
reason = "I have fever"

User: "I want sick leave tomorrow because I have school function"
leave_type = null
reason = "I have school function"

User: "casual leave"
leave_type = casual

User: "EMP003"
emp_id = EMP003
                                              
- If user previously gave a reason/date in the current unfinished leave flow, keep it unless user replaces it.


IT RULES:
- Extract issue_type, priority, reason.
- issue_type examples: laptop, vpn, outlook, email, printer, network, software installation.
- priority: low, medium, high, urgent.

ASSET RULES:
- Extract asset_type: laptop, monitor, keyboard, mouse, vpn token, software license.

ADMIN RULES:
- Extract request_id from messages like approve leave request 1, ticket 2, asset request 3.
- Extract engineer_name from messages like assign ticket 1 to Rahul.

Return:
emp_id, date, reason, leave_type, request_id, issue_type, priority, asset_type, engineer_name.

Recent conversation:
{history}

Current message:
{user_input}
""")

    chain = prompt | llm.with_structured_output(DetailExtraction)

    result = chain.invoke({
        "history": active_leave_history_to_text(state.chat_history),
        "user_input": state.user_input
    })

    # start with previous state (IMPORTANT)
    data = {
        "emp_id": state.emp_id,
        "date": state.date,
        "reason": state.reason,
        "leave_type": state.leave_type,
        "request_id": state.request_id,
        "issue_type": state.issue_type,
        "priority": state.priority,
        "asset_type": state.asset_type,
        "engineer_name": state.engineer_name,
    }

    # -------- UPDATE ONLY NEW VALUES --------
    if result.emp_id:
        data["emp_id"] = result.emp_id

    if result.date:
        data["date"] = result.date

    if result.reason:
        data["reason"] = result.reason

    if result.leave_type:
        data["leave_type"] = result.leave_type

    if result.request_id:
        data["request_id"] = result.request_id

    if result.issue_type:
        data["issue_type"] = result.issue_type

    if result.priority:
        data["priority"] = result.priority

    if result.asset_type:
        data["asset_type"] = result.asset_type

    if result.engineer_name:
        data["engineer_name"] = result.engineer_name

    # -------- FALLBACK EXTRACTION (CRITICAL) --------
    text = state.user_input.lower()

    # leave type
    if "casual" in text:
        data["leave_type"] = "casual"
    elif "sick" in text:
        data["leave_type"] = "sick"

    # date
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if date_match:
        data["date"] = date_match.group(0)

    # reason
    if "because" in text or "due to" in text:
        parts = re.split(r"because|due to", text)
        if len(parts) > 1:
            data["reason"] = parts[-1].strip()

    # EMP ID (ONLY FROM CURRENT MESSAGE — IMPORTANT)
    emp_match = re.search(r"\bEMP\d{3}\b", state.user_input, re.IGNORECASE)
    if emp_match:
        data["emp_id"] = emp_match.group(0).upper()

    # -------- NORMALIZE --------
    if data["emp_id"]:
        data["emp_id"] = data["emp_id"].upper().strip()

    if data["leave_type"]:
        data["leave_type"] = data["leave_type"].lower().strip()

    if data["priority"]:
        data["priority"] = data["priority"].lower().strip()

    if data["asset_type"]:
        data["asset_type"] = data["asset_type"].lower().strip()

    if data["issue_type"]:
        data["issue_type"] = data["issue_type"].lower().strip()

    return data

# ---------------- RAG ----------------
def leave_summary_node(state: AgentState) -> dict:
    from database import get_all_leave_requests

    if state.role not in ["manager", "hr", "admin"]:
        return {"response": "Only Manager, HR, or Admin can view leave summary."}

    requests = get_all_leave_requests()

    total = len(requests)
    pending = sum(1 for r in requests if r["status"] == "pending")
    approved = sum(1 for r in requests if r["status"] == "approved")
    rejected = sum(1 for r in requests if r["status"] == "rejected")

    return {
        "response": (
            f"Team Leave Summary:\n"
            f"Total Requests: {total}\n"
            f"Pending: {pending}\n"
            f"Approved: {approved}\n"
            f"Rejected: {rejected}\n\n"
            f"Tip: Ask 'show pending leave requests' to approve or reject."
        )
    }
def rag_node(state: AgentState) -> dict:
    answer = answer_policy_question(
        question=state.user_input,
        role=state.role,
        chat_history=state.chat_history
    )

    return {"response": answer}

def small_talk_node(state: AgentState) -> dict:
    prompt = ChatPromptTemplate.from_template("""
Reply naturally and briefly to this user message.

User message:
{message}
""")
    result = (prompt | llm).invoke({"message": state.user_input})
    return {"response": result.content}

# ---------------- LEAVE NODES ----------------
def make_missing_leave_response(missing: list) -> str:
    prompt = ChatPromptTemplate.from_template("""
You are a friendly HR assistant.

The user is applying for leave, but some required details are missing.

Missing fields:
{missing}

Ask naturally for ONLY the missing details.
Do not repeat "Sure I can help" every time.
Keep it short.
""")

    chain = prompt | llm

    result = chain.invoke({
        "missing": ", ".join(missing)
    })

    return result.content

def validate_leave_type_with_llm(leave_type: str, reason: str) -> LeaveTypeValidation:
    prompt = ChatPromptTemplate.from_template("""
You are an HR leave validation assistant.

Check whether the selected leave type matches the reason.

Selected leave type:
{leave_type}

Reason:
{reason}

Rules:
- Sick leave is for health-related reasons such as fever, illness, medical appointment, doctor visit, hospital, recovery, injury, etc.
- Casual leave is for non-medical reasons such as school function, family function, travel, personal work, event, ceremony, wedding, etc.
- If selected leave type does not match the reason, mark is_valid as false.
- Suggest the correct leave type.

Return structured output.
""")

    chain = prompt | llm.with_structured_output(LeaveTypeValidation)

    return chain.invoke({
        "leave_type": leave_type,
        "reason": reason
    })

def validate_leave_node(state: AgentState) -> dict:
    missing = []

    if not state.emp_id:
        missing.append("employee ID")

    if not state.leave_type:
        missing.append("leave type (sick or casual)")

    if not state.date:
        missing.append("date")

    if not state.reason:
        missing.append("reason")

    if missing:
        return {
            "missing_fields": missing,
            "response": f"Please provide: {', '.join(missing)}."
        }

    # LLM validation must be created BEFORE using validation
    validation = validate_leave_type_with_llm(
        leave_type=state.leave_type,
        reason=state.reason
    )

    if not validation.is_valid:
        return {
            "missing_fields": ["leave type confirmation"],
            "response": (
                f"{validation.explanation}\n\n"
                f"Suggested leave type: {validation.suggested_leave_type}\n"
                f"Please confirm the correct leave type and reason again."
            )
        }

    return {"missing_fields": []}
    
def employee_details_node(state: AgentState) -> dict:
    from database import get_employee, get_all_employees

    if state.role not in ["hr", "admin"]:
        return {
            "response": "Only HR or Admin can view employee details."
        }

    # If HR asks all employees
    if "all employee" in state.user_input.lower() or "list employee" in state.user_input.lower():
        employees = get_all_employees()

        if not employees:
            return {"response": "No employee records found."}

        lines = ["Employee records:"]

        for emp in employees:
            lines.append(
                f"ID: {emp['emp_id']} | Name: {emp['name']} | Role: {emp['role']} | Manager: {emp['manager_name']}"
            )

        return {"response": "\n".join(lines)}

    if not state.emp_id:
        return {
            "response": "Please provide the employee ID. Example: EMP003"
        }

    emp = get_employee(state.emp_id)

    if not emp:
        return {
            "response": f"No employee found with ID {state.emp_id}."
        }

    prompt = ChatPromptTemplate.from_template("""
You are a professional HR assistant.

Rewrite the employee details clearly and naturally.
Do not invent extra data.

Employee ID: {emp_id}
Name: {name}
Role: {role}
Manager: {manager}

Give a concise HR-style response.
""")

    chain = prompt | llm

    result = chain.invoke({
        "emp_id": emp["emp_id"],
        "name": emp["name"],
        "role": emp["role"],
        "manager": emp["manager_name"]
    })

    return {"response": result.content}

def apply_leave_node(state: AgentState) -> dict:
    result = apply_leave_tool.invoke({
        "emp_id": state.emp_id,
        "leave_type": state.leave_type,
        "date": state.date,
        "reason": state.reason,
    })

    return {"response": result["message"]}


def view_leave_node(state: AgentState) -> dict:
    if not state.emp_id:
        return {"response": "Please provide your employee ID."}

    result = view_leave_history_with_role(state.emp_id, state.role)
    history = result["history"]

    if not history:
        return {"response": "No leave history found."}

    lines = ["Your Leave History:"]

    for item in history:
        lines.append(
            f"ID {item['id']} | {item['leave_type']} | {item['date']} | {item['status']}"
        )

    return {"response": "\n".join(lines)}


def pending_leave_node(state: AgentState) -> dict:
    if state.role not in ["manager", "admin", "hr"]:
        return {"response": "Only Manager, HR, or Admin can view pending leave requests."}

    result = get_pending_leaves_tool()
    requests = result["requests"]

    if not requests:
        return {"response": "No pending leave requests."}

    lines = ["Pending Leave Requests:"]

    for item in requests:
        lines.append(
            f"Request ID: {item['id']}\n"
            f"Employee: {item['employee_name']} ({item['emp_id']})\n"
            f"Type: {item['leave_type']}\n"
            f"Date: {item['date']}\n"
            f"Reason: {item['reason']}\n"
            f"Status: {item['status']}\n"
            f"Action: approve leave {item['id']} / reject leave {item['id']}\n"
        )

    return {"response": "\n".join(lines)}

def leave_balance_node(state: AgentState) -> dict:
    import re

    # extract EMP ID from current message
    match = re.search(r"\bEMP\d{3}\b", state.user_input, re.IGNORECASE)

    emp_id = match.group(0).upper() if match else state.emp_id

    if not emp_id:
        return {"response": "Please provide your employee ID. Example: EMP002"}

    result = check_balance_tool.invoke({
        "emp_id": emp_id
    })

    balance = result["balance"]

    if not balance.get("success"):
        return {"response": balance["message"]}

    return {
        "response": (
            f"Leave Balance for {balance['employee_name']} ({balance['emp_id']}):\n\n"
            f"Casual: {balance['casual_remaining']} remaining\n"
            f"Sick: {balance['sick_remaining']} remaining"
        )
    }

def leave_status_node(state: AgentState) -> dict:
    import re
    from database import get_leave_history

    emp_id = state.emp_id
    request_id = state.request_id

    # Extract EMP ID directly from current user message
    emp_match = re.search(r"\bEMP\d{3}\b", state.user_input, re.IGNORECASE)
    if emp_match:
        emp_id = emp_match.group(0).upper()

    # Extract request ID if user says request 3 / ID 3
    id_match = re.search(r"\b(?:request|id)\s*(\d+)\b", state.user_input, re.IGNORECASE)
    if id_match:
        request_id = int(id_match.group(1))

    # If specific request ID is given
    if request_id:
        result = check_leave_status_with_role(
            request_id=request_id,
            emp_id=emp_id or "",
            role=state.role
        )

        if not result["success"]:
            return {"response": result["message"]}

        request = result["request"]

        return {
            "response": (
                f"Leave Request Status:\n"
                f"Request ID: {request['id']}\n"
                f"Employee: {request['employee_name']} ({request['emp_id']})\n"
                f"Leave Type: {request['leave_type']}\n"
                f"Date: {request['date']}\n"
                f"Reason: {request['reason']}\n"
                f"Status: {request['status']}"
            )
        }

    # If EMP ID is given, show latest leave request
    if emp_id:
        history = get_leave_history(emp_id)

        if not history:
            return {"response": f"No leave requests found for {emp_id}."}

        latest = history[0]

        return {
            "response": (
                f"Latest Leave Request Status:\n"
                f"Request ID: {latest['id']}\n"
                f"Employee: {latest['employee_name']} ({latest['emp_id']})\n"
                f"Leave Type: {latest['leave_type']}\n"
                f"Date: {latest['date']}\n"
                f"Reason: {latest['reason']}\n"
                f"Status: {latest['status']}"
            )
        }

    return {
        "response": "Please provide your employee ID. Example: EMP001"
    }

def cancel_leave_node(state: AgentState) -> dict:
    if not state.request_id:
        return {"response": "Please provide the leave request ID to cancel."}

    result = cancel_leave_with_role(
        request_id=state.request_id,
        emp_id=state.emp_id or "",
        role=state.role
    )

    return {"response": result["message"]}


def approve_leave_node(state: AgentState) -> dict:
    import re

    if state.role not in ["manager", "admin"]:
        return {"response": "Only Manager or Admin can approve leave requests."}

    request_id = state.request_id

    match = re.search(r"\d+", state.user_input)
    if match:
        request_id = int(match.group(0))

    if not request_id:
        return {"response": "Please provide the leave request ID. Example: approve leave 2"}

    result = approve_leave_with_role(request_id, state.role)
    return {"response": result["message"]}


def reject_leave_node(state: AgentState) -> dict:
    import re

    if state.role not in ["manager", "admin"]:
        return {"response": "Only Manager or Admin can reject leave requests."}

    request_id = state.request_id

    match = re.search(r"\d+", state.user_input)
    if match:
        request_id = int(match.group(0))

    if not request_id:
        return {"response": "Please provide the leave request ID. Example: reject leave 2"}

    result = reject_leave_with_role(request_id, state.role)
    return {"response": result["message"]}




# ---------------- IT NODES ----------------

def validate_it_ticket_node(state: AgentState) -> dict:
    missing = []

    if not state.issue_type:
        missing.append("issue type")

    if not state.priority:
        missing.append("priority")

    if not state.reason:
        missing.append("issue description")

    if missing:
        return {
            "missing_fields": missing,
            "response": f"Please provide the missing IT ticket details: {', '.join(missing)}."
        }

    return {"missing_fields": []}


def raise_it_ticket_node(state: AgentState) -> dict:
    result = raise_it_ticket_tool.invoke({
        "employee_name": state.emp_id or state.name,
        "issue_type": state.issue_type,
        "priority": state.priority,
        "reason": state.reason,
    })

    return {"response": result["message"]}


def view_it_tickets_node(state: AgentState) -> dict:
    result = view_it_tickets_with_role(state.emp_id or state.name, state.role)
    tickets = result["tickets"]

    if not tickets:
        return {"response": "No IT tickets found."}

    lines = ["IT Tickets:"]

    for item in tickets:
        lines.append(
            f"ID: {item['id']} | Employee: {item['employee_name']} | "
            f"Issue: {item['issue_type']} | Priority: {item['priority']} | "
            f"Status: {item['status']} | Engineer: {item['assigned_engineer']}"
        )

    return {"response": "\n".join(lines)}


def it_ticket_status_node(state: AgentState) -> dict:
    if not state.request_id:
        return {"response": "Please provide the IT ticket ID."}

    result = check_it_ticket_status_with_role(
        ticket_id=state.request_id,
        employee_name=state.emp_id or state.name,
        role=state.role
    )

    if not result["success"]:
        return {"response": result["message"]}

    ticket = result["ticket"]

    return {
        "response": (
            f"IT Ticket Status:\n"
            f"Ticket ID: {ticket['id']}\n"
            f"Employee: {ticket['employee_name']}\n"
            f"Issue: {ticket['issue_type']}\n"
            f"Priority: {ticket['priority']}\n"
            f"Status: {ticket['status']}\n"
            f"Assigned Engineer: {ticket['assigned_engineer']}"
        )
    }


def assign_it_ticket_node(state: AgentState) -> dict:
    if not state.request_id:
        return {"response": "Please provide the ticket ID."}

    if not state.engineer_name:
        return {"response": "Please provide the engineer name."}

    result = assign_it_ticket_with_role(
        ticket_id=state.request_id,
        engineer_name=state.engineer_name,
        role=state.role
    )

    return {"response": result["message"]}


def resolve_it_ticket_node(state: AgentState) -> dict:
    if not state.request_id:
        return {"response": "Please provide the ticket ID to resolve."}

    result = resolve_it_ticket_with_role(
        ticket_id=state.request_id,
        role=state.role
    )

    return {"response": result["message"]}


# ---------------- ASSET NODES ----------------

def validate_asset_node(state: AgentState) -> dict:
    missing = []

    if not state.asset_type:
        missing.append("asset type")

    if not state.reason:
        missing.append("reason")

    if missing:
        return {
            "missing_fields": missing,
            "response": f"Please provide the missing asset request details: {', '.join(missing)}."
        }

    return {"missing_fields": []}


def request_asset_node(state: AgentState) -> dict:
    result = request_asset_tool.invoke({
        "employee_name": state.emp_id or state.name,
        "asset_type": state.asset_type,
        "reason": state.reason,
    })

    return {"response": result["message"]}


def asset_status_node(state: AgentState) -> dict:
    if not state.request_id:
        return {"response": "Please provide the asset request ID."}

    result = check_asset_status_with_role(
        request_id=state.request_id,
        employee_name=state.emp_id or state.name,
        role=state.role
    )

    if not result["success"]:
        return {"response": result["message"]}

    asset = result["asset"]

    return {
        "response": (
            f"Asset Request Status:\n"
            f"Request ID: {asset['id']}\n"
            f"Employee: {asset['employee_name']}\n"
            f"Asset: {asset['asset_type']}\n"
            f"Reason: {asset['reason']}\n"
            f"Status: {asset['status']}\n"
            f"Manager Approval: {asset['manager_approval']}\n"
            f"IT Approval: {asset['it_approval']}\n"
            f"Inventory Status: {asset['inventory_status']}"
        )
    }


def approve_asset_node(state: AgentState) -> dict:
    if not state.request_id:
        return {"response": "Please provide the asset request ID to approve."}

    result = approve_asset_with_role(state.request_id, state.role)
    return {"response": result["message"]}


def reject_asset_node(state: AgentState) -> dict:
    if not state.request_id:
        return {"response": "Please provide the asset request ID to reject."}

    result = reject_asset_with_role(state.request_id, state.role)
    return {"response": result["message"]}


# ---------------- UNKNOWN ----------------

def unknown_node(state: AgentState) -> dict:
    return {
        "response": "I can help with HR policies, leave management, IT tickets, and asset requests."
    }