from unittest import result

from dotenv import load_dotenv
import re

load_dotenv()

from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from database import check_duplicate_ticket
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
            ("apply" in content and "leave" in content)
            or ("take" in content and "leave" in content)
            or ("want" in content and "leave" in content)
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
    current_text = state.user_input.strip().lower()
    if state.missing_fields:
        return {"intent": state.intent}
    full_history = "\n".join(
        [f"{m.get('role', '')}: {m.get('content', '')}" for m in state.chat_history[-8:]]
    )
    
    if re.search(r"\bEMP\d{3}\b", state.user_input, re.IGNORECASE):
        last_assistant_msg = ""

        for msg in reversed(state.chat_history):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg.get("content", "").lower()
                break

        if "leave balance" in last_assistant_msg:
            return {"intent": "leave_balance"}

        if "leave request status" in last_assistant_msg or "status of my leave" in last_assistant_msg:
            return {"intent": "leave_status"}

        if "leave history" in last_assistant_msg:
            return {"intent": "view_leave"}

        if "view your it tickets" in last_assistant_msg:
            return {"intent": "view_it_tickets"}

        if "it ticket status" in last_assistant_msg:
            return {"intent": "it_ticket_status"}
            
    prompt = ChatPromptTemplate.from_template("""
You are the intent router for an Enterprise HR + IT Assistant.

Your job is to classify the CURRENT user message into exactly ONE intent.

Available intents:
small_talk
rag
apply_leave
view_leave
cancel_leave
leave_status
leave_balance
approve_leave
reject_leave
pending_leaves
employee_details
raise_it_ticket
view_it_tickets
it_ticket_status
assign_it_ticket
resolve_it_ticket
request_asset
asset_status
approve_asset
reject_asset
add_employee
delete_employee
unknown

IMPORTANT RULES:

1. Current message has highest priority.
Do not blindly follow old history.

2. Apply leave:
Use apply_leave when user wants to apply/take/request leave.
Examples:
- I want to apply leave
- apply sick leave
- I want casual leave tomorrow
- EMP001 after assistant asked for missing leave application details
- sick leave on 2026-05-10 because fever

3. Leave balance:
Use leave_balance when user asks about balance/remaining leaves.
Examples:
- check leave balance
- how many leaves are left
- pending leave balance
- EMP001 after assistant asked employee ID for leave balance

4. Leave status:
Use leave_status when user asks status/latest request/request approval state.
Examples:
- check my leave status
- check my leave request
- status of my leave
- is my leave approved
- EMP001 after assistant asked employee ID for leave status

5. View leave:
Use view_leave when user asks history/list of previous leave records.
Examples:
- show my leave history
- view my applied leaves
- list leaves of EMP003

6. Pending leaves:
Use pending_leaves when manager/hr/admin asks to see all pending requests.
Examples:
- show pending leave requests
- show all leave requests
- pending approvals

7. Approve/reject leave:
Use approve_leave or reject_leave when manager/admin approves or rejects.
Examples:
- approve leave 1
- approve leave request 1
- reject leave 2

8. Employee details:
Use employee_details when HR/Admin asks employee info/list employees.

9. RAG:
Use rag for policy questions.
Examples:
- what is notice period?
- explain leave policy
- what is WFH policy?

IMPORTANT:
keyboard, mouse, monitor, vpn token, software license are asset requests, not IT support tickets.

If user says "I need keyboard" classify as request_asset.
If user says "keyboard not working" classify as raise_it_ticket only if they report a problem.

10. IT tickets:

raise_it_ticket:
Use this intent when the user wants to raise/create/open/log a support ticket OR reports a problem.

This is VERY IMPORTANT:
If the user says:
- I want to raise a ticket
- raise a ticket
- create ticket
- open ticket
- new ticket
- support request
→ ALWAYS classify as raise_it_ticket

Even if no issue details are provided yet.

Also use raise_it_ticket if user describes a problem like:
- my laptop is not working
- VPN not connecting
- printer issue
- network problem
- email/outlook issue
- software installation issue

view_it_tickets:
Use view_it_tickets when user asks to show/list/view IT tickets.
Examples:
- show my IT tickets
- show all IT tickets
- list tickets

it_ticket_status:
Use it_ticket_status when user asks ticket status.
Examples:
- status of ticket 1
- show status of my IT tickets

assign_it_ticket:
Use assign_it_ticket when IT/Admin assigns ticket to engineer.

resolve_it_ticket:
Use resolve_it_ticket when IT/Admin resolves/closes ticket.
                                              
11. Asset:
request_asset: request laptop/monitor/keyboard/mouse/vpn token/software license
asset_status: asset request status
approve_asset: approve asset request
reject_asset: reject asset request

12. small_talk:
Greetings or casual messages.
Examples: hi, hello, thanks, okay, bye, yey

13.Employee management:
add_employee: HR/Admin adds a new employee.
Examples:
- add new employee
- create employee EMP011 named Zoya email zoya@test.com role employee

delete_employee: HR/Admin deletes/removes employee.
Examples:
- delete employee EMP011
- remove employee EMP005

14 unknown:
Anything outside HR/IT/asset/policy.

Recent conversation:
{history}

Current user message:
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

Your job is ONLY to extract values. Do not answer the user.

GENERAL RULES:
- Current user message has highest priority.
- Use recent conversation ONLY when the current task is unfinished.
- If the user corrects a previous value, use the latest correction.
- Do not reuse details from completed tasks.
- If recent conversation contains LEAVE_FLOW_COMPLETED, ignore leave details before that marker.
- Return null only when the value is not available in current message or unfinished recent conversation.

LEAVE RULES:
- Extract emp_id, leave_type, date, reason.
- emp_id must look like EMP001, EMP002, EMP003, etc.
- leave_type must be sick, casual, or null.
- date should be extracted in YYYY-MM-DD format when present.
- reason should be the actual reason text.
- If current message only provides emp_id, keep leave_type/date/reason from recent unfinished leave conversation.
- If current message only provides leave_type/date/reason, keep emp_id from recent unfinished leave conversation.
- If user gives all details in one message, extract all details.
- If any detail is missing, return null for only that missing field.
- Do not invent missing values.
- Do not clear previous unfinished values unless user starts a completely new leave request.

LEAVE TYPE VALIDATION:
- Sick leave is for medical reasons: fever, headache, illness, doctor visit, hospital, injury, recovery.
- Casual leave is for non-medical reasons: family function, marriage, travel, personal work, ceremony, event.
- If leave type and reason conflict, set leave_type = null and keep the reason.
- Handle typos:
  - sixk, sik, sic = sick
  - causual, casul = casual

LEAVE EXAMPLES:
User: "I want sick leave on 2026-05-07 because of fever"
emp_id = null
leave_type = sick
date = 2026-05-07
reason = fever

Recent: user asked sick leave on 2026-05-07 because fever
Current: "EMP001"
emp_id = EMP001
leave_type = sick
date = 2026-05-07
reason = fever

Recent: user gave EMP003
Current: "casual leave on 2026-05-10 due to family function"
emp_id = EMP003
leave_type = casual
date = 2026-05-10
reason = family function

User: "I want sick leave on 2026-05-10 because of family function"
emp_id = null
leave_type = null
date = 2026-05-10
reason = family function

IT RULES:
- Extract issue_type, priority, reason.
- issue_type examples: laptop, vpn, outlook, email, printer, network, software installation.
- priority must be low, medium, high, urgent, or null.
- If priority is missing, return null.

ASSET RULES:
- Extract asset_type.
- Valid asset_type: laptop, monitor, keyboard, mouse, vpn token, software license.

ADMIN RULES and HR RULES:
- Extract request_id from messages like approve leave request 1, reject leave 2, ticket 3, asset request 4.
- Extract engineer_name from messages like assign ticket 1 to Rahul.

EMPLOYEE MANAGEMENT RULES:
- For add_employee extract emp_id, new_emp_name, new_emp_email, new_emp_role.
- Role can be employee, hr, manager, it, admin.
- For delete_employee extract emp_id.
                                              
Return exactly:
emp_id, date, reason, leave_type, request_id, issue_type, priority, asset_type, engineer_name.

Recent unfinished conversation:
{history}

Current message:
{user_input}

Recent conversation:
{history}

Current message:
{user_input}
""")

    chain = prompt | llm.with_structured_output(DetailExtraction)

    result = chain.invoke({
        "history": "\n".join(
            f"{m['role']}: {m['content']}"
            for m in state.chat_history[-8:]
        ),
        "user_input": state.user_input
    })

    # start with previous state (IMPORTANT)
    data = {
        "emp_id": None,
        "date": result.date,
        "reason": result.reason,
        "leave_type": result.leave_type,
        "request_id": result.request_id,
        "issue_type": result.issue_type,
        "priority": result.priority,
        "asset_type": result.asset_type,
        "engineer_name": result.engineer_name,
    }
       # -------- UPDATE ONLY NEW VALUES --------

    # IMPORTANT:
    # Do NOT take emp_id from LLM result because it may come from old chat history.
    # emp_id must come only from the CURRENT user message.
    emp_match = re.search(r"\bEMP\d{3}\b", state.user_input, re.IGNORECASE)

    if emp_match:
        data["emp_id"] = emp_match.group(0).upper()
    else:
        data["emp_id"] = None

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

    # -------- FALLBACK EXTRACTION --------
    text = state.user_input.lower()

    if "casual" in text:
        data["leave_type"] = "casual"
    elif "sick" in text:
        data["leave_type"] = "sick"

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if date_match:
        data["date"] = date_match.group(0)

    if "because" in text or "due to" in text:
        parts = re.split(r"because|due to", text)
        if len(parts) > 1:
            data["reason"] = parts[-1].strip()

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
    from holidays import is_invalid_leave_date
    # Check holiday / weekend
    if state.date:
        invalid, message = is_invalid_leave_date(state.date)

        if invalid:
            return {
                "missing_fields": ["valid working day"],
                "response": f"Cannot apply leave: {message}"
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

    emp_match = re.search(r"\bEMP\d{3}\b", state.user_input, re.IGNORECASE)
    if emp_match:
        emp_id = emp_match.group(0).upper()

    request_id = None
    id_match = re.search(
        r"\b(?:leave\s*request\s*id|request\s*id|id)\s*[:#-]?\s*(\d+)\b",
        state.user_input,
        re.IGNORECASE
    )
    if id_match:
        request_id = int(id_match.group(1))

    if not emp_id and state.role not in ["manager", "hr", "admin"]:
        return {"response": "Please provide your employee ID. Example: EMP001"}

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
    prompt = ChatPromptTemplate.from_template("""
You are an IT support assistant.

Check which details are missing to raise an IT ticket.

Required details:
1. employee ID like EMP001
2. issue type: laptop, vpn, outlook, email, printer, network, software installation
3. issue description / reason
4. priority: low, medium, high, urgent

Rules:
- If priority is missing, default it to medium.
- If issue description is already clear, do not ask again.
- Ask only for missing fields.
- Keep response short and natural.

IMPORTANT:
- If user asks for keyboard, mouse, monitor, VPN token, or software license → this is NOT an IT ticket.
- It is an ASSET REQUEST. Ask them to request asset instead.

Current extracted details:
Employee ID: {emp_id}
Issue type: {issue_type}
Priority: {priority}
Reason: {reason}

User message:
{user_input}
""")

    missing = []

    # ---------- ASSET DETECTION ----------
    text = (state.user_input or "").lower()
    asset_keywords = ["keyboard", "mouse", "monitor", "vpn token", "software license"]

    if any(asset in text for asset in asset_keywords):
        return {
            "missing_fields": [],
            "response": (
                "It looks like you are requesting an asset.\n"
                "Please say 'request keyboard' or 'request asset' to proceed."
            )
        }

    # ---------- ACTIVE TICKET CHECK ----------
    if state.emp_id:
        duplicate = check_duplicate_ticket(state.emp_id, state.issue_type or "")
        if duplicate:
            return {
                "missing_fields": [],
                "response": (
                    f"You already have an active IT ticket.\n"
                    f"Ticket ID: {duplicate['id']}\n"
                    f"Issue: {duplicate['issue_type']}\n"
                    f"Status: {duplicate['status']}\n\n"
                    f"Please wait until it is resolved before raising a new one."
                )
            }

    # ---------- VALIDATION ----------
    valid_issues = ["laptop", "vpn", "outlook", "email", "printer", "network", "software installation"]

    if not state.emp_id:
        missing.append("employee ID")

    if not state.issue_type:
        missing.append("issue type")
    elif state.issue_type not in valid_issues:
        return {
            "missing_fields": ["valid issue type"],
            "response": (
                "Please choose a valid issue type: laptop, VPN, Outlook/email, printer, "
                "network, or software installation."
            )
        }

    if not state.reason:
        missing.append("issue description")

    priority = state.priority or "medium"

    if missing:
        chain = prompt | llm
        result = chain.invoke({
            "emp_id": state.emp_id,
            "issue_type": state.issue_type,
            "priority": priority,
            "reason": state.reason,
            "user_input": state.user_input
        })

        return {
            "missing_fields": missing,
            "priority": priority,
            "response": result.content
        }

    return {
        "missing_fields": [],
        "priority": priority
    }

def raise_it_ticket_node(state: AgentState) -> dict:
    if not state.emp_id:
        return {"response": "Please provide your employee ID. Example: EMP001"}

    if not state.issue_type:
        return {"response": "Please tell me the issue type, like laptop, VPN, printer, network, Outlook, or software installation."}

    if not state.reason:
        return {"response": "Please briefly describe the issue."}

    result = raise_it_ticket_tool.invoke({
        "emp_id": state.emp_id,
        "issue_type": state.issue_type,
        "priority": state.priority or "medium",
        "reason": state.reason,
    })

    return {"response": result["message"]}

def view_it_tickets_node(state: AgentState) -> dict:
    if state.role not in ["it", "admin"] and not state.emp_id:
        return {
            "response": "Please provide your employee ID to view your IT tickets. Example: EMP001"
        }

    result = view_it_tickets_with_role(
        state.emp_id or "",
        state.role
    )

    tickets = result["tickets"]

    if not tickets:
        return {"response": "No IT tickets found."}

    lines = ["IT Tickets:"]

    for item in tickets:
        lines.append(
            f"ID: {item['id']} | Employee: {item['employee_name']} ({item['emp_id']}) | "
            f"Issue: {item['issue_type']} | Priority: {item['priority']} | "
            f"Status: {item['status']} | Engineer: {item['assigned_engineer']}"
        )

    return {"response": "\n".join(lines)}


def it_ticket_status_node(state: AgentState) -> dict:
    # Employee: ask EMP ID first
    if state.role not in ["it", "admin"] and not state.emp_id:
        return {"response": "Please provide your employee ID to view your IT ticket status. Example: EMP001"}

    # If ticket ID is given, show that ticket
    if state.request_id:
        result = check_it_ticket_status_with_role(
            ticket_id=state.request_id,
            emp_id=state.emp_id or "",
            role=state.role
        )

        if not result["success"]:
            return {"response": result["message"]}

        ticket = result["ticket"]
        return {
            "response": (
                f"IT Ticket Status:\n"
                f"Ticket ID: {ticket['id']}\n"
                f"Employee: {ticket['employee_name']} ({ticket['emp_id']})\n"
                f"Issue: {ticket['issue_type']}\n"
                f"Priority: {ticket['priority']}\n"
                f"Status: {ticket['status']}\n"
                f"Assigned Engineer: {ticket['assigned_engineer']}"
            )
        }

    # If no ticket ID, show latest/own tickets
    result = view_it_tickets_with_role(state.emp_id or "", state.role)
    tickets = result["tickets"]

    if not tickets:
        return {"response": "No IT tickets found."}

    lines = ["Your IT Ticket Status:"]

    for item in tickets[:5]:
        lines.append(
            f"Ticket ID: {item['id']} | Issue: {item['issue_type']} | "
            f"Priority: {item['priority']} | Status: {item['status']} | "
            f"Engineer: {item['assigned_engineer']}"
        )

    return {"response": "\n".join(lines)}


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
        "emp_id": state.emp_id or "",
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

def add_employee_node(state: AgentState) -> dict:
    from database import add_employee

    if state.role not in ["hr", "admin"]:
        return {"response": "Only HR or Admin can add employees."}

    missing = []

    if not state.emp_id:
        missing.append("employee ID")
    if not state.new_emp_name:
        missing.append("employee name")
    if not state.new_emp_email:
        missing.append("email")
    if not state.new_emp_role:
        state.new_emp_role = "employee"

    if missing:
        return {"response": f"Please provide: {', '.join(missing)}."}

    result = add_employee(
        emp_id=state.emp_id,
        name=state.new_emp_name,
        email=state.new_emp_email,
        role=state.new_emp_role
    )

    return {"response": result["message"]}


def delete_employee_node(state: AgentState) -> dict:
    from database import delete_employee

    if state.role not in ["hr", "admin"]:
        return {"response": "Only HR or Admin can delete employees."}

    if not state.emp_id:
        return {"response": "Please provide the employee ID to delete. Example: EMP011"}

    result = delete_employee(state.emp_id)

    return {"response": result["message"]}
# ---------------- UNKNOWN ----------------

def unknown_node(state: AgentState) -> dict:
    return {
        "response": "I can help with HR policies, leave management, IT tickets, and asset requests."
    }