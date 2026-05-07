from typing import Optional, Literal, List, Dict
from pydantic import BaseModel, Field


IntentType = Literal[
    "rag",
    "small_talk",

    # Leave
    "apply_leave",
    "view_leave",
    "leave_summary",
    "cancel_leave",
    "leave_status",
    "leave_balance",
    "approve_leave",
    "reject_leave",
    "pending_leaves",
    "employee_details",

    # IT tickets
    "raise_it_ticket",
    "view_it_tickets",
    "it_ticket_status",
    "assign_it_ticket",
    "resolve_it_ticket",

    # Asset requests
    "request_asset",
    "asset_status",
    "manager_approve_asset",
    "manager_reject_asset",
    "it_approve_asset",
    "it_reject_asset",
    "validate_inventory",
    "fulfill_asset",

    "add_employee",
    "delete_employee",

    "unknown",
]

#main memory/state object passed inside LangGraph.pydantic model

class AgentState(BaseModel):
    user_input: str
    intent: Optional[IntentType] = None
    response: Optional[str] = None

    emp_id: Optional[str] = None
    name: str = "Employee"
    role: str = "employee"

    session_id: str = "default"
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

    # Leave
    date: Optional[str] = None
    reason: Optional[str] = None
    leave_type: Optional[str] = None
    request_id: Optional[int] = None

    # IT ticket
    issue_type: Optional[str] = None
    priority: Optional[str] = None
    engineer_name: Optional[str] = None

    # Asset
    asset_type: Optional[str] = None

    missing_fields: List[str] = Field(default_factory=list)
    new_emp_name: Optional[str] = None
    new_emp_email: Optional[str] = None
    new_emp_role: Optional[str] = None

#defines what frontend sends to backend

class ChatRequest(BaseModel):
    message: str
    emp_id: Optional[str] = None
    name: str = "Employee"
    role: str = "employee"
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

#response from the backend

class ChatResponse(BaseModel):
    intent: str
    response: str


class RouterOutput(BaseModel):
    intent: IntentType


class LeaveTypeValidation(BaseModel):
    is_valid: bool
    suggested_leave_type: Optional[str] = None
    explanation: str

#Extraction

class DetailExtraction(BaseModel):
    emp_id: Optional[str] = None

    # Leave
    date: Optional[str] = None
    reason: Optional[str] = None
    leave_type: Optional[str] = None
    request_id: Optional[int] = None

    # IT ticket
    issue_type: Optional[str] = None
    priority: Optional[str] = None
    engineer_name: Optional[str] = None

    # Asset
    asset_type: Optional[str] = None

#tool input models

class ApplyLeaveInput(BaseModel):
    emp_id: str
    leave_type: str
    date: str
    reason: str


class EmployeeIdInput(BaseModel):
    emp_id: str


class RequestIdInput(BaseModel):
    request_id: int


class RaiseITTicketInput(BaseModel):
    emp_id: str
    issue_type: str
    priority: str = "medium"
    reason: str


class AssetRequestInput(BaseModel):
    emp_id: str
    asset_type: str
    reason: str