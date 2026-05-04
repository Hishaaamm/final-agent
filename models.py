from typing import Optional, Literal, List, Dict
from pydantic import BaseModel, Field


IntentType = Literal[
    "rag",

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
    "small_talk",
    "raise_it_ticket",
    "view_it_tickets",
    "it_ticket_status",
    "assign_it_ticket",
    "resolve_it_ticket",

    "request_asset",
    "asset_status",
    "approve_asset",
    "reject_asset",

    "unknown",
]


class AgentState(BaseModel):
    user_input: str
    intent: Optional[IntentType] = None
    response: Optional[str] = None

    emp_id: Optional[str] = None
    name: str = "Employee"
    role: str = "employee"

    session_id: str = "default"
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

    date: Optional[str] = None
    reason: Optional[str] = None
    leave_type: Optional[str] = None
    request_id: Optional[int] = None

    issue_type: Optional[str] = None
    priority: Optional[str] = None
    asset_type: Optional[str] = None
    engineer_name: Optional[str] = None

    missing_fields: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    emp_id: str = "EMP001"
    name: str = "Employee"
    role: str = "employee"
    chat_history: List[Dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    intent: str
    response: str


class RouterOutput(BaseModel):
    intent: IntentType

class LeaveTypeValidation(BaseModel):
    is_valid: bool
    suggested_leave_type: Optional[str] = None
    explanation: str

class DetailExtraction(BaseModel):
    emp_id: Optional[str] = None
    date: Optional[str] = None
    reason: Optional[str] = None
    leave_type: Optional[str] = None
    request_id: Optional[int] = None

    issue_type: Optional[str] = None
    priority: Optional[str] = None
    asset_type: Optional[str] = None
    engineer_name: Optional[str] = None


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
    employee_name: str
    issue_type: str
    priority: str
    reason: str


class AssetRequestInput(BaseModel):
    employee_name: str
    asset_type: str
    reason: str