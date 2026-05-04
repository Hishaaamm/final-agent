from langgraph.graph import StateGraph, END
from models import AgentState

from nodes import (
    router_node,
    extract_details_node,
    rag_node,

    small_talk_node,
    validate_leave_node,
    apply_leave_node,
    view_leave_node,
    cancel_leave_node,
    leave_status_node,
    leave_balance_node,
    pending_leave_node,
    approve_leave_node,
    reject_leave_node,
    employee_details_node,

    validate_it_ticket_node,
    raise_it_ticket_node,
    view_it_tickets_node,
    it_ticket_status_node,
    assign_it_ticket_node,
    resolve_it_ticket_node,

    validate_asset_node,
    request_asset_node,
    asset_status_node,
    approve_asset_node,
    reject_asset_node,

    unknown_node,
)


def route_after_router(state: AgentState):
    if state.intent == "rag":
        return "rag"

    if state.intent in [
        "apply_leave",
        "cancel_leave",
        "leave_status",
        "approve_leave",
        "reject_leave",
        "raise_it_ticket",
        "it_ticket_status",
        "assign_it_ticket",
        "resolve_it_ticket",
        "request_asset",
        "asset_status",
        "approve_asset",
        "reject_asset",
    ]:
        return "extract_details"
    if state.intent == "small_talk":
        return "small_talk"

    if state.intent == "view_leave":
        return "view_leave"

    if state.intent == "leave_balance":
        return "leave_balance"

    if state.intent == "pending_leaves":
        return "pending_leaves"

    if state.intent == "view_it_tickets":
        return "view_it_tickets"

    if state.intent == "employee_details":
        return "extract_details"

    return "unknown"


def route_after_extract(state: AgentState):
    if state.intent == "apply_leave":
        return "validate_leave"

    if state.intent == "cancel_leave":
        return "cancel_leave"

    if state.intent == "leave_status":
        return "leave_status"

    if state.intent == "approve_leave":
        return "approve_leave"

    if state.intent == "reject_leave":
        return "reject_leave"

    if state.intent == "leave_balance":
        return "leave_balance"

    if state.intent == "employee_details":
        return "employee_details"
    
    if state.intent == "raise_it_ticket":
        return "validate_it_ticket"

    if state.intent == "it_ticket_status":
        return "it_ticket_status"

    if state.intent == "assign_it_ticket":
        return "assign_it_ticket"

    if state.intent == "resolve_it_ticket":
        return "resolve_it_ticket"

    if state.intent == "request_asset":
        return "validate_asset"

    if state.intent == "asset_status":
        return "asset_status"

    if state.intent == "approve_asset":
        return "approve_asset"

    if state.intent == "reject_asset":
        return "reject_asset"

    return "unknown"


def route_after_leave_validation(state: AgentState):
    if state.missing_fields:
        return "end"
    return "apply_leave"


def route_after_it_ticket_validation(state: AgentState):
    if state.missing_fields:
        return "end"
    return "raise_it_ticket"


def route_after_asset_validation(state: AgentState):
    if state.missing_fields:
        return "end"
    return "request_asset"


workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("extract_details", extract_details_node)
workflow.add_node("rag", rag_node)

workflow.add_node("small_talk", small_talk_node)
workflow.add_node("employee_details", employee_details_node)
workflow.add_node("validate_leave", validate_leave_node)
workflow.add_node("apply_leave", apply_leave_node)
workflow.add_node("view_leave", view_leave_node)
workflow.add_node("cancel_leave", cancel_leave_node)
workflow.add_node("leave_status", leave_status_node)
workflow.add_node("leave_balance", leave_balance_node)
workflow.add_node("pending_leaves", pending_leave_node)
workflow.add_node("approve_leave", approve_leave_node)
workflow.add_node("reject_leave", reject_leave_node)

workflow.add_node("validate_it_ticket", validate_it_ticket_node)
workflow.add_node("raise_it_ticket", raise_it_ticket_node)
workflow.add_node("view_it_tickets", view_it_tickets_node)
workflow.add_node("it_ticket_status", it_ticket_status_node)
workflow.add_node("assign_it_ticket", assign_it_ticket_node)
workflow.add_node("resolve_it_ticket", resolve_it_ticket_node)

workflow.add_node("validate_asset", validate_asset_node)
workflow.add_node("request_asset", request_asset_node)
workflow.add_node("asset_status", asset_status_node)
workflow.add_node("approve_asset", approve_asset_node)
workflow.add_node("reject_asset", reject_asset_node)

workflow.add_node("unknown", unknown_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_after_router,
    {
        "rag": "rag",
        "small_talk": "small_talk",
        "extract_details": "extract_details",
        "view_leave": "view_leave",
        "leave_balance": "leave_balance",
        "pending_leaves": "pending_leaves",
        "view_it_tickets": "view_it_tickets",
        "employee_details": "employee_details",
        "unknown": "unknown",
    },
)

workflow.add_conditional_edges(
    "extract_details",
    route_after_extract,
    {
        "validate_leave": "validate_leave",
        "cancel_leave": "cancel_leave",
        "leave_status": "leave_status",
        "leave_balance": "leave_balance",
        "approve_leave": "approve_leave",
        "reject_leave": "reject_leave",
        "employee_details": "employee_details",
        "validate_it_ticket": "validate_it_ticket",
        "it_ticket_status": "it_ticket_status",
        "assign_it_ticket": "assign_it_ticket",
        "resolve_it_ticket": "resolve_it_ticket",

        "validate_asset": "validate_asset",
        "asset_status": "asset_status",
        "approve_asset": "approve_asset",
        "reject_asset": "reject_asset",

        "unknown": "unknown",
    },
)

workflow.add_conditional_edges(
    "validate_leave",
    route_after_leave_validation,
    {
        "apply_leave": "apply_leave",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "validate_it_ticket",
    route_after_it_ticket_validation,
    {
        "raise_it_ticket": "raise_it_ticket",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "validate_asset",
    route_after_asset_validation,
    {
        "request_asset": "request_asset",
        "end": END,
    },
)

for node in [
    "small_talk",
    "rag",
    "apply_leave",
    "view_leave",
    "employee_details",
    "cancel_leave",
    "leave_status",
    "leave_balance",
    "pending_leaves",
    "approve_leave",
    "reject_leave",
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
]:
    workflow.add_edge(node, END)

enterprise_graph = workflow.compile()