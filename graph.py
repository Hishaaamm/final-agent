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

    add_employee_node,
    delete_employee_node,

    unknown_node,
)


# ---------------- ROUTING ----------------

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
        "employee_details",
        "add_employee",
        "delete_employee",
    ]:
        return "extract_details"

    if state.intent == "small_talk":
        return "small_talk"

    if state.intent == "view_leave":
        return "extract_details"

    if state.intent == "leave_balance":
        return "leave_balance"

    if state.intent == "pending_leaves":
        return "pending_leaves"

    if state.intent == "view_it_tickets":
        return "extract_details"

    return "unknown"


def route_after_extract(state: AgentState):
    mapping = {
        "apply_leave": "validate_leave",
        "cancel_leave": "cancel_leave", 
        "leave_status": "leave_status",
        "approve_leave": "approve_leave",
        "view_leave": "view_leave",
        "reject_leave": "reject_leave",
        "leave_balance": "leave_balance",
        "employee_details": "employee_details",
        "add_employee": "add_employee",
        "delete_employee": "delete_employee",
        "raise_it_ticket": "validate_it_ticket",
        "view_it_tickets": "view_it_tickets",
        "it_ticket_status": "it_ticket_status",
        "assign_it_ticket": "assign_it_ticket",
        "resolve_it_ticket": "resolve_it_ticket",
        "request_asset": "validate_asset",
        "asset_status": "asset_status",
        "approve_asset": "approve_asset",
        "reject_asset": "reject_asset",
    }
    return mapping.get(state.intent, "unknown")


def route_after_leave_validation(state: AgentState):
    # Extra safety check (prevents DB errors)
    if not state.emp_id:
        return "end"

    if state.missing_fields:
        return "end"

    return "apply_leave"


def route_after_it_ticket_validation(state: AgentState):
    return "raise_it_ticket" if not state.missing_fields else "end"


def route_after_asset_validation(state: AgentState):
    return "request_asset" if not state.missing_fields else "end"


# ---------------- BUILD GRAPH ----------------

workflow = StateGraph(AgentState)

# Nodes
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
workflow.add_node("add_employee", add_employee_node)
workflow.add_node("delete_employee", delete_employee_node)

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

# Entry
workflow.set_entry_point("router")

# Router edges
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

# Extract edges
workflow.add_conditional_edges(
    "extract_details",
    route_after_extract,
    {
        "validate_leave": "validate_leave",
        "cancel_leave": "cancel_leave",
        "leave_status": "leave_status",
        "leave_balance": "leave_balance",
        "approve_leave": "approve_leave",
        "view_leave": "view_leave",

        "reject_leave": "reject_leave",
        "employee_details": "employee_details",
        "add_employee": "add_employee",
        "delete_employee": "delete_employee",
        
        "validate_it_ticket": "validate_it_ticket",
        "it_ticket_status": "it_ticket_status",
        "view_it_tickets": "view_it_tickets",
        "assign_it_ticket": "assign_it_ticket",
        "resolve_it_ticket": "resolve_it_ticket",
        "validate_asset": "validate_asset",
        "asset_status": "asset_status",
        "approve_asset": "approve_asset",
        "reject_asset": "reject_asset",
        
        "unknown": "unknown",
    },
)

# Validation edges
workflow.add_conditional_edges(
    "validate_leave",
    route_after_leave_validation,
    {"apply_leave": "apply_leave", "end": END},
)

workflow.add_conditional_edges(
    "validate_it_ticket",
    route_after_it_ticket_validation,
    {"raise_it_ticket": "raise_it_ticket", "end": END},
)

workflow.add_conditional_edges(
    "validate_asset",
    route_after_asset_validation,
    {"request_asset": "request_asset", "end": END},
)

# End nodes
for node in [
    "small_talk", "rag", "apply_leave", "view_leave", "employee_details",
    "cancel_leave", "leave_status", "leave_balance", "pending_leaves",
    "approve_leave", "reject_leave", "raise_it_ticket", "view_it_tickets",
    "it_ticket_status", "assign_it_ticket", "resolve_it_ticket",
    "request_asset", "asset_status", "approve_asset", "reject_asset","add_employee", "delete_employee",
    "unknown",
]:
    workflow.add_edge(node, END)

# Compile
enterprise_graph = workflow.compile()


# ---------------- PRINT GRAPH ----------------

if __name__ == "__main__":
    graph = enterprise_graph.get_graph()
    print(graph.draw_ascii())