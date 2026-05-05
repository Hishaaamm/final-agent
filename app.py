from urllib import response

from dotenv import load_dotenv
import gradio as gr

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from database import (
    create_tables,
    save_message,
    load_memory,
    get_pending_leave_requests,
    approve_leave_request,
    reject_leave_request,
    get_it_tickets,
    assign_it_ticket,
    resolve_it_ticket,
    get_asset_request_status,
    approve_asset_request,
    reject_asset_request
)

from graph import enterprise_graph
from models import ChatRequest, ChatResponse


load_dotenv()
create_tables()

app = FastAPI(title="Enterprise HR + IT Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = enterprise_graph.invoke({
        "user_input": request.message,
        "emp_id": getattr(request, "emp_id", None),
        "name": request.name,
        "role": request.role,
        "chat_history": request.chat_history
    })

    return ChatResponse(
        intent=result.get("intent", "unknown"),
        response=result.get("response", "No response generated.")
    )


@app.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Connected to Enterprise Assistant.")

    chat_history = []

    while True:
        message = await websocket.receive_text()

        await websocket.send_text("Routing request...")
        await websocket.send_text("Processing with LangGraph...")

        result = enterprise_graph.invoke({
            "user_input": message,
            "emp_id": "EMP001",
            "name": "EMP001",
            "role": "employee",
            "chat_history": chat_history
        })

        response = result.get("response", "No response generated.")

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": response})

        await websocket.send_text(response)


def gradio_chat(message, history, role, emp_id):
    emp_id = emp_id.upper().strip() if emp_id else None
    session_id = f"{emp_id}_{role}_session" if emp_id else f"{role}_session"

    db_history = load_memory(session_id, limit=10)

    result = enterprise_graph.invoke({
        "user_input": message,
        "emp_id": emp_id,
        "name": emp_id or "User",
        "role": role,
        "chat_history": db_history,
        "session_id": session_id
    })

    response = result.get("response", "No response generated.")

    save_message(session_id, "user", message)
    save_message(session_id, "assistant", response)

    if "leave request submitted" in response.lower():
        save_message(session_id, "system", "LEAVE_FLOW_COMPLETED")

    return response


def dashboard_data():
    requests = get_pending_leave_requests()

    if not requests:
        return "No pending leave requests."

    text = ""

    for r in requests:
        text += (
            f"Request ID: {r['id']}\n"
            f"Employee: {r['emp_id']} - {r['employee_name']}\n"
            f"Type: {r['leave_type']}\n"
            f"Date: {r['date']}\n"
            f"Reason: {r['reason']}\n"
            f"Status: {r['status']}\n"
            f"{'-' * 40}\n"
        )

    return text


def dashboard_approve(request_id):
    if request_id is None:
        return "Please enter a request ID.", dashboard_data()

    result = approve_leave_request(int(request_id), "manager")
    return result["message"], dashboard_data()


def dashboard_reject(request_id):
    if request_id is None:
        return "Please enter a request ID.", dashboard_data()

    result = reject_leave_request(int(request_id), "manager")
    return result["message"], dashboard_data()

def it_dashboard_data():
    tickets = get_it_tickets(emp_id="", role="it")

    if not tickets:
        return "No IT tickets found."

    text = ""

    for t in tickets:
        text += (
            f"Ticket ID: {t['id']}\n"
            f"Employee: {t['employee_name']} ({t['emp_id']})\n"
            f"Issue Type: {t['issue_type']}\n"
            f"Priority: {t['priority']}\n"
            f"Reason: {t['reason']}\n"
            f"Status: {t['status']}\n"
            f"Assigned Engineer: {t['assigned_engineer']}\n"
            f"{'-' * 45}\n"
        )

    return text


def dashboard_assign_ticket(ticket_id, engineer_name):
    if ticket_id is None:
        return "Please enter a ticket ID.", it_dashboard_data()

    if not engineer_name or not engineer_name.strip():
        return "Please enter engineer name.", it_dashboard_data()

    result = assign_it_ticket(
        ticket_id=int(ticket_id),
        engineer_name=engineer_name.strip(),
        role="it"
    )

    return result["message"], it_dashboard_data()


def dashboard_resolve_ticket(ticket_id):
    if ticket_id is None:
        return "Please enter a ticket ID.", it_dashboard_data()

    result = resolve_it_ticket(
        ticket_id=int(ticket_id),
        role="it"
    )

    return result["message"], it_dashboard_data()
with gr.Blocks(title="Enterprise HR + IT Assistant") as demo:
    gr.Markdown("# Enterprise HR + IT Assistant")
    gr.Markdown("HR + IT assistant with RAG, leave management, IT tickets, assets, RBAC, and manager approval dashboard.")

    with gr.Tab("Chat Assistant"):
        gr.ChatInterface(
            fn=gradio_chat,
            additional_inputs=[
                gr.Dropdown(
                    choices=["employee", "hr", "manager", "it", "admin"],
                    value="employee",
                    label="Role"
                )
            ],
            description="Ask HR policy questions, apply leave, raise IT tickets, request assets, and track status."
        )

    with gr.Tab("Manager Leave Dashboard"):
        gr.Markdown("## Pending Leave Requests")

        refresh_btn = gr.Button("Refresh Requests")

        pending_box = gr.Textbox(
            label="Pending Leave Requests",
            value=dashboard_data,
            lines=15
        )

        request_id = gr.Number(label="Request ID", precision=0)

        with gr.Row():
            approve_btn = gr.Button("Approve Leave")
            reject_btn = gr.Button("Reject Leave")

        result_box = gr.Textbox(label="Action Result")

        refresh_btn.click(
            fn=dashboard_data,
            inputs=None,
            outputs=pending_box
        )

        approve_btn.click(
            fn=dashboard_approve,
            inputs=request_id,
            outputs=[result_box, pending_box]
        )

        reject_btn.click(
            fn=dashboard_reject,
            inputs=request_id,
            outputs=[result_box, pending_box]
        )
    with gr.Tab("IT Ticket Dashboard"):
        gr.Markdown("## IT Ticket Dashboard")
        gr.Markdown(
            "View all IT tickets, assign tickets to engineers, and resolve completed tickets."
        )

        refresh_it_btn = gr.Button("Refresh Tickets")

        it_ticket_box = gr.Textbox(
            label="All IT Tickets",
            value=it_dashboard_data,
            lines=18
        )

        with gr.Row():
            ticket_id_input = gr.Number(
                label="Ticket ID",
                precision=0
            )

            engineer_input = gr.Textbox(
                label="Engineer Name",
                placeholder="Example: Rahul"
            )

        with gr.Row():
            assign_ticket_btn = gr.Button("Assign Ticket")
            resolve_ticket_btn = gr.Button("Resolve Ticket")

        it_result_box = gr.Textbox(label="Action Result")

        refresh_it_btn.click(
            fn=it_dashboard_data,
            inputs=None,
            outputs=it_ticket_box
        )

        assign_ticket_btn.click(
            fn=dashboard_assign_ticket,
            inputs=[ticket_id_input, engineer_input],
            outputs=[it_result_box, it_ticket_box]
        )

        resolve_ticket_btn.click(
            fn=dashboard_resolve_ticket,
            inputs=ticket_id_input,
            outputs=[it_result_box, it_ticket_box]
        )

app = gr.mount_gradio_app(app, demo, path="/ui")