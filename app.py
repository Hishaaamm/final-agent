from dotenv import load_dotenv
load_dotenv()
import gradio as gr
import json
import re
import speech_recognition as sr
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from database import (
    create_tables,
    get_pending_leave_requests,
    approve_leave_request,
    reject_leave_request,
    get_it_tickets,
    assign_it_ticket,
    resolve_it_ticket,
    get_activity_logs,
    save_activity_log,
    get_asset_request_status,
    approve_asset_request,
    reject_asset_request,

)

from graph import enterprise_graph

#pydantic models
from models import ChatRequest, ChatResponse



create_tables()
#creates fastapi backend app

app = FastAPI(title="Enterprise HR + IT Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#request will come here first
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
    print("✅ WebSocket endpoint was called")
    await websocket.accept()
    await websocket.send_text("Connected to Enterprise Assistant.")

    chat_history = []

    while True:
        message = await websocket.receive_text()

        await websocket.send_text("Routing request...")
        await websocket.send_text("Processing with LangGraph...")

        result = enterprise_graph.invoke({
            "user_input": message,
            "emp_id": None,
            "name": "Employee",
            "role": "employee",
            "chat_history": chat_history
        })
        response = result.get("response", "No response generated.")

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": response})

        await websocket.send_text(response)

def extract_question_index_with_llm(message: str):
    """
    Uses LLM to understand which previous question/query/message the user is asking for.
    """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
You are a memory index extractor.

Your job is to check whether the user is asking about a previous question/query/message.

Return ONLY valid JSON.

Rules:
- If user asks for first question/query/message, return {{"index": 0}}
- If user asks for second question/query/message, return {{"index": 1}}
- If user asks for third question/query/message, return {{"index": 2}}
- If user asks for 10th question/query/message, return {{"index": 9}}
- If user asks for last/latest/previous question/query/message, return {{"index": -1}}
- If user is not asking about a previous question/query/message, return {{"index": null}}

User message:
"{message}"
"""

    try:
        result = llm.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)

        match = re.search(r"\{.*\}", content, re.DOTALL)

        if not match:
            return None

        data = json.loads(match.group())
        return data.get("index")

    except Exception as e:
        print("LLM index extraction failed:", e)
        return None


def handle_previous_question_request_with_llm(message: str, user_msgs: list):
    index = extract_question_index_with_llm(message)

    if index is None:
        return None

    if not user_msgs:
        return "No previous questions found."

    if index == -1:
        return f"Your last question was: {user_msgs[-1]}"

    if not isinstance(index, int):
        return None

    if index < 0:
        return "Invalid question number."

    if index >= len(user_msgs):
        return (
            f"I could not find question number {index + 1}. "
            f"You have only asked {len(user_msgs)} question(s) so far."
        )

    return f"Your question number {index + 1} was: {user_msgs[index]}"

def gradio_chat(message, history, role, session_state):
    import re

    # If no session, create fresh session
    if session_state is None:
        session_state = {
            "current_role": role,
            "emp_id": None,
            "chat_history": []
        }

    # Safety: make sure required keys exist
    session_state.setdefault("chat_history", [])
    session_state.setdefault("emp_id", None)
    session_state.setdefault("current_role", role)

    # If role changed, start fresh chat
    if session_state.get("current_role") != role:
        session_state["chat_history"] = []
        session_state["emp_id"] = None
        session_state["current_role"] = role

    # Search for employee ID pattern like EMP001
    emp_match = re.search(r"\bEMP\d{3}\b", message, re.IGNORECASE)

    if emp_match:
        new_emp = emp_match.group(0).upper()

        # If employee changed, clear old chat history
        if session_state.get("emp_id") and session_state["emp_id"] != new_emp:
            session_state["chat_history"] = []

        session_state["emp_id"] = new_emp

    # Get only previous user messages
    user_msgs = [
        chat["content"]
        for chat in session_state.get("chat_history", [])
        if chat.get("role") == "user"
    ]

    # Check if user is asking about previous question/query/message
    previous_question_response = handle_previous_question_request_with_llm(
        message,
        user_msgs
    )

    if previous_question_response:
        # Save user's current memory question also
        session_state["chat_history"].append({
            "role": "user",
            "content": message
        })

        session_state["chat_history"].append({
            "role": "assistant",
            "content": previous_question_response
        })

        return previous_question_response, session_state

    # Normal flow: save user message
    session_state["chat_history"].append({
        "role": "user",
        "content": message
    })

    # Send to LangGraph first step
    result = enterprise_graph.invoke({
        "user_input": message,
        "emp_id": session_state.get("emp_id"),
        "name": "Employee",
        "role": role,
        "chat_history": session_state["chat_history"],
        "session_id": "gradio_session"
    })

    response = result.get("response", "No response generated.")

    # Update emp_id if graph extracted one
    if result.get("emp_id"):
        session_state["emp_id"] = result.get("emp_id")

    # Save assistant response
    session_state["chat_history"].append({
        "role": "assistant",
        "content": response
    })

    session_state["current_role"] = role

    return response, session_state
def dashboard_data():
    requests = get_pending_leave_requests()

    if not requests:
        return """
        <div style="
            padding:20px;
            border-radius:16px;
            background:#1f2937;
            color:white;
            font-size:16px;
        ">
            No pending leave requests.
        </div>
        """

    html = """
    <div style="
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
        gap:18px;
        margin-top:10px;
    ">
    """
#fstring f""" inserts Python values into HTML.
    for r in requests:
        html += f""" 
        <div style="
            background:#1f2937;
            border:1px solid #374151;
            border-radius:18px;
            padding:20px;
            color:white;
            box-shadow:0 10px 25px rgba(0,0,0,0.25);
        ">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                margin-bottom:14px;
            ">
                <span style="
                    background:#f59e0b;
                    color:#111827;
                    padding:5px 12px;
                    border-radius:999px;
                    font-size:12px;
                    font-weight:bold;
                ">
                    PENDING
                </span>

                <span style="
                    color:#9ca3af;
                    font-size:13px;
                ">
                    Request #{r['id']}
                </span>
            </div>

            <h3 style="
                margin:0;
                margin-bottom:12px;
                font-size:22px;
            ">
                {r['employee_name']}
            </h3>

            <div style="color:#d1d5db;line-height:1.9;">
                <div><b>Employee ID:</b> {r['emp_id']}</div>
                <div><b>Leave Type:</b> {r['leave_type'].title()}</div>
                <div><b>Date:</b> {r['date']}</div>
                <div><b>Reason:</b> {r['reason']}</div>
                <div><b>Status:</b> {r['status']}</div>
            </div>

        </div>
        """

    html += "</div>"

    return html


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
        return """
        <div style="padding:20px;border-radius:16px;background:#1f2937;color:white;">
            No IT tickets found.
        </div>
        """

    html = """
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px;margin-top:10px;">
    """

    for t in tickets:
        html += f"""
        <div style="background:#1f2937;border:1px solid #374151;border-radius:18px;padding:20px;color:white;">
            <div style="display:flex;justify-content:space-between;margin-bottom:12px;">
                <span style="background:#3b82f6;color:white;padding:5px 12px;border-radius:999px;font-size:12px;font-weight:bold;">
                    {t['status'].upper()}
                </span>
                <span style="color:#9ca3af;">Ticket #{t['id']}</span>
            </div>

            <h3 style="margin:0 0 12px 0;">{t['employee_name']} ({t['emp_id']})</h3>

            <div style="color:#d1d5db;line-height:1.8;">
                <div><b>Issue Type:</b> {t['issue_type']}</div>
                <div><b>Priority:</b> {t['priority']}</div>
                <div><b>Reason:</b> {t['reason']}</div>
                <div><b>Assigned Engineer:</b> {t['assigned_engineer']}</div>
            </div>
        </div>
        """

    html += "</div>"
    return html


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
    
def switch_role(role, session_state):
    session_state = {
        "current_role": role,
        "emp_id": None,
        "chat_history": []
    }

    return [], session_state

#connects with 
def chat_submit(message, role, session_state):
    if not message or not message.strip():
        return "", [], session_state

    response, session_state = gradio_chat(message, [], role, session_state)

    ui_history = []

    history = session_state["chat_history"]

    for i in range(0, len(history) - 1, 2):
        if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
            ui_history.append((history[i]["content"], history[i + 1]["content"]))

    return "", ui_history, session_state
#shows user message first
def add_user_message(message, role, session_state, chatbot):
    if not message or not message.strip():
        return "", chatbot, session_state

    chatbot = chatbot + [(message, None)]

    return "", chatbot, session_state


def process_bot_response(chatbot, role, session_state):
    import time

    if not chatbot:
        yield chatbot, session_state
        return

    message = chatbot[-1][0]

    chatbot[-1] = (message, "🧠 AI analyzing request...")
    yield chatbot, session_state

    time.sleep(0.15)

    chatbot[-1] = (message, "🔍 Checking enterprise systems...")
    yield chatbot, session_state

    time.sleep(0.15)

    chatbot[-1] = (message, "⚡ Agent preparing response...")
    yield chatbot, session_state

    response, session_state = gradio_chat(
        message,
        [],
        role,
        session_state
    )
    save_activity_log(
        role=role,
        intent=session_state.get("last_intent", "unknown"),
        user_message=message,
        assistant_response=response
    )

    chatbot[-1] = (message, response)

    yield chatbot, session_state

def transcribe_voice(audio_path):
    if audio_path is None:
        return ""

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)

        return text

    except Exception as e:
        return f"Voice recognition failed: {str(e)}"
    
def show_ai_workflow():
    return """
    <div style="font-family:Arial;padding:20px;background:#111827;color:white;border-radius:18px;">
        <h2 style="color:#60a5fa;">Enterprise AI Workflow</h2>

        <pre style="
            background:#1f2937;
            padding:20px;
            border-radius:14px;
            font-size:15px;
            line-height:1.8;
            overflow-x:auto;
        ">
User Query
   |
   v
Router Agent
   |
   +--> Small Talk / Unknown Handler
   |
   +--> RBAC Validation
           |
           +--> HR Agent
           |      |
           |      +--> Leave Apply
           |      +--> Leave Balance
           |      +--> Leave Status
           |      +--> Leave Approval
           |
           +--> IT Agent
           |      |
           |      +--> Raise Ticket
           |      +--> Assign Ticket
           |      +--> Resolve Ticket
           |
           +--> Asset Agent
           |      |
           |      +--> Request Asset
           |      +--> Approve / Reject Asset
           |
           +--> RAG Agent
                  |
                  +--> HR / IT Policy Search

Memory + Logs + Power Automate Emails run across workflows.
        </pre>

        <h3 style="color:#34d399;">What this shows</h3>
        <ul style="line-height:1.8;">
            <li><b>Router Agent:</b> Detects user intent.</li>
            <li><b>RBAC:</b> Checks if the role is allowed.</li>
            <li><b>HR Agent:</b> Handles leave and employee workflows.</li>
            <li><b>IT Agent:</b> Handles tickets and issue resolution.</li>
            <li><b>Asset Agent:</b> Handles asset requests and approvals.</li>
            <li><b>RAG Agent:</b> Answers company policy questions.</li>
            <li><b>Power Automate:</b> Sends email notifications.</li>
        </ul>
    </div>
    """

def format_logs():

    logs = get_activity_logs()

    if not logs:
        return "<h3>No logs available.</h3>"

    html = """
    <div style='padding:20px;color:white;'>
    <h2 style='color:#60a5fa;'>Enterprise Activity Logs</h2>
    """

    for log in logs:

        html += f"""
        <div style='
            background:#1f2937;
            padding:15px;
            margin-bottom:15px;
            border-radius:12px;
            border-left:4px solid #3b82f6;
        '>

        <p><b>Time:</b> {log['timestamp']}</p>
        <p><b>Role:</b> {log['role']}</p>
        <p><b>Intent:</b> {log['intent']}</p>
        <p><b>User:</b> {log['user_message']}</p>
        <p><b>Assistant:</b> {log['assistant_response']}</p>

        </div>
        """

    html += "</div>"

    return html

#gradio ui

with gr.Blocks(title="Enterprise HR + IT Assistant") as demo:
    gr.Markdown("# Enterprise HR + IT Assistant")
    gr.Markdown("HR + IT assistant with RAG, leave management, IT tickets, assets, RBAC, and manager approval dashboard.")

    with gr.Tab("Chat Assistant"):

        session_state = gr.State({
            "current_role": "employee",
            "emp_id": None,
            "chat_history": []
        })

        role_dropdown = gr.Dropdown(
            choices=["employee", "hr", "manager", "it", "admin"],
            value="employee",
            label="Role"
        )

        chatbot = gr.Chatbot(
            label="Chatbot",
            height=420
        )

        msg = gr.Textbox(
            placeholder="Type your message here...",
            label="Message"
        )
        voice_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Voice Input"
        )

        voice_to_text_btn = gr.Button("Convert Voice to Text")
        voice_to_text_btn.click(
            fn=transcribe_voice,
            inputs=voice_input,
            outputs=msg
        )
        send_btn = gr.Button("Send")

        role_dropdown.change(
            fn=switch_role,
            inputs=[role_dropdown, session_state],
            outputs=[chatbot, session_state]
        )

        send_btn.click(
            fn=add_user_message,
            inputs=[msg, role_dropdown, session_state, chatbot],
            outputs=[msg, chatbot, session_state],
            queue=False
        ).then(
            fn=process_bot_response,
            inputs=[chatbot, role_dropdown, session_state],
            outputs=[chatbot, session_state]
        )

        msg.submit(
            fn=add_user_message,
            inputs=[msg, role_dropdown, session_state, chatbot],
            outputs=[msg, chatbot, session_state],
            queue=False
        ).then(
            fn=process_bot_response,
            inputs=[chatbot, role_dropdown, session_state],
            outputs=[chatbot, session_state]
        )
    # with gr.Tab("AI Workflow"):
    #     gr.Markdown("## LangGraph Workflow Visualization")

    #     workflow_btn = gr.Button("Show AI Workflow")

    #     workflow_box = gr.HTML()

    #     workflow_btn.click(
    #         fn=show_ai_workflow,
    #         inputs=None,
    #         outputs=workflow_box
    #     )       

    with gr.Tab("Activity Logs"):

        refresh_logs_btn = gr.Button("Refresh Logs")

        logs_output = gr.HTML()

        refresh_logs_btn.click(
            fn=format_logs,
            inputs=None,
            outputs=logs_output
        )

    with gr.Tab("Manager Leave Dashboard"):
        gr.Markdown("## Pending Leave Requests")

        refresh_btn = gr.Button("Refresh Requests")

        pending_box = gr.HTML(
            value=dashboard_data()
        )

        request_id = gr.Number(
            label="Request ID",
            precision=0
        )

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
#it dashboard
    with gr.Tab("IT Ticket Dashboard"):
        gr.Markdown("## IT Ticket Dashboard")
        gr.Markdown(
            "View all IT tickets, assign tickets to engineers, and resolve completed tickets."
        )

        refresh_it_btn = gr.Button("Refresh Tickets")

        it_ticket_box = gr.HTML(
            value=it_dashboard_data()
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
        
demo.queue()
app = gr.mount_gradio_app(app, demo, path="/ui")