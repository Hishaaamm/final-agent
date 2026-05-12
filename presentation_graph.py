from graphviz import Digraph

def create_presentation_graph():
    dot = Digraph(comment="Enterprise AI Copilot Architecture")

    dot.attr(rankdir="TB")
    dot.attr("node", shape="box", style="rounded,filled", color="#7c3aed", fillcolor="#ede9fe")

    dot.node("START", "User Query")
    dot.node("ROUTER", "Router Agent\n(Intent Detection)")
    dot.node("RBAC", "RBAC Validation\n(Role Check)")

    dot.node("HR", "HR Agent")
    dot.node("IT", "IT Agent")
    dot.node("ASSET", "Asset Agent")
    dot.node("RAG", "RAG Agent\n(Policy Q&A)")
    dot.node("UNKNOWN", "Unknown / Out of Scope")

    dot.node("HR_TOOLS", "Leave Tools\nApply / Balance / Status / Approval")
    dot.node("IT_TOOLS", "IT Tools\nRaise / Assign / Resolve Ticket")
    dot.node("ASSET_TOOLS", "Asset Tools\nRequest / Approve / Reject")
    dot.node("RAG_TOOLS", "Vector DB + Documents\nHR / IT Policy Retrieval")

    dot.node("EMAIL", "Power Automate\nEmail Notification")
    dot.node("MEMORY", "Memory + Activity Logs")
    dot.node("END", "Final Response")

    dot.edge("START", "ROUTER")
    dot.edge("ROUTER", "RBAC")

    dot.edge("RBAC", "HR")
    dot.edge("RBAC", "IT")
    dot.edge("RBAC", "ASSET")
    dot.edge("RBAC", "RAG")
    dot.edge("RBAC", "UNKNOWN")

    dot.edge("HR", "HR_TOOLS")
    dot.edge("IT", "IT_TOOLS")
    dot.edge("ASSET", "ASSET_TOOLS")
    dot.edge("RAG", "RAG_TOOLS")

    dot.edge("HR_TOOLS", "EMAIL")
    dot.edge("IT_TOOLS", "EMAIL")
    dot.edge("ASSET_TOOLS", "EMAIL")

    dot.edge("HR_TOOLS", "MEMORY")
    dot.edge("IT_TOOLS", "MEMORY")
    dot.edge("ASSET_TOOLS", "MEMORY")
    dot.edge("RAG_TOOLS", "MEMORY")
    dot.edge("UNKNOWN", "MEMORY")

    dot.edge("EMAIL", "END")
    dot.edge("MEMORY", "END")

    dot.render("enterprise_architecture_graph", format="png", cleanup=True)

    print("Graph saved as enterprise_architecture_graph.png")


if __name__ == "__main__":
    create_presentation_graph()