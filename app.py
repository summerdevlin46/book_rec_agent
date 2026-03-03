"""
Gradio Frontend for the Book Recommendation Agent
Run with: python frontend/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from langchain_core.messages import HumanMessage
from agent import recommendation_graph, AgentState


# ── State Management ───────────────────────────────────────────────────────────

def get_initial_state() -> AgentState:
    return {
        "messages": [],
        "user_preferences": {},
        "recommendations": [],
    }


# ── Core Chat Function ─────────────────────────────────────────────────────────

def chat(user_message: str, history: list, agent_state: dict) -> tuple:
    """
    Called on every user message.
    Returns: (updated_history, updated_state, recommendations_html)
    """
    if not user_message.strip():
        return history, agent_state, ""

    # Add user message to agent state
    agent_state["messages"].append(HumanMessage(content=user_message))

    # Run the graph
    result = recommendation_graph.invoke(agent_state)

    # Update state
    agent_state.update(result)

    # Extract the last AI response for display
    ai_response = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            if msg.content:  # Skip tool-call-only messages
                ai_response = msg.content
                break

    # Update Gradio chat history
    history.append((user_message, ai_response))

    # Build recommendations HTML card display
    recs_html = build_recommendations_html(result.get("recommendations", []))

    return history, agent_state, recs_html


def build_recommendations_html(recommendations: list) -> str:
    """Render book recommendation cards as HTML."""
    if not recommendations:
        return ""

    cards = []
    for book in recommendations:
        cards.append(f"""
        <div style="
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
            background: #fafafa;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        ">
            <h3 style="margin: 0 0 4px 0; color: #1a1a1a;">📖 {book.get('title', 'Unknown')}</h3>
            <p style="margin: 0 0 8px 0; color: #555; font-style: italic;">
                by {book.get('author', 'Unknown')} · {book.get('genre', '')} · {book.get('published', '')}
            </p>
            <p style="margin: 0 0 8px 0; color: #333;">💡 {book.get('why', '')}</p>
            <p style="margin: 0; color: #888; font-size: 0.85em;">
                ⭐ {book.get('rating', 'N/A')} &nbsp;|&nbsp; ISBN: {book.get('isbn', 'N/A')}
            </p>
        </div>
        """)

    return f"""
    <div style="padding: 8px;">
        <h2 style="color: #2d2d2d; border-bottom: 2px solid #6366f1; padding-bottom: 8px;">
            📚 Your Recommendations
        </h2>
        {"".join(cards)}
    </div>
    """


def clear_chat():
    return [], get_initial_state(), ""


# ── Gradio UI ──────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "I love sci-fi with AI themes, recently read Klara and the Sun",
    "Looking for a cozy mystery series I can binge",
    "Best fantasy books published in the last 2 years",
    "Non-fiction books about psychology and human behavior",
    "Short books I can finish in a weekend — literary fiction",
]

with gr.Blocks(
    title="📚 AI Book Recommender",
    theme=gr.themes.Soft(primary_hue="violet"),
    css="""
    .gradio-container { max-width: 1100px !important; }
    #chatbot { height: 480px; }
    """,
) as demo:
    # Header
    gr.Markdown("""
    # 📚 AI Book Recommender
    *Powered by LangGraph + Google Books + Tavily*
    
    Tell me what kind of books you enjoy, your current mood, or books you've loved — 
    I'll find your next great read!
    """)

    # Hidden state
    agent_state = gr.State(get_initial_state())

    with gr.Row():
        # Left: Chat
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Chat",
                bubble_full_width=False,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=book"),
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="e.g. 'I loved Dune, what should I read next?'",
                    label="",
                    scale=4,
                    container=False,
                )
                send_btn = gr.Button("Send →", variant="primary", scale=1)
            
            clear_btn = gr.Button("🗑️ New Conversation", variant="secondary", size="sm")

            gr.Examples(
                examples=EXAMPLE_QUERIES,
                inputs=msg_input,
                label="Try these examples:",
            )

        # Right: Recommendation Cards
        with gr.Column(scale=1):
            recs_display = gr.HTML(
                value="<p style='color:#999; padding:20px;'>Your recommendations will appear here after chatting... 📖</p>",
                label="Recommendations",
            )

    # Event bindings
    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot, agent_state],
        outputs=[chatbot, agent_state, recs_display],
    ).then(lambda: "", outputs=msg_input)  # Clear input after send

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot, agent_state],
        outputs=[chatbot, agent_state, recs_display],
    ).then(lambda: "", outputs=msg_input)

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, agent_state, recs_display],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,           # Set True to get a public Gradio link
        show_error=True,
    )
