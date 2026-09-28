import os
import sqlite3
import asyncio
from google import genai
from google.genai import types
import gradio as gr
from playwright.async_api import async_playwright

# --- Database Setup for Long-Term Memory ---
DB_PATH = "memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def load_chat_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM memory")
    rows = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

def save_message(role, content):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO memory (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()


# --- Playwright Browser Automation & Gemini Reasoning ---
async def run_browser_agent(instruction: str, api_key: str):
    if not api_key:
        yield "Error: Please enter your Gemini API Key.", None
        return

    client = genai.Client(api_key=api_key)
    yield "Launching browser automation...", None

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
            )
            context = await browser.new_context(viewport={"width": 1280, "height": 800})
            page = await context.new_page()

            yield f"Navigating web for: {instruction}", None
            
            if "search" in instruction.lower() or "google" in instruction.lower():
                query = instruction.replace("search", "").replace("for", "").strip()
                await page.goto(f"https://www.google.com/search?q={query}")
            else:
                await page.goto("https://www.google.com")
                
            await page.wait_for_load_state("networkidle")
            
            screenshot_path = "current_view.png"
            await page.screenshot(path=screenshot_path, full_page=False)
            await browser.close()

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"The user wants: {instruction}. I have browsed the web. Provide the final answer and summary.",
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.3
                )
            )
            
            answer = response.text
            save_message("user", instruction)
            save_message("assistant", answer)
            
            yield answer, screenshot_path

    except Exception as e:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=instruction,
            config=types.GenerateContentConfig(tools=[{"google_search": {}}])
        )
        answer = response.text
        save_message("user", instruction)
        save_message("assistant", answer)
        yield answer, None


# --- Gradio Mobile-Friendly UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Web Browser Agent") as demo:
    gr.Markdown("# 🤖 Mobile Web Browser Agent")
    gr.Markdown("Powered by Gemini 2.5 Flash, Playwright, and Persistent Memory.")

    with gr.Row():
        api_key_input = gr.Textbox(
            label="Gemini API Key", 
            type="password", 
            placeholder="Enter your Gemini API Key...",
            value=os.environ.get("GEMINI_API_KEY", "")
        )

    chatbot = gr.Chatbot(
        label="Conversation History",
        value=[(msg["content"] if msg["role"]=="user" else None, msg["content"] if msg["role"]=="assistant" else None) for msg in load_chat_history()],
        height=400,
        type="tuples"
    )

    with gr.Row():
        msg_input = gr.Textbox(
            label="Prompt / Instruction", 
            placeholder="e.g., Search Google for recent AI developments...",
            scale=4
        )
        submit_btn = gr.Button("Run Agent", variant="primary", scale=1)

    with gr.Row():
        browser_status = gr.Textbox(label="Agent Status / Action Log", interactive=False)
        screenshot_output = gr.Image(label="Live Browser View", type="filepath")

    gr.HTML("""
        <script>
        function speakText(text) {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                let utterance = new SpeechSynthesisUtterance(text);
                window.speechSynthesis.speak(utterance);
            } else {
                alert('Text-to-speech not supported on this browser.');
            }
        }
        </script>
    """)

    def handle_interaction(prompt, api_key, history):
        if not prompt.strip():
            return history, "", "Please enter a prompt.", None

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        gen = loop.run_until_complete(
            [res async for res in run_browser_agent(prompt, api_key)]
        )
        
        status_update = gen[-2] if len(gen) >= 2 else "Processing..."
        screenshot = gen[-1] if len(gen) >= 2 else None
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM memory WHERE role='assistant' ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        answer = row[0] if row else "Done."
        updated_history = history + [(prompt, answer)]
        return updated_history, "", status_update, screenshot

    submit_btn.click(
        fn=handle_interaction,
        inputs=[msg_input, api_key_input, chatbot],
        outputs=[chatbot, msg_input, browser_status, screenshot_output]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
