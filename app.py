import os
import gradio as gr
import google.genai as genai

# Load Gemini API Key from environment
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("⚠️ GEMINI_API_KEY is not set. Please add it in the environment or Hugging Face secrets.")

# Initialize Gemini Client
client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = """
You are a helpful Medical & Diet Assistant.

Rules:
- You provide ONLY general health education and diet/lifestyle information.
- You do NOT diagnose any condition.
- You do NOT prescribe or adjust medications, tests, or treatments.
- If the user mentions emergency symptoms (e.g., chest pain, breathing difficulty, severe pain,
  suicidal thoughts, sudden weakness on one side of the body, confusion, or blood in vomit/stool),
  tell them clearly to seek immediate medical help or go to the nearest hospital.
- Always encourage consulting a real doctor for any serious or persistent issue.
- At the end of every answer, add this line:
  "Note: I am an AI assistant and not a substitute for a doctor. Please consult a healthcare professional."
- Use simple language and, when helpful, give examples using common Indian foods
  (idli, dosa, rice, sambar, chapati, curd, etc.).
"""

def medidiet_assistant(user_message: str, mode: str) -> str:
    if not user_message or user_message.strip() == "":
        return "Please enter your question or details."

    full_prompt = (
        f"Mode: {mode}\n"
        f"User message: {user_message}\n\n"
        "Follow the system rules strictly."
    )

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[SYSTEM_PROMPT, full_prompt],
        )
        return resp.text
    except Exception as e:
        return f"Error contacting Gemini model: {e}"

def chat_fn(user_message, mode):
    return medidiet_assistant(user_message, mode)

with gr.Blocks() as demo:
    gr.Markdown("## 🩺 MediDiet Assist (Gemini)\nGeneral health & diet guidance assistant.")

    mode = gr.Dropdown(
        ["health_qa", "diet_plan", "food_check"],
        value="health_qa",
        label="Mode"
    )

    user_input = gr.Textbox(
        lines=5,
        label="Ask your question",
        placeholder="Eg: I have acidity at night, what can I change in my diet?"
    )

    output = gr.Textbox(
        label="Assistant Reply",
        lines=12
    )

    submit_btn = gr.Button("Ask MediDiet")

    submit_btn.click(
        fn=chat_fn,
        inputs=[user_input, mode],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
