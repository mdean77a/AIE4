import chainlit as cl
from chainlit.input_widget import Select
import openai
from openai import AsyncOpenAI

@cl.on_chat_start
async def start():
    AI_PROVIDER = "ollama"
    AI_MODEL = "Claudius the Third"

    welcome_message = f"""
# 🪙 Welcome to the University of Utah DCC Clinical Protocol Implementation Assistant!

Hello! I'm your friendly AI assistant specializing in starting up clinical trials. 
I'm here to chat about anything, but I have particular expertise in:

1. 📈 Informed consent document development
2. 🔗 Eligibility checklists
3. 💼 Site initiation processes
4. ⛏️ Looking up CDE definitions for the database
5. 🌐 Developing risk management plans

Feel free to ask me about these topics or anything else you'd like to discuss!

---

**Current Configuration:**
- AI Provider: `{AI_PROVIDER.capitalize()}`
- AI Model: `{AI_MODEL}`

---

💡 **Tip**: Upload the protocol as a PDF document.
"""

    await cl.Message(content=welcome_message).send()

    tasks = await cl.ChatSettings(
        [
            Select(
                id="Task",
                label="To Do List",
                values=["Consent Documents", "ELigibility Checklist", "Site Initiation", "CDE Lookup", "Risk Management"],
                initial_index = None,
            )
        ]
    ).send()
    value = tasks["Task"]
    print(value)