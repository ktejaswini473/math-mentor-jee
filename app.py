import streamlit as st
import os
import json
import base64
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import tempfile

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY missing in .env — add valid key and restart.")
    st.stop()

client = OpenAI()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

KB_PATH = "kb_docs.json"
MEMORY_PATH = "memory.jsonl"

with open(KB_PATH) as f:
    docs = json.load(f)
kb_context = "\n\n".join([f"Title: {d['title']}\nContent: {d['content']}" for d in docs])

def save_memory(entry: dict):
    with open(MEMORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

class ProblemState(BaseModel):
    raw_input: str
    input_type: str
    extracted_text: str
    edited_text: Optional[str] = None
    structured_problem: Optional[dict] = None
    solution: Optional[str] = None
    verification: Optional[str] = None
    explanation: Optional[str] = None
    confidence: float = 1.0
    hitl_triggered: bool = False
    feedback: Optional[str] = None

parser_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert math parser. Output ONLY valid JSON with keys: problem_text, topic, variables, constraints, needs_clarification."),
    ("human", "{text}")
])
parser_chain = parser_prompt | llm | StrOutputParser()

solver_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are a JEE math solver. Use this knowledge base:
{kb_context}

Think step-by-step, then give final answer in \\boxed{{}}."""),
    ("human", "{problem}")
])

verifier_prompt = ChatPromptTemplate.from_messages([
    ("system", "Verify the solution for correctness, units, domain, edge cases. Output JSON: {{'verdict': 'correct'/'incorrect'/'unsure', 'comment': '...' }}"),
    ("human", "Problem: {problem}\nSolution: {solution}")
])

explainer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Explain step-by-step like a friendly tutor. Be clear, use LaTeX."),
    ("human", "{problem}\nSolution steps: {solution}")
])

def parse_node(state: ProblemState):
    text = state.edited_text or state.extracted_text
    structured = json.loads(parser_chain.invoke({"text": text}))
    return state.copy(update={
        "structured_problem": structured,
        "hitl_triggered": structured.get("needs_clarification", False)
    })

def solve_node(state: ProblemState):
    solution = (solver_prompt | llm | StrOutputParser()).invoke({
        "problem": state.structured_problem["problem_text"]
    })
    return state.copy(update={"solution": solution})

def verify_node(state: ProblemState):
    result = json.loads((verifier_prompt | llm).invoke({
        "problem": state.structured_problem["problem_text"],
        "solution": state.solution
    }))
    return state.copy(update={
        "verification": result["comment"],
        "confidence": 0.3 if result["verdict"] == "unsure" else 0.9 if result["verdict"] == "correct" else 0.1,
        "hitl_triggered": result["verdict"] == "unsure"
    })

def explain_node(state: ProblemState):
    explanation = (explainer_prompt | llm | StrOutputParser()).invoke({
        "problem": state.structured_problem["problem_text"],
        "solution": state.solution
    })
    return state.copy(update={"explanation": explanation})

workflow = StateGraph(ProblemState)
workflow.add_node("parse", parse_node)
workflow.add_node("solve", solve_node)
workflow.add_node("verify", verify_node)
workflow.add_node("explain", explain_node)

workflow.set_entry_point("parse")
workflow.add_edge("parse", "solve")
workflow.add_edge("solve", "verify")
workflow.add_conditional_edges(
    "verify",
    lambda s: "hitl" if s.hitl_triggered else "explain",
    {"hitl": END, "explain": "explain"}
)
workflow.add_edge("explain", END)

app = workflow.compile()

# UI
st.title("Reliable Multimodal Math Mentor 🚀")

input_mode = st.radio("Input mode", ["Text", "Image", "Audio"])

extracted = ""

if input_mode == "Text":
    raw = st.text_area("Type your math question")
    extracted = raw
elif input_mode == "Image":
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if uploaded:
        bytes_data = uploaded.getvalue()
        base64_image = base64.b64encode(bytes_data).decode('utf-8')
        with st.spinner("Extracting text with GPT-4o vision..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Extract the exact math problem text from the image. Preserve LaTeX if present. Transcribe handwritten accurately."},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
                ],
                max_tokens=500
            )
        extracted = response.choices[0].message.content.strip()
        st.image(uploaded, caption="Uploaded image")
else:  # Audio
    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        with st.spinner("Transcribing with Whisper API..."):
            with open(tmp_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        extracted = transcript.text
        os.unlink(tmp_path)

if extracted:
    st.subheader("Extracted text (edit if needed)")
    edited = st.text_area("Edit extracted text", value=extracted, height=200)
    if st.button("Solve"):
        with st.spinner("Running agents..."):
            initial_state = ProblemState(
                raw_input="" if input_mode == "Text" else uploaded.name,
                input_type=input_mode.lower(),
                extracted_text=extracted,
                edited_text=edited if edited != extracted else None
            )
            result = app.invoke(initial_state)

        if result.hitl_triggered:
            st.warning("HITL triggered — low confidence or ambiguity detected.")
            st.info("In production, route to human reviewer here.")

        st.subheader("Structured Problem")
        st.json(result.structured_problem)

        st.subheader("Knowledge Base Used")
        st.text(kb_context)

        st.subheader(f"Final Answer (Confidence: {result.confidence:.2f})")
        st.markdown(result.solution)

        st.subheader("Step-by-Step Explanation")
        st.markdown(result.explanation)

        st.subheader("Feedback")
        col1, col2 = st.columns(2)
        if col1.button("✅ Correct"):
            save_memory({
                "problem_text": result.structured_problem["problem_text"],
                "solution": result.solution,
                "explanation": result.explanation,
                "feedback": "correct"
            })
            st.success("Saved as positive example")
        if col2.button("❌ Incorrect"):
            comment = st.text_input("What was wrong?")
            if st.button("Submit correction"):
                save_memory({
                    "problem_text": result.structured_problem["problem_text"],
                    "solution": result.solution,
                    "explanation": result.explanation,
                    "feedback": "incorrect",
                    "correction_comment": comment
                })
                st.success("Saved correction — future solutions will improve")
