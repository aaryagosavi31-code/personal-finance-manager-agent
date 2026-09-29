import os
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Field, SQLModel, create_engine, Session, select
from google import genai
from google.genai import types

load_dotenv()

# ---------------------------------------------------------
# 1. Database Model Setup (SQLite)
# ---------------------------------------------------------
class Expense(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}  # Fixes the Python 3.13 reloader crash
    id: Optional[int] = Field(default=None, primary_key=True)
    amount: float
    category: str
    description: str

sqlite_file_name = "finance.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url)

# Modern FastAPI Lifespan Handler (Replaces @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    yield

# ---------------------------------------------------------
# 2. Database Action Functions (Executed by Agent)
# ---------------------------------------------------------
def add_expense(amount: float, category: str, description: str) -> dict:
    """Logs a new financial expense into the database."""
    with Session(engine) as session:
        expense = Expense(amount=amount, category=category.capitalize(), description=description)
        session.add(expense)
        session.commit()
        session.refresh(expense)
        return {
            "status": "success",
            "message": f"Logged ${amount:.2f} under '{category}' ({description}).",
            "id": expense.id
        }

def get_summary(category: Optional[str] = None) -> dict:
    """Retrieves spending summaries, totals, or filtered expense lists."""
    with Session(engine) as session:
        statement = select(Expense)
        if category:
            statement = statement.where(Expense.category.ilike(f"%{category}%"))
        
        results = session.exec(statement).all()
        total = sum(item.amount for item in results)
        
        return {
            "status": "success",
            "total_spent": total,
            "count": len(results),
            "items": [item.model_dump() for item in results]
        }

TOOL_MAP = {
    "add_expense": add_expense,
    "get_summary": get_summary
}

# ---------------------------------------------------------
# 3. Gemini Client & Tools Setup
# ---------------------------------------------------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tools_list = [add_expense, get_summary]

# ---------------------------------------------------------
# 4. FastAPI Setup & Chat Route
# ---------------------------------------------------------
app = FastAPI(title="Personal Finance Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=request.message,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a helpful Personal Finance Assistant. "
                    "When users mention spending money, use the add_expense tool. "
                    "When users ask about totals, summaries, or past spending, use get_summary."
                ),
                tools=tools_list,
            )
        )

        if response.function_calls:
            call = response.function_calls[0]
            fn_name = call.name
            fn_args = call.args

            if fn_name in TOOL_MAP:
                tool_result = TOOL_MAP[fn_name](**fn_args)
                return {
                    "reply": tool_result.get("message", f"Total spent: ${tool_result.get('total_spent', 0):.2f}"),
                    "tool_called": fn_name,
                    "data": tool_result
                }

        return {"reply": response.text, "tool_called": None}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)