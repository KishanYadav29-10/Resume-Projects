from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from typing import TypedDict  # 📌 Place this at the top of your file

from suggest import suggest_outfit
from memory import update_memory, get_user_preferences

# Define the output schema for parsing
class OutfitInput(BaseModel):
    clothing_item: str = Field(default="")
    color: str = Field(default="")
    occasion: str = Field(default="")

    class Config:
        orm_mode = True  # 🔧 Needed for from_orm to work

# Use the schema for structured output parsing
parser = PydanticOutputParser(pydantic_object=OutfitInput)


# Initialize LLM
llm = ChatOllama(model="mistral")
USER_ID = "kishan"

# Define the prompt
prompt = PromptTemplate.from_template("""
You are a wardrobe assistant. Extract the following details strictly in **valid JSON** format only:
- clothing_item (e.g., shirt, kurta, top)
- color (e.g., red, green, navy blue)
- occasion (e.g., party, interview)
Do not include markdown, explanations, or comments. Only return raw JSON that matches the structure.

Input: "{input}"
{format_instructions}
""").partial(format_instructions=parser.get_format_instructions())


# Node: Extract details
def extract_details(state):
    user_input = state["input"]
    chain = prompt | llm | parser
    parsed_obj = chain.invoke({"input": user_input})

    # ✅ Convert Pydantic object to dict
    parsed = parsed_obj.dict()

    memory = get_user_preferences(USER_ID)
    for key in ["clothing_item", "color", "occasion"]:
        if not parsed.get(key) and memory.get(f"last_{key}"):
            parsed[key] = memory[f"last_{key}"]

    updated_fields = {f"last_{k}": v for k, v in parsed.items() if v}
    update_memory(USER_ID, updated_fields)

    return {**state, **parsed}


# Node: Generate outfit suggestion
def generate_suggestion(state):
    clothing_item = state.get("clothing_item", "")
    color = state.get("color", "")
    occasion = state.get("occasion", "")
    prefs = get_user_preferences(USER_ID)

    suggestion = suggest_outfit(clothing_item, color, occasion, prefs)
    return {**state, "generate_suggestion": suggestion}

# Build LangGraph flow

# 🧠 Define the state structure for LangGraph
class State(TypedDict):
    input: str
    clothing_item: str
    color: str
    occasion: str
    generate_suggestion: str

# ✅ Create the graph with defined state schema
builder = StateGraph(State)
builder.add_node("extract_details", extract_details)
builder.add_node("generate", generate_suggestion)  # ✅ Renamed here

builder.set_entry_point("extract_details")
builder.add_edge("extract_details", "generate")
builder.add_edge("generate", END)


graph = builder.compile()
