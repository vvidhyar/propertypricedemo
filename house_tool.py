from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import requests
import json
import os

# Define the input schema for the tool
class HouseFeatures(BaseModel):
    city: str
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: float
    view: float
    condition: float
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: float

class HousePricePredictorTool(BaseTool):
    name: str = "HousePricePredictor"
    description: str = "Predicts the house price based on its features"
    args_schema: Type[BaseModel] = HouseFeatures

    def _run(self, **kwargs):
        url = "https://adb-1805611446428304.4.azuredatabricks.net/serving-endpoints/housepricedisplayinchatbot/invocations"
        headers = {
            "Authorization": f"Bearer {os.getenv('DATABRICKS_TOKEN')}",
            "Content-Type": "application/json"
        }
        defaults = {
                "floors": 1.0,
                "waterfront": 0.0,
                "view": 0.0,
                "condition": 3.0,
                "sqft_above": kwargs.get("sqft_living", 1800),
                "sqft_basement": 0,
                "yr_built": 2000,
                "yr_renovated": 0.0
            }
        for key, value in defaults.items():
            kwargs.setdefault(key, value)    
        payload = {
            "dataframe_split": {
                "columns": list(kwargs.keys()),
                "data": [list(kwargs.values())]
            }
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            prediction = response.json().get("predictions", [None])[0]

            if prediction is not None:
                return f"The estimated house price is ${prediction:,.2f}"
            else:
                return "Prediction failed: No prediction returned from model."
        except Exception as e:
            return f"Unexpected error occurred: {str(e)}"
    def _arun(self, **kwargs):
        raise NotImplementedError("Async not implemented")
from langchain.agents import initialize_agent, AgentType
#from langchain.agents import create_openai_functions_agent, AgentExecutor   #use this when initialize_agent gets deprecated
from langchain.chat_models import ChatOpenAI
#from langchain_openai import ChatOpenAI   #use this when langchain_openai gets deprecated
# Initialize your OpenAI model
llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[HousePricePredictorTool()],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
