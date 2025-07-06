# Second redeploy attempt
from flask import Flask, request, jsonify
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from house_tool import HousePricePredictorTool  
import os

app = Flask(__name__)

# Initialize LLM and agent
llm = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[HousePricePredictorTool()],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=False
)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = agent.run(user_input)
    return jsonify({"response": response})
if __name__ == "__main__":
    app.run(debug=True)