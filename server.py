import random
from pathlib import Path
import os
import json
import pandas as pd
import requests
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger
from dotenv import load_dotenv
load_dotenv()
import traceback
logger = get_logger(__name__)

logger.info('Starting MCP MCP Server')

host = 'localhost'
port = 8000
transport = 'sse' # server sent event
mcp = FastMCP('my-mcp', host=host, port=port)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_csv_agent
import json


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, max_tokens=4096)


@mcp.tool()
def csv_query_agent(filenames: list[str], query: str, structure_output: dict) -> str:
    """This tool is for calculate a <operation> b

    Args:
        filenames: list of file.csv
        query: CSV query
        structure_output: format to response
    """
    try:
        # Load the CSV agent
        agent_executor = create_csv_agent(
            llm,
            # ["/content/warehouse_agent.csv"],
            ["/src/Inbound.csv", "/src/Outbound.csv", "/src/Inventory.csv"],
            agent_type="zero-shot-react-description",
            verbose=True,
            allow_dangerous_code=True
        )

        prompt = f"""
        Query: {query}

        Goal:
        1. Generate csv query to get enough information for the following query
        2. If facing some error, try to fix step by step before response.
        3. Try to print the all final result variable for traceability in single print() function using string format to concatenate all result at once.
        4. Response with structure_output

        Remember to answer with this follow structure json format:
        {structure_output}
        """
        response = agent_executor.invoke(prompt)
        response['output'] = json.loads(response['output'].replace("```json", "").replace("```", "").strip())
        print(json.dumps(response, indent = 2, ensure_ascii=False))
        return response['output']
    
    except Exception as e:
        print("Error: ", e)
        traceback.print_exc()
    

logger.info(
        f'MCP MCP Server at {host}:{port} and transport {transport}'
)

mcp.run(transport=transport)
