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
import time
from typing import List, Dict, Any, Optional

logger = get_logger(__name__)

logger.info('Starting MCP MCP Server')

host = 'localhost'
port = 8000
transport = 'sse' # server sent event
mcp = FastMCP('my-mcp', host=host, port=port)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import json

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, max_tokens=4096)

df_template = """```python
{df_name}.head().to_markdown()
>>> {df_head}
``` """

# System prompt for better reasoning
SYSTEM_PROMPT =    """
You are an expert data analyst working with pandas DataFrames. 
You have access to a number of pandas dataframes. 
Here is a sample of rows from each dataframe and the python code that was used to generate the sample:
{df_context}

Your task is to:

1. Analyze the query carefully and understand what information is needed
2. Generate appropriate pandas operations to extract the required data (Don't assume you have access to any libraries other than built-in Python ones and pandas. )
3. If the query involves time periods (like "monthly", "weekly"), ensure you properly group and aggregate the data for the ENTIRE requested period
4. Always explain your reasoning step by step, especially when dealing with time-based queries
5. If your initial result seems incomplete or doesn't match the expected time period, investigate why and adjust your approach
6. Provide detailed explanations for any data limitations or filtering decisions
7. When dealing with large datasets, summarize key insights while maintaining accuracy
8. Use only existed dataframe and do not mock up any Sample data.
"""

FEW_SHOT = """
Here is possible scenario to generate different type of chart.

Example Output for generate bar chart:
{{
  "explanation": "your explaination of final answer and how you retrive the data"
  "type": "bar",
  "data": {{
    "labels": ["26-May", "2-Jun", "9-Jun"],
    "datasets": [
      {{
        "label": "Total Inbound",
        "data": [8000, 12000, 9500],
        "backgroundColor": "#5bc0de",
        "stack": "stack1"
      }},
      {{
        "label": "Max Capacity (MT)",
        "data": [50000, 50000, 50000],
        "borderColor": "#d9534f",
      }}
    ]
  }}
}}

Example Output for generate bar and line chart in the same chart:
{{
  "explanation": "your explaination of final answer and how you retrive the data"
  "type": "bar",
  "data": {{
    "labels": ["26-May", "2-Jun", "9-Jun"],
    "datasets": [
      {{
        "label": "Total Inbound",
        "data": [8000, 12000, 9500],
        "backgroundColor": "#5bc0de",
        "stack": "stack1"
      }},
      {{
        "label": "Max Capacity (MT)",
        "data": [50000, 50000, 50000],
        "type": "line",
        "borderColor": "#d9534f",
        "fill": false
      }}
    ]
  }}
}}
"""

def check_dataframe_size(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Check dataframe size and provide recommendations"""
    rows, cols = df.shape
    memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    
    size_info = {
        'name': name,
        'rows': rows,
        'columns': cols,
        'memory_mb': round(memory_usage, 2),
        'is_large': rows > 10000 or memory_usage > 50,
        'recommendations': []
    }
    
    if rows > 50000:
        size_info['recommendations'].append("Consider sampling or chunking for better performance")
    if memory_usage > 100:
        size_info['recommendations'].append("High memory usage - consider data type optimization")
    if cols > 50:
        size_info['recommendations'].append("Many columns - consider selecting only relevant ones")
    
    return size_info

def load_and_analyze_dataframes(filenames: List[str]) -> tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
    """Load dataframes and analyze their sizes"""
    dataframes = []
    size_analyses = []
    df_context_list = []
    
    for filename in filenames:
        try:
            print("filename: ", filename)
            df = pd.read_csv(filename)
            print(df.head())
            dataframes.append(df)
            df_context_list.append(df_template.format(df_head=df.head().to_markdown(), df_name=filename.split(".")[0]))
            
            # Analyze size
            size_info = check_dataframe_size(df, filename)
            size_analyses.append(size_info)
            
            logger.info(f"Loaded {filename}: {size_info['rows']} rows, {size_info['columns']} cols, {size_info['memory_mb']} MB")
            
            # Log warnings for large datasets
            if size_info['is_large']:
                logger.warning(f"Large dataset detected: {filename}")
                for rec in size_info['recommendations']:
                    logger.warning(f"  Recommendation: {rec}")
                    
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise
    df_context = "\n\n".join(df_context_list)
    return dataframes, size_analyses, df_context

def execute_with_retry(agent_executor, prompt: str, max_retries: int = 3, delay: float = 1.0) -> Dict[str, Any]:
    """Execute agent with retry logic"""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Executing query (attempt {attempt + 1}/{max_retries})")
            response = agent_executor.invoke(prompt)
            return response
            
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logger.error(f"All {max_retries} attempts failed")
                
    # If all retries failed, raise the last exception
    raise last_exception

@mcp.tool()
def pandas_query_agent(filenames: list[str], query: str) -> str:
    """Enhanced pandas agent for querying multiple CSV files with better error handling and reasoning
    
    Args:
        filenames: List of CSV file paths
        query: Natural language query about the data
        
    Returns:
        JSON string with structured results of explaination and chart to be plot
    """
    try:
        # Use default files if none provided
        if not filenames or True:
            filenames = ["./src/Inbound.csv", "./src/Outbound.csv", "./src/Inventory.csv"]
        
        # Load and analyze dataframes
        logger.info(f"Loading dataframes from: {filenames}")
        dataframes, size_analyses, df_context = load_and_analyze_dataframes(filenames)
        
        # Log size analysis
        total_rows = sum(analysis['rows'] for analysis in size_analyses)
        total_memory = sum(analysis['memory_mb'] for analysis in size_analyses)
        
        logger.info(f"Total dataset: {total_rows} rows, {total_memory:.2f} MB")
        
        # Check if we need to handle large datasets differently
        has_large_datasets = any(analysis['is_large'] for analysis in size_analyses)
        if has_large_datasets:
            logger.warning("Large datasets detected - performance may be impacted")
        
        final_prompt = SYSTEM_PROMPT.format(df_context=df_context)+"\n"+FEW_SHOT
        print(final_prompt)
        # Create pandas agent with enhanced configuration
        agent_executor = create_pandas_dataframe_agent(
            llm,
            dataframes,
            agent_type="tool-calling",  # Changed from zero-shot-react-description
            verbose=True,
            # handle_parsing_errors=True,  # Added for better error handling
            suffix=final_prompt,  # Added system prompt
            return_intermediate_steps=True,  # For better debugging
            max_iterations=10,  # Prevent infinite loops
            max_execution_time=300,  # 5 minute timeout
            allow_dangerous_code=True
        )
        
        # Execute with retry logic
        response = execute_with_retry(agent_executor,query, max_retries=3)
        print(response)
        # Process response
        try:
            # Extract the output and clean it
            output = response.get('output', '')
            
            # Try to parse JSON from output
            if isinstance(output, str):
                # Remove markdown formatting if present
                cleaned_output = output.replace("```json", "").replace("```", "").strip()
                
                # Try to find JSON in the output
                import re
                json_match = re.search(r'\{.*\}', cleaned_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_output = json.loads(json_str)
                else:
                    # If no JSON found, create a structured response
                    parsed_output = {
                        "result": cleaned_output,
                        "reasoning": "Output was not in JSON format",
                        "metadata": {
                            "total_rows_processed": total_rows,
                            "total_memory_mb": total_memory,
                            "large_datasets": has_large_datasets
                        }
                    }
            else:
                parsed_output = output
            
            # Add metadata to response
            if isinstance(parsed_output, dict):
                parsed_output['metadata'] = {
                    "execution_info": {
                        "total_rows_processed": total_rows,
                        "total_memory_mb": total_memory,
                        "large_datasets": has_large_datasets,
                        "files_processed": filenames
                    },
                    "size_analysis": size_analyses
                }
            
            # Log the final response
            logger.info("Query executed successfully")
            logger.info(f"Response structure: {type(parsed_output)}")
            
            return json.dumps(parsed_output, indent=2, ensure_ascii=False)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            # Return the raw output with error information
            error_response = {
                "error": "JSON parsing failed",
                "raw_output": str(response.get('output', '')),
                "parsing_error": str(e),
                "metadata": {
                    "total_rows_processed": total_rows,
                    "files_processed": filenames
                }
            }
            return json.dumps(error_response, indent=2, ensure_ascii=False)
    
    except Exception as e:
        logger.error(f"Error in pandas_query_agent: {e}")
        traceback.print_exc()
        
        # Return structured error response
        error_response = {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "query": query,
            "files": filenames
        }
        return json.dumps(error_response, indent=2, ensure_ascii=False)

# Keep the old function for backward compatibility
# @mcp.tool()
# def csv_query_agent(filenames: list[str], query: str) -> str:
#     """Legacy CSV agent - deprecated, use pandas_query_agent instead"""
#     logger.warning("csv_query_agent is deprecated. Use pandas_query_agent instead.")
#     return pandas_query_agent(filenames, query)

logger.info(f'MCP MCP Server at {host}:{port} and transport {transport}')

mcp.run(transport=transport)