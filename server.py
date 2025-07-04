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
import io
import sys

logger = get_logger(__name__)

logger.info('Starting MCP MCP Server')

host = 'localhost'
port = 8000
transport = 'sse' # server sent event
mcp = FastMCP('my-mcp', host=host, port=port)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.schema import HumanMessage
import json
import google.generativeai as genai

# Hardcoded API Key (not recommended for production)
GOOGLE_API_KEY = "GOOGLE API KEY"
# Configure Google API
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found")
    raise ValueError("GOOGLE_API_KEY is required")

# Configure the API key for google-generativeai
genai.configure(api_key=GOOGLE_API_KEY)

# Test API Key first
try:
    test_model = genai.GenerativeModel('gemini-2.0-flash-exp')
    test_response = test_model.generate_content("Hello")
    logger.info("Google API Key validated successfully")
except Exception as e:
    logger.error(f"Google API Key validation failed: {e}")
    logger.info("Trying alternative model...")
    try:
        test_model = genai.GenerativeModel('gemini-1.5-flash')
        test_response = test_model.generate_content("Hello")
        logger.info("Alternative model works")
    except Exception as e2:
        logger.error(f"All models failed: {e2}")
        raise

# Initialize LLM with proper configuration
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", 
        temperature=0.4, 
        max_tokens=4096,
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True
    )
    logger.info("LLM initialized with gemini-2.0-flash-exp")
except Exception as e:
    logger.warning(f"Failed to initialize with gemini-2.0-flash-exp: {e}")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.4, 
            max_tokens=4096,
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True
        )
        logger.info("LLM initialized with gemini-1.5-flash")
    except Exception as e2:
        logger.error(f"Failed to initialize LLM with any model: {e2}")
        raise

# Template for DataFrame context
df_template = """
{df_name}.head().to_markdown()
{df_head}

{df_name}.info()
{df_info}

{df_name}.dtypes
{df_dtypes}
"""

def get_df_info_string(df):
    """Get DataFrame info as string"""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

# Enhanced system prompt with DataFrame context
ENHANCED_SYSTEM_PROMPT = """You have access to pandas dataframes with the following structure:

{df_context}

Given a user question about the dataframes, write Python code to answer it.

IMPORTANT RULES:
1. Use only the dataframe variables shown above (inbound_df, outbound_df, inventory_df)
2. Don't assume you have access to any libraries other than pandas and built-in Python
3. Always store your final answer in a variable called 'result' 
4. Always print(result) at the end
5. Write concise code without comments
6. Use only the existing dataframes - do not create mock data
7. Handle date filtering carefully using pandas datetime operations
8. For time-based queries, ensure you capture the full requested period

When dealing with time periods:
- Use pd.to_datetime() if dates aren't already datetime objects
- Use proper date filtering with .loc or boolean indexing
- For monthly/weekly aggregations, use .groupby() with appropriate time periods
- Always verify your date filtering logic covers the entire requested period

Data Quality Notes:
- inventory_df has a typo: 'UNRESRICTED_STOCK' (missing 'T')
- inventory_df currency column may be truncated as 'CURRENC'
- Date formats are typically YYYY/MM/DD
"""

def check_dataframe_size(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Check dataframe size and provide recommendations"""
    rows, cols = df.shape
    memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    
    size_info = {
        'name': str(name),                    
        'rows': int(rows),                   
        'columns': int(cols),                 
        'memory_mb': float(round(memory_usage, 2)),  
        'is_large': bool(rows > 10000 or memory_usage > 50),  
        'recommendations': []
    }
    
    if rows > 50000:
        size_info['recommendations'].append("Consider sampling or chunking for better performance")
    if memory_usage > 100:
        size_info['recommendations'].append("High memory usage - consider data type optimization")
    if cols > 50:
        size_info['recommendations'].append("Many columns - consider selecting only relevant ones")
    
    return size_info

def make_json_safe(obj):
    """Convert any object to be JSON serializable"""
    import numpy as np
    import pandas as pd
    
    if obj is None:
        return None
    elif isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    else:
        return str(obj)  # Convert everything else to string

def analyze_result_data_usage(result, execution_env):
    """Analyze what data was actually used in the result"""
    analysis = {
        "result_type": type(result).__name__,
        "result_size": None,
        "data_sources_used": [],
        "estimated_rows_in_result": None
    }
    
    try:
        if isinstance(result, pd.DataFrame):
            analysis["result_size"] = [int(result.shape[0]), int(result.shape[1])]
            analysis["estimated_rows_in_result"] = int(len(result))
            analysis["columns_in_result"] = result.columns.tolist()
            
            # Try to identify which source DataFrames might have contributed
            for df_name in execution_env.keys():
                if df_name.endswith('_df') and df_name in execution_env:
                    source_df = execution_env[df_name]
                    if isinstance(source_df, pd.DataFrame) and not source_df.empty:
                        # Check if result columns match source columns
                        common_columns = set(result.columns) & set(source_df.columns)
                        if common_columns:
                            analysis["data_sources_used"].append({
                                "dataframe": df_name,
                                "common_columns": list(common_columns),
                                "source_file": execution_env.get('_file_mapping', {}).get(df_name, {}).get('source_file', 'unknown')
                            })
        
        elif isinstance(result, (int, float)):
            analysis["result_size"] = "scalar"
            analysis["estimated_rows_in_result"] = 1
            analysis["result_value"] = float(result) if isinstance(result, (int, float)) else str(result)
            
        elif isinstance(result, dict):
            analysis["result_size"] = len(result)
            analysis["result_keys"] = list(result.keys())
            
        elif isinstance(result, list):
            analysis["result_size"] = len(result)
            analysis["estimated_rows_in_result"] = len(result)
    
    except Exception as e:
        analysis["analysis_error"] = str(e)
    
    return analysis

def extract_operations_from_script(script):
    """Extract pandas operations from the script"""
    operations = []
    
    # Common pandas operations to look for
    operation_patterns = {
        'groupby': r'\.groupby\(',
        'merge': r'\.merge\(',
        'join': r'\.join\(',
        'filter': r'\[.*\]',
        'sort': r'\.sort_values\(',
        'aggregate': r'\.(sum|mean|count|max|min|std)\(',
        'datetime_filter': r'\.dt\.',
        'pivot': r'\.pivot',
        'resample': r'\.resample\('
    }
    
    import re
    for op_name, pattern in operation_patterns.items():
        if re.search(pattern, script):
            operations.append(op_name)
    
    return operations

def determine_dataframe_name(filename):
    """Determine the DataFrame name from filename"""
    df_name_mapping = {
        ".venv/bootcathon-2025/src/Inbound.csv": "inbound_df",
        ".venv/bootcathon-2025/src/Outbound.csv": "outbound_df", 
        ".venv/bootcathon-2025/src/Inventory.csv": "inventory_df",
        "Inbound.csv": "inbound_df",
        "Outbound.csv": "outbound_df",
        "Inventory.csv": "inventory_df",
        "Inbound": "inbound_df",
        "Outbound": "outbound_df",
        "Inventory": "inventory_df"
    }
    
    if filename in df_name_mapping:
        return df_name_mapping[filename]
    
    base_filename = os.path.basename(filename)
    if base_filename in df_name_mapping:
        return df_name_mapping[base_filename]
    
    # Fallback logic
    lower_filename = filename.lower()
    if 'inbound' in lower_filename:
        return "inbound_df"
    elif 'outbound' in lower_filename:
        return "outbound_df"
    elif 'inventory' in lower_filename:
        return "inventory_df"
    else:
        return os.path.splitext(base_filename)[0] + "_df"

def estimate_query_complexity(query):
    """Estimate query complexity based on keywords"""
    complexity_indicators = {
        'simple': ['total', 'sum', 'count', 'average', 'show'],
        'medium': ['group', 'filter', 'where', 'by month', 'by day', 'aggregate'],
        'complex': ['join', 'merge', 'pivot', 'correlation', 'trend', 'forecast']
    }
    
    query_lower = query.lower()
    
    if any(indicator in query_lower for indicator in complexity_indicators['complex']):
        return "complex"
    elif any(indicator in query_lower for indicator in complexity_indicators['medium']):
        return "medium"
    else:
        return "simple"

def load_and_analyze_dataframes(filenames: List[str]) -> tuple[List[pd.DataFrame], List[Dict[str, Any]], str]:
    """Load dataframes and analyze their sizes with enhanced context"""
    dataframes = []
    size_analyses = []
    df_context_list = []
    
    # Enhanced DataFrame name mapping
    df_name_mapping = {
        # Full paths
        ".venv/bootcathon-2025/src/Inbound.csv": "inbound_df",
        ".venv/bootcathon-2025/src/Outbound.csv": "outbound_df", 
        ".venv/bootcathon-2025/src/Inventory.csv": "inventory_df",
        "./src/Inbound.csv": "inbound_df",
        "./src/Outbound.csv": "outbound_df", 
        "./src/Inventory.csv": "inventory_df",
        # Just filenames
        "Inbound.csv": "inbound_df",
        "Outbound.csv": "outbound_df",
        "Inventory.csv": "inventory_df",
        # Without extensions (fix for the current issue)
        "Inbound": "inbound_df",
        "Outbound": "outbound_df",
        "Inventory": "inventory_df"
    }
    
    for filename in filenames:
        try:
            # Add .csv extension if missing
            if not filename.endswith('.csv'):
                filename_with_ext = filename + '.csv'
                logger.info(f"Adding .csv extension: {filename} -> {filename_with_ext}")
            else:
                filename_with_ext = filename
            
            # Try different possible paths for the file
            possible_paths = [
                filename,  # Original
                filename_with_ext,  # With .csv extension
                f"./src/{os.path.basename(filename_with_ext)}",
                f".venv/bootcathon-2025/src/{os.path.basename(filename_with_ext)}",
                f"/home/gat/Documents/bootcathon2025/.venv/bootcathon-2025/src/{os.path.basename(filename_with_ext)}",
                f"/home/gat/Documents/bootcathon2025/src/{os.path.basename(filename_with_ext)}"
            ]
            
            df = None
            actual_path = None
            
            logger.info(f"Searching for file: {filename}")
            for path in possible_paths:
                logger.debug(f"  Trying: {path}")
                if os.path.exists(path):
                    logger.info(f"  Found file at: {path}")
                    df = pd.read_csv(path)
                    actual_path = path
                    break
            
            if df is None:
                logger.warning(f"File not found: {filename}")
                logger.info(f"Searched paths: {possible_paths}")
                
                # List what files ARE available in each directory
                for search_dir in [".", "./src/", ".venv/bootcathon-2025/src/"]:
                    if os.path.exists(search_dir):
                        available_files = [f for f in os.listdir(search_dir) if f.endswith('.csv')]
                        if available_files:
                            logger.info(f"Available CSV files in {search_dir}: {available_files}")
                continue
            
            # Convert date columns to datetime if they exist
            date_columns = [col for col in df.columns if 'DATE' in col.upper()]
            for date_col in date_columns:
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    logger.info(f"Converted {date_col} to datetime")
                except Exception as date_error:
                    logger.warning(f"Could not convert {date_col} to datetime: {date_error}")
            
            dataframes.append(df)
            
            # Get proper DataFrame name using improved logic
            df_name = None
            
            # Try exact match first (including both original filename and with extension)
            for name_to_try in [filename, filename_with_ext, actual_path]:
                if name_to_try in df_name_mapping:
                    df_name = df_name_mapping[name_to_try]
                    break
            
            if df_name is None:
                # Try basename match
                base_filename = os.path.basename(actual_path)
                base_without_ext = os.path.splitext(base_filename)[0]
                
                if base_filename in df_name_mapping:
                    df_name = df_name_mapping[base_filename]
                elif base_without_ext in df_name_mapping:
                    df_name = df_name_mapping[base_without_ext]
                else:
                    # Check if filename contains key terms
                    lower_filename = actual_path.lower()
                    if 'inbound' in lower_filename:
                        df_name = "inbound_df"
                    elif 'outbound' in lower_filename:
                        df_name = "outbound_df"
                    elif 'inventory' in lower_filename:
                        df_name = "inventory_df"
                    else:
                        df_name = base_without_ext + "_df"
            
            logger.info(f"Mapped {filename} -> {df_name} (loaded from {actual_path})")
            
            # Create context with sample data and structure info
            df_context = df_template.format(
                df_name=df_name,
                df_head=df.head().to_markdown(),
                df_info=get_df_info_string(df),
                df_dtypes=df.dtypes.to_string()
            )
            df_context_list.append(df_context)
            
            # Analyze size
            size_info = check_dataframe_size(df, actual_path)
            size_analyses.append(size_info)
            
            logger.info(f"Loaded {actual_path}: {size_info['rows']} rows, {size_info['columns']} cols, {size_info['memory_mb']} MB")
            
            # Log warnings for large datasets
            if size_info['is_large']:
                logger.warning(f"Large dataset detected: {actual_path}")
                for rec in size_info['recommendations']:
                    logger.warning(f"  Recommendation: {rec}")
                    
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            continue
    
    if not dataframes:
        logger.error("No valid dataframes could be loaded!")
        logger.info("Run list_available_files() to see what CSV files are actually available")
        raise ValueError("No valid dataframes could be loaded")
    
    # Combine all DataFrame contexts
    combined_context = "\n\n".join(df_context_list)
    
    return dataframes, size_analyses, combined_context

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
def pandas_query_agent(filenames: list[str], query: str, structure_output: dict) -> str:
    """Enhanced pandas agent with DataFrame context for better code generation
    
    Args:
        filenames: List of CSV file paths (optional, defaults to standard files)
        query: Natural language query about the data
        structure_output: Expected JSON structure for the response
        
    Returns:
        JSON string with structured results
    """
    try:
        # Use default files if none provided
        if not filenames:
            filenames = [".venv/bootcathon-2025/src/Inbound.csv", ".venv/bootcathon-2025/src/Outbound.csv", ".venv/bootcathon-2025/src/Inventory.csv"]

        # Load and analyze dataframes with context
        logger.info(f"Loading dataframes from: {filenames}")
        dataframes, size_analyses, df_context = load_and_analyze_dataframes(filenames)
        
        # Create enhanced system prompt with DataFrame context
        enhanced_system_prompt = ENHANCED_SYSTEM_PROMPT.format(df_context=df_context)
        
        # Log size analysis
        total_rows = sum(analysis['rows'] for analysis in size_analyses)
        total_memory = sum(analysis['memory_mb'] for analysis in size_analyses)
        
        logger.info(f"Total dataset: {total_rows} rows, {total_memory:.2f} MB")
        
        # Check if we need to handle large datasets differently
        has_large_datasets = any(analysis['is_large'] for analysis in size_analyses)
        if has_large_datasets:
            logger.warning("Large datasets detected - performance may be impacted")
        
        # Create pandas agent with multiple fallback strategies
        agent_executor = None
        
        # Strategy 1: Try tool-calling with full features
        try:
            agent_executor = create_pandas_dataframe_agent(
                llm,
                dataframes,
                agent_type="tool-calling",
                verbose=True,
                prefix=enhanced_system_prompt,  # Use prefix instead of suffix
                return_intermediate_steps=True,
                max_iterations=8,
                max_execution_time=300,
                allow_dangerous_code=True,
            )
            logger.info("Created pandas agent with tool-calling and DataFrame context")
        except Exception as e:
            logger.warning(f"Failed to create tool-calling agent: {e}")
            
        # Strategy 2: Try zero-shot-react-description (more compatible)
        if agent_executor is None:
            try:
                agent_executor = create_pandas_dataframe_agent(
                    llm,
                    dataframes,
                    agent_type="zero-shot-react-description",
                    verbose=True,
                    prefix=enhanced_system_prompt,
                    max_iterations=6,
                    allow_dangerous_code=True,
                )
                logger.info("Created pandas agent with zero-shot-react-description")
            except Exception as e:
                logger.warning(f"Failed to create zero-shot agent: {e}")
        
        # Strategy 3: Minimal configuration fallback
        if agent_executor is None:
            try:
                agent_executor = create_pandas_dataframe_agent(
                    llm,
                    dataframes,
                    verbose=True,
                    max_iterations=5,
                    allow_dangerous_code=True,
                )
                logger.info("Created pandas agent with minimal configuration")
            except Exception as e:
                logger.error(f"Failed to create any pandas agent: {e}")
                raise
        
        # Enhanced query with structure requirements and context awareness
        enhanced_query = f"""
        DataFrame Context Available: You have access to properly structured DataFrames with sample data and schema information.
        
        User Query: {query}
        
        Expected Output Structure: {json.dumps(structure_output, indent=2)}
        
        Dataset Information:
        {json.dumps(size_analyses, indent=2)}
        
        Instructions:
        1. Analyze the query carefully using the DataFrame context provided
        2. Identify which dataframes are needed (inbound_df, outbound_df, inventory_df)
        3. Write efficient pandas code using the exact column names shown in the context
        4. Handle date filtering properly - ensure full time period coverage
        5. Store your final answer in 'result' variable
        6. Always print(result) at the end
        7. If dealing with time periods, verify you're capturing the complete requested timeframe
        8. Use proper aggregation methods for summary statistics
        
        Remember: You have full DataFrame schema and sample data available - use it!
        """
        
        # Execute with retry logic
        response = execute_with_retry(agent_executor, enhanced_query, max_retries=3)
        
        # Process response
        try:
            # Extract the output and clean it
            output = response.get('output', '')
            
            # Try to parse JSON from output
            if isinstance(output, str):
                # Remove markdown formatting if present
                cleaned_output = output.replace("```json", "").replace("```python", "").replace("```", "").strip()
                
                # Try to find JSON in the output
                import re
                json_match = re.search(r'\{.*\}', cleaned_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    try:
                        parsed_output = json.loads(json_str)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try to extract key information
                        parsed_output = {
                            "result": cleaned_output,
                            "reasoning": "Pandas analysis completed with DataFrame context",
                            "query": query,
                            "execution_method": "context_aware_pandas_agent"
                        }
                else:
                    # If no JSON found, create a structured response
                    parsed_output = {
                        "result": cleaned_output,
                        "reasoning": "Analysis completed using enhanced DataFrame context",
                        "query": query,
                        "raw_output": output,
                        "execution_method": "context_aware_pandas_agent"
                    }
            else:
                parsed_output = output
            
            # Add comprehensive metadata to response
            if isinstance(parsed_output, dict):
                # Make size_analyses JSON safe before adding to metadata
                safe_size_analyses = make_json_safe(size_analyses)
                
                # Track file-to-dataframe mapping
                file_mapping = {}
                for i, filename in enumerate(filenames):
                    if i < len(dataframes):
                        df_name = determine_dataframe_name(filename)
                        file_mapping[df_name] = {
                            "source_file": filename,
                            "file_path": os.path.abspath(filename) if os.path.exists(filename) else filename,
                            "rows": int(len(dataframes[i])),
                            "columns": int(len(dataframes[i].columns)),
                            "column_names": dataframes[i].columns.tolist(),
                            "sample_data": dataframes[i].head(2).to_dict('records'),
                            "file_size_mb": float(round(os.path.getsize(filename) / 1024 / 1024, 2)) if os.path.exists(filename) else 0.0
                        }
                
                parsed_output['enhanced_metadata'] = {
                    "query_analysis": {
                        "original_query": query,
                        "query_length": len(query),
                        "query_complexity": estimate_query_complexity(query)
                    },
                    
                    "data_sources": {
                        "files_used": file_mapping,
                        "total_rows_available": int(sum(info["rows"] for info in file_mapping.values())),
                        "total_columns_available": int(sum(info["columns"] for info in file_mapping.values())),
                        "file_paths": [info["file_path"] for info in file_mapping.values()]
                    },
                    
                    "execution_details": {
                        "dataframes_loaded": int(len(dataframes)),
                        "total_memory_mb": float(total_memory),
                        "large_datasets": [analysis['name'] for analysis in size_analyses if analysis['is_large']],
                        "execution_method": "enhanced_pandas_query_agent_with_metadata"
                    }
                }
                
                parsed_output['metadata'] = {
                    "execution_info": {
                        "total_rows_processed": int(total_rows),
                        "total_memory_mb": float(total_memory),
                        "large_datasets": bool(has_large_datasets),
                        "dataframes_loaded": int(len(dataframes)),
                        "files_processed": [str(f) for f in filenames if os.path.exists(f)]
                    },
                    "enhancement_features": {
                        "dataframe_context_provided": True,
                        "automatic_date_conversion": True,
                        "schema_aware_execution": True,
                        "sample_data_available": True
                    },
                    "size_analysis": safe_size_analyses,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            
            # Log the final response
            logger.info("Query executed successfully with DataFrame context")
            logger.info(f"Response type: {type(parsed_output)}")
            
            return json.dumps(parsed_output, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Response processing error: {e}")
            # Return the raw output with error information
            error_response = {
                "error": "Response processing failed",
                "raw_output": str(response.get('output', '')),
                "processing_error": str(e),
                "query": query,
                "execution_method": "context_aware_pandas_agent",
                "metadata": {
                    "total_rows_processed": total_rows,
                    "files_processed": [f for f in filenames if os.path.exists(f)],
                    "context_provided": True,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
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
            "files": filenames,
            "execution_method": "context_aware_pandas_agent",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return json.dumps(error_response, indent=2, ensure_ascii=False)

@mcp.tool()
def execute_pandas_script(script: str, filenames: list[str] = None) -> str:
    """Execute pandas script with detailed usage tracking"""
    try:
        # Use default files if none provided
        if not filenames:
            filenames = [".venv/bootcathon-2025/src/Inbound.csv", ".venv/bootcathon-2025/src/Outbound.csv", ".venv/bootcathon-2025/src/Inventory.csv"]

        # Load dataframes with file mapping
        dataframes, size_analyses, df_context = load_and_analyze_dataframes(filenames)
        
        # Create enhanced execution environment with tracking
        execution_env = {
            'pd': pd,
            'numpy': __import__('numpy'),
            'json': json,
            'datetime': __import__('datetime'),
            '_usage_tracker': [],  # Track operations
            '_file_mapping': {},   # Track file to dataframe mapping
        }
        
        # Enhanced DataFrame name mapping with file tracking
        df_name_mapping = {
            ".venv/bootcathon-2025/src/Inbound.csv": "inbound_df",
            ".venv/bootcathon-2025/src/Outbound.csv": "outbound_df", 
            ".venv/bootcathon-2025/src/Inventory.csv": "inventory_df",
            "./src/Inbound.csv": "inbound_df",
            "./src/Outbound.csv": "outbound_df", 
            "./src/Inventory.csv": "inventory_df",
            "Inbound.csv": "inbound_df",
            "Outbound.csv": "outbound_df",
            "Inventory.csv": "inventory_df",
            "Inbound": "inbound_df",
            "Outbound": "outbound_df",
            "Inventory": "inventory_df"
        }
        
        # Add dataframes to environment with enhanced tracking
        file_metadata = {}
        for i, filename in enumerate(filenames):
            if i < len(dataframes):
                df_name = None
                
                # Get proper DataFrame name
                if filename in df_name_mapping:
                    df_name = df_name_mapping[filename]
                else:
                    base_filename = os.path.basename(filename)
                    if base_filename in df_name_mapping:
                        df_name = df_name_mapping[base_filename]
                    else:
                        lower_filename = filename.lower()
                        if 'inbound' in lower_filename:
                            df_name = "inbound_df"
                        elif 'outbound' in lower_filename:
                            df_name = "outbound_df"
                        elif 'inventory' in lower_filename:
                            df_name = "inventory_df"
                        else:
                            df_name = f"df_{i}"
                
                # Add DataFrame with metadata
                execution_env[df_name] = dataframes[i]
                
                # Track file-to-dataframe mapping
                file_metadata[df_name] = {
                    "source_file": filename,
                    "file_size_bytes": int(os.path.getsize(filename)) if os.path.exists(filename) else 0,
                    "rows": int(len(dataframes[i])),
                    "columns": int(len(dataframes[i].columns)),
                    "column_names": dataframes[i].columns.tolist(),
                    "date_columns": [col for col in dataframes[i].columns if 'DATE' in col.upper()],
                    "numeric_columns": dataframes[i].select_dtypes(include=['number']).columns.tolist(),
                    "memory_usage_mb": float(round(dataframes[i].memory_usage(deep=True).sum() / 1024 / 1024, 2))
                }
                
                execution_env['_file_mapping'][df_name] = file_metadata[df_name]
                
                logger.info(f"Added DataFrame {df_name} from {filename} (shape: {dataframes[i].shape})")
        
        # Always ensure the standard names exist (even if empty)
        for df_name in ['inbound_df', 'outbound_df', 'inventory_df']:
            if df_name not in execution_env:
                execution_env[df_name] = pd.DataFrame()
                file_metadata[df_name] = {
                    "source_file": "empty_dataframe",
                    "file_size_bytes": 0,
                    "rows": 0,
                    "columns": 0,
                    "column_names": [],
                    "date_columns": [],
                    "numeric_columns": [],
                    "memory_usage_mb": 0.0
                }
                logger.warning(f"{df_name} not found, created empty DataFrame")
        
        # Execute the script with operation tracking
        logger.info(f"Executing pandas script: {script[:100]}...")
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Execute the script
            exec(script, execution_env)
            result = execution_env.get('result', 'Script executed successfully')
            
        finally:
            sys.stdout = old_stdout
        
        # Get captured print output
        printed_output = captured_output.getvalue()
        
        # Analyze what data was actually used in the result
        result_metadata = analyze_result_data_usage(result, execution_env)
        
        # Prepare enhanced response with detailed metadata
        response = {
            "execution_status": "success",
            "result": make_json_safe(result),
            "printed_output": printed_output,
            "script_executed": script,
            
            # Enhanced metadata
            "data_usage": {
                "files_loaded": file_metadata,
                "dataframes_available": [k for k in execution_env.keys() if k.endswith('_df')],
                "result_analysis": result_metadata,
                "operations_performed": extract_operations_from_script(script)
            },
            
            "file_details": [
                {
                    "filename": filename,
                    "dataframe_name": next((k for k, v in file_metadata.items() if v["source_file"] == filename), "unknown"),
                    "rows_loaded": file_metadata.get(next((k for k, v in file_metadata.items() if v["source_file"] == filename), ""), {}).get("rows", 0),
                    "columns_loaded": file_metadata.get(next((k for k, v in file_metadata.items() if v["source_file"] == filename), ""), {}).get("columns", 0),
                    "file_path": os.path.abspath(filename) if os.path.exists(filename) else filename
                }
                for filename in filenames
            ],
            
            "metadata": {
                "total_rows_processed": int(sum(info["rows"] for info in file_metadata.values())),
                "total_files_processed": int(len(file_metadata)),
                "execution_method": "enhanced_pandas_script_with_tracking",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        logger.info("Enhanced pandas script executed successfully with metadata tracking")
        return json.dumps(response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error executing pandas script: {e}")
        traceback.print_exc()
        
        error_response = {
            "execution_status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "script_executed": script,
            "file_details": [
                {
                    "filename": filename,
                    "attempted_load": True,
                    "file_exists": os.path.exists(filename),
                    "file_path": os.path.abspath(filename) if os.path.exists(filename) else filename
                }
                for filename in (filenames or [])
            ],
            "execution_method": "enhanced_pandas_script_with_tracking",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return json.dumps(error_response, indent=2, ensure_ascii=False)

@mcp.tool()
def generate_and_execute_pandas(query: str, filenames: list[str] = None) -> str:
    """Generate pandas script using LLM, then execute it directly
    
    Args:
        query: Natural language query
        filenames: List of CSV files to load (optional)
        
    Returns:
        JSON string with generation and execution results
    """
    try:
        # Load dataframes for context
        if not filenames:
            filenames = [".venv/bootcathon-2025/src/Inbound.csv", ".venv/bootcathon-2025/src/Outbound.csv", ".venv/bootcathon-2025/src/Inventory.csv"]
        
        dataframes, size_analyses, df_context = load_and_analyze_dataframes(filenames)
        
        # Generate pandas script using LLM
        script_generation_prompt = f"""
        You are a pandas expert. Generate Python code to answer this query:
        
        Query: {query}
        
        Available DataFrames (ALREADY LOADED):
        {df_context}
        
        CRITICAL REQUIREMENTS:
        1. DO NOT use pd.read_csv() - the DataFrames are already loaded
        2. Use ONLY these pre-loaded variables: inbound_df, outbound_df, inventory_df
        3. Store your final answer in a variable called 'result'
        4. Add print(result) at the end
        5. Write clean, efficient pandas code
        6. Handle dates properly (they are already datetime objects)
        7. Use exact column names as shown in the context
        8. Return ONLY the Python code, no explanations or markdown
        
        EXAMPLE:
        # Good - uses existing DataFrames:
        result = inbound_df['NET_QUANTITY_MT'].sum()
        print(result)
        
        # Bad - tries to load files (DON'T DO THIS):
        # inbound_df = pd.read_csv('Inbound.csv')
        
        Code:
        """
        
        # Get script from LLM
        response = llm.invoke([HumanMessage(content=script_generation_prompt)])
        generated_script = response.content.strip()
        
        # Clean up the script (remove markdown formatting)
        generated_script = generated_script.replace("```python", "").replace("```", "").strip()
        
        logger.info(f"Generated script: {generated_script[:200]}...")
        
        # Execute the generated script
        execution_result = execute_pandas_script(generated_script, filenames)
        execution_data = json.loads(execution_result)
        
        # Combine generation and execution results
        combined_response = {
            "query": query,
            "generated_script": generated_script,
            "execution_result": execution_data,
            "method": "llm_generated_direct_execution",
            "success": execution_data.get("execution_status") == "success",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info("Generate and execute pandas completed successfully")
        return json.dumps(combined_response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in generate_and_execute_pandas: {e}")
        traceback.print_exc()
        
        error_response = {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "query": query,
            "method": "llm_generated_direct_execution",
            "success": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return json.dumps(error_response, indent=2, ensure_ascii=False)

# Keep the old function for backward compatibility
@mcp.tool()
def csv_query_agent(filenames: list[str], query: str, structure_output: dict) -> str:
    """Legacy CSV agent - deprecated, use pandas_query_agent instead"""
    logger.warning("csv_query_agent is deprecated. Use pandas_query_agent instead.")
    return pandas_query_agent(filenames, query, structure_output)

# Enhanced test function
@mcp.tool()
def test_connection() -> str:
    """Test if the MCP server is working with DataFrame context"""
    try:
        # Create test dataframes similar to your actual data structure
        test_inbound = pd.DataFrame({
            'INBOUND_DATE': ['2024/01/01', '2024/01/02'],
            'PLANT_NAME': ['TEST-WAREHOUSE', 'TEST-WAREHOUSE'],
            'MATERIAL_NAME': ['MAT-001', 'MAT-002'],
            'NET_QUANTITY_MT': [10.5, 15.2]
        })
        
        test_outbound = pd.DataFrame({
            'OUTBOUND_DATE': ['2024/01/01', '2024/01/02'],
            'PLANT_NAME': ['TEST-WAREHOUSE', 'TEST-WAREHOUSE'],
            'MATERIAL_NAME': ['MAT-001', 'MAT-002'],
            'NET_QUANTITY_MT': [5.0, 8.5]
        })
        
        # Convert dates
        test_inbound['INBOUND_DATE'] = pd.to_datetime(test_inbound['INBOUND_DATE'])
        test_outbound['OUTBOUND_DATE'] = pd.to_datetime(test_outbound['OUTBOUND_DATE'])
        
        # Test direct script execution
        test_script = """
result = {
    'total_inbound': inbound_df['NET_QUANTITY_MT'].sum(),
    'total_outbound': outbound_df['NET_QUANTITY_MT'].sum(),
    'net_flow': inbound_df['NET_QUANTITY_MT'].sum() - outbound_df['NET_QUANTITY_MT'].sum()
}
print(f"Test completed: {result}")
"""
        
        # Create test files temporarily
        test_inbound.to_csv('./test_inbound.csv', index=False)
        test_outbound.to_csv('./test_outbound.csv', index=False)
        
        try:
            # Test direct execution
            direct_result = execute_pandas_script(test_script, ['./test_inbound.csv', './test_outbound.csv'])
            
            # Test LLM generation
            llm_result = generate_and_execute_pandas(
                "What is the total inbound quantity?", 
                ['./test_inbound.csv', './test_outbound.csv']
            )
            
            # Clean up test files
            os.remove('./test_inbound.csv')
            os.remove('./test_outbound.csv')
            
            return json.dumps({
                "status": "success",
                "message": "MCP server with all execution methods is working correctly",
                "tests_completed": [
                    "DataFrame context generation",
                    "Date conversion",
                    "Direct script execution",
                    "LLM script generation and execution"
                ],
                "direct_execution_result": json.loads(direct_result),
                "llm_generation_result": json.loads(llm_result),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, indent=2)
            
        except Exception as cleanup_error:
            # Clean up test files if they exist
            for test_file in ['./test_inbound.csv', './test_outbound.csv']:
                if os.path.exists(test_file):
                    os.remove(test_file)
            raise cleanup_error
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, indent=2)

@mcp.tool()
def list_available_files() -> str:
    """List all available CSV files and their locations"""
    try:
        search_paths = [
            ".",
            "./src/",
            ".venv/bootcathon-2025/src/",
            "/home/gat/Documents/bootcathon2025/.venv/bootcathon-2025/src/",
            "/home/gat/Documents/bootcathon2025/src/"
        ]
        
        found_files = {}
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                files_in_path = []
                try:
                    for file in os.listdir(search_path):
                        if file.endswith('.csv'):
                            full_path = os.path.join(search_path, file)
                            files_in_path.append({
                                "filename": file,
                                "full_path": full_path,
                                "size_bytes": os.path.getsize(full_path)
                            })
                    found_files[search_path] = files_in_path
                except PermissionError:
                    found_files[search_path] = "Permission denied"
            else:
                found_files[search_path] = "Path does not exist"
        
        return json.dumps({
            "search_results": found_files,
            "current_directory": os.getcwd(),
            "total_csv_files": sum(len(files) if isinstance(files, list) else 0 for files in found_files.values())
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "current_directory": os.getcwd()
        }, indent=2)

logger.info(f'MCP MCP Server at {host}:{port} and transport {transport}')

# Test the server before starting
try:
    logger.info("Testing all server components...")
    test_result = test_connection()
    logger.info(f"Test result preview: {test_result[:500]}...")
except Exception as e:
    logger.warning(f"Server test failed: {e}")

logger.info("Server ready with all execution methods:")
logger.info("- pandas_query_agent: LangChain pandas agent with DataFrame context")
logger.info("- execute_pandas_script: Direct pandas script execution")
logger.info("- generate_and_execute_pandas: LLM generates code + direct execution")

mcp.run(transport=transport)