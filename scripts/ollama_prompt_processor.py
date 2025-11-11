import sys
import os
import asyncio
import json
import flatdict
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
from pathlib import Path
from string import Formatter

from ollama import AsyncClient


# Dependencies: pandas openpyxl flatdict ollama

class PromptProcessor:
    def __init__(self, input_file: str, output_dir: str, model: str = "llama3"):
        """
        Initialize the prompt processor.
        
        Args:
            input_file: Path to the input file (CSV, Excel, etc.)
            output_dir: Directory to save output results
            model: LLM model to use for processing
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.model = model
        
    async def process_json_responses(self, 
                                    responses: List[Dict], 
                                    ids: List[Any], 
                                    prompt_type: str, 
                                    json_key: str = None) -> List[Dict]:
        """
        Process responses and flatten extracted JSON structures.
        
        Args:
            responses: List of responses from the LLM
            ids: List of identifiers corresponding to each response
            prompt_type: Type of prompt used (for tracking)
            json_key: Optional key to extract from JSON response
            
        Returns:
            List of processed and flattened dictionaries
        """
        processed = []

        for i, response in enumerate(responses):
            output = {}
            
            # Handle None responses (failed prompts)
            if response is None:
                output = {
                    "item_id": ids[i],
                    "prompt_type": prompt_type,
                    "error": "Prompt failed after retry"
                }
                processed.append(output)
                continue
                
            # Extract content from ollama response
            if "message" in response and "content" in response["message"]:
                content = response["message"]["content"]
                try:
                    response_json = json.loads(content)
                    if json_key and json_key in response_json:
                        nested_dicts = response_json[json_key]
                        if isinstance(nested_dicts, list):
                            flat_dicts = [flatdict.FlatDict(d, delimiter=".") for d in nested_dicts]
                            for d in flat_dicts:
                                output.update(d)
                        elif isinstance(nested_dicts, dict):
                            flat_dict = flatdict.FlatDict(nested_dicts, delimiter=".")
                            output.update(flat_dict)
                    else:
                        # If no json_key specified or not found, use the whole response
                        flat_dict = flatdict.FlatDict(response_json, delimiter=".")
                        output.update(flat_dict)
                except (json.JSONDecodeError, TypeError):
                    output["json_parse_error"] = content
            
            # Include usage info if available
            for metric in ["eval_count", "prompt_eval_count", "total_duration"]:
                if metric in response:
                    output[metric] = response[metric]
                
            # Add metadata
            output.update({
                "item_id": ids[i],
                "prompt_type": prompt_type,
            })
            
            processed.append(output)
        return processed

    async def execute_prompt_with_retry(self, 
                                       ollama_client: AsyncClient, 
                                       model: str,
                                       system_message: str, 
                                       user_message: str) -> Dict:
        """
        Execute a single prompt with one retry on failure.
        
        Args:
            ollama_client: AsyncClient instance for Ollama
            model: Model name to use
            system_message: System message for the LLM
            user_message: User message for the LLM
            
        Returns:
            Response dictionary or None if failed after retry
        """
        try:
            response = await ollama_client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                format="json",  # Request JSON format response
            )
            return response
        except Exception as e:
            print(f"Prompt failed with error: {str(e)}. Retrying once...")
            try:
                # Retry once
                response = await ollama_client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    format="json",  # Request JSON format response
                )
                return response
            except Exception as retry_error:
                print(f"Retry failed with error: {str(retry_error)}. Skipping this prompt.")
                return None

    async def run_prompt_batch(self, 
                              ollama_client: AsyncClient, 
                              system_message: str, 
                              user_message_template: str, 
                              prompt_name: str, 
                              items: List[Dict[str, Any]], 
                              ids: List[Any] = None, 
                              json_key: str = None) -> List[Dict]:
        """
        Execute concurrent prompts and process JSON responses.
        
        Args:
            ollama_client: AsyncClient instance for Ollama
            system_message: System message for the LLM
            user_message_template: Template string with {variable} placeholders
            prompt_name: Name of the prompt for tracking
            items: List of dictionaries containing variables for the template
            ids: Optional list of identifiers for each item
            json_key: Optional key to extract from JSON response
            
        Returns:
            List of processed responses
        """
        if ids is None:
            ids = list(range(len(items)))
        
        # Build tasks list
        tasks = []
        for item, item_id in zip(items, ids):
            # Format the user message by replacing variables
            user_msg = user_message_template
            for key, value in item.items():
                placeholder = f"{{{key}}}"
                if placeholder in user_msg:
                    user_msg = user_msg.replace(placeholder, str(value))
            
            task = self.execute_prompt_with_retry(
                ollama_client,
                self.model,
                system_message,
                user_msg
            )
            tasks.append(task)
        
        # Process items one by one with progress updates
        responses = []
        total = len(tasks)
        for i, task in enumerate(tasks):
            response = await task
            responses.append(response)
            progress = (i + 1) / total * 100
            print(f"Progress: {progress:.1f}% - Processed {i+1}/{total} items")
        
        # Process structured JSON responses
        return await self.process_json_responses(responses, ids, prompt_name, json_key)

    async def process_data(self, 
                          system_message: str, 
                          user_message_template: str, 
                          prompt_name: str,
                          id_column: str = None,
                          json_key: str = None,
                          batch_size: int = None,
                          prompt_vars: list = [None],
                          output_filename: str = None) -> pd.DataFrame:
        """
        Process data from the input file using the provided prompt templates.
        
        Args:
            system_message: System message for the LLM
            user_message_template: Template string with {variable} placeholders
            prompt_name: Name of the prompt for tracking
            id_column: Column to use as identifier (defaults to index if None)
            json_key: Optional key to extract from JSON response
            batch_size: Optional batch size for processing (None processes all at once)
            output_filename: Base filename for output files
            
        Returns:
            DataFrame with processed results
        """
        try:
            print(f"Loading input file: {self.input_file}")
            
            # Load input file based on extension
            file_ext = os.path.splitext(self.input_file)[1].lower()
            if file_ext == '.xlsx' or file_ext == '.xls':
                df = pd.read_excel(self.input_file)
            elif file_ext == '.csv':
                df = pd.read_csv(self.input_file)
            elif file_ext == '.json':
                df = pd.read_json(self.input_file)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            print(f"Found {len(df)} items to process")
            
            # Initialize Ollama client
            client = AsyncClient()
            
            required_vars = prompt_vars
            
            # Prepare items and IDs
            items = df_to_prompt_items(df, required_vars)
            ids = df[id_column].tolist() if id_column and id_column in df.columns else list(range(len(df)))
            
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Set up output filename
            if output_filename is None:
                output_filename = f"{prompt_name}_results"
            
            # Process in batches if specified
            all_results = []
            if batch_size:
                for batch_num, i in enumerate(range(0, len(items), batch_size)):
                    batch_items = items[i:i+batch_size]
                    batch_ids = ids[i:i+batch_size]
                    print(f"Processing batch {batch_num + 1} ({len(batch_items)} items)...")
                    
                    # Process this batch
                    batch_results = await self.run_prompt_batch(
                        client, system_message, user_message_template, 
                        prompt_name, batch_items, batch_ids, json_key
                    )
                    
                    # Save this batch's results immediately
                    batch_df = pd.DataFrame(batch_results)
                    batch_output_path = os.path.join(
                        self.output_dir, 
                        f"{output_filename}_batch_{batch_num+1}.xlsx"
                    )
                    batch_df.to_excel(batch_output_path, index=False)
                    print(f"Batch {batch_num+1} results saved to: {batch_output_path}")
                    
                    # Add to cumulative results
                    all_results.extend(batch_results)
                    
                    # Also save cumulative results after each batch
                    cumulative_df = pd.DataFrame(all_results)
                    cumulative_output_path = os.path.join(
                        self.output_dir, 
                        f"{output_filename}_cumulative.xlsx"
                    )
                    cumulative_df.to_excel(cumulative_output_path, index=False)
                    print(f"Cumulative results updated at: {cumulative_output_path}")
            else:
                # Process all at once
                print("Starting data processing...")
                all_results = await self.run_prompt_batch(
                    client, system_message, user_message_template, 
                    prompt_name, items, ids, json_key
                )
                
                # Save results immediately
                results_df = pd.DataFrame(all_results)
                output_path = os.path.join(self.output_dir, f"{output_filename}.xlsx")
                results_df.to_excel(output_path, index=False)
                print(f"Results saved to: {output_path}")
            
            # Convert final results to DataFrame
            results_df = pd.DataFrame(all_results)
            
            return results_df
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            # If we have partial results, save them before raising the exception
            if all_results:
                try:
                    partial_df = pd.DataFrame(all_results)
                    recovery_path = os.path.join(self.output_dir, f"{output_filename}_partial_recovery.xlsx")
                    partial_df.to_excel(recovery_path, index=False)
                    print(f"Partial results saved to: {recovery_path}")
                except Exception as save_error:
                    print(f"Failed to save partial results: {str(save_error)}")
            raise

    async def run(self, 
                 system_message: str, 
                 user_message_template: str, 
                 prompt_name: str,
                 id_column: str = None,
                 json_key: str = None,
                 output_filename: str = None,
                 prompt_vars: list = [None],
                 batch_size: int = None) -> bool:
        """
        Run the prompt processor with the given configuration.
        
        Args:
            system_message: System message for the LLM
            user_message_template: Template string with {variable} placeholders
            prompt_name: Name of the prompt for tracking
            id_column: Column to use as identifier (defaults to index if None)
            json_key: Optional key to extract from JSON response
            output_filename: Custom filename for output (defaults to prompt_name)
            batch_size: Optional batch size for processing
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            if output_filename is None:
                output_filename = f"{prompt_name}_results"
                
            results_df = await self.process_data(
                system_message, 
                user_message_template, 
                prompt_name,
                id_column,
                json_key,
                batch_size,
                prompt_vars,
                output_filename
            )
            
            # Final save is now handled in process_data
            print(f"Processing complete!")
            return True
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return False


def df_to_prompt_items(df: pd.DataFrame, columns: List[str] = None) -> List[Dict[str, Any]]:
    """
    Transform dataframe rows into a format suitable for prompt templates.
    
    Args:
        df: Input dataframe
        columns: List of columns to include (None for all columns)
        
    Returns:
        List of dictionaries with column values
    """
    if columns is None or columns == [None]:
        columns = df.columns.tolist()
    
    # Convert dataframe to list of dictionaries with only the required columns
    items = []
    for _, row in df.iterrows():
        item = {col: row[col] for col in columns if col in row}
        items.append(item)
    
    return items

async def main(input_file: str, output_dir: str, output_file: str, model: str, system_message: str, user_message: str, column_mapping: dict, prompt_vars: list, batch_size: int = 10):

    # Initialize processor
    processor = PromptProcessor(
        input_file=input_file,
        output_dir=output_dir,
        model=model
    )
    
    success = await processor.run(
        system_message=system_message,
        user_message_template=user_message,
        prompt_name=column_mapping["prompt_name"],
        id_column=column_mapping["id_column"],
        json_key=column_mapping["json_key"],
        output_filename=output_file,
        prompt_vars=prompt_vars,
        batch_size=batch_size,        
    )
    
    if success:
        print("Data processing completed successfully!")
    else:
        print("Data processing failed.")