import sys
import os
import asyncio
import json
import flatdict
from typing import Any, Dict, List, Optional
import pandas as pd
from pathlib import Path

from ollama import AsyncClient


#Dependencies: pandas openpyxl flatdict ollama

class RequirementsProcessor:
    def __init__(self, excel_file, output_dir, model="llama3"):
        self.excel_file = excel_file
        self.output_dir = output_dir
        self.model = model
        
    async def process_json_responses(self, responses, ids, prompt_type, json_key="requirements_review"):
        """Process responses and flatten extracted JSON structures."""
        processed = []

        for i, response in enumerate(responses):
            output = {}
            
            # Extract content from ollama response
            if "message" in response and "content" in response["message"]:
                content = response["message"]["content"]
                try:
                    response_json = json.loads(content)
                    if json_key in response_json:
                        nested_dicts = response_json[json_key]
                        flat_dicts = [flatdict.FlatDict(d, delimiter=".") for d in nested_dicts]
                        for d in flat_dicts:
                            output.update(d)
                except (json.JSONDecodeError, TypeError):
                    output["json_parse_error"] = content
            
            # Include usage info if available
            if "eval_count" in response:
                output["eval_count"] = response["eval_count"]
            if "prompt_eval_count" in response:
                output["prompt_eval_count"] = response["prompt_eval_count"]
            if "total_duration" in response:
                output["total_duration"] = response["total_duration"]
                
            output.update(
                {
                    "requirement_id": ids[i],
                    "prompt_type": prompt_type,
                }
            )
            processed.append(output)
        return processed

    async def run_requirement_review(self, ollama_client, system_message, user_message, 
                                    prompt_name, requirements, ids=None, json_key="requirements_review"):
        """Execute concurrent review prompts and process JSON responses."""
        if ids is None:
            ids = list(range(len(requirements)))
        
        # Build tasks list
        tasks = []
        for req, req_id in zip(requirements, ids):
            task = ollama_client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user",
                        "content": user_message
                        .replace("{requirements}", f"{req_id}: {req}")
                        .replace("{enable_split}", "True"),
                    },
                ],
                format="json",  # Request JSON format response
            )
            tasks.append(task)
        
        # Process requirements one by one with progress updates
        responses = []
        total = len(tasks)
        for i, task in enumerate(tasks):
            response = await task
            responses.append(response)
            progress = (i + 1) / total * 100
            print(f"Progress: {progress:.1f}% - Processed {i+1}/{total} requirements")
        
        # Process structured JSON responses
        return await self.process_json_responses(responses, ids, prompt_name, json_key)

    async def process_requirements(self):
        try:
            print(f"Loading Excel file: {self.excel_file}")
            # Load Excel file
            df = pd.read_excel(self.excel_file)
            
            print(f"Found {len(df)} requirements to process")
            # Extract requirements and IDs
            requirements = df['requirement_text'].tolist()
            ids = df['requirement_id'].tolist() if 'requirement_id' in df.columns else None
            
            # Initialize Ollama client
            client = AsyncClient()
            
            # Define system and user messages
            system_message = """You are a requirements analysis expert. Analyze the given requirement 
                              and provide structured feedback in JSON format."""
            
            user_message = """Please analyze the following requirement and provide a detailed review:
                           {requirements}
                           
                           Return your analysis in JSON format with the key 'requirements_review' 
                           containing an array of objects with your findings."""
            
            # Process requirements
            print("Starting requirements processing...")
            results = await self.run_requirement_review(
                client, 
                system_message, 
                user_message, 
                "requirement_review", 
                requirements, 
                ids
            )
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            
            return results_df
            
        except Exception as e:
            print(f"Error processing requirements: {str(e)}")
            raise

    async def run(self):
        try:
            results_df = await self.process_requirements()
            
            # Save results to output directory
            output_path = os.path.join(self.output_dir, 'requirements_analysis_results.xlsx')
            results_df.to_excel(output_path, index=False)
            
            print(f"Processing complete! Results saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return False


async def main():
    if len(sys.argv) < 3:
        print("Usage: python requirements_processor.py <excel_file> <output_dir> [model]")
        return
    
    excel_file = sys.argv[1]
    output_dir = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "llama3"
    
    processor = RequirementsProcessor(excel_file, output_dir, model)
    success = await processor.run()
    
    if success:
        print("Requirements analysis completed successfully!")
    else:
        print("Requirements analysis failed.")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())