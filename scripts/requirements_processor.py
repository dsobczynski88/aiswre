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
            system_message = """
            You are a Senior Requirements Quality Analyst and technical editor. 
            You specialize in detecting and fixing requirement defects using authoritative quality rules. 
            Be rigorous, consistent, and concise. Maintain the author's technical intent while removing ambiguity. 
            Do not add new functionality. Ask targeted clarification questions when needed.

            Response Format (produce exactly this JSON structure):
            {
            "requirements_review": [
                {
                "requirement_id": "<ID>",
                "original": "<original requirement>",
                "checks": {
                    "R2": {"status": "pass|fail", "active_voice": ["<issues>"], "explanation": "<brief>"},
                    "R3": {"status": "pass|fail", "appropriate_subj_verb": ["<issues>"], "explanation": "<brief>"},
                    "R5": {"status": "pass|fail", "definite_articles": ["<issues>"], "explanation": "<brief>"},
                    "R6": {"status": "pass|fail", "units": ["<issues>"], "explanation": "<brief>"},
                    "R7": {"status": "pass|fail", "vague terms": ["<issues>"], "explanation": "<brief>"},
                    "R8": {"status": "pass|fail", "escape_clauses": ["<issues>"], "explanation": "<brief>"},
                    "R9": {"status": "pass|fail", "open_ended_clauses": ["<issues>"], "explanation": "<brief>"}
                },
                "proposed_rewrite": "<single improved requirement that resolves all detected issues>",
                "split_recommendation": {
                    "needed": true|false,
                    "because": "<why>",
                    "split_into": ["<Req A>", "<Req B>"]
                },
                }
            ]
            }

            Evaluation method:
            1) Parse inputs and normalize IDs. 
            2) For each requirement, test 2, R3, R5, R6, R7, R8, R9. 
            3) Explain each failure succinctly. 
            4) Rewrite to a single, verifiable sentence unless a split is recommended. 
            5) Apply glossary rules for abbreviations; on first use of allowed abbreviations, prefer the expanded form with abbreviation in parentheses. 
            6) If required numbers are missing and no defaults are provided, use TBD placeholders and ask explicit questions to resolve them. 
            7) Summarize compliance.

            Important: If {requirements} is empty, respond with a single clarifying question requesting requirements to review and stop.           
            """
            
            user_message = """
            Task: Review and improve the following requirement statements using the provided variables.
            Variables:
            - Requirements (list or newline-separated; may include IDs):
            {requirements}
            - Enable split recommendations (true|false; default true): {enable_split}

            Produce output strictly in the Response Format JSON. Do not use Markdown.

            Now perform the review on the provided inputs and return only the Response Format JSON.
            """

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