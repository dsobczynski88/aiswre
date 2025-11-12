import sys
import os
import asyncio
import json
import flatdict
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
from pathlib import Path
from string import Formatter

# LangChain imports
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# PDF processing
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Dependencies: pandas openpyxl flatdict langchain langchain-community langchain-text-splitters faiss-cpu pypdf

class RAGProcessor:
    """Handles the RAG (Retrieval Augmented Generation) functionality"""
    
    def __init__(self, pdf_directory: Optional[str] = None, model_name: str = "llama3"):
        """
        Initialize the RAG processor.
        
        Args:
            pdf_directory: Directory containing PDF files (optional)
            model_name: Name of the model to use for embeddings
        """
        self.pdf_directory = pdf_directory
        self.model_name = model_name
        self.vectorstore = None
        
    def load_documents(self, url="https://specinnovations.com/blog/how-to-verify-and-validate-requirements"):
        """Load content from a web URL"""
        print(f"Loading content from {url}...")
        
        try:
            # Updated import for WebBaseLoader
            from langchain_community.document_loaders import WebBaseLoader
            
            # Load content from the URL
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            if not documents:
                print("Warning: No content was loaded from the URL.")
                return []
                
            print(f"Loaded {len(documents)} document(s) from the URL")
            return documents
        except Exception as e:
            print(f"Error loading web content: {str(e)}")
            raise

    def create_vectorstore(self, documents):
        """Create a vector store from the documents"""
        print("Creating vector store...")
        
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        try:
            # Updated imports
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.embeddings import OllamaEmbeddings
            from langchain_community.vectorstores import FAISS
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                raise ValueError("Document splitting resulted in 0 chunks")
                
            print(f"Split into {len(chunks)} chunks")
            
            # Create embeddings and vector store
            embeddings = OllamaEmbeddings(model=self.model_name)
            
            # Add error handling for empty documents
            filtered_chunks = []
            for chunk in chunks:
                if chunk.page_content and len(chunk.page_content.strip()) > 0:
                    filtered_chunks.append(chunk)
                else:
                    print("Warning: Empty chunk detected and filtered out")
                    
            if not filtered_chunks:
                raise ValueError("All chunks were empty after filtering")
                
            vectorstore = FAISS.from_documents(filtered_chunks, embeddings)
            print("Vector store created successfully")
            return vectorstore
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            raise
        
    def initialize(self):
        """Initialize the RAG system"""
        documents = self.load_documents()
        self.vectorstore = self.create_vectorstore(documents)
        return self.vectorstore
    
    def get_retriever(self):
        """Get the retriever from the vector store"""
        if not self.vectorstore:
            self.initialize()
        return self.vectorstore.as_retriever(search_kwargs={"k": 5})


class PromptProcessor:
    def __init__(self, input_file: str, output_dir: str, model: str = "llama3", pdf_directory: Optional[str] = None, use_rag: bool = True):
        """
        Initialize the prompt processor.
        
        Args:
            input_file: Path to the input file (CSV, Excel, etc.)
            output_dir: Directory to save output results
            model: LLM model to use for processing
            pdf_directory: Directory containing PDF files for RAG (optional)
            use_rag: Whether to use RAG functionality even if pdf_directory is not provided
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.model = model
        self.pdf_directory = pdf_directory
        self.rag_processor = None
        
        # Initialize RAG if use_rag is True (regardless of pdf_directory)
        if use_rag:
            self.rag_processor = RAGProcessor(pdf_directory, model)
            
    def process_json_responses(self, 
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
                
            try:
                # Extract content from response
                if isinstance(response, str):
                    content = response
                elif isinstance(response, dict) and "content" in response:
                    content = response["content"]
                else:
                    content = str(response)
                    
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
            except Exception as e:
                output["processing_error"] = str(e)
                output["raw_response"] = str(response)
                
            # Add metadata
            output.update({
                "item_id": ids[i],
                "prompt_type": prompt_type,
            })
            
            processed.append(output)
        return processed

    def execute_prompt_with_retry(self, 
                                 llm, 
                                 system_message: str, 
                                 user_message: str,
                                 retriever=None) -> Dict:
        """
        Execute a single prompt with one retry on failure.
        
        Args:
            llm: LangChain LLM instance
            system_message: System message for the LLM
            user_message: User message for the LLM
            retriever: Optional retriever for RAG
            
        Returns:
            Response dictionary or None if failed after retry
        """
        try:
            # If retriever is provided, use RAG
            if retriever:
                # Get relevant context from the retriever
                docs = retriever.invoke(user_message)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Add context to the user message
                enhanced_user_message = f"""
                Context information from documents:
                {context}
                
                User query:
                {user_message}
                """
                
                # Create messages with context
                messages = [
                    SystemMessage(content=system_message),
                    HumanMessage(content=enhanced_user_message)
                ]
            else:
                # Standard messages without RAG
                messages = [
                    SystemMessage(content=system_message),
                    HumanMessage(content=user_message)
                ]
            
            # Execute the prompt
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Prompt failed with error: {str(e)}. Retrying once...")
            try:
                # Retry once
                if retriever:
                    # Get relevant context from the retriever
                    docs = retriever.get_relevant_documents(user_message)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Add context to the user message
                    enhanced_user_message = f"""
                    Context information from documents:
                    {context}
                    
                    User query:
                    {user_message}
                    """
                    
                    # Create messages with context
                    messages = [
                        SystemMessage(content=system_message),
                        HumanMessage(content=enhanced_user_message)
                    ]
                else:
                    # Standard messages without RAG
                    messages = [
                        SystemMessage(content=system_message),
                        HumanMessage(content=user_message)
                    ]
                
                response = llm.invoke(messages)
                return response.content
            except Exception as retry_error:
                print(f"Retry failed with error: {str(retry_error)}. Skipping this prompt.")
                return None

    def run_prompt_batch(self, 
                        llm, 
                        system_message: str, 
                        user_message_template: str, 
                        prompt_name: str, 
                        items: List[Dict[str, Any]], 
                        ids: List[Any] = None, 
                        json_key: str = None) -> List[Dict]:
        """
        Execute concurrent prompts and process JSON responses.
        
        Args:
            llm: LangChain LLM instance
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
        
        # Get retriever if RAG is enabled
        retriever = None
        if self.rag_processor:
            retriever = self.rag_processor.get_retriever()
        
        # Process items one by one with progress updates
        responses = []
        total = len(items)
        for i, (item, item_id) in enumerate(zip(items, ids)):
            # Format the user message by replacing variables
            user_msg = user_message_template
            for key, value in item.items():
                placeholder = f"{{{key}}}"
                if placeholder in user_msg:
                    user_msg = user_msg.replace(placeholder, str(value))
            
            # Execute the prompt
            response = self.execute_prompt_with_retry(
                llm,
                system_message,
                user_msg,
                retriever
            )
            responses.append(response)
            
            # Show progress
            progress = (i + 1) / total * 100
            print(f"Progress: {progress:.1f}% - Processed {i+1}/{total} items")
        
        # Process structured JSON responses
        return self.process_json_responses(responses, ids, prompt_name, json_key)

    def process_data(self, 
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
            
            # Initialize LangChain LLM
            llm = ChatOllama(
                model=self.model,
                format="json",
                temperature=0.1
            )
            
            # Initialize RAG if needed
            if self.rag_processor:
                print("Initializing RAG system...")
                self.rag_processor.initialize()
            
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
                    batch_results = self.run_prompt_batch(
                        llm, system_message, user_message_template, 
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
                all_results = self.run_prompt_batch(
                    llm, system_message, user_message_template, 
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

    def run(self, 
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
                
            results_df = self.process_data(
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

def main(input_file: str, output_dir: str, output_file: str, model: str, system_message: str, 
         user_message: str, column_mapping: dict, prompt_vars: list, batch_size: int = 10, 
         pdf_directory: str = None):

    # Initialize processor with RAG if pdf_directory is provided
    processor = PromptProcessor(
        input_file=input_file,
        output_dir=output_dir,
        model=model,
        pdf_directory=pdf_directory
    )
    
    success = processor.run(
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

if __name__ == "__main__":
    # Example usage
    input_file = "testcases.xlsx"
    output_dir = "../output"
    output_file = "testcase-analysis"
    model = "llama3.1"
    pdf_directory = None  # Directory containing PDF files for RAG
    
    system_message = """
    You are a Senior Test Verification Traceability Analyst with expertise in software quality assurance.
    You specialize in critically analyzing test cases against requirements to ensure complete coverage according to industry best practices.
    Be thorough, methodical, and precise in your assessment. Focus on identifying gaps in test coverage while maintaining technical accuracy.
    Do not invent new requirements or test cases. Ask targeted clarification questions when information is insufficient.

    Response Format (produce exactly this JSON structure):
    {
    "traceability_review": {
        "requirement": "<requirement text>",
        "coverage_summary": "complete|partial|inadequate",
        "analysis": {
        "functional_coverage": {
            "status": "complete|partial|inadequate",
            "covered_aspects": ["<aspect 1>", "<aspect 2>", ...],
            "gaps": ["<gap 1>", "<gap 2>", ...],
            "explanation": "<brief assessment>"
        },
        "non_functional_coverage": {
            "status": "complete|partial|inadequate|not_applicable",
            "covered_aspects": ["<aspect 1>", "<aspect 2>", ...],
            "gaps": ["<gap 1>", "<gap 2>", ...],
            "explanation": "<brief assessment>"
        },
        "test_case_quality": {
            "status": "good|needs_improvement|poor",
            "strengths": ["<strength 1>", "<strength 2>", ...],
            "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
            "explanation": "<brief assessment>"
        }
        },
        "test_cases_assessment": [
        {
            "test_case_id": "<ID or index>",
            "coverage_aspects": ["<aspect 1>", "<aspect 2>", ...],
            "missing_aspects": ["<aspect 1>", "<aspect 2>", ...],
            "test_type": "positive|negative|boundary|performance|other",
            "quality_issues": ["<issue 1>", "<issue 2>", ...]
        }
        ],
        "recommendations": {
        "additional_test_cases": ["<test case 1>", "<test case 2>", ...],
        "test_case_improvements": ["<improvement 1>", "<improvement 2>", ...],
        "priority_gaps": ["<gap 1>", "<gap 2>", ...]
        }
    }
    }

    Evaluation method:
    1) Parse the requirement and test cases
    2) Identify all testable aspects of the requirement
    3) Map test cases to requirement aspects to determine coverage
    4) Evaluate test cases for positive, negative, and boundary conditions
    5) Assess if non-functional requirements are adequately tested
    6) Identify specific gaps in coverage and missing edge cases
    7) Provide actionable recommendations for improving test coverage

    Important: If the requirement or test cases are empty or unclear, respond with a single clarifying question requesting the necessary information and stop.
    
    USE PROVIDED CONTEXT WHEN PERFORMING YOUR ASSESSMENT
    """
    
    user_message = """
    Task: Review the test cases for complete verification traceability against the requirement.
    Variables:
    - Requirement:
    {requirement}
    - Test Cases (list or newline-separated; may include IDs):
    {test_cases}

    Analyze whether these test cases completely verify all aspects of the requirement. Identify any gaps in coverage, missing edge cases, or aspects of the requirement that aren't adequately tested.

    Produce output strictly in the Response Format JSON. Do not use Markdown.

    Now perform the traceability review on the provided inputs and return only the Response Format JSON.
    """
    
    column_mapping = {
        "prompt_name": "testcase-analysis",
        "id_column": "requirement_id",
        "json_key": "traceability_review"
    }
    
    prompt_vars = ["requirement","test_cases"]
    batch_size = 5
    
    main(
        input_file=input_file,
        output_dir=output_dir,
        output_file=output_file,
        model=model,
        system_message=system_message,
        user_message=user_message,
        column_mapping=column_mapping,
        prompt_vars=prompt_vars,
        batch_size=batch_size,
        pdf_directory=pdf_directory
    )