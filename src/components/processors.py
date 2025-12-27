import os
import re
import json
import asyncio
import time
import ast
import pandas as pd
import flatdict
from typing import Any, Dict, List, Optional, Sequence, Union
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from src.components.clients import RateLimitOpenAIClient

def parse_llm_json_like(raw: str) -> Dict[str, Any]:
    """
    Robustly parses JSON-like strings produced by LLMs.
    Handles:
      - Escaped quotes
      - Python dict literals
      - Mixed quoting
    """

    if not raw or not isinstance(raw, str):
        raise ValueError("Input must be a non-empty string")

    text = raw.strip()

    # ----------------------------------------------------
    # Step 1: Unwrap if the entire payload is quoted
    # ----------------------------------------------------
    if (text.startswith("'") and text.endswith("'")) or \
       (text.startswith('"') and text.endswith('"')):
        text = text[1:-1]

    # Unescape common LLM escape patterns
    text = text.replace("\\'", "'").replace('\\"', '"')

    # ----------------------------------------------------
    # Step 2: Attempt strict JSON
    # ----------------------------------------------------
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ----------------------------------------------------
    # Step 3: Attempt Python literal parsing (SAFE)
    # ----------------------------------------------------
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
        raise ValueError("Parsed value is not a dictionary")
    except Exception:
        pass

    # ----------------------------------------------------
    # Step 4: Last-resort fixups → then JSON
    # ----------------------------------------------------
    repaired = text

    # Python → JSON boolean
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)

    # Convert single quotes to double quotes conservatively
    repaired = re.sub(r"(?<!\\)'", '"', repaired)

    return json.loads(repaired)


# -----------------------------------------------------------------------------
# Common utility functions for data processing
# -----------------------------------------------------------------------------
def load_input_data(input_file: str) -> pd.DataFrame:
    """
    Load CSV or Excel file into a pandas DataFrame.

    Args:
        input_file: Path to CSV or Excel file

    Returns:
        DataFrame with loaded data

    Raises:
        ValueError: If file extension is not supported or file path is invalid

    Example:
        >>> df = load_input_data("data/requirements.xlsx")
        >>> df = load_input_data("data/test_cases.csv")
    """
    if not input_file:
        raise ValueError("input_file is not provided")

    _, ext = os.path.splitext(input_file)
    ext = ext.lower()

    if ext == ".csv":
        return pd.read_csv(input_file)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(input_file)
    raise ValueError(f"Unsupported input file extension: {ext}")


def df_to_prompt_items(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None
) -> List[Dict[str, Any]]:
    """
    Convert each row of DataFrame into a dict suitable for processing.

    This utility function extracts specified columns from a DataFrame and converts
    each row into a dictionary. Keys are explicitly coerced to strings for
    compatibility with various processing functions.

    Args:
        df: Input DataFrame
        columns: List of column names to extract. If None, uses all columns.

    Returns:
        List of dictionaries, one per DataFrame row

    Raises:
        ValueError: If any specified columns are missing from the DataFrame

    Example:
        >>> df = pd.DataFrame({
        ...     "id": ["REQ-001", "REQ-002"],
        ...     "text": ["System shall...", "User shall..."],
        ...     "priority": ["high", "medium"]
        ... })
        >>> items = df_to_prompt_items(df, columns=["id", "text"])
        >>> # Returns: [
        >>> #   {"id": "REQ-001", "text": "System shall..."},
        >>> #   {"id": "REQ-002", "text": "User shall..."}
        >>> # ]
    """
    if columns is None:
        columns = df.columns.tolist()

    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    records = df[columns].to_dict(orient="records")

    # Explicitly coerce keys to str to satisfy type checker and intended use
    return [{str(k): v for k, v in record.items()} for record in records]


class BasicOpenAIProcessor:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.previous_response_ids: List[str] = []
        self.previous_responses: List[str] = []

    def get_response(
        self,
        input: Union[List, str],
        print_response: bool = True,
        store: bool = True,  # Kept for backwards compatibility
        previous_response_id: Optional[str] = None,  # Kept for backwards compatibility
        **kwargs
        ):
        # Provide the essentials, then allow override via kwargs
        params = {
            "model": self.model,
            "input": input,
            "store": store,
            "previous_response_id": previous_response_id,
        }
        params.update(kwargs)  # allow all other supported/needed arguments

        response = self.client.responses.create(**params)
        self.previous_responses.append(response.output_text)
        self.previous_response_ids.append(response.id)
        if print_response:
            print("Printing `response.output_text`:\n\n", response.output_text)
        return response

    def get_structured_response(
        self,
        messages,
        response_format,
        **kwargs
        ):
        params = {
            "model": self.model,
            "messages": messages,
            "response_format": response_format,
        }
        params.update(kwargs)

        completion = self.client.beta.chat.completions.parse(**params)
        return completion.choices[0].message

    @staticmethod
    def check_structured_output(completion):
        # If the model refuses to respond, you will get a refusal message
        if getattr(completion, "refusal", False):
            print(completion.refusal)
        else:
            print(getattr(completion, "parsed", completion))

class PromptProcessor:
    """
    Process prompts asynchronously using RateLimitOpenAIClient with token and request throttling.
    Handles prompt creation, OpenAI API calls, and JSON response normalization.
    """

    def __init__(
        self,
        client: RateLimitOpenAIClient,
        input_file: Optional[str] = None,
        output_dir: str = "./output",
        model: str = "gpt-4o-mini",
        pdf_directory: Optional[str] = None,
        use_rag: bool = False,
        input_df: Optional[pd.DataFrame] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        ) -> None:
        """
        Args:
            client: Initialized RateLimitOpenAIClient instance.
            input_file: Optional CSV/Excel file path.
            output_dir: Output directory for any saved results.
            model: OpenAI model name (e.g., "gpt-4o-mini").
            pdf_directory: Path for optional RAG context (future extension).
            use_rag: Whether to use retrieval-augmented generation components.
            input_df: Preloaded DataFrame of prompt data.
            model_kwargs: Parameters for OpenAI completions (temperature, max_tokens, etc.).
        """
        self.client = client
        self.input_file = input_file
        self.output_dir = output_dir
        self.model = model
        self.pdf_directory = pdf_directory
        self.use_rag = use_rag
        self.input_df = input_df
        self.model_kwargs = model_kwargs

        # Load data if needed
        if self.input_df is None and self.input_file:
            self.input_df = load_input_data(self.input_file)
        elif self.input_df is None and not self.input_file:
            raise ValueError("Either input_file or input_df must be provided")

    async def _call_openai(self, system_message: str, user_prompt: str) -> str:
        """
        Make a single asynchronous call to the OpenAI API with proper rate limiting.

        Returns:
            The assistant message content (string) or an error JSON.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]
        try:
            completion = await self.client.chat_completion_parse(
                model=self.model,
                messages=messages,
                **self.model_kwargs,
            )
            return completion
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def process_json_responses(
        self,
        responses: Sequence[Any],
        ids: Sequence[Any],
        prompt_type: str,
        ) -> List[Dict[str, Any]]:

        processed: List[Dict[str, Any]] = []

        for i, response in enumerate(responses):

            base_output: Dict[str, Any] = {
                "item_id": ids[i],
                "prompt_type": prompt_type,
            }

            # -------------------------------------------------
            # Handle None response
            # -------------------------------------------------
            if response is None:
                processed.append({
                    **base_output,
                    "error": "Prompt failed after retry",
                })
                continue

            # -------------------------------------------------
            # Extract content
            # -------------------------------------------------
            try:
                if isinstance(response, ParsedChatCompletion):
                    content = response.choices[0].message.content
                elif isinstance(response, dict) and "response" in response:
                    content = response["response"].content
                else:
                    content = str(response)
            except Exception as e:
                processed.append({
                    **base_output,
                    "processing_error": str(e),
                    "raw_response": str(response),
                })
                continue

            # -------------------------------------------------
            # Parse JSON (robust)
            # -------------------------------------------------
            try:
                response_json = parse_llm_json_like(content)
            except Exception as e:
                processed.append({
                    **base_output,
                    "json_parse_error": str(e),
                    "raw_response": content,
                })
                continue

            # -------------------------------------------------
            # Collect shared + row-level structures
            # -------------------------------------------------
            shared_flat: Dict[str, Any] = {}
            row_expanders: List[List[Dict[str, Any]]] = []

            for key, value in response_json.items():

                # ------------------------------
                # Case 1: dict → shared columns
                # ------------------------------
                if isinstance(value, dict):
                    flat = flatdict.FlatDict(value, delimiter=".")
                    shared_flat.update({f"{key}.{k}": v for k, v in flat.items()})

                # -------------------------------------------------------
                # Case 2: list of dicts → row-expanding structure
                # -------------------------------------------------------
                elif (
                    isinstance(value, list)
                    and value
                    and all(isinstance(v, dict) for v in value)
                ):
                    expanded_rows: List[Dict[str, Any]] = []
                    for idx, item in enumerate(value):
                        flat = flatdict.FlatDict(item, delimiter=".")
                        expanded_rows.append(
                            {
                                f"{key}.{k}": v
                                for k, v in flat.items()
                            }
                        )
                    row_expanders.append(expanded_rows)

                # ------------------------------
                # Case 3: scalar
                # ------------------------------
                else:
                    shared_flat[key] = value

            # -------------------------------------------------
            # Combine shared + expanded rows
            # -------------------------------------------------

            if row_expanders:
                # Currently supports 1 expanding list cleanly
                for idx, row_payload in enumerate(row_expanders[0]):
                    final_row = {
                        **base_output,
                        **shared_flat,
                        **row_payload,
                        "raw_response": content,
                    }
                    processed.append(final_row)
            else:
                processed.append({
                    **base_output,
                    **shared_flat,
                    "raw_response": content,
                })

            # -------------------------------------------------
            # Token usage (if available)
            # -------------------------------------------------
            usage = getattr(response, "usage", None)
            if usage:
                last_rows = processed[-len(row_expanders[0]):] if row_expanders else [processed[-1]]
                for row in last_rows:
                    try:
                        row.update(dict(usage))
                        for sub_key in ("prompt_tokens_details", "completion_tokens_details"):
                            sub = getattr(usage, sub_key, None)
                            if sub:
                                row.update(dict(sub))
                    except Exception:
                        pass

        return processed

    async def run_prompt_batch(
        self,
        system_message: str,
        user_message_template: str,
        prompt_name: str,
        items: Sequence[Dict[str, Any]],
        ids: Optional[Sequence[Any]] = None,
        json_key: Optional[str] = None,
        ) -> List[Dict[str, Any]]:
        """
        Run multiple prompts asynchronously through the rate-limited OpenAI backend.
        """
        ids = list(ids) if ids is not None else list(range(len(items)))

        formatted_prompts = []
        print(f"Items: {items}")
        for item in items:
            msg = user_message_template
            for k, v in item.items():
                msg = msg.replace(f"{{{k}}}", str(v))
            formatted_prompts.append(msg)

        # Execute async API calls concurrently
        tasks = [self._call_openai(system_message, user_msg) for user_msg in formatted_prompts]
        responses = await asyncio.gather(*tasks)
        return await self.process_json_responses(responses, ids, prompt_name)


# -----------------------------------------------------------------------------
# GraphProcessor - For running LangGraph graphs asynchronously
# -----------------------------------------------------------------------------
class GraphProcessor:
    """
    Process graph executions asynchronously using LangGraph runnables.

    Similar to PromptProcessor but designed for running LangGraph graphs instead of
    OpenAI API calls. Handles DataFrame input, graph execution, and result processing.
    """

    def __init__(
        self,
        graph_runnable: Any,
        input_file: Optional[str] = None,
        output_dir: str = "./output",
        input_df: Optional[pd.DataFrame] = None,
        graph_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            graph_runnable: LangGraph runnable instance (e.g., TestCaseReviewerRunnable)
            input_file: Optional CSV/Excel file path
            output_dir: Output directory for any saved results
            input_df: Preloaded DataFrame of input data
            graph_kwargs: Additional parameters to pass to graph execution
        """
        self.graph_runnable = graph_runnable
        self.input_file = input_file
        self.output_dir = output_dir
        self.input_df = input_df
        self.graph_kwargs = graph_kwargs or {}

        # Load data if needed
        if self.input_df is None and self.input_file:
            self.input_df = load_input_data(self.input_file)
        elif self.input_df is None and not self.input_file:
            raise ValueError("Either input_file or input_df must be provided")

    async def _run_graph(self, graph_input: Dict[str, Any]) -> Any:
        """
        Run a single graph execution asynchronously.

        Args:
            graph_input: Dictionary of input parameters for the graph

        Returns:
            The graph output (structure depends on graph implementation)
        """
        try:
            # Run the graph with provided input and any additional kwargs
            result = await self.graph_runnable.arun(**graph_input, **self.graph_kwargs)
            return result
        except Exception as e:
            return {"error": str(e), "input": graph_input}

    async def process_graph_results(
        self,
        results: Sequence[Any],
        ids: Sequence[Any],
        graph_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Process graph execution results into structured dictionaries.

        Similar to process_json_responses but for graph outputs.

        Args:
            results: Sequence of graph execution results
            ids: Sequence of IDs corresponding to each result
            graph_name: Name/identifier for the graph type

        Returns:
            List of dictionaries with flattened result data
        """
        processed: List[Dict[str, Any]] = []

        for i, result in enumerate(results):
            base_output: Dict[str, Any] = {
                "item_id": ids[i],
                "graph_name": graph_name,
            }

            # -------------------------------------------------
            # Handle None or error results
            # -------------------------------------------------
            if result is None:
                processed.append({
                    **base_output,
                    "error": "Graph execution returned None",
                })
                continue

            if isinstance(result, dict) and "error" in result:
                processed.append({
                    **base_output,
                    **result,
                })
                continue

            # -------------------------------------------------
            # Process successful graph results
            # -------------------------------------------------
            try:
                # If result is a Pydantic model or has __dict__, extract attributes
                if hasattr(result, "__dict__"):
                    result_dict = {}
                    for key, value in result.__dict__.items():
                        # Skip private attributes
                        if not key.startswith("_"):
                            result_dict[key] = value

                    # Flatten nested structures using flatdict
                    flat = flatdict.FlatDict(result_dict, delimiter=".")
                    processed.append({
                        **base_output,
                        **dict(flat),
                        "raw_result": str(result),
                    })

                # If result is already a dict
                elif isinstance(result, dict):
                    flat = flatdict.FlatDict(result, delimiter=".")
                    processed.append({
                        **base_output,
                        **dict(flat),
                    })

                # If result is a simple type
                else:
                    processed.append({
                        **base_output,
                        "result": result,
                    })

            except Exception as e:
                processed.append({
                    **base_output,
                    "processing_error": str(e),
                    "raw_result": str(result),
                })

        return processed

    async def run_graph_batch(
        self,
        items: Sequence[Dict[str, Any]],
        ids: Optional[Sequence[Any]] = None,
        graph_name: str = "graph_execution",
        ) -> List[Dict[str, Any]]:
        """
        Run multiple graph executions asynchronously.

        This is the graph equivalent of PromptProcessor.run_prompt_batch().

        Args:
            items: Sequence of input dictionaries for graph execution
            ids: Optional sequence of IDs for tracking results
            graph_name: Name/identifier for the graph type

        Returns:
            List of processed result dictionaries

        Example:
            >>> processor = GraphProcessor(graph_runnable=reviewer_graph)
            >>> items = [
            ...     {"requirement": req1, "test": tc1},
            ...     {"requirement": req2, "test": tc2},
            ... ]
            >>> results = await processor.run_graph_batch(items, ids=["TC-001", "TC-002"])
        """
        ids = list(ids) if ids is not None else list(range(len(items)))

        print(f"🚀 Starting graph execution for {len(items)} items...")
        start_time = time.time()

        # Execute graph runs concurrently
        tasks = [self._run_graph(item) for item in items]
        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start_time
        print(f"✅ Completed {len(results)} graph executions in {elapsed:.2f} seconds")

        # Process and flatten results
        return await self.process_graph_results(results, ids, graph_name)