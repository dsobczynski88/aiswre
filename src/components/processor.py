import os
import re
import json
import asyncio
import time
import ast
import pandas as pd
import flatdict
from typing import Any, Dict, List, Optional, Sequence
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
            self.input_df = self.load_input_data()
        elif self.input_df is None and not self.input_file:
            raise ValueError("Either input_file or input_df must be provided")

    def load_input_data(self) -> pd.DataFrame:
        """Load CSV or Excel file into a pandas DataFrame."""
        if not self.input_file:
            raise ValueError("input_file is not provided")

        _, ext = os.path.splitext(self.input_file)
        ext = ext.lower()

        if ext == ".csv":
            return pd.read_csv(self.input_file)
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(self.input_file)
        raise ValueError(f"Unsupported input file extension: {ext}")

    def df_to_prompt_items(
        self,
        df: pd.DataFrame,
        columns: Optional[Sequence[str]] = None
        ) -> List[Dict[str, Any]]:
        """Convert each row of dataframe into a dict suitable for prompt templating."""
        if columns is None:
            columns = df.columns.tolist()

        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        records = df[columns].to_dict(orient="records")

        # Explicitly coerce keys to str to satisfy type checker and intended use
        return [{str(k): v for k, v in record.items()} for record in records]

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
# Asynchronous top-level runner with dependency injection
# -----------------------------------------------------------------------------
async def main(
    system_message: str = "You are a code review assistant.",
    user_template: str = "Analyze this code snippet:\nLanguage: {language}\nCode:\n{code_snippet}",
    input_df: Optional[pd.DataFrame] = None,
    prompt_name: str = "code-review",
    model: str = "gpt-4o-mini",
    output_dir: str = "./output",
    output_filename: str = "results.xlsx",
    *,
    client: Optional[RateLimitOpenAIClient] = None,
    processor: Optional[PromptProcessor] = None,
    ) -> pd.DataFrame:
    """
    Main orchestrator for running batched LLM prompt processing.

    Either an initialized `processor` or a `client` must be supplied.

    Returns:
        A pandas DataFrame with all processed results.
    """
    os.makedirs(output_dir, exist_ok=True)

    if input_df is None:
        raise ValueError("input_df must be provided to run main().")

    if processor is not None:
        proc = processor
    else:
        if client is None:
            raise ValueError("Either `processor` or `client` must be provided.")
        proc = PromptProcessor(client=client, input_df=input_df, model=model, output_dir=output_dir)

    items = proc.df_to_prompt_items(input_df)
    ids = list(range(len(items)))

    print(f"🚀 Starting code review for {len(items)} items...")
    start_time = time.time()

    results = await proc.run_prompt_batch(
        system_message=system_message,
        user_message_template=user_template,
        prompt_name=prompt_name,
        items=items,
        ids=ids,
    )

    elapsed = time.time() - start_time
    print(f"✅ Processed {len(results)} items in {elapsed:.2f} seconds total")

    df_out = pd.DataFrame(results)
    output_path = os.path.join(output_dir, output_filename)
    df_out.to_excel(output_path, index=False)
    print(f"📁 Results saved to {output_path}")

    return df_out