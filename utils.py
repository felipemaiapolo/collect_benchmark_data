from __future__ import annotations

import json
import gzip
import os
import time
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Iterator, List, Union

import anthropic
import google.generativeai as genai
from huggingface_hub import InferenceClient
from mistralai import Mistral, UserMessage
import openai
from openai import OpenAI
import requests
from together import Together

from config import (
    MODEL_CONFIGS, MODEL_API_NAMES, QWEN3_MODELS, HF_ENDPOINTS, BATCH_MODELS,
    JUDGE_CONFIG, OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY,
    TOGETHER_API_KEY, MISTRAL_API_KEY, HF_TOKEN, RITS_API_KEY, OPENROUTER_API_KEY
)


MODELS = {
    'openai': [
        'gpt-3.5-turbo-1106',
        'gpt-4-0613',
        'gpt-4o-2024-11-20',
        'gpt-4o-mini-2024-07-18',
        'gpt-5-2025-08-07',
        'gpt-5-mini-2025-08-07',
        'gpt-5-nano-2025-08-07',
        'gpt-5-2025-08-07-thinking',
        'o3-2025-04-16',
        'o4-mini-2025-04-16',
    ],
    'anthropic': [
        'claude-3-haiku-20240307',
        'claude-3-5-haiku-20241022',
        'claude-opus-4-1-20250805',
        'claude-3-7-sonnet-20250219',
        'claude-sonnet-4-20250514',
    ],
    'google': [
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite',
        'gemini-2.5-pro',
        'gemma-3-1b-it',
    ],
    'mistral': [
        "mistral-medium-2505",
    ],
    'togetherai': [
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        'google/gemma-2-27b-it',
    ],
    'hf': [
        'deepseek-ai/DeepSeek-R1',
        'deepseek-ai/DeepSeek-V3',
        'openai/gpt-oss-20b',
        'openai/gpt-oss-120b',
        'meta-llama/Llama-3.1-8B-Instruct',
        'meta-llama/Llama-3.1-70B-Instruct',
        #'Qwen/Qwen2-7B-Instruct',
        #'Qwen/Qwen2-72B-Instruct',
        'Qwen/Qwen3-32B',
        'Qwen/Qwen3-14B',
        'Qwen/QwQ-32B',
    ],
    'openrouter': [
        'meta-llama/Llama-3.1-405B-Instruct',
         'meta-llama/llama-4-scout',
         'meta-llama/llama-4-maverick',
         #'qwen/qwq-32b',
         'Qwen/Qwen3-235B-A22B-Instruct-2507',
    ],
    'rits': [
        "ibm-granite/granite-3.0-8b-instruct",
    ],
    'local': [
        'ibm-granite/granite-3.0-2b-instruct',
    ],
    'hf_endpoint': []
}

# Populate HF endpoint models
for endpoint_name, endpoint_config in HF_ENDPOINTS.items():
    MODELS['hf_endpoint'].extend(endpoint_config['models'])


class ModelLookupError(ValueError):
    """Raised when a model is missing or ambiguous."""


def GetAPI(model_name: str, catalog: Dict[str, List[str]] = MODELS) -> str:
    """
    Return the provider/API name that hosts `model_name`.

    Parameters
    ----------
    model_name : str
        The exact model identifier to look up.
    catalog : dict[str, list[str]], optional
        Mapping of provider → list of model names.  Defaults to ``MODELS``.

    Returns
    -------
    str
        The single provider that contains the model.

    Raises
    ------
    ModelLookupError
        If the model is not found or appears under more than one provider.
    """
    matches = [provider for provider, names in catalog.items() if model_name in names]

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise ModelLookupError(f"Model '{model_name}' not found in any provider.")

    raise ModelLookupError(
        f"Model '{model_name}' appears in more than one provider: {matches}"
    )


def GetAPIModelName(model: str) -> str:
    """
    Get the actual API model name from internal model name.
    
    For models with date suffixes or special variants (e.g., 'gpt-5-2025-08-07-thinking'),
    this returns the actual model name to use in API calls (e.g., 'gpt-5').
    
    Parameters
    ----------
    model : str
        Internal model name used in your scripts
    
    Returns
    -------
    str
        Actual model name to use in API calls
    """
    return MODEL_API_NAMES.get(model, model)


def GetHFEndpoint(model_name: str) -> Dict[str, str]:
    """Get HuggingFace endpoint configuration for a model"""
    for endpoint_name, endpoint_config in HF_ENDPOINTS.items():
        if model_name in endpoint_config['models']:
            return {
                'url': endpoint_config['url'],
                'api_key': endpoint_config['api_key']
            }
    raise ValueError(f"Model {model_name} not found in HF_ENDPOINTS")


def ApplyPromptModifications(prompt: str, model: str) -> str:
    """Apply model-specific prompt modifications"""
    # Add suffix for Qwen 3 models
    if model in QWEN3_MODELS:
        prompt = prompt + "\n/set nothink"
    
    return prompt


def GetModelConfig(model: str) -> Dict[str, Any]:
    """Get configuration parameters for a specific model"""
    return MODEL_CONFIGS.get(model, {}).copy()


def GetJudgeConfig() -> Dict[str, Any]:
    """Get configuration for judge model"""
    return JUDGE_CONFIG.copy()


def IsBatchModel(model: str) -> tuple[bool, str]:
    """Check if model supports batch API and return provider"""
    for provider, models in BATCH_MODELS.items():
        if model in models:
            return True, provider
    return False, None


def SaveJson(
    data: Dict[str, Any],
    path: Union[str, Path],
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Write a Python dictionary to a JSON file.

    Parameters
    ----------
    data         : The dictionary (or any JSON-serialisable object) to save.
    path         : Destination filename or `pathlib.Path`.
    indent       : How many spaces to indent.  Use `None` for the most compact form.
    ensure_ascii : If True, escape non-ASCII characters (default: keep them as-is).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # Create folders if missing
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def LoadJson(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a JSON file back into a Python dictionary.

    Parameters
    ----------
    path : Filename or `pathlib.Path`.

    Returns
    -------
    dict  : The JSON contents parsed into a dictionary (or whatever object was saved).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    

def SaveJsonl(
    data: Iterable[Mapping],
    path: str | Path,
    *,
    compress: bool = False,
    ensure_ascii: bool = False,
) -> None:
    """
    Save an iterable of dictionaries (or any JSON-serialisable objects)
    to a JSON Lines file.

    Parameters
    ----------
    data
        Iterable whose elements will become one JSON object per line.
    path
        Destination filename. If ``compress=True`` and the name does not
        already end in ``.gz``, the suffix ``.gz`` is appended.
    compress
        Write with gzip compression (text mode).  Useful for large files.
    ensure_ascii
        Passed to ``json.dump``.  Keep it ``False`` to preserve non-ASCII
        characters; set ``True`` to escape them (ASCII-only output).

    Examples
    --------
    >>> records = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    >>> save_jsonl(records, "records.jsonl")          # plain text
    >>> save_jsonl(records, "records.jsonl", compress=True)   # gzip
    """

    path = Path(path)
    if compress and path.suffix != ".gz":
        path = path.with_suffix(path.suffix + ".gz")

    open_fn = gzip.open if compress else open

    with open_fn(path, "wt", encoding="utf-8") as f:
        for obj in data:
            json.dump(obj, f, ensure_ascii=ensure_ascii)
            f.write("\n")


JsonDict = Mapping[str, Any]
def LoadJsonl(
    path: Union[str, Path],
    *,
    compress: bool | None = None,
    discard_blank: bool = True,
    skip_invalid: bool = False,
) -> List[JsonDict]:
    """
    Read a JSON-Lines file written by ``SaveJsonl`` and return a list of dicts.

    Parameters
    ----------
    path
        File to read.  If ``compress`` is *None*, a ``.gz`` suffix is taken
        to mean gzip compression.
    compress
        • True  — force gzip mode  
        • False — force plain-text mode  
        • None  — infer from file name (default)
    discard_blank
        Ignore empty / whitespace-only lines instead of raising an error.
    skip_invalid
        Skip lines that fail JSON parsing instead of raising.

    Returns
    -------
    list[dict]
        One element per line of the input file.
    """
    path = Path(path)
    if compress is None:
        compress = path.suffix == ".gz"

    open_fn = gzip.open if compress else open
    records: List[JsonDict] = []

    with open_fn(path, "rt", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line and discard_blank:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                if skip_invalid:
                    continue
                raise ValueError(f"Invalid JSON on line {lineno}: {e}") from e

    return records


def QueryModel(prompt, model):
    """Query a model with automatic configuration and prompt modifications"""
    
    # Apply prompt modifications (e.g., Qwen suffix)
    prompt = ApplyPromptModifications(prompt, model)
    
    # Get model configuration
    model_config = GetModelConfig(model)
    
    # Get actual API model name
    api_model_name = GetAPIModelName(model)
    
    api = GetAPI(model)

    if api == 'openai':
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=api_model_name,
            messages=[{"role": "user", "content": prompt}],
            **model_config
        )
        resp = response.choices[0].message.content

    elif api == 'google':
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Extract generation config from model_config
        generation_config = {}
        safety_settings = model_config.pop('safety_settings', [])
        
        # Add other generation parameters
        if 'temperature' in model_config:
            generation_config['temperature'] = model_config['temperature']
        if 'top_p' in model_config:
            generation_config['top_p'] = model_config['top_p']
        if 'max_output_tokens' in model_config:
            generation_config['max_output_tokens'] = model_config['max_output_tokens']

        model_instance = genai.GenerativeModel(
            model_name=api_model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
    
        response = model_instance.generate_content(prompt)
        try:
            resp = response.text.strip()
        except ValueError as e:
            # Response was blocked by safety filters
            print(f"  ⚠️  Gemini blocked response for {model}: {str(e)[:100]}")
            
            # Log details about why it was blocked
            if hasattr(response, 'prompt_feedback'):
                print(f"  ⚠️  Prompt feedback: {response.prompt_feedback}")
            
            if response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'safety_ratings'):
                    blocked_categories = [
                        rating.category.name 
                        for rating in candidate.safety_ratings 
                        if rating.probability.name in ['HIGH', 'MEDIUM']
                    ]
                    if blocked_categories:
                        print(f"  ⚠️  Blocked categories: {blocked_categories}")
                if hasattr(candidate, 'finish_reason'):
                    print(f"  ⚠️  Finish reason: {candidate.finish_reason}")
            
            # Return marker for blocked content
            resp = "[BLOCKED_BY_SAFETY_FILTER]"

    elif api == 'mistral':
        client = Mistral(api_key=MISTRAL_API_KEY)
        messages = [{"role": "user", "content": prompt}]
        
        chat_response = client.chat.complete(
            model=api_model_name,
            messages=messages,
            **model_config
        )
        resp = chat_response.choices[0].message.content
        
    elif api == 'anthropic':
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Anthropic requires messages in the config
        messages = [{"role": "user", "content": prompt}]
        
        message = client.messages.create(
            model=api_model_name,
            messages=messages,
            **model_config
        )
        resp = message.content[0].text

    elif api == 'togetherai':
        client = Together(api_key=TOGETHER_API_KEY)
        
        # Extract stop sequences if present
        stop_sequences = model_config.pop('stop', ["<|eot_id|>", "<|eom_id|>"])

        stream = client.chat.completions.create(
            model=api_model_name,
            messages=[{"role": "user", "content": prompt}],
            stop=stop_sequences,
            stream=True,
            **model_config
        )

        answer = ""
        for chunk in stream:
            answer += chunk.choices[0].delta.content or ""
        
        resp = answer.strip()

    elif api == 'hf':
        # Models that use HF router (OpenAI-compatible endpoint)
        hf_router_models = {
            'deepseek-ai/DeepSeek-R1',
            'deepseek-ai/DeepSeek-V3',
            'openai/gpt-oss-20b',
            'openai/gpt-oss-120b',
            'meta-llama/Llama-3.1-8B-Instruct',
            'meta-llama/Llama-3.1-70B-Instruct',
            #'Qwen/Qwen2-7B-Instruct',
            #'Qwen/Qwen2-72B-Instruct',
            'Qwen/Qwen3-32B',
            'Qwen/Qwen3-14B',
            'Qwen/QwQ-32B',
        }
        
        # Check if this model uses HF router
        if model in hf_router_models:
            # Use OpenAI-compatible HF router
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=HF_TOKEN,
            )
            
            response = client.chat.completions.create(
                model=api_model_name,
                messages=[{"role": "user", "content": prompt}],
                **model_config
            )
            resp = response.choices[0].message.content
        
        else:
            # Existing InferenceClient logic for old models
            model_dict = {
                "Qwen/Qwen2.5-7B-Instruct": "together",
                "Qwen/Qwen2.5-72B-Instruct": "hyperbolic",
                "Qwen/Qwen2.5-32B-Instruct": "featherless-ai",
                "google/gemma-2-2b-it": "nebius",
                "google/gemma-2-9b-it": "nebius"
            }
            
            provider = model_dict.get(model)
            if provider:
                client = InferenceClient(provider=provider, api_key=HF_TOKEN)
            else:
                client = InferenceClient(api_key=HF_TOKEN)

            if model in ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-32B-Instruct"]:
                completion = client.chat.completions.create(
                    model=api_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **model_config
                )
                resp = completion.choices[0].message.content

            elif model in ["Qwen/Qwen2.5-72B-Instruct"]:
                # Remove streaming-incompatible params
                stream_config = {k: v for k, v in model_config.items() if k in ['temperature', 'top_p']}
                
                completion = client.chat.completions.create(
                    model=api_model_name,
                    stream=True,
                    messages=[{"role": "user", "content": prompt}],
                    **stream_config
                )
                resp = ""
                for chunk in completion:
                    content = chunk.choices[0].delta.content
                    if content:
                        resp += content

            elif model in ["google/gemma-2-2b-it"]:
                completion = client.chat.completions.create(
                    model=api_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **model_config
                )
                resp = completion.choices[0].message.content

            elif model in ["google/gemma-2-9b-it"]:
                # Extract stop sequences
                stop_sequences = model_config.pop('stop', ["<|eot_id|>", "<|eom_id|>"])
                stream_config = {k: v for k, v in model_config.items() if k in ['temperature', 'top_p']}
                
                completion = client.chat.completions.create(
                    model=api_model_name,
                    stop=stop_sequences,
                    stream=True,
                    messages=[{"role": "user", "content": prompt}],
                    **stream_config
                )
                resp = ""
                for chunk in completion:
                    content = chunk.choices[0].delta.content
                    if content:
                        resp += content
            else:
                raise ValueError(f"Unknown HF model: {model}")
            
    elif api == 'hf_endpoint':
        # Use custom HuggingFace endpoint
        endpoint_config = GetHFEndpoint(model)
        
        client = OpenAI(
            base_url=endpoint_config['url'],
            api_key=endpoint_config['api_key']
        )
        base_models = ['ibm-granite/granite-3.3-8b-base', 'ibm-granite/granite-3.3-2b-base']
        if model in base_models:
            # Use completions API for base models
            response = client.completions.create(
                model=api_model_name,
                prompt=prompt
                # max_tokens=model_config.get('max_tokens', 2048),
                # temperature=model_config.get('temperature', 0.7),
                # stream=False
            )
            resp = response.choices[0].text.strip()
        else:
            response = client.chat.completions.create(
                model=api_model_name,  # HF endpoints typically use "tgi" as model name
                messages=[{"role": "user", "content": prompt}],
                **model_config
                )
            resp = response.choices[0].message.content
        
        # response = client.chat.completions.create(
        #     model=api_model_name,  # HF endpoints typically use "tgi" as model name
        #     messages=[{"role": "user", "content": prompt}],
        #     **model_config
        # )
        # resp = response.choices[0].message.content

    elif api == 'openrouter':
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/your-username/your-repo",  # Optional
                "X-Title": "AI Benchmark Research",  # Optional
            },
            model=api_model_name,
            messages=[{"role": "user", "content": prompt}],
            **model_config
        )
        resp = response.choices[0].message.content 
           
    elif api == 'rits':
        RITS_ENVS = { 
            "ibm-granite/granite-3.0-8b-instruct": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-0-8b-instruct/v1"
        }
        client = OpenAI(
            api_key=RITS_API_KEY,  
            base_url=RITS_ENVS[model],
            default_headers={'RITS_API_KEY': RITS_API_KEY}
        )

        messages = [{"role": "user", "content": prompt}]

        completion = client.chat.completions.create(
            model=api_model_name,
            messages=messages,
            **model_config
        )

        resp = completion.choices[0].message.content.strip()
    else:
        raise ValueError(f"Unknown API: {api}")
        
    return resp


def QueryModelError(prompt, model, max_tries=10):
    """Query model with retry logic"""
    success = False
    tries = 0
    while not success:
        if tries == max_tries - 1:
            return "NA"
        try:
            resp = QueryModel(prompt, model)
            success = True
        except Exception as e:
            print(f"Error querying {model}: {e}")
            time.sleep(5)
            tries += 1
    return resp


def QueryJudgeError(prompt, max_tries=10):
    """Query judge model with retry logic and timeout handling"""
    judge_config = GetJudgeConfig()
    judge_model = judge_config.pop('model')
    
    success = False
    tries = 0
    while not success:
        if tries == max_tries:
            print(f"  ⚠️  Judge model failed after {max_tries} attempts, returning NA")
            return "NA"
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                timeout=60.0,  # Add explicit 60 second timeout
                **judge_config
            )
            resp = response.choices[0].message.content
            success = True
        except Exception as e:
            error_msg = str(e).lower()
            print(f"  ⚠️  Error querying judge model (attempt {tries+1}/{max_tries}): {e}")
            
            # Different wait times for different errors
            if "timeout" in error_msg:
                wait_time = 10  # Wait 10 seconds for timeout
            elif "rate" in error_msg or "limit" in error_msg:
                wait_time = 30  # Wait 30 seconds for rate limits
            else:
                wait_time = 5  # Wait 5 seconds for other errors
            
            print(f"  ⏳ Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
            tries += 1
    
    return resp


def GetJudgePrompt(prompt, model_response, gold_response):
    """Generate judge prompt for evaluation"""
    MODELTASK = prompt
    MODELRESPONSE = model_response
    GOLDENRESPONSE = gold_response
                   
    judge_prompt = r'''I want you to act as a judge for how well a model did answering a user-defined task.
You will be provided with a user-defined task given to the model, its golden answer(s), and the model's answer. The context of the task may not be given here. Your task is to first analyze and justify in detail how the model’s answer conforms to or contradicts the golden answer(s). Please note that the formats of the gold and model responses may differ; therefore, focus primarily on the content rather than on formatting discrepancies. After providing your analysis, assign a correctness score indicating how correct the model’s answer is, based on your justification. The correctness score should be one of the following:
0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right).
Your response must strictly follow this format:
{{
   "justification": your_justification,
   "correctness_score": your_score
}}
Note that each one of the golden answers is considered correct. Thus if the Model’s Answer matches any
one of the golden answers, it should be considered correct. First, I will provide you some examples on how to judge properly.
<dmf>assistant
Certainly! I will act as a judge and adhere to the format you requested. Under oath, I will judge without consulting my internal knowledge of facts under any circumstances.
<dmf>user
User-defined task --- Sandy bought 1 million Safe Moon tokens. She has 4 siblings. She wants to keep half of them
to herself and divide the remaining tokens among her siblings. After splitting it up, how many more tokens
will she have than any of her siblings?
Golden Answer(s) --- \<answer 1\> 375000
Model’s Answer --- Sandy will have more tokens than any sibling by 3/8 million.
<dmf>assistant
{{
    "justification": "The golden answer states that Sandy will have 375,000 more tokens than any of her
siblings, which is a precise numerical value. The model’s answer translates this scenario into a fraction
of the total, saying Sandy will have more tokens than any sibling by 3/8 million. 1 million tokens * 3/8 =
375,000 tokens. So the model provided an answer in fractional form that, when converted to a numerical
value, exactly matches the golden answer’s quantity.",
    "correctness_score": 1.0
}}
<dmf>user
User-defined task --- Extract the names and emails mentioned in the following text.
Richard Matthew Stallman (/ˈstɔːlmən/ STAWL-mən; born March 16, 1953), also known by his initials, rms,[1] is an American free software movement activist and programmer. He can be reached at rms@gnu.org based on the information on his webpage.
Golden Answer(s) --- {{"names":  ["Richard Matthew Stallman"], "emails": ["rms@gnu.org"]}}
Model’s Answer --- names: Richard Matthew Stallman
<dmf>assistant
{{
    "justification": "The model correctly identified the names but failed to extract the emails.",
    "correctness_score": 0.5
}}
<dmf>user
User defined task --- Question: In 2015 Edgar Lungu became prime minister of?
Golden Answer: <answer 1> Zambia; <answer 2> Zamibia; <answer 3> People of Zambia; <answer
4> Zambian cuisine; <answer 5> Zambians; <answer 6> Culture of Zambia; <answer 7> Etymology of
Zambia; <answer 8> Zambia; <answer 9> Health care in Zambia; <answer 10> ISO 3166-1:ZM; <answer
11> Republic Of Zambia; <answer 12> Cuisine of Zambia; <answer 13> Sport in Zambia; <answer 14>
Republic of Zambia; <answer 15> Zambian people; <answer 16> Name of Zambia
Model’s Answer: Prime Minister
<dmf>assistant
{{
    "justification": "The golden answers provide a detailed list of entities all relating to Zambia, indicating
that Edgar Lungu became the leader (specifically, they mentioned \"prime minister\") of Zambia in 2015.
The model’s answer, \"Prime Minister,\" merely repeats part of the question without answering it.",
    "correctness_score": 0.0
}}
<dmf>user
User defined task --- Give a one-line concise summary of the following news
PARIS (Reuters) - French President Emmanuel Macron will set out plans for reforming the European Union on Tuesday, including proposals for a separate eurozone budget, despite a German election result that is likely to complicate his far-reaching ambitions. German Chancellor Angela Merkel s conservatives saw their support slide in Sunday s election, though they remain the biggest parliamentary bloc. She is expected to seek a coalition with the liberal Free Democrats (FDP) - who have criticized Macron s ideas for Europe - and the Greens. Elysee officials said Macron, who has promised sweeping reforms to Europe s monetary union in coordination with Merkel, hoped the issues to be raised in his speech would be taken into account in Germany s coalition negotiations. One Elysee official said a eurozone budget, one of Macron s most contentious ideas, would be necessary in due course and that the president would therefore raise the issue in his speech, to be delivered at the Sorbonne University in Paris. Since his election in May, Macron has made the overhaul of the EU and its institutions one of his major themes. As well as his eurozone budget idea, he wants to see the appointment of a eurozone finance minister and the creation of a rescue fund that would preemptively help countries facing economic trouble. Ahead of Sunday s election, Merkel had indicated her willingness to work with Macron on a reform agenda, even if her own ideas may not reach as far as his. But the election results have left Merkel facing a difficult coalition-building task which is in turn likely to limit her flexibility on Europe. A coalition of Merkel s CDU/CSU bloc, the FDP and the Greens is unprecedented at the national level - and any attempt by the chancellor and Macron to press for greater EU integration will face opposition from the new German lower house Bundestag. The FDP has called for a phasing out of Europe s ESM bailout fund and changes to EU treaties that would allow countries to leave the euro zone. And the far-right, eurosceptic Alternative for Germany is now the third biggest party in the Bundestag, further curbing Merkel s room for maneuver. But Elysee officials noted that the FDP had reaffirmed its attachment to the EU and to strong Franco-German relations, a point Macron was likely to emphasize in his speech while at the same time not seeking to impose anything on his partners. Macron, the sources said, would propose that the whole EU move forward together, and that those who did not want to should not stand in the way of those that did. Coming just two days after the German election, Macron s speech is likely to be interpreted in Germany as an attempt to shape the debate before the coalition talks begin in earnest. German coalition agreements are strict, with the contours set out in them limiting the government s room for maneuver. In that respect, Merkel s ability to work with Macron on EU reform will be pre-determined by whatever coalition deal is struck. FDP leader Christian Lindner said on Monday he would not agree to any coalition that did not promise a change in the German government s direction. While that appeared to set the stage for tough talks, he also offered hope for Macron, saying when asked about the French president s eurozone budget ideas that the FDP had a strong interest in the strength of France.
Golden Answer(s) --- After German election, Macron to set out his vision for Europe
Model's Answer --- French President Emmanuel Macron to introduce plans for reforming the European Union amid the uncertain aftermath of German elections
<dmf>assistant
{{
    "justification": "While the golden answer is more concise, the model's answer is largely similar to the golden answer in its meaning.",
    "correctness_score": 0.6   
}}
<dmf>user
User defined task --- {MODELTASK}
Golden Answer(s) --- {GOLDENRESPONSE}
Model’s Answer --- {MODELRESPONSE}
<dmf>assistant'''.format(MODELTASK=MODELTASK,GOLDENRESPONSE=GOLDENRESPONSE,MODELRESPONSE=MODELRESPONSE)
    
    return judge_prompt