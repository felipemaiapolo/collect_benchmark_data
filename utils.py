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

MODELS = {'openai': ['o1-preview-2024-09-12',
                     'gpt-4o-mini-2024-07-18',
                     'gpt-4o-2024-05-13'],
         'anthropic': ['claude-3-5-sonnet-20241022'],
         'google':['gemini-1.5-pro-002',
                   'gemini-1.5-flash-002',
                   'gemini-1.5-flash-8b-001'],
         'mistral':["mistral-large-2407",
                    "mistral-small-2409"],
         'togetherai':["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                       "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                       "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                       'google/gemma-2-27b-it'],
         'hf':['Qwen/Qwen2.5-72B-Instruct',
               'Qwen/Qwen2.5-32B-Instruct',
               'Qwen/Qwen2.5-7B-Instruct',
               'google/gemma-2-9b-it',
               'google/gemma-2-2b-it'],
         'rits':["ibm-granite/granite-3.0-8b-instruct"],
         'local':['ibm-granite/granite-3.0-2b-instruct']}

OPENAI_API_KEY=''
ANTHROPIC_API_KEY=''
GEMINI_API_KEY=''
TOGETHER_API_KEY=''
MISTRAL_API_KEY =''
HF_TOKEN=''
RITS_API_KEY=''

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
        • True  – force gzip mode  
        • False – force plain-text mode  
        • None  – infer from file name (default)
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
    api = GetAPI(model)

    if api == 'openai':
        client = OpenAI(api_key=OPENAI_API_KEY)  # the key is picked up automatically

        response = client.chat.completions.create(
            model = model,     # pick any model you have access to
            messages=[
                {"role": "user",   "content": prompt}
            ]                  
        )
        resp = response.choices[0].message.content

    elif api=='google':
        genai.configure(api_key=GEMINI_API_KEY)
        generation_config = {
            "temperature": 1.0,
            "top_p": 1,
            "top_k": 1
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"}
        ]

        model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
    
        # Generate response
        response = model.generate_content(prompt)
        resp = response.text.strip()

    elif api=='mistral':
        api_key = MISTRAL_API_KEY
        client = Mistral(api_key=api_key)
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        chat_response = client.chat.complete(
            model=model,
            messages=messages,
        )
        resp = chat_response.choices[0].message.content
        
    elif api=='anthropic':
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
                            model=model,
                            max_tokens=3000,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
        resp = message.content[0].text

    elif api=='togetherai':
        client = Together(api_key=TOGETHER_API_KEY)

        stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1,
                stop=["<|eot_id|>", "<|eom_id|>"],
                stream=True
        )

        # Concatenate the streamed response content
        answer = ""
        for chunk in stream:
            # Safely extract the content from each streamed chunk
            answer += chunk.choices[0].delta.content or ""
        
        # Replace "Yes" with the generated answer in the specific column
        resp = answer.strip()

    elif api=='hf':
        model_dict = {"Qwen/Qwen2.5-7B-Instruct":"together",
                "Qwen/Qwen2.5-72B-Instruct":"hyperbolic",
                "Qwen/Qwen2.5-32B-Instruct":"featherless-ai",
                "google/gemma-2-2b-it":"nebius",
                "google/gemma-2-9b-it":"nebius"}

        provider = model_dict[model]

        client = InferenceClient(
            provider=provider,
            api_key=HF_TOKEN,
        )

        if model in ["Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2.5-32B-Instruct"]:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user","content": prompt}],
            )
            resp = completion.choices[0].message.content

        elif model in ["Qwen/Qwen2.5-72B-Instruct"]:
            completion = client.chat.completions.create(
                temperature=0.7,
                top_p=0.9,
                model=model,
                stream=True,
                messages=[{"role": "user","content": prompt}],
            )
            resp = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content  # Get the current content
                resp += content  # Append to the full response

        elif model in ["google/gemma-2-2b-it"]:
            completion = client.chat.completions.create(
                temperature=0.7,
                top_p=0.9,
                model=model,
                messages=[{"role": "user","content": prompt}],
            )
            resp = completion.choices[0].message.content

        elif model in ["google/gemma-2-9b-it"]:
            completion = client.chat.completions.create(
                temperature=0.7,
                top_p=0.7,
                #top_k=50,
                #repetition_penalty=1,
                model=model,
                stop=["<|eot_id|>", "<|eom_id|>"],
                stream=True,
                messages=[{"role": "user","content": prompt}],
            )
            resp = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content  # Get the current content
                resp += content  # Append to the full response
        else:
            raise()
    elif api=='rits':
        RITS_ENVS = { 
            "ibm-granite/granite-3.0-8b-instruct":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-0-8b-instruct/v1"
            }
        client = OpenAI(
                    api_key=RITS_API_KEY,  
                    base_url=RITS_ENVS[model],
                    default_headers={'RITS_API_KEY': RITS_API_KEY}
                )

        messages = [
                    {"role": "user", "content": prompt}
                ]

        # Make the API call using the provided model and messages.
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            n=1
        )

        resp = completion.choices[0].message.content.strip()
    else:
        raise()
        
    return resp
    
def QueryModelError(prompt, model, max_tries=10):
    success = False
    tries = 0
    while not success:
        if tries==max_tries-1:
            return "NA"
        try:
            resp = QueryModel(prompt, model)
            success = True
        except:
            time.sleep(5)
            tries+=1
    return resp

def QueryJudgeError(prompt, max_tries=10):
    JUDGE_MODEL = 'gpt-4o-mini-2024-07-18'
    success = False
    tries = 0
    while not success:
        if tries==max_tries:
            return "NA"
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)  
            response = client.chat.completions.create(
                model = JUDGE_MODEL,     # pick any model you have access to
                temperature = 0,
                messages=[
                    {"role": "user",   "content": prompt}
                ]                  
            )
            resp = response.choices[0].message.content
            success = True
        except:
            time.sleep(5)
            tries+=1
    return resp

def GetJudgePrompt(prompt, model_response, gold_response):
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