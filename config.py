"""
Configuration file for model settings, endpoints, and batch parameters
"""

# ============================================================================
# MODEL API NAME MAPPING
# ============================================================================
# Maps internal model names (with dates/suffixes) to actual API model names
MODEL_API_NAMES = {
    # GPT-5 family - internal names with dates map to simple API names
    'gpt-5-2025-08-07': 'gpt-5',
    'gpt-5-2025-08-07-thinking': 'gpt-5',
    #'Qwen/Qwen2-7B-Instruct': 'Qwen/Qwen2-7B-Instruct:featherless-ai',
    #'Qwen/Qwen2-72B-Instruct': 'Qwen/Qwen2-72B-Instruct:featherless-ai',
    'Qwen/Qwen3-235B-A22B-Instruct-2507': 'qwen/qwen3-235b-a22b-2507',
    'meta-llama/Llama-3.1-70B-Instruct':'meta-llama/Llama-3.1-70B-Instruct:scaleway',
    'meta-llama/Llama-3.1-405B-Instruct':'meta-llama/llama-3.1-405b-instruct',
    'Qwen/QwQ-32B': 'Qwen/QwQ-32B:featherless-ai',
    # All other models map to themselves (identity mapping)
    
}

# ============================================================================
# MODEL-SPECIFIC CONFIGURATIONS
# ============================================================================
MODEL_CONFIGS = {
    # ============================================================================
    # OpenAI Models
    # ============================================================================
    'gpt-3.5-turbo-1106': {
    },
    'gpt-4-0613': {
    },
    'gpt-4o-2024-11-20': {
    },
    'gpt-4o-mini-2024-07-18': {
    },
    'o3-2025-04-16': {
    },
    'o4-mini-2025-04-16':{
    },
    
    # GPT-5 Standard (low reasoning)
    'gpt-5-2025-08-07': {
        'reasoning_effort': 'low',
        'verbosity': 'low',
    },
    'gpt-5-mini-2025-08-07': {
        'reasoning_effort': 'low',
        'verbosity': 'low',
    },
    'gpt-5-nano-2025-08-07': {
        'reasoning_effort': 'low',
        'verbosity': 'low',
    },
    
    # GPT-5 "Thinking" Variants (high reasoning)
    'gpt-5-2025-08-07-thinking': {
        'reasoning_effort': 'high',
        'verbosity': 'high',
    },
    
    # ============================================================================
    # Anthropic Models
    # ============================================================================
    'claude-3-haiku-20240307': {
        'max_tokens': 2048,
    },
    'claude-3-5-haiku-20241022': {
        'max_tokens': 2048,
    },
    'claude-opus-4-1-20250805': {
        'max_tokens': 2048,
    },
    'claude-3-7-sonnet-20250219': {
        'max_tokens': 2048,
    },
    'claude-sonnet-4-20250514': {
        'max_tokens': 2048,
    },
    
    # ============================================================================
    # Google Models
    # ============================================================================
    'gemini-2.5-flash': {
        'max_output_tokens': 4000,
        'safety_settings': [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    },
    'gemini-2.5-flash-lite': {
        'max_output_tokens': 3000,
        'safety_settings': [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    },
    'gemini-2.5-pro': {
        'max_output_tokens': 5091,
        'safety_settings': [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    },
    'gemma-3-1b-it':{
        'max_output_tokens': 3000,
        'temperature': 0.7,
    },

    # ============================================================================
    # Mistral Models
    # ============================================================================
    "mistral-medium-2505": {
    },
    
    # ============================================================================
    # Together AI Models
    # ============================================================================
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {
        'max_tokens': 4096,
        'temperature': 0.7,
        'top_p': 0.7,
        'top_k': 50,
        'repetition_penalty': 1,
        'stop': ["<|eot_id|>", "<|eom_id|>"],
    },
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {
        'max_tokens': 4096,
        'temperature': 0.7,
        'top_p': 0.7,
        'top_k': 50,
        'repetition_penalty': 1,
        'stop': ["<|eot_id|>", "<|eom_id|>"],
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {
        'max_tokens': 4096,
        'temperature': 0.7,
        'top_p': 0.7,
        'top_k': 50,
        'repetition_penalty': 1,
        'stop': ["<|eot_id|>", "<|eom_id|>"],
    },
    'google/gemma-2-27b-it': {
        'max_tokens': 4096,
        'temperature': 0.7,
        'top_p': 0.7,
        'top_k': 50,
        'repetition_penalty': 1,
        'stop': ["<|eot_id|>", "<|eom_id|>"],
    },
    
    # ============================================================================
    # HuggingFace Models
    # ============================================================================
    'deepseek-ai/DeepSeek-R1':{
    },
    'deepseek-ai/DeepSeek-V3':{
    },
    'openai/gpt-oss-20b':{
    },
    'openai/gpt-oss-120b':{
    },
    'meta-llama/Llama-3.1-8B-Instruct':{
    },
    'meta-llama/Llama-3.1-70B-Instruct':{
    },
    #'Qwen/Qwen2-7B-Instruct':{},
    #'Qwen/Qwen2-72B-Instruct':{},
    'Qwen/Qwen3-32B':{
    },
    'Qwen/Qwen3-14B':{
    },
    'Qwen/QwQ-32B':{},

    # ============================================================================
    # OpenRouter Models
    # ============================================================================
    'meta-llama/Llama-3.1-405B-Instruct':{
    },
    'meta-llama/llama-4-scout':{},
    'meta-llama/llama-4-maverick':{},
    #'qwen/qwq-32b':{},
    'Qwen/Qwen3-235B-A22B-Instruct-2507':{},
    
    # ============================================================================
    # RITS Models
    # ============================================================================
    "ibm-granite/granite-3.0-8b-instruct": {
        'temperature': 0.7,
        'top_p': 0.9,
        'max_tokens': 4096,
        'n': 1,
    },
    
    # ============================================================================
    # Local Models
    # ============================================================================
    'ibm-granite/granite-3.0-2b-instruct': {
        'temperature': 0.7,
        'top_p': 0.9,
        'max_tokens': 2048,
    },
}

# ============================================================================
# JUDGE MODEL CONFIGURATION (separate from evaluation models)
# ============================================================================
JUDGE_CONFIG = {
    'model': 'gpt-4o-mini-2024-07-18',
    'temperature': 0,
}

# ============================================================================
# MODELS WITH SPECIAL PROMPT SUFFIXES
# ============================================================================
# Models that need "/set nothink" appended to each prompt
QWEN3_MODELS = [
    'Qwen/Qwen3-32B',
    'Qwen/Qwen3-14B',
]

# ============================================================================
# HUGGINGFACE ENDPOINT CONFIGURATIONS
# ============================================================================
HF_ENDPOINTS = {
    'endpoint_1': {
        'url': 'https://fy7gvl58xwxra03w.us-east-1.aws.endpoints.huggingface.cloud/v1/',
        'api_key': 'API KEY',  # Replace with your actual key
        'models': ['Qwen/Qwen2-7B-Instruct']
    },
    'endpoint_2': {
        'url': 'https://zt8t014juaznxzgq.us-east-2.aws.endpoints.huggingface.cloud/v1/',
        'api_key': 'API KEY',  # Replace with your actual key
        'models': ['meta-llama/Llama-2-70b-chat-hf']
    },
    'endpoint_3': {
        'url': 'https://kxvf3b7m6u0dhjji.us-east-1.aws.endpoints.huggingface.cloud/v1/',
        'api_key': 'API KEY',  # Replace with your actual key
        'models': ['meta-llama/Llama-2-7b-chat-hf']
    },
    'endpoint_4': {
        'url': 'https://uc8xay8eplu92cwr.us-east-1.aws.endpoints.huggingface.cloud/v1/',
        'api_key': 'API KEY',  # Replace with your actual key
        'models': ['ibm-granite/granite-3.3-2b-base']
    },
    'endpoint_5': {
        'url': 'https://cg9i5aq57zko2hzm.us-east-1.aws.endpoints.huggingface.cloud/v1/',
        'api_key': 'API KEY',  # Replace with your actual key
        'models': ['ibm-granite/granite-3.3-8b-base']
    },
    'endpoint_6': {
        'url': 'https://q75wtw8kpxwxzq32.us-east-2.aws.endpoints.huggingface.cloud/v1/',
        'api_key': 'API KEY',  # Replace with your actual key
        'models': ['Qwen/Qwen2-72B-Instruct']
    },
    # Add more endpoints as needed
}

# ============================================================================
# BATCH API CONFIGURATIONS
# ============================================================================
BATCH_CONFIG = {
    'openai': {
        'batch_size': 500,  # Max requests per batch file
        'parallel_batches': 30,  # How many batches to submit in parallel
        'poll_interval': 60,  # Seconds between status checks
        'completion_window': '24h',  # OpenAI batch completion window
    },
    'anthropic': {
        'batch_size': 500,  # Max requests per batch
        'parallel_batches': 30,  # How many batches to submit in parallel
        'poll_interval': 60,  # Seconds between status checks
    }
}

# ============================================================================
# BATCH-ENABLED MODELS
# ============================================================================
BATCH_MODELS = {
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
    ]
}

# ============================================================================
# API KEYS (Keep your existing keys)
# ============================================================================
OPENAI_API_KEY = 'API KEY'
ANTHROPIC_API_KEY = 'API KEY'
GEMINI_API_KEY = 'API KEY'
TOGETHER_API_KEY = ''
MISTRAL_API_KEY = 'API KEY'
HF_TOKEN = 'API KEY'
RITS_API_KEY = ''
OPENROUTER_API_KEY = 'API KEY'