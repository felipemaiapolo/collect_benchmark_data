from datasets import load_dataset
from tqdm.auto import tqdm
import time
import numpy as np
from utils import (
    QueryModelError, QueryJudgeError, GetJudgePrompt, 
    SaveJson, LoadJson, IsBatchModel, GetModelConfig,
    ApplyPromptModifications
)
from batch_manager import BatchManager
from config import BATCH_CONFIG
import uuid
from pathlib import Path
import os
from filelock import FileLock


def run(benchmark, model, reps=2, n=100, use_batch_api=True):
    """
    Main evaluation function with batch API support
    
    Parameters
    ----------
    benchmark : str
        Benchmark name
    model : str
        Model name
    reps : int
        Number of repetitions per question
    n : int
        Number of questions to sample
    use_batch_api : bool
        Whether to use batch API for supported models (OpenAI/Anthropic)
    """
    
    # Check if this model supports batch API
    is_batch, batch_provider = IsBatchModel(model)
    
    if is_batch and use_batch_api:
        print(f"\n{'='*60}")
        print(f"Model {model} supports batch API ({batch_provider})")
        print(f"Using batch submission mode")
        print(f"{'='*60}\n")
        return run_with_batch(benchmark, model, reps, n, batch_provider)
    else:
        print(f"\n{'='*60}")
        print(f"Model {model} using standard API calls")
        print(f"{'='*60}\n")
        return run_standard(benchmark, model, reps, n)


def run_standard(benchmark, model, reps=2, n=100):
    """Standard evaluation without batch API"""
    
    # Load benchmark data
    if benchmark == 'ifeval':
        from ifeval.utils import process_results
        bench_data = load_dataset("google/IFEval")['train']
    elif benchmark == 'math':
        from mathbench.utils import doc_to_text
        bench_data = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")
        bench_data = [d for d in bench_data['test'] if d['level'] == 'Level 5']
    elif benchmark == 'musr':
        from musr.utils import doc_to_text
        bench_data = load_dataset("TAUR-Lab/MuSR")
        bench_data_aux = []
        for subject in ['murder_mysteries', 'object_placements', 'team_allocation']:
            bench_data_aux2 = list(bench_data[subject])
            for d in bench_data_aux2:
                d['subject'] = subject
            bench_data_aux += bench_data_aux2
        bench_data = bench_data_aux
        del(bench_data_aux)
        del(bench_data_aux2)
    elif benchmark == 'gpqa':
        from gpqa.utils import process_docs, doc_to_text
        bench_data = load_dataset("Idavidrein/gpqa", 'gpqa_main')['train']
        bench_data = list(process_docs(bench_data))
    elif benchmark == 'bbh':
        from bbh.utils import doc_to_text
        bench_data = []
        subs = ['boolean_expressions', 'causal_judgement', 'date_understanding',
               'disambiguation_qa', 'formal_fallacies', 'geometric_shapes',
               'hyperbaton', 'logical_deduction_five_objects',
               'logical_deduction_seven_objects',
               'logical_deduction_three_objects', 'movie_recommendation',
               'navigate', 'object_counting', 'penguins_in_a_table',
               'reasoning_about_colored_objects', 'ruin_names',
               'salient_translation_error_detection', 'snarks',
               'sports_understanding', 'temporal_sequences',
               'tracking_shuffled_objects_five_objects',
               'tracking_shuffled_objects_seven_objects',
               'tracking_shuffled_objects_three_objects', 'web_of_lies']
        for subject in tqdm(subs, desc='loading bbh data'):
            bench_data_aux = list(load_dataset("SaylorTwift/bbh", subject)['test'])
            for d in bench_data_aux:
                d['subject'] = subject
            bench_data += bench_data_aux
        del(bench_data_aux)
    elif benchmark == 'mmlu-pro':
        from mmlupro.utils import doc_to_text, doc_to_choice
        bench_data = list(load_dataset("TIGER-Lab/MMLU-Pro")['test'])
    else:
        raise NotImplementedError(f"Benchmark {benchmark} not implemented")
        
    # Sample questions
    m = len(bench_data) 
    ids = np.unique(np.round(np.linspace(0, m-1, n)))
    ids = [int(x) for x in ids if int(x) <= m-1]

    # Process each question
    for id in tqdm(ids, desc=f"{benchmark} - {model}"):
        for rep in range(reps):
            model_name = model.replace("/", ".").replace("_", ".")
            bench_name = benchmark.replace("/", ".").replace("_", ".")
            file_path = f"results/{bench_name}/{model_name}/{bench_name}_{model_name}_{id}_{rep}.json"
    
            try:
                output = LoadJson(file_path)
                prompt = output['prompt']
                gold_response = output['gold_response']
                doc = output['doc']
    
                if output['model_response'] == "NA":
                    model_response = QueryModelError(prompt, model)
                    output['model_response'] = model_response
                    SaveJson(output, file_path)
                else:
                    model_response = output['model_response']
                    
                if model_response != "NA":
                    if output['scores'] == "NA":
                        if benchmark == 'ifeval':
                            scores = process_results(doc, [model_response])
                            judge_prompt = "NA"
                        else:
                            judge_prompt = GetJudgePrompt(prompt, model_response, gold_response)
                            scores = QueryJudgeError(judge_prompt)
                        output['scores'] = scores
                        output['judge_prompt'] = judge_prompt
                        SaveJson(output, file_path)
            except:    
                # Getting prompt, gold resp, subject, and id bench
                doc = bench_data[id]
                if benchmark == 'ifeval':
                    prompt = doc['prompt']
                    gold_response = ''
                    subject = 'ifeval'
                    id_bench = doc['key']
                elif benchmark == 'math':
                    prompt = doc_to_text(doc)
                    gold_response = doc['solution']
                    subject = doc['type'] + "_" + doc['level'].replace(" ", "-")
                    id_bench = ''
                elif benchmark == 'musr':
                    prompt = doc_to_text(doc)
                    gold_response = doc['answer_choice']
                    subject = doc['subject']
                    id_bench = ''
                elif benchmark == 'gpqa':
                    prompt = doc_to_text(doc)
                    gold_response = doc['answer']
                    subject = 'main'
                    id_bench = ''
                elif benchmark == 'bbh':
                    prompt = doc_to_text(doc)
                    gold_response = doc['target']
                    subject = doc['subject']
                    id_bench = ''
                elif benchmark == 'mmlu-pro':
                    prompt = doc_to_text(doc)
                    gold_response = doc['answer']
                    subject = doc['category']
                    id_bench = doc['question_id']
                    
                # Prompting model and evaluating
                model_response = QueryModelError(prompt, model)
                if model_response == "NA":
                    scores = "NA"
                    judge_prompt = "NA"
                else:
                    if benchmark == 'ifeval':
                        scores = process_results(doc, [model_response])
                        judge_prompt = "NA"
                    else:
                        judge_prompt = GetJudgePrompt(prompt, model_response, gold_response)
                        scores = QueryJudgeError(judge_prompt)
                        
                # Output
                output = {
                    'model': model,
                    'benchmark': benchmark,
                    'id': id,
                    'rep': rep,
                    'id_bench': id_bench,
                    'prompt': prompt,
                    'model_response': model_response,
                    'gold_response': gold_response,
                    'subject': subject,
                    'scores': scores,
                    'judge_prompt': judge_prompt,
                    'doc': doc
                }
                
                SaveJson(output, file_path)


def run_with_batch(benchmark, model, reps=2, n=100, provider='openai'):
    """Evaluation using batch API"""
    
    print(f"Starting batch evaluation for {model} on {benchmark}")
    print(f"Provider: {provider}")
    print(f"Batch size: {BATCH_CONFIG[provider]['batch_size']}")
    print(f"Parallel batches: {BATCH_CONFIG[provider]['parallel_batches']}\n")
    
    # Initialize batch manager
    batch_manager = BatchManager()
    
    # ========================================================================
    # CRITICAL FIX: Wrap entire read-modify-write in lock
    # ========================================================================
    with batch_manager.lock:
        # Load or create batch state
        batch_state = batch_manager.load_batch_state()
        
        # Create state key for this model-benchmark combination
        state_key = f"{model}_{benchmark}".replace("/", "_").replace(".", "_")
        
        if state_key not in batch_state[provider]:
            batch_state[provider][state_key] = {
                'submitted_batches': {},
                'completed_batches': {},
                'processed_items': []
            }
        
        job_state = batch_state[provider][state_key]
        
        # Step 1: Check existing batches and retrieve completed results
        print("Step 1: Checking existing batch submissions...")
        retrieve_completed_batches(batch_manager, provider, job_state, benchmark, model, reps)
        
        # Step 2: Collect items that need processing
        print("\nStep 2: Collecting items that need processing...")
        items_to_process = collect_items_to_process(benchmark, model, reps, n, job_state)
        
        if not items_to_process:
            print("✓ All items already processed!")
            batch_manager.save_batch_state(batch_state)
            return
        
        print(f"Found {len(items_to_process)} items to process")
        
        # Step 3: Create batch requests
        print("\nStep 3: Creating batch requests...")
        batch_requests = create_batch_requests(items_to_process, model, provider)
        
        # Step 4: Submit batches
        print("\nStep 4: Submitting batches...")
        submit_batches(batch_manager, provider, batch_requests, job_state, batch_state)
        
        # Save state before releasing lock
        batch_manager.save_batch_state(batch_state)
    
    # Lock released here
    print(f"\n{'='*60}")
    print(f"Batch submission complete for {model} on {benchmark}")
    print(f"Submitted {len(job_state['submitted_batches'])} batches")
    print(f"Run this function again to check status and retrieve results")
    print(f"{'='*60}\n")


def retrieve_completed_batches(batch_manager, provider, job_state, benchmark, model, reps):
    """Check and retrieve results from completed batches"""
    
    submitted_batches = list(job_state['submitted_batches'].items())
    
    for batch_key, batch_info in submitted_batches:
        batch_id = batch_info['batch_id']
        
        print(f"  Checking batch {batch_key} (ID: {batch_id})...")
        
        try:
            status_info = batch_manager.check_batch_status(provider, batch_id)
            status = status_info['status']
            
            print(f"    Status: {status}")
            print(f"    Progress: {status_info['request_counts']}")
            
            if status in ['completed', 'ended']:
                print(f"    ✓ Batch completed! Retrieving results...")
                
                # Retrieve results
                if provider == 'openai':
                    results = batch_manager.retrieve_batch_results(
                        provider, batch_id, status_info['output_file_id']
                    )
                else:
                    results = batch_manager.retrieve_batch_results(provider, batch_id)
                
                # Process and save results
                process_batch_results(results, provider, benchmark, model, reps, job_state)
                
                # Move to completed
                job_state['completed_batches'][batch_key] = batch_info
                del job_state['submitted_batches'][batch_key]
                
                print(f"    ✓ Results saved!")
                
            elif status in ['failed', 'cancelled', 'canceled', 'expired']:
                print(f"    ✗ Batch failed with status: {status}")
                # Keep in submitted for retry
                
        except Exception as e:
            print(f"    ⚠ Error checking batch: {e}")


def load_benchmark_data(benchmark):
    """Load benchmark dataset safely with locking"""
    
    # Create a lock specifically for dataset loading
    # This forces workers to queue up if they try to load simultaneously
    lock_path = os.path.expanduser(f"~/.cache/huggingface/{benchmark}_load.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    
    with FileLock(lock_path):
        if benchmark == 'ifeval':
            bench_data = load_dataset("google/IFEval")['train']
        elif benchmark == 'math':
            bench_data = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")
            bench_data = [d for d in bench_data['test'] if d['level'] == 'Level 5']
        elif benchmark == 'musr':
            bench_data = load_dataset("TAUR-Lab/MuSR")
            bench_data_aux = []
            for subject in ['murder_mysteries', 'object_placements', 'team_allocation']:
                bench_data_aux2 = list(bench_data[subject])
                for d in bench_data_aux2:
                    d['subject'] = subject
                bench_data_aux += bench_data_aux2
            bench_data = bench_data_aux
        elif benchmark == 'gpqa':
            from gpqa.utils import process_docs
            bench_data = load_dataset("Idavidrein/gpqa", 'gpqa_main')['train']
            bench_data = list(process_docs(bench_data))
        elif benchmark == 'bbh':
            # BBH loads many sub-files, so we lock the whole block
            bench_data = []
            subs = ['boolean_expressions', 'causal_judgement', 'date_understanding',
                   'disambiguation_qa', 'formal_fallacies', 'geometric_shapes',
                   'hyperbaton', 'logical_deduction_five_objects',
                   'logical_deduction_seven_objects',
                   'logical_deduction_three_objects', 'movie_recommendation',
                   'navigate', 'object_counting', 'penguins_in_a_table',
                   'reasoning_about_colored_objects', 'ruin_names',
                   'salient_translation_error_detection', 'snarks',
                   'sports_understanding', 'temporal_sequences',
                   'tracking_shuffled_objects_five_objects',
                   'tracking_shuffled_objects_seven_objects',
                   'tracking_shuffled_objects_three_objects', 'web_of_lies']
            for subject in subs:
                bench_data_aux = list(load_dataset("SaylorTwift/bbh", subject)['test'])
                for d in bench_data_aux:
                    d['subject'] = subject
                bench_data += bench_data_aux
        elif benchmark == 'mmlu-pro':
            bench_data = list(load_dataset("TIGER-Lab/MMLU-Pro")['test'])
        else:
            raise NotImplementedError(f"Benchmark {benchmark} not implemented")
        
        return bench_data


def get_prompt_and_metadata(doc, benchmark):
    """Extract prompt and metadata from document"""
    
    if benchmark == 'ifeval':
        from ifeval.utils import process_results
        prompt = doc['prompt']
        gold_response = ''
        subject = 'ifeval'
        id_bench = doc['key']
    elif benchmark == 'math':
        from mathbench.utils import doc_to_text
        prompt = doc_to_text(doc)
        gold_response = doc['solution']
        subject = doc['type'] + "_" + doc['level'].replace(" ", "-")
        id_bench = ''
    elif benchmark == 'musr':
        from musr.utils import doc_to_text
        prompt = doc_to_text(doc)
        gold_response = doc['answer_choice']
        subject = doc['subject']
        id_bench = ''
    elif benchmark == 'gpqa':
        from gpqa.utils import doc_to_text
        prompt = doc_to_text(doc)
        gold_response = doc['answer']
        subject = 'main'
        id_bench = ''
    elif benchmark == 'bbh':
        from bbh.utils import doc_to_text
        prompt = doc_to_text(doc)
        gold_response = doc['target']
        subject = doc['subject']
        id_bench = ''
    elif benchmark == 'mmlu-pro':
        from mmlupro.utils import doc_to_text
        prompt = doc_to_text(doc)
        gold_response = doc['answer']
        subject = doc['category']
        id_bench = doc['question_id']
    
    return prompt, gold_response, subject, id_bench


def collect_items_to_process(benchmark, model, reps, n, job_state):
    """Collect all items that need processing"""
    
    # Load benchmark data
    bench_data = load_benchmark_data(benchmark)
    
    # Sample questions
    m = len(bench_data)
    ids = np.unique(np.round(np.linspace(0, m-1, n)))
    ids = [int(x) for x in ids if int(x) <= m-1]
    
    # Collect items
    items = []
    processed_set = set(job_state['processed_items'])
    
    for id in ids:
        for rep in range(reps):
            item_key = f"{id}_{rep}"
            
            # Skip if already processed
            if item_key in processed_set:
                continue
            
            # Check if file exists and is complete
            model_name = model.replace("/", ".").replace("_", ".")
            bench_name = benchmark.replace("/", ".").replace("_", ".")
            file_path = f"results/{bench_name}/{model_name}/{bench_name}_{model_name}_{id}_{rep}.json"
            
            try:
                output = LoadJson(file_path)
                if output.get('model_response') != "NA" and output.get('scores') != "NA":
                    # Already complete
                    job_state['processed_items'].append(item_key)
                    continue
            except:
                pass
            
            # Get doc and prompt
            doc = bench_data[id]
            prompt, gold_response, subject, id_bench = get_prompt_and_metadata(doc, benchmark)
            
            items.append({
                'id': id,
                'rep': rep,
                'item_key': item_key,
                'prompt': prompt,
                'gold_response': gold_response,
                'subject': subject,
                'id_bench': id_bench,
                'doc': doc
            })
    
    return items


def create_batch_requests(items, model, provider):
    """Create batch API requests from items"""
    
    model_config = GetModelConfig(model)
    requests = []
    
    for item in items:
        # Apply prompt modifications
        prompt = ApplyPromptModifications(item['prompt'], model)
        
        # Create request in provider format
        if provider == 'openai':
            request = {
                'custom_id': item['item_key'],
                'model': model,  # Store internal model name
                'messages': [{'role': 'user', 'content': prompt}],
                'params': model_config,
                'metadata': item  # Store metadata for later
            }
        elif provider == 'anthropic':
            request = {
                'custom_id': item['item_key'],
                'model': model,  # Store internal model name
                'messages': [{'role': 'user', 'content': prompt}],
                'params': model_config,
                'metadata': item
            }
        
        requests.append(request)
    
    return requests


def submit_batches(batch_manager, provider, batch_requests, job_state, batch_state):
    """Submit batch requests in chunks"""
    
    batch_size = BATCH_CONFIG[provider]['batch_size']
    
    # Split into batches
    for i in range(0, len(batch_requests), batch_size):
        batch_chunk = batch_requests[i:i + batch_size]
        batch_key = str(uuid.uuid4())[:8]
        
        print(f"  Submitting batch {batch_key} with {len(batch_chunk)} requests...")
        
        try:
            batch_id = batch_manager.submit_batch(provider, batch_chunk, batch_key)
            
            # Save batch info
            job_state['submitted_batches'][batch_key] = {
                'batch_id': batch_id,
                'provider': provider,
                'num_requests': len(batch_chunk),
                'submitted_at': time.time(),
                'requests': batch_chunk  # Store for reference
            }
            
            # DO NOT save state here - let the caller handle it
            
            print(f"    ✓ Batch submitted: {batch_id}")
            
            # Add small delay between submissions to avoid rate limits
            time.sleep(2)
            
        except Exception as e:
            print(f"    ✗ Error submitting batch: {e}")
            # Re-raise critical errors
            if any(keyword in str(e).lower() for keyword in ['authentication', 'quota', 'forbidden', 'unauthorized']):
                raise


def process_batch_results(results, provider, benchmark, model, reps, job_state):
    """Process and save batch results"""
    
    for result in results:
        custom_id = result['custom_id']
        
        # Parse custom_id to get id and rep
        id_str, rep_str = custom_id.split('_')
        id = int(id_str)
        rep = int(rep_str)
        
        # Get response
        if provider == 'openai':
            if 'response' in result and 'body' in result['response']:
                model_response = result['response']['body']['choices'][0]['message']['content']
            else:
                model_response = "NA"
        elif provider == 'anthropic':
            if result.get('result') and hasattr(result['result'], 'type') and result['result'].type == 'succeeded':
                model_response = result['result'].message.content[0].text
            else:
                model_response = "NA"
        
        # Load existing data if available
        model_name = model.replace("/", ".").replace("_", ".")
        bench_name = benchmark.replace("/", ".").replace("_", ".")
        file_path = f"results/{bench_name}/{model_name}/{bench_name}_{model_name}_{id}_{rep}.json"
        
        try:
            output = LoadJson(file_path)
        except:
            # Need to recreate output - find metadata in submitted batches
            output = None
            for batch_info in job_state['submitted_batches'].values():
                for req in batch_info['requests']:
                    if req['custom_id'] == custom_id:
                        metadata = req['metadata']
                        output = {
                            'model': model,
                            'benchmark': benchmark,
                            'id': id,
                            'rep': rep,
                            'id_bench': metadata['id_bench'],
                            'prompt': metadata['prompt'],
                            'model_response': "NA",
                            'gold_response': metadata['gold_response'],
                            'subject': metadata['subject'],
                            'scores': "NA",
                            'judge_prompt': "NA",
                            'doc': metadata['doc']
                        }
                        break
                if output:
                    break
            
            # Also check completed batches
            if not output:
                for batch_info in job_state['completed_batches'].values():
                    for req in batch_info['requests']:
                        if req['custom_id'] == custom_id:
                            metadata = req['metadata']
                            output = {
                                'model': model,
                                'benchmark': benchmark,
                                'id': id,
                                'rep': rep,
                                'id_bench': metadata['id_bench'],
                                'prompt': metadata['prompt'],
                                'model_response': "NA",
                                'gold_response': metadata['gold_response'],
                                'subject': metadata['subject'],
                                'scores': "NA",
                                'judge_prompt': "NA",
                                'doc': metadata['doc']
                            }
                            break
                    if output:
                        break
        
        if output is None:
            print(f"  ⚠ Could not find metadata for {custom_id}")
            continue
        
        # Update with response
        output['model_response'] = model_response
        
        # Evaluate if response is not NA
        if model_response != "NA":
            if benchmark == 'ifeval':
                from ifeval.utils import process_results
                scores = process_results(output['doc'], [model_response])
                judge_prompt = "NA"
            else:
                judge_prompt = GetJudgePrompt(output['prompt'], model_response, output['gold_response'])
                scores = QueryJudgeError(judge_prompt)
            
            output['scores'] = scores
            output['judge_prompt'] = judge_prompt
        
        # Save
        SaveJson(output, file_path)
        
        # Mark as processed
        if custom_id not in job_state['processed_items']:
            job_state['processed_items'].append(custom_id)