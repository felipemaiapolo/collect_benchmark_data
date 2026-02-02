"""
Batch API manager for OpenAI and Anthropic
Handles batch submission, status checking, and result retrieval
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import openai
from openai import OpenAI
import anthropic
from datetime import datetime
from filelock import FileLock

from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, BATCH_CONFIG
from utils import GetAPIModelName

class BatchManager:
    """Manages batch operations for OpenAI and Anthropic APIs"""
    
    BATCH_STATE_FILE = 'batch_state.json'
    BATCH_LOCK_FILE = 'batch_state.json.lock'
    BATCH_REQUESTS_DIR = Path('batch_requests')
    BATCH_RESULTS_DIR = Path('batch_results')
    
    def __init__(self):
        self.BATCH_REQUESTS_DIR.mkdir(exist_ok=True)
        self.BATCH_RESULTS_DIR.mkdir(exist_ok=True)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.lock = FileLock(self.BATCH_LOCK_FILE)
        
    def load_batch_state(self) -> Dict[str, Any]:
        """Load existing batch state from file SAFELY"""
        # Wait for lock before reading
        with self.lock:
            if Path(self.BATCH_STATE_FILE).exists():
                try:
                    with open(self.BATCH_STATE_FILE, 'r') as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    # Fallback if file was previously corrupted
                    return {'openai': {}, 'anthropic': {}}
            return {'openai': {}, 'anthropic': {}}
    
    def save_batch_state(self, state: Dict[str, Any]):
        """Save batch state to file SAFELY"""
        # Wait for lock before writing
        with self.lock:
            with open(self.BATCH_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
    
    # ========================================================================
    # OPENAI BATCH API
    # ========================================================================
    
    def create_openai_batch_file(self, requests: List[Dict], batch_id: str) -> str:
        """Create JSONL file for OpenAI batch"""
        file_path = self.BATCH_REQUESTS_DIR / f"openai_batch_{batch_id}.jsonl"
        
        with open(file_path, 'w') as f:
            for req in requests:
                # Get actual API model name
                api_model_name = GetAPIModelName(req['model'])
                
                # OpenAI batch format
                batch_req = {
                    "custom_id": req['custom_id'],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": api_model_name,
                        "messages": req['messages'],
                        **req.get('params', {})
                    }
                }
                f.write(json.dumps(batch_req) + '\n')
        
        return str(file_path)
    
    def submit_openai_batch(self, requests: List[Dict], batch_id: str) -> str:
        """Submit batch to OpenAI"""
        print(f"Creating OpenAI batch file with {len(requests)} requests...")
        
        # Create batch file
        file_path = self.create_openai_batch_file(requests, batch_id)
        
        # Upload file
        print("Uploading batch file...")
        with open(file_path, 'rb') as f:
            batch_input_file = self.openai_client.files.create(
                file=f,
                purpose="batch"
            )
        
        # Create batch
        print("Creating batch...")
        batch = self.openai_client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window=BATCH_CONFIG['openai']['completion_window']
        )
        
        print(f"✓ OpenAI batch submitted: {batch.id}")
        return batch.id
    
    def check_openai_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Check status of OpenAI batch"""
        batch = self.openai_client.batches.retrieve(batch_id)
        
        return {
            'status': batch.status,
            'request_counts': {
                'total': batch.request_counts.total,
                'completed': batch.request_counts.completed,
                'failed': batch.request_counts.failed
            },
            'output_file_id': batch.output_file_id if batch.status == 'completed' else None,
            'error_file_id': batch.error_file_id if batch.status == 'completed' else None
        }
    
    def retrieve_openai_batch_results(self, batch_id: str, output_file_id: str) -> List[Dict]:
        """Retrieve results from completed OpenAI batch"""
        print(f"Downloading results for batch {batch_id}...")
        
        # Download results
        file_response = self.openai_client.files.content(output_file_id)
        results_path = self.BATCH_RESULTS_DIR / f"openai_batch_{batch_id}_results.jsonl"
        
        with open(results_path, 'wb') as f:
            f.write(file_response.content)
        
        # Parse results
        results = []
        with open(results_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        
        print(f"✓ Retrieved {len(results)} results")
        return results
    
    # ========================================================================
    # ANTHROPIC BATCH API
    # ========================================================================
    
    def submit_anthropic_batch(self, requests: List[Dict], batch_id: str) -> str:
        """Submit batch to Anthropic"""
        print(f"Creating Anthropic batch with {len(requests)} requests...")
        
        # Convert to Anthropic format
        formatted_requests = []
        for req in requests:
            # Get actual API model name
            api_model_name = GetAPIModelName(req['model'])
            
            formatted_requests.append({
                "custom_id": req['custom_id'],
                "params": {
                    "model": api_model_name,
                    "messages": req['messages'],
                    **req.get('params', {})
                }
            })
        
        # Create batch
        batch = self.anthropic_client.messages.batches.create(
            requests=formatted_requests
        )
        
        print(f"✓ Anthropic batch submitted: {batch.id}")
        return batch.id
    
    def check_anthropic_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Check status of Anthropic batch"""
        batch = self.anthropic_client.messages.batches.retrieve(batch_id)
        
        return {
            'status': batch.processing_status,
            'request_counts': {
                'total': batch.request_counts.processing + batch.request_counts.succeeded + batch.request_counts.errored + batch.request_counts.canceled + batch.request_counts.expired,
                'completed': batch.request_counts.succeeded,
                'failed': batch.request_counts.errored + batch.request_counts.canceled + batch.request_counts.expired
            }
        }
    
    def retrieve_anthropic_batch_results(self, batch_id: str) -> List[Dict]:
        """Retrieve results from completed Anthropic batch"""
        print(f"Retrieving results for Anthropic batch {batch_id}...")
        
        results = []
        
        # Iterate through all results
        for result in self.anthropic_client.messages.batches.results(batch_id):
            results.append({
                'custom_id': result.custom_id,
                'result': result.result if result.result.type == 'succeeded' else None,
                'error': result.result.error if result.result.type == 'errored' else None
            })
        
        print(f"✓ Retrieved {len(results)} results")
        return results
    
    # ========================================================================
    # UNIFIED INTERFACE
    # ========================================================================
    
    def submit_batch(self, provider: str, requests: List[Dict], batch_id: str) -> str:
        """Submit batch to specified provider"""
        if provider == 'openai':
            return self.submit_openai_batch(requests, batch_id)
        elif provider == 'anthropic':
            return self.submit_anthropic_batch(requests, batch_id)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def check_batch_status(self, provider: str, batch_id: str) -> Dict[str, Any]:
        """Check batch status for specified provider"""
        if provider == 'openai':
            return self.check_openai_batch_status(batch_id)
        elif provider == 'anthropic':
            return self.check_anthropic_batch_status(batch_id)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def retrieve_batch_results(self, provider: str, batch_id: str, output_file_id: Optional[str] = None) -> List[Dict]:
        """Retrieve batch results for specified provider"""
        if provider == 'openai':
            return self.retrieve_openai_batch_results(batch_id, output_file_id)
        elif provider == 'anthropic':
            return self.retrieve_anthropic_batch_results(batch_id)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def poll_batch_until_complete(self, provider: str, batch_id: str, max_wait_hours: int = 24) -> Dict[str, Any]:
        """Poll batch status until complete or timeout"""
        poll_interval = BATCH_CONFIG[provider]['poll_interval']
        max_polls = int(max_wait_hours * 3600 / poll_interval)
        
        print(f"Polling {provider} batch {batch_id} (checking every {poll_interval}s)...")
        
        for i in range(max_polls):
            status_info = self.check_batch_status(provider, batch_id)
            status = status_info['status']
            
            if status in ['completed', 'ended']:
                print(f"✓ Batch {batch_id} completed!")
                return status_info
            elif status in ['failed', 'cancelled', 'canceled', 'expired']:
                print(f"✗ Batch {batch_id} failed with status: {status}")
                return status_info
            
            # Show progress
            counts = status_info['request_counts']
            print(f"  Progress: {counts['completed']}/{counts['total']} completed "
                  f"({counts['failed']} failed) - Status: {status}")
            
            time.sleep(poll_interval)
        
        print(f"⚠ Timeout waiting for batch {batch_id}")
        return self.check_batch_status(provider, batch_id)