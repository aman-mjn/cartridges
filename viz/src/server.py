#!/usr/bin/env python3
"""
Backend API server for the dataset visualization app.
Provides endpoints for dataset discovery and loading.
"""

from functools import lru_cache
import os
import pickle
import glob
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from transformers import AutoTokenizer
from cartridges.structs import read_conversations
import pandas as pd

app = FastAPI(title="Dataset Visualization API", version="1.0.0")

# Configuration from environment variables
CORS_ENABLED = os.getenv('CORS_ENABLED', 'true').lower() == 'true'
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8000'))
RELOAD = os.getenv('RELOAD', 'false').lower() == 'true'
OUTPUTS_ROOT = os.getenv('CARTRIDGES_OUTPUT_DIR', '/root/cartridges-v2/outputs')

# CORS middleware configuration
if CORS_ENABLED:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@lru_cache(maxsize=5)
def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from pickle or parquet file."""
    try:
        # Use the new read_conversations function that handles both formats
        conversations = read_conversations(file_path)
        return conversations
    except ImportError as e:
        print(f"Missing dependency for {file_path}: {e}")
        print("Please install required dependencies: pip install pyarrow pandas")
        return []
    except Exception as e:
        print(f"Error loading dataset {file_path}: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        # Fallback to old pickle loading for backwards compatibility
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different data formats
            if isinstance(data, dict):
                if 'rows' in data:
                    return data['rows']
                elif 'examples' in data:
                    return data['examples']
                elif 'data' in data:
                    return data['data']
                else:
                    # Try to extract first list value
                    for value in data.values():
                        if isinstance(value, list):
                            return value
                    return []
            elif isinstance(data, list):
                return data
            else:
                return []
        except Exception as e2:
            print(f"Error loading dataset with fallback {file_path}: {e2}")
            return []

def serialize_training_example(example, tokenizer: AutoTokenizer=None, include_logprobs=False) -> Dict[str, Any]:
    """Convert TrainingExample to JSON-serializable format."""
    try:
        messages = []
        for msg in example.messages:
            token_ids = msg.token_ids.tolist() if hasattr(msg.token_ids, "tolist") else msg.token_ids
            message_data = {
                'content': msg.content,
                'role': msg.role,
                'token_ids': token_ids,
                'token_strs': [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids] if tokenizer else None,
                'top_logprobs': None
            }
            
            # Handle logprobs if they exist and are requested
            if include_logprobs and hasattr(msg, 'top_logprobs') and msg.top_logprobs is not None:
                # Use the original structure to get all top-k alternatives for each position
                top_logprobs_matrix = msg.top_logprobs  # This is the original TopLogprobs object
                
                # Create list of lists, same length as token_ids
                token_idx_to_logprobs = [[] for _ in range(len(token_ids))]
                
                for token_idx, token_id, logprobs in zip(top_logprobs_matrix.token_idx, top_logprobs_matrix.token_id, top_logprobs_matrix.logprobs):
                    token_idx_to_logprobs[token_idx].append({
                        'token_id': int(token_id),
                        "token_str": tokenizer.decode([token_id], skip_special_tokens=False) if tokenizer else None,
                        'logprob': float(logprobs)
                    })
                            
                result = []
                for token_idx, logprobs in enumerate(token_idx_to_logprobs):
                    # Sort by logprob (highest first)
                    logprobs.sort(key=lambda x: x['logprob'], reverse=True)
                    result.append(logprobs)
                
                message_data['top_logprobs'] = result
            
            messages.append(message_data)
        
        # Serialize metadata to handle numpy arrays and other non-serializable objects
        serialized_metadata = {}
        if example.metadata:
            for key, value in example.metadata.items():
                try:
                    # Handle numpy arrays
                    if hasattr(value, 'tolist'):
                        serialized_metadata[key] = value.tolist()
                    # Handle other numpy types
                    elif hasattr(value, 'item'):
                        serialized_metadata[key] = value.item()
                    # Handle regular serializable objects
                    else:
                        # Test if it's JSON serializable
                        import json
                        json.dumps(value)
                        serialized_metadata[key] = value
                except (TypeError, ValueError):
                    # Convert to string if not serializable
                    serialized_metadata[key] = str(value)
        
        
        return {
            'messages': messages,
            'system_prompt': example.system_prompt,
            'type': example.type,
            'metadata': serialized_metadata
        }
    except Exception as e:
        print(f"Error serializing example: {e}")
        return {
            'messages': [],
            'system_prompt': '',
            'type': 'unknown',
            'metadata': {}
        }

def quick_check_dataset(file_path: str) -> Optional[int]:
    """Quickly check if a file is a valid dataset and return approximate size."""
    try:
        # For parquet files, we can get row count without loading all data
        if file_path.endswith('.parquet'):
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(file_path)
                return table.num_rows
            except ImportError:
                print(f"Warning: pyarrow not available, falling back to loading full dataset for {file_path}")
                # Fallback to loading full dataset
                conversations = load_dataset(file_path)
                return len(conversations)
        
        # For pickle files, we need to load to check
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different data formats and get count without loading all examples
        if isinstance(data, dict):
            if 'rows' in data and isinstance(data['rows'], list):
                return len(data['rows'])
            elif 'examples' in data and isinstance(data['examples'], list):
                return len(data['examples'])
            elif 'data' in data and isinstance(data['data'], list):
                return len(data['data'])
            else:
                # Try to find first list value
                for value in data.values():
                    if isinstance(value, list):
                        return len(value)
                return 0
        elif isinstance(data, list):
            return len(data)
        else:
            return 0
    except Exception as e:
        print(f"Error quick-checking dataset {file_path}: {e}")
        return None

@app.get("/api/datasets")
def discover_datasets(output_dir: Optional[str] = Query(None)):
    """Discover and return available datasets without loading full content."""
    
    # If no output_dir specified, try common locations
    search_paths = []
    if output_dir:
        search_paths.append(output_dir)
    
    # Add some common search paths
    search_paths.extend([
        os.path.expanduser('~/code/cartridges/outputs'),
        os.path.expanduser('~/outputs'),
        '/tmp/cartridges_output',
        './outputs'
    ])
    
    # Also check environment variables
    env_output_dir = os.environ.get('CARTRIDGES_OUTPUT_DIR')
    if env_output_dir:
        search_paths.insert(0, env_output_dir)
    
    datasets = []
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        # Find all .pkl and .parquet files recursively
        pkl_files = glob.glob(os.path.join(search_path, '**/*.pkl'), recursive=True)
        parquet_files = glob.glob(os.path.join(search_path, '**/*.parquet'), recursive=True)
        all_files = pkl_files + parquet_files
        
        for file_path in all_files:
            try:
                # Quick check if it's a valid dataset
                size_bytes = os.path.getsize(file_path)
                size_gb = size_bytes / (1024 ** 3) if size_bytes is not None else None
                if size_gb is not None and size_gb > 0:
                    file_obj = Path(file_path)
                    dataset_name = file_obj.stem
                    
                    # Calculate relative path from search_path
                    try:
                        relative_path = str(file_obj.relative_to(search_path))
                    except ValueError:
                        # If relative_to fails, just use the filename
                        relative_path = file_obj.name
                    
                    datasets.append({
                        'name': dataset_name,
                        'path': file_path,
                        'relative_path': relative_path,
                        'size': size_gb,
                        'directory': str(file_obj.parent)
                    })
            except Exception as e:
                print(f"Error checking {file_path}: {e}")
                continue
    
    # Sort datasets by relative path for consistent ordering
    datasets.sort(key=lambda d: d['relative_path'])
    
    return datasets

@app.get("/api/dataset/{dataset_path:path}/info")
def get_dataset_info(dataset_path: str):
    """Get dataset metadata without loading examples."""
    try:
        # Decode the path
        import urllib.parse
        dataset_path = urllib.parse.unquote(dataset_path)
        
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get total count efficiently
        total_count = quick_check_dataset(dataset_path)
        if total_count is None:
            # Fallback: load dataset to get count
            examples = load_dataset(dataset_path)
            total_count = len(examples)
        
        return {
            'path': dataset_path,
            'total_count': total_count,
            'file_size': os.path.getsize(dataset_path)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/{dataset_path:path}")
def get_dataset_page(
    dataset_path: str, 
    page: int = Query(0), 
    page_size: int = Query(12),
    search: Optional[str] = Query(None),
    search_messages: Optional[str] = Query('true'),
    search_system_prompt: Optional[str] = Query('false'),
    search_metadata: Optional[str] = Query('false')
):
    """Load and return a specific page of a dataset with optional search."""
    try:
        # Decode the path
        import urllib.parse
        dataset_path = urllib.parse.unquote(dataset_path)
        
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert search field parameters to booleans
        search_messages_bool = search_messages and search_messages.lower() == 'true'
        search_system_prompt_bool = search_system_prompt and search_system_prompt.lower() == 'true'
        search_metadata_bool = search_metadata and search_metadata.lower() == 'true'
        print(f"Search fields - messages: {search_messages_bool}, system_prompt: {search_system_prompt_bool}, metadata: {search_metadata_bool}")
        
        # Load all examples
        t0 = time.time()
        examples = load_dataset(dataset_path)
        print(f"Loaded dataset in {time.time() - t0} seconds")
        
        # Apply search filter if provided
        if search and search.strip():
            t0 = time.time()
            search_query = search.strip().lower()
            filtered_examples = []
            
            for example in examples:
                matches = []
                
                # Search in message contents (if enabled)
                if search_messages_bool:
                    message_match = any(
                        search_query in msg.content.lower() 
                        for msg in example.messages
                    )
                    matches.append(message_match)
                
                # Search in system prompt (if enabled)
                if search_system_prompt_bool:
                    system_prompt_match = (
                        example.system_prompt and 
                        search_query in example.system_prompt.lower()
                    )
                    matches.append(system_prompt_match)
                
                # Search in metadata (if enabled)
                if search_metadata_bool:
                    metadata_match = False
                    if example.metadata:
                        metadata_match = any(
                            search_query in str(value).lower() 
                            for value in example.metadata.values()
                        )
                    matches.append(metadata_match)
                
                # Include example if any enabled field matches
                if any(matches):
                    filtered_examples.append(example)
            
            examples = filtered_examples
            print(f"Filtered {len(examples)} examples in {time.time() - t0} seconds")
        
        total_count = len(examples)
        
        # Calculate pagination
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, total_count)
        
        # Only serialize the requested page
        t0 = time.time()
        page_examples = examples[start_idx:end_idx]
        serialized_examples = []
        for example in page_examples:
            serialized_examples.append(serialize_training_example(example))
        print(f"Serialized examples in {time.time() - t0} seconds")
        
        return {
            'examples': serialized_examples,
            'total_count': total_count,
            'page': page,
            'page_size': page_size,
            'total_pages': (total_count + page_size - 1) // page_size,
            'path': dataset_path,
            'search': search,
            'search_fields': {
                'messages': search_messages_bool,
                'system_prompt': search_system_prompt_bool,
                'metadata': search_metadata_bool
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dataset/example")
def get_dataset_example_with_logprobs(request: Dict[str, Any]):
    """Get a single example with logprobs included."""
    try:
        dataset_path = request.get('dataset_path')

        tokenizer = _get_tokenizer(dataset_path)

        example_index = request.get('example_index')
        
        if not dataset_path:
            raise HTTPException(status_code=400, detail="dataset_path is required")
        if example_index is None:
            raise HTTPException(status_code=400, detail="example_index is required")
                
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load the examples
        examples = load_dataset(dataset_path)
        
        if example_index < 0 or example_index >= len(examples):
            raise HTTPException(status_code=404, detail=f"Example index {example_index} not found (dataset has {len(examples)} examples)")
        
        example = examples[example_index]
        serialized_example = serialize_training_example(example, include_logprobs=True, tokenizer=tokenizer)
        
        return {
            'example': serialized_example,
            'index': example_index,
            'total_count': len(examples)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in get_dataset_example_with_logprobs: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    print("Health check called!")
    return {'status': 'healthy'}


def _get_dataset_config(dataset_path: str) -> Dict[str, Any]:
    # Look for config.yaml in the same directory as the dataset
    dataset_dir = os.path.dirname(dataset_path)
    config_path = os.path.join(dataset_dir, 'config.yaml')
    
    if not os.path.exists(config_path):
        # Also try the parent directory (common pattern)
        parent_config_path = os.path.join(os.path.dirname(dataset_dir), 'config.yaml')
        if os.path.exists(parent_config_path):
            config_path = parent_config_path
        else:
            return {'config': None, 'path': None}
    
    # Load the YAML config
    import yaml
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return {
        'config': config_data,
        'path': config_path,
        'exists': True
    }

@app.post("/api/dataset/config")
def get_dataset_config(request: Dict[str, Any]):
    """Get the SynthesizeConfig for a dataset if it exists."""
    try:
        dataset_path = request.get('dataset_path')
        if not dataset_path:
            raise HTTPException(status_code=400, detail="dataset_path is required")
        
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        return _get_dataset_config(dataset_path)
    
    except Exception as e:
        print(f"Error loading config: {e}")
        return {'config': None, 'path': None, 'exists': False, 'error': str(e)}

@lru_cache(maxsize=3)
def _get_tokenizer(dataset_path: str):
    """Decode token IDs to text using the specified tokenizer."""
    config = _get_dataset_config(dataset_path)["config"]
    tokenizer_name = config["synthesizer"]["client"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return tokenizer


@app.get("/api/dashboards")
def get_dashboards():
    """Return a minimal local dashboard definition so UI can proceed without W&B."""
    return {'dashboards': [
        {
            'name': 'Local',
            'filters': {},
            'table': 'eval_metrics.csv',
            'score_metric': 'loss',
            'step': 'step',
        }
    ]}

@app.post("/api/dashboard/analyze")
def analyze_run_with_dashboard(request: Dict[str, Any]):
    print(f"Analyzing run with dashboard: {request}")
    """Analyze a local run: return plot metadata from metrics.csv and no tables by default."""
    try:
        run_id = request.get('run_id')
        dashboard_name = request.get('dashboard_name')
        if not all([run_id, dashboard_name]):
            raise HTTPException(status_code=400, detail="run_id and dashboard_name are required")
        # Build plot metadata by inspecting local CSVs
        run_dir = Path(OUTPUTS_ROOT) / run_id
        plots_meta = []
        metrics_path = run_dir / 'metrics.csv'
        eval_path = run_dir / 'eval_metrics.csv'
        if metrics_path.exists():
            plots_meta.append({
                'id': 'local/loss',
                'plot_name': 'Training Loss',
                'x_col': 'step',
                'y_col': 'loss',
            })
            # Training perplexity (if present)
            try:
                df_head = pd.read_csv(metrics_path, nrows=1)
                if 'perplexity' in df_head.columns:
                    plots_meta.append({
                        'id': 'local/ppl',
                        'plot_name': 'Training Perplexity',
                        'x_col': 'step',
                        'y_col': 'perplexity',
                    })
            except Exception:
                pass
        if eval_path.exists():
            try:
                df_eval = pd.read_csv(eval_path)
                if 'eval_name' in df_eval.columns:
                    for name in sorted(df_eval['eval_name'].dropna().unique()):
                        plots_meta.append({
                            'id': f'eval/{name}',
                            'plot_name': f'Eval: {name}',
                            'x_col': 'step',
                            'y_col': 'loss',
                        })
                        # Add eval perplexity plot if available
                        if 'perplexity' in df_eval.columns:
                            plots_meta.append({
                                'id': f'eval_ppl/{name}',
                                'plot_name': f'Perplexity: {name}',
                                'x_col': 'step',
                                'y_col': 'perplexity',
                            })
            except Exception:
                pass
        return {
            'plots': plots_meta,
            'tables': [],
            'dashboard_name': dashboard_name,
            'run_id': run_id
        }
        
    except Exception as e:
        print(f"Error in analyze_run_with_dashboard: {str(e)}")
        return {'error': str(e)}

@app.post("/api/dashboard/table")
def get_table_data(request: Dict[str, Any]):
    """Load specific table data on demand."""
    try:
        run_id = request.get('run_id')
        table_path = request.get('table_path')
        table_step = request.get('table_step')
        
        # Get entity and project from environment variables
        entity = os.getenv('WANDB_ENTITY', 'hazy-research')
        project = os.getenv('WANDB_PROJECT', 'cartridges')
        
        if not all([run_id, table_path]):
            raise HTTPException(status_code=400, detail="run_id and table_path are required")
        
        # Import wandb and get the run
        import wandb
        try:
            api = wandb.Api()
        except Exception as e:
            print(f"Failed to initialize W&B API: {str(e)}")
            return {'error': f'Failed to initialize W&B API. Make sure WANDB_API_KEY is set: {str(e)}'}
        
        # Get the run
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Create a TableSpec and materialize it
        from dashboards.base import TableSpec
        table_spec = TableSpec(
            run=run,
            path=table_path,
            step=table_step
        )
        
        # Materialize the table data
        df = table_spec.materialize()
        
        # Handle NaN values that can't be JSON serialized
        df = df.fillna('')  # Replace NaN with empty strings
        
        return {
            'data': df.to_dict('records'),
            'step': table_step,
            'path': table_path
        }
        
    except Exception as e:
        print(f"Error in get_table_data: {str(e)}")
        return {'error': str(e)}

@app.post("/api/dashboard/plots")
def get_plot_data(request: Dict[str, Any]):
    """Load plot data from local CSVs for a run (training, lr, evals)."""
    try:
        run_id = request.get('run_id')
        dashboard_name = request.get('dashboard_name')
        if not all([run_id, dashboard_name]):
            raise HTTPException(status_code=400, detail="run_id and dashboard_name are required")
        run_dir = Path(OUTPUTS_ROOT) / run_id
        metrics_path = run_dir / 'metrics.csv'
        eval_path = run_dir / 'eval_metrics.csv'
        plots = []
        # Training loss
        if metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path).fillna('')
                if {'step','loss'}.issubset(df.columns):
                    plots.append({
                        'id': 'local/loss',
                        'plot_name': 'Training Loss',
                        'x_col': 'step',
                        'y_col': 'loss',
                        'data': df[['step','loss']].to_dict('records')
                    })
                if {'step','perplexity'}.issubset(df.columns):
                    plots.append({
                        'id': 'local/ppl',
                        'plot_name': 'Training Perplexity',
                        'x_col': 'step',
                        'y_col': 'perplexity',
                        'data': df[['step','perplexity']].to_dict('records')
                    })
            except Exception as e:
                return {'error': f'Failed to read metrics: {e}'}
        # Eval plots (holdout, replay, etc.)
        if eval_path.exists():
            try:
                df_eval = pd.read_csv(eval_path).fillna('')
                if {'step','eval_name','loss'}.issubset(df_eval.columns):
                    for name in sorted(df_eval['eval_name'].dropna().unique()):
                        sub = df_eval[df_eval['eval_name'] == name]
                        plots.append({
                            'id': f'eval/{name}',
                            'plot_name': f'Eval: {name}',
                            'x_col': 'step',
                            'y_col': 'loss',
                            'data': sub[['step','loss']].to_dict('records')
                        })
                        if {'perplexity'}.issubset(sub.columns):
                            plots.append({
                                'id': f'eval_ppl/{name}',
                                'plot_name': f'Perplexity: {name}',
                                'x_col': 'step',
                                'y_col': 'perplexity',
                                'data': sub[['step','perplexity']].to_dict('records')
                            })
            except Exception as e:
                return {'error': f'Failed to read eval metrics: {e}'}
        return {'plots': plots, 'dashboard_name': dashboard_name, 'run_id': run_id}
        
    except Exception as e:
        print(f"Error in get_plot_data: {str(e)}")
        return {'error': str(e)}

@app.get('/api/local/runs')
def list_local_runs(root: Optional[str] = None):
    """List local runs by scanning OUTPUTS_ROOT (or provided root) for subdirectories with metrics.csv."""
    base = Path(root or OUTPUTS_ROOT)
    runs = []
    if not base.exists():
        return {'runs': [], 'root': str(base)}

    # Collect all directories that contain metrics.csv or eval_metrics.csv at any depth
    run_dirs = set()
    for metrics_file in base.rglob('metrics.csv'):
        run_dirs.add(metrics_file.parent)
    for eval_file in base.rglob('eval_metrics.csv'):
        run_dirs.add(eval_file.parent)

    for run_dir in run_dirs:
        metrics_path = run_dir / 'metrics.csv'
        eval_path = run_dir / 'eval_metrics.csv'
        # Use relative path from base as stable ID (handles nested date/uuid structure)
        try:
            rel_id = str(run_dir.relative_to(base))
        except ValueError:
            rel_id = run_dir.name
        runs.append({
            'id': rel_id,
            'name': run_dir.name,
            'path': str(run_dir),
            'createdAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(run_dir.stat().st_mtime)),
            'has_metrics': metrics_path.exists(),
            'has_eval': eval_path.exists(),
        })

    # sort by mtime desc
    runs.sort(key=lambda r: r['createdAt'], reverse=True)
    return {'runs': runs, 'root': str(base)}

@app.get('/api/local/metrics')
def get_local_metrics(run_id: str, root: Optional[str] = None):
    """Return metrics.csv and eval_metrics.csv contents as JSON arrays."""
    base = Path(root or OUTPUTS_ROOT) / run_id
    result: Dict[str, Any] = {'run_id': run_id, 'root': str(base)}
    try:
        if (base / 'metrics.csv').exists():
            df = pd.read_csv(base / 'metrics.csv')
            result['metrics'] = df.to_dict('records')
        else:
            result['metrics'] = []
        if (base / 'eval_metrics.csv').exists():
            df_eval = pd.read_csv(base / 'eval_metrics.csv')
            result['eval_metrics'] = df_eval.to_dict('records')
        else:
            result['eval_metrics'] = []
        return result
    except Exception as e:
        return {'error': str(e), 'run_id': run_id, 'root': str(base)}

@app.post("/api/dashboard/slices")
def get_table_slices(request: Dict[str, Any]):
    """Get slices for a specific table using dashboard slice functions."""
    try:
        run_id = request.get('run_id')
        dashboard_name = request.get('dashboard_name')
        table_path = request.get('table_path')
        table_step = request.get('table_step')
        
        # Get entity and project from environment variables
        entity = os.getenv('WANDB_ENTITY', 'hazy-research')
        project = os.getenv('WANDB_PROJECT', 'cartridges')
        
        if not all([run_id, dashboard_name, table_path]):
            raise HTTPException(status_code=400, detail="run_id, dashboard_name, and table_path are required")
        
        # Import wandb and dashboard registry
        import wandb
        try:
            from dashboards.base import registry
            
            # Dynamically import all dashboard modules to ensure they're registered
            dashboards_dir = os.path.join(os.path.dirname(__file__), 'dashboards')
            dashboards_dir = os.path.abspath(dashboards_dir)
            
            if os.path.exists(dashboards_dir):
                for filename in os.listdir(dashboards_dir):
                    if filename.endswith('.py') and filename not in ['__init__.py', 'base.py']:
                        module_name = filename[:-3]  # Remove .py extension
                        try:
                            importlib.import_module(f"dashboards.{module_name}")
                        except Exception:
                            # Ignore import errors for individual modules
                            pass
        except ImportError as e:
            print(f"Failed to import dashboard registry: {str(e)}")
            return {'error': f'Failed to import dashboard registry: {str(e)}'}
        
        # Get the dashboard
        if dashboard_name not in registry.dashboards:
            raise HTTPException(status_code=404, detail=f"Dashboard '{dashboard_name}' not found")
        
        dashboard = registry.dashboards[dashboard_name]
        
        # Initialize wandb API and get the run
        try:
            api = wandb.Api()
        except Exception as e:
            print(f"Failed to initialize W&B API: {str(e)}")
            return {'error': f'Failed to initialize W&B API. Make sure WANDB_API_KEY is set: {str(e)}'}
        
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Create a TableSpec and materialize the data
        from dashboards.base import TableSpec
        table_spec = TableSpec(
            run=run,
            path=table_path,
            step=table_step
        )
        
        # Materialize the table data
        df = table_spec.materialize()
        
        # Get slices from the dashboard
        slices = dashboard.slices(df)
        
        # Convert slices to JSON-serializable format
        slices_data = []
        for slice_obj in slices:
            # Handle NaN values
            slice_df = slice_obj.df.fillna('')
            
            # Convert metrics to JSON-serializable format (handle NaN values)
            serialized_metrics = {}
            for key, value in slice_obj.metrics.items():
                if pd.isna(value):
                    serialized_metrics[key] = None
                else:
                    serialized_metrics[key] = float(value) if isinstance(value, (int, float)) else value
            
            slices_data.append({
                'name': slice_obj.name,
                'data': slice_df.to_dict('records'),
                'count': len(slice_obj.df),
                'metrics': serialized_metrics
            })
        
        return {
            'slices': slices_data,
            'total_count': len(df)
        }
        
    except Exception as e:
        print(f"Error in get_table_slices: {str(e)}")
        return {'error': str(e)}

@app.post("/api/dashboard/slice-metrics")
def get_slice_metrics_over_time(request: Dict[str, Any]):
    """Compute slice metrics across all table steps for a run and dashboard."""
    try:
        run_id = request.get('run_id')
        dashboard_name = request.get('dashboard_name')
        
        # Get entity and project from environment variables
        entity = os.getenv('WANDB_ENTITY', 'hazy-research')
        project = os.getenv('WANDB_PROJECT', 'cartridges')
        
        if not all([run_id, dashboard_name]):
            raise HTTPException(status_code=400, detail="run_id and dashboard_name are required")
        
        # Import wandb and dashboard registry
        import wandb
        try:
            from dashboards.base import registry
            
            # Dynamically import all dashboard modules to ensure they're registered
            dashboards_dir = os.path.join(os.path.dirname(__file__), 'dashboards')
            dashboards_dir = os.path.abspath(dashboards_dir)
            
            if os.path.exists(dashboards_dir):
                for filename in os.listdir(dashboards_dir):
                    if filename.endswith('.py') and filename not in ['__init__.py', 'base.py']:
                        module_name = filename[:-3]  # Remove .py extension
                        try:
                            importlib.import_module(f"dashboards.{module_name}")
                        except Exception:
                            # Ignore import errors for individual modules
                            pass
        except ImportError as e:
            print(f"Failed to import dashboard registry: {str(e)}")
            return {'error': f'Failed to import dashboard registry: {str(e)}'}
        
        # Get the dashboard
        if dashboard_name not in registry.dashboards:
            raise HTTPException(status_code=404, detail=f"Dashboard '{dashboard_name}' not found")
        
        dashboard = registry.dashboards[dashboard_name]
        
        # Initialize wandb API and get the run
        try:
            api = wandb.Api()
        except Exception as e:
            print(f"Failed to initialize W&B API: {str(e)}")
            return {'error': f'Failed to initialize W&B API. Make sure WANDB_API_KEY is set: {str(e)}'}
        
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Get all table specs from dashboard
        table_specs = dashboard.tables(run)
        print(f"Found {len(table_specs)} table specs for slice metrics computation")
        
        # Compute slice metrics for each table step
        slice_metrics_over_time = {}
        step_data = []
        
        for table_spec in table_specs:
            try:
                print(f"Computing slice metrics for step {table_spec.step}")
                
                # Materialize the table data
                df = table_spec.materialize()
                
                # Get slices from the dashboard
                slices = dashboard.slices(df)
                
                # Store metrics for each slice at this step
                step_metrics = {'step': table_spec.step}
                
                for slice_obj in slices:
                    slice_name = slice_obj.name
                    
                    # Initialize slice tracking if not exists
                    if slice_name not in slice_metrics_over_time:
                        slice_metrics_over_time[slice_name] = {
                            'name': slice_name,
                            'data': []
                        }
                    
                    # Convert metrics to JSON-serializable format
                    serialized_metrics = {}
                    for key, value in slice_obj.metrics.items():
                        if pd.isna(value):
                            serialized_metrics[key] = None
                        else:
                            serialized_metrics[key] = float(value) if isinstance(value, (int, float)) else value
                    
                    # Add step and metrics to slice data
                    step_data_point = {
                        'step': table_spec.step,
                        **serialized_metrics
                    }
                    slice_metrics_over_time[slice_name]['data'].append(step_data_point)
                    
                    # Also add to step metrics for easier access
                    step_metrics[slice_name] = serialized_metrics
                
                step_data.append(step_metrics)
                
            except Exception as e:
                print(f"Error computing slice metrics for step {table_spec.step}: {e}")
                continue
        
        # Convert to list format
        slice_metrics_list = list(slice_metrics_over_time.values())
        
        print(f"Successfully computed slice metrics for {len(slice_metrics_list)} slices across {len(step_data)} steps")
        
        return {
            'slice_metrics': slice_metrics_list,
            'step_count': len(step_data)
        }
        
    except Exception as e:
        print(f"Error in get_slice_metrics_over_time: {str(e)}")
        return {'error': str(e)}

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Visualization Server')
    parser.add_argument('--host', default=HOST, help='Host to bind to')
    parser.add_argument('--port', type=int, default=PORT, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', default=RELOAD, help='Enable auto-reload')
    parser.add_argument('--no-cors', action='store_true', help='Disable CORS')
    parser.add_argument('--cors-origins', default=','.join(CORS_ORIGINS), 
                       help='Comma-separated list of allowed CORS origins')
    
    args = parser.parse_args()
    
    # Override configuration with CLI args
    cors_enabled = not args.no_cors and CORS_ENABLED
    cors_origins = args.cors_origins.split(',') if args.cors_origins != ','.join(CORS_ORIGINS) else CORS_ORIGINS
    
    # Configure CORS middleware
    if cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    print(f"Starting server on {args.host}:{args.port}")
    print(f"CORS enabled: {cors_enabled}")
    if cors_enabled:
        print(f"CORS origins: {cors_origins}")
    
    print(f"Start the frontend with: VITE_API_TARGET=http://localhost:{args.port} npm run dev")
    print("If you are on a remote machine, you need to forward the port to your local machine and run the frontend on your local machine.")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)