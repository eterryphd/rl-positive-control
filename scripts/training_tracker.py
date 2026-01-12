# scripts/training_tracker.py
"""
Training tracker for logging all examples and tracking seen problems.

This provides:
1. Complete audit trail of everything sent to/received from the model
2. Set of all problems seen during training (for held-out validation)
3. Compressed logging to manage storage

All logging happens on rank 0 only to avoid file conflicts.
"""

import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class TrainingExample:
    """Single training example with full context."""
    timestamp: str
    global_step: int
    problem: str
    prompt: str
    completion: str
    answer: Any
    predicted: Any
    reward: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'global_step': self.global_step,
            'problem': self.problem,
            'prompt': self.prompt,
            'completion': self.completion,
            'answer': self.answer,
            'predicted': self.predicted,
            'reward': self.reward,
        }


class TrainingTracker:
    """
    Tracks all training examples for reproducibility and validation exclusion.
    
    Features:
    - Logs every prompt/completion/reward to compressed JSONL
    - Maintains set of seen problems for held-out validation
    - Thread-safe for multi-process logging
    - Automatically handles rank 0 logging
    
    Usage:
        tracker = TrainingTracker(output_dir, is_main_process=True)
        tracker.open()
        
        # In reward_fn:
        tracker.log_examples(problems, prompts, completions, answers, predictions, rewards, step)
        
        # For validation:
        held_out = generator.generate_held_out(n, tracker.seen_problems, config)
        
        tracker.close()
    """
    
    def __init__(self, output_dir: Path, is_main_process: bool = True):
        """
        Args:
            output_dir: Directory to write logs
            is_main_process: Only rank 0 should write to avoid conflicts
        """
        self.output_dir = Path(output_dir)
        self.is_main_process = is_main_process
        self.seen_problems: Set[str] = set()
        self.log_file: Optional[gzip.GzipFile] = None
        self.lock = Lock()
        self.total_examples = 0
        self.current_step = 0
        print(f">>> DEBUG TrainingTracker init: is_main_process={is_main_process}")
        
    def open(self):
        """Open log file for writing."""
        if not self.is_main_process:
            return
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use timestamp in filename to avoid overwrites on resume
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = self.output_dir / f'training_log_{timestamp}.jsonl.gz'
        
        self.log_file = gzip.open(log_path, 'wt', encoding='utf-8')
        
        # Write header with metadata
        header = {
            'type': 'header',
            'start_time': datetime.now().isoformat(),
            'version': '1.0',
        }
        self.log_file.write(json.dumps(header) + '\n')
        self.log_file.flush()
        
        print(f">>> Training log: {log_path}")
    
    def close(self):
        """Close log file and write summary."""
        if not self.is_main_process or self.log_file is None:
            return
        
        # Write footer with summary
        footer = {
            'type': 'footer',
            'end_time': datetime.now().isoformat(),
            'total_examples': self.total_examples,
            'unique_problems': len(self.seen_problems),
            'final_step': self.current_step,
        }
        self.log_file.write(json.dumps(footer) + '\n')
        self.log_file.close()
        self.log_file = None
        
        print(f">>> Training log closed: {self.total_examples} examples, {len(self.seen_problems)} unique problems")
    
    def log_examples(
        self,
        problems: List[str],
        prompts: List[str],
        completions: List[str],
        answers: List[Any],
        predictions: List[Any],
        rewards: List[float],
        global_step: int,
    ):
        """
        Log a batch of training examples.
        
        Called from reward_fn with all completions for a batch.
        Note: With GRPO num_generations > 1, multiple completions per problem.
        
        Args:
            problems: Original problem strings
            prompts: Full prompts sent to model
            completions: Model outputs
            answers: Ground truth answers
            predictions: Extracted predictions
            rewards: Computed rewards
            global_step: Current training step
        """
        # DEBUG
        if global_step == 0:
            print(f">>> DEBUG log_examples called: is_main={self.is_main_process}, file={self.log_file is not None}")
            print(f"    problems[0] if any: {problems[0] if problems else 'EMPTY'}")
            print(f"    len(problems): {len(problems)}")
        
        timestamp = datetime.now().isoformat()
        self.current_step = global_step
        
        with self.lock:
            for prob, prompt, comp, ans, pred, rew in zip(
                problems, prompts, completions, answers, predictions, rewards
            ):
                # Track unique problems
                self.seen_problems.add(prob)
                
                # Log if main process
                if self.is_main_process and self.log_file is not None:
                    example = TrainingExample(
                        timestamp=timestamp,
                        global_step=global_step,
                        problem=prob,
                        prompt=prompt,
                        completion=comp,
                        answer=ans,
                        predicted=pred,
                        reward=rew,
                    )
                    self.log_file.write(json.dumps(example.to_dict()) + '\n')
                    self.total_examples += 1
            
            # Flush after every write for debugging
            if self.is_main_process and self.log_file is not None:
                self.log_file.flush()
    
    def get_stats(self) -> Dict:
        """Get current tracking statistics."""
        return {
            'total_examples': self.total_examples,
            'unique_problems': len(self.seen_problems),
            'current_step': self.current_step,
        }


def load_seen_problems(log_dir: Path) -> Set[str]:
    """
    Load seen problems from all training logs in a directory.
    
    Useful for resuming training and ensuring validation excludes
    all previously seen problems.
    
    Robust to corrupt/incomplete gzip files from crashed runs.
    
    Args:
        log_dir: Directory containing training_log_*.jsonl.gz files
    
    Returns:
        Set of all problem strings seen in training
    """
    seen = set()
    log_files = sorted(log_dir.glob('training_log_*.jsonl.gz'))
    
    for log_file in log_files:
        try:
            print(f"    Loading seen problems from {log_file.name}...")
            with gzip.open(log_file, 'rt', encoding='utf-8') as f:
                for line# scripts/training_tracker.py
"""
Training tracker for logging all examples and tracking seen problems.

This provides:
1. Complete audit trail of everything sent to/received from the model
2. Set of all problems seen during training (for held-out validation)
3. Compressed logging to manage storage

All logging happens on rank 0 only to avoid file conflicts.
"""

import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class TrainingExample:
    """Single training example with full context."""
    timestamp: str
    global_step: int
    problem: str
    prompt: str
    completion: str
    answer: Any
    predicted: Any
    reward: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'global_step': self.global_step,
            'problem': self.problem,
            'prompt': self.prompt,
            'completion': self.completion,
            'answer': self.answer,
            'predicted': self.predicted,
            'reward': self.reward,
        }


class TrainingTracker:
    """
    Tracks all training examples for reproducibility and validation exclusion.
    
    Features:
    - Logs every prompt/completion/reward to compressed JSONL
    - Maintains set of seen problems for held-out validation
    - Thread-safe for multi-process logging
    - Automatically handles rank 0 logging
    
    Usage:
        tracker = TrainingTracker(output_dir, is_main_process=True)
        tracker.open()
        
        # In reward_fn:
        tracker.log_examples(problems, prompts, completions, answers, predictions, rewards, step)
        
        # For validation:
        held_out = generator.generate_held_out(n, tracker.seen_problems, config)
        
        tracker.close()
    """
    
    def __init__(self, output_dir: Path, is_main_process: bool = True):
        """
        Args:
            output_dir: Directory to write logs
            is_main_process: Only rank 0 should write to avoid conflicts
        """
        self.output_dir = Path(output_dir)
        self.is_main_process = is_main_process
        self.seen_problems: Set[str] = set()
        self.log_file: Optional[gzip.GzipFile] = None
        self.lock = Lock()
        self.total_examples = 0
        self.current_step = 0
        print(f">>> DEBUG TrainingTracker init: is_main_process={is_main_process}")
        
    def open(self):
        """Open log file for writing."""
        if not self.is_main_process:
            return
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use timestamp in filename to avoid overwrites on resume
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = self.output_dir / f'training_log_{timestamp}.jsonl.gz'
        
        self.log_file = gzip.open(log_path, 'wt', encoding='utf-8')
        
        # Write header with metadata
        header = {
            'type': 'header',
            'start_time': datetime.now().isoformat(),
            'version': '1.0',
        }
        self.log_file.write(json.dumps(header) + '\n')
        self.log_file.flush()
        
        print(f">>> Training log: {log_path}")
    
    def close(self):
        """Close log file and write summary."""
        if not self.is_main_process or self.log_file is None:
            return
        
        # Write footer with summary
        footer = {
            'type': 'footer',
            'end_time': datetime.now().isoformat(),
            'total_examples': self.total_examples,
            'unique_problems': len(self.seen_problems),
            'final_step': self.current_step,
        }
        self.log_file.write(json.dumps(footer) + '\n')
        self.log_file.close()
        self.log_file = None
        
        print(f">>> Training log closed: {self.total_examples} examples, {len(self.seen_problems)} unique problems")
    
    def log_examples(
        self,
        problems: List[str],
        prompts: List[str],
        completions: List[str],
        answers: List[Any],
        predictions: List[Any],
        rewards: List[float],
        global_step: int,
    ):
        """
        Log a batch of training examples.
        
        Called from reward_fn with all completions for a batch.
        Note: With GRPO num_generations > 1, multiple completions per problem.
        
        Args:
            problems: Original problem strings
            prompts: Full prompts sent to model
            completions: Model outputs
            answers: Ground truth answers
            predictions: Extracted predictions
            rewards: Computed rewards
            global_step: Current training step
        """
        # DEBUG
        if global_step == 0:
            print(f">>> DEBUG log_examples called: is_main={self.is_main_process}, file={self.log_file is not None}")
            print(f"    problems[0] if any: {problems[0] if problems else 'EMPTY'}")
            print(f"    len(problems): {len(problems)}")
        
        timestamp = datetime.now().isoformat()
        self.current_step = global_step
        
        with self.lock:
            for prob, prompt, comp, ans, pred, rew in zip(
                problems, prompts, completions, answers, predictions, rewards
            ):
                # Track unique problems
                self.seen_problems.add(prob)
                
                # Log if main process
                if self.is_main_process and self.log_file is not None:
                    example = TrainingExample(
                        timestamp=timestamp,
                        global_step=global_step,
                        problem=prob,
                        prompt=prompt,
                        completion=comp,
                        answer=ans,
                        predicted=pred,
                        reward=rew,
                    )
                    self.log_file.write(json.dumps(example.to_dict()) + '\n')
                    self.total_examples += 1
            
            # Flush after every write for debugging
            if self.is_main_process and self.log_file is not None:
                self.log_file.flush()
    
    def get_stats(self) -> Dict:
        """Get current tracking statistics."""
        return {
            'total_examples': self.total_examples,
            'unique_problems': len(self.seen_problems),
            'current_step': self.current_step,
        }


def load_seen_problems(log_dir: Path) -> Set[str]:
    """
    Load seen problems from all training logs in a directory.
    
    Useful for resuming training and ensuring validation excludes
    all previously seen problems.
    
    Robust to corrupt/incomplete gzip files from crashed runs.
    
    Args:
        log_dir: Directory containing training_log_*.jsonl.gz files
    
    Returns:
        Set of all problem strings seen in training
    """
    seen = set()
    log_files = sorted(log_dir.glob('training_log_*.jsonl.gz'))
    
    for log_file in log_files:
        try:
            print(f"    Loading seen problems from {log_file.name}...")
            with gzip.open(log_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get('type') in ('header', 'footer'):
                            continue
                        if 'problem' in record:
                            seen.add(record['problem'])
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
        except EOFError:
            # Gzip file was truncated (crash during write)
            print(f"    Warning: {log_file.name} is truncated (crashed run?), skipping rest of file")
            continue
        except Exception as e:
            # Any other error - skip this file
            print(f"    Warning: Could not read {log_file.name}: {e}")
            continue
    
    print(f"    Loaded {len(seen)} unique problems from {len(log_files)} log files")
    return seen in f:
                    try:
                        record = json.loads(line)
                        if record.get('type') in ('header', 'footer'):
                            continue
                        if 'problem' in record:
                            seen.add(record['problem'])
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
        except EOFError:
            # Gzip file was truncated (crash during write)
            print(f"    Warning: {log_file.name} is truncated (crashed run?), skipping rest of file")
            continue
        except Exception as e:
            # Any other error - skip this file
            print(f"    Warning: Could not read {log_file.name}: {e}")
            continue
    
    print(f"    Loaded {len(seen)} unique problems from {len(log_files)} log files")
    return seen