#  Copyright 2025 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import psutil


class MemoryProfiler:
    """Lightweight memory profiler for tracking memory usage during evaluation."""
    
    def __init__(self, output_dir: Path = Path("/output"), enable_wandb: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = self.output_dir / "memory_profile.csv"
        self.enable_wandb = enable_wandb
        self.start_time = time.time()
        self._lock = Lock()
        self._continuous_logging_thread = None
        self._stop_continuous_logging = threading.Event()
        
        # Initialize wandb if enabled
        self.wandb = None
        if self.enable_wandb:
            try:
                import wandb
                self.wandb = wandb
                if not wandb.run:
                    wandb.init(project="unicorn-eval-memory", name=f"memory-profile-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            except ImportError:
                print("Warning: wandb not available, falling back to CSV only")
                self.enable_wandb = False
        
        # Initialize CSV file
        self._initialize_csv()
        
        # Record initial memory state
        self.log_memory("initialization")
    
    def _initialize_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'elapsed_seconds',
                'event',
                'process_memory_mb',
                'system_memory_percent',
                'available_memory_mb',
                'cpu_percent'
            ])
    
    def log_memory(self, event: str, task_name: Optional[str] = None):
        """Log current memory usage with event description."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            cpu_percent = process.cpu_percent()
            
            current_time = time.time()
            elapsed_seconds = current_time - self.start_time
            
            # Format event name
            if task_name:
                event = f"{event}:{task_name}"
            
            memory_data = {
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': round(elapsed_seconds, 2),
                'event': event,
                'process_memory_mb': round(memory_info.rss / 1024 / 1024, 2),
                'system_memory_percent': round(system_memory.percent, 2),
                'available_memory_mb': round(system_memory.available / 1024 / 1024, 2),
                'cpu_percent': round(cpu_percent, 2)
            }
            
            # Thread-safe CSV writing
            with self._lock:
                with open(self.csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        memory_data['timestamp'],
                        memory_data['elapsed_seconds'],
                        memory_data['event'],
                        memory_data['process_memory_mb'],
                        memory_data['system_memory_percent'],
                        memory_data['available_memory_mb'],
                        memory_data['cpu_percent']
                    ])
            
            # Log to wandb if enabled
            if self.enable_wandb and self.wandb and self.wandb.run:
                self.wandb.log({
                    'memory/process_memory_mb': memory_data['process_memory_mb'],
                    'memory/system_memory_percent': memory_data['system_memory_percent'],
                    'memory/available_memory_mb': memory_data['available_memory_mb'],
                    'cpu/cpu_percent': memory_data['cpu_percent'],
                    'event': event
                }, step=int(elapsed_seconds))
        
        except Exception as e:
            # Fail silently to avoid disrupting evaluation
            print(f"Memory profiling error: {e}")
    
    def _continuous_logging_worker(self):
        """Worker function for continuous memory logging every second."""
        while not self._stop_continuous_logging.wait(1.0):  # Wait 1 second or until stop event
            self.log_memory("continuous")
    
    def start_continuous_logging(self):
        """Start continuous memory logging every second in a background thread."""
        if self._continuous_logging_thread is None or not self._continuous_logging_thread.is_alive():
            self._stop_continuous_logging.clear()
            self._continuous_logging_thread = threading.Thread(
                target=self._continuous_logging_worker,
                daemon=True,
                name="MemoryProfiler-Continuous"
            )
            self._continuous_logging_thread.start()
            print("Started continuous memory logging (every 1 second)")
    
    def stop_continuous_logging(self):
        """Stop continuous memory logging."""
        if self._continuous_logging_thread and self._continuous_logging_thread.is_alive():
            self._stop_continuous_logging.set()
            self._continuous_logging_thread.join(timeout=2)
            print("Stopped continuous memory logging")
    
    def log_peak_memory(self, event: str):
        """Log peak memory usage since last call."""
        try:
            process = psutil.Process()
            # Get peak memory if available (Linux only)
            if hasattr(process, "memory_full_info"):
                memory_info = process.memory_full_info()
                if hasattr(memory_info, 'peak_nonpaged_pool'):
                    self.log_memory(f"{event}_peak")
            else:
                # Fallback to current memory
                self.log_memory(f"{event}_current")
        except Exception:
            # Fallback to regular memory logging
            self.log_memory(event)
    
    def get_memory_summary(self) -> dict:
        """Get summary statistics from the CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_file)
            return {
                'max_memory_mb': df['process_memory_mb'].max(),
                'avg_memory_mb': df['process_memory_mb'].mean(),
                'min_available_memory_mb': df['available_memory_mb'].min(),
                'max_system_memory_percent': df['system_memory_percent'].max(),
                'total_events': len(df),
                'duration_seconds': df['elapsed_seconds'].max()
            }
        except ImportError:
            # Fallback without pandas
            max_memory = 0
            event_count = 0
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    max_memory = max(max_memory, float(row['process_memory_mb']))
                    event_count += 1
            return {
                'max_memory_mb': max_memory,
                'total_events': event_count
            }
    
    def finalize(self):
        """Finalize profiling and optionally close wandb."""
        self.stop_continuous_logging()
        self.log_memory("finalization")
        summary = self.get_memory_summary()
        print(f"Memory profiling complete. Max memory usage: {summary['max_memory_mb']:.2f} MB")
        print(f"Memory profile saved to: {self.csv_file}")
        
        if self.enable_wandb and self.wandb and self.wandb.run:
            # Log final summary
            self.wandb.log(summary)
            self.wandb.finish()


# Global profiler instance
_profiler: Optional[MemoryProfiler] = None


def init_memory_profiler(output_dir: Path = Path("/tmp"), enable_wandb: bool = False) -> MemoryProfiler:
    """Initialize global memory profiler."""
    global _profiler
    _profiler = MemoryProfiler(output_dir=output_dir, enable_wandb=enable_wandb)
    return _profiler


def log_memory(event: str, task_name: Optional[str] = None):
    """Log memory usage if profiler is initialized."""
    if _profiler:
        _profiler.log_memory(event, task_name)


def log_peak_memory(event: str):
    """Log peak memory usage if profiler is initialized."""
    if _profiler:
        _profiler.log_peak_memory(event)


def get_profiler() -> Optional[MemoryProfiler]:
    """Get the global profiler instance."""
    return _profiler


def start_continuous_logging():
    """Start continuous memory logging if profiler is initialized."""
    if _profiler:
        _profiler.start_continuous_logging()


def stop_continuous_logging():
    """Stop continuous memory logging if profiler is initialized."""
    if _profiler:
        _profiler.stop_continuous_logging()


def finalize_memory_profiler():
    """Finalize and cleanup global memory profiler."""
    global _profiler
    if _profiler:
        _profiler.finalize()
        _profiler = None