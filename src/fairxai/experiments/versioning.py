"""Experiment versioning and result management system."""

import json
import yaml
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
import subprocess


class ExperimentVersioning:
    """
    Manages experiment versioning with latest_run and archived runs.
    """
    
    def __init__(self, base_results_dir: Path):
        """
        Initialize versioning system.
        
        Args:
            base_results_dir: Base directory for experiment results
                             (e.g., results/experiments/)
        """
        self.base_dir = Path(base_results_dir)
        self.latest_dir = self.base_dir / "latest_run"
        self.archives_dir = self.base_dir / "archived_runs"
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        self.archives_dir.mkdir(parents=True, exist_ok=True)
        (self.latest_dir / "manifests").mkdir(exist_ok=True)
        (self.latest_dir / "results").mkdir(exist_ok=True)
        (self.latest_dir / "predictions").mkdir(exist_ok=True)
        (self.latest_dir / "models").mkdir(exist_ok=True)
    
    def generate_experiment_id(self) -> str:
        """
        Generate unique experiment ID.
        
        Returns:
            UUID string (8 characters)
        """
        return str(uuid.uuid4())[:8]
    
    def get_git_commit(self) -> Optional[str]:
        """
        Get current git commit SHA.
        
        Returns:
            Commit SHA or None if not a git repo
        """
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def get_git_status(self) -> Dict[str, Any]:
        """
        Get git repository status.
        
        Returns:
            Dictionary with commit, branch, and dirty status
        """
        try:
            commit = self.get_git_commit()
            
            # Get branch name
            branch_result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            branch = branch_result.stdout.strip()
            
            # Check if working directory is dirty
            status_result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True
            )
            is_dirty = len(status_result.stdout.strip()) > 0
            
            return {
                'commit': commit,
                'branch': branch,
                'is_dirty': is_dirty
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {
                'commit': None,
                'branch': None,
                'is_dirty': None
            }
    
    def save_manifest(
        self,
        exp_id: str,
        config: Dict[str, Any],
        execution_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save experiment manifest with full configuration.
        
        Args:
            exp_id: Experiment ID
            config: Experiment configuration dictionary
            execution_metadata: Optional execution metadata
            
        Returns:
            Path to saved manifest file
        """
        manifest = {
            'experiment_id': exp_id,
            'timestamp': datetime.now().isoformat(),
            'git': self.get_git_status(),
            'configuration': config
        }
        
        if execution_metadata:
            manifest['execution'] = execution_metadata
        
        manifest_path = self.latest_dir / "manifests" / f"experiment_{exp_id}.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"✓ Saved manifest: {manifest_path}")
        return manifest_path
    
    def save_results(
        self,
        exp_id: str,
        results: Dict[str, Any],
        format: str = 'json'
    ) -> Path:
        """
        Save experiment results.
        
        Args:
            exp_id: Experiment ID
            results: Results dictionary
            format: Output format ('json' or 'yaml')
            
        Returns:
            Path to saved results file
        """
        results_path = self.latest_dir / "results" / f"results_{exp_id}.{format}"
        
        if format == 'json':
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif format == 'yaml':
            with open(results_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.debug(f"✓ Saved results: {results_path}")
        return results_path
    
    def save_predictions(
        self,
        exp_id: str,
        predictions: Any,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save model predictions.
        
        Args:
            exp_id: Experiment ID
            predictions: Predictions (DataFrame or dict)
            filename: Optional custom filename
            
        Returns:
            Path to saved predictions file
        """
        if filename is None:
            filename = f"predictions_{exp_id}.csv"
        
        pred_path = self.latest_dir / "predictions" / filename
        
        if hasattr(predictions, 'to_csv'):
            predictions.to_csv(pred_path, index=False)
        else:
            with open(pred_path, 'w') as f:
                json.dump(predictions, f, indent=2, default=str)
        
        self.logger.debug(f"✓ Saved predictions: {pred_path}")
        return pred_path
    
    def archive_previous_run(self) -> Optional[Path]:
        """
        Archive the current latest_run to archived_runs/run_{datetime}.
        
        Returns:
            Path to archived directory or None if nothing to archive
        """
        # Check if latest_run has any results
        results_dir = self.latest_dir / "results"
        if not results_dir.exists() or not list(results_dir.glob("*.json")):
            self.logger.info("No previous run to archive")
            return None
        
        # Create archive directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.archives_dir / f"run_{timestamp}"
        
        # Move latest_run to archive
        shutil.copytree(self.latest_dir, archive_path)
        
        # Clean latest_run (keep directory structure)
        for subdir in ['manifests', 'results', 'predictions', 'models']:
            subdir_path = self.latest_dir / subdir
            if subdir_path.exists():
                for file in subdir_path.glob('*'):
                    file.unlink()
        
        self.logger.info(f"✓ Archived previous run to: {archive_path}")
        return archive_path
    
    def load_experiment(self, exp_id: str, from_archive: bool = False) -> Dict[str, Any]:
        """
        Load experiment manifest and results.
        
        Args:
            exp_id: Experiment ID
            from_archive: Whether to load from archives
            
        Returns:
            Dictionary with manifest and results
        """
        if from_archive:
            # Search in archives
            manifest_files = list(self.archives_dir.glob(f"*/manifests/experiment_{exp_id}.yaml"))
            result_files = list(self.archives_dir.glob(f"*/results/results_{exp_id}.json"))
        else:
            # Load from latest_run
            manifest_files = [self.latest_dir / "manifests" / f"experiment_{exp_id}.yaml"]
            result_files = [self.latest_dir / "results" / f"results_{exp_id}.json"]
        
        if not manifest_files:
            raise FileNotFoundError(f"Manifest not found for experiment: {exp_id}")
        
        # Load manifest
        with open(manifest_files[0], 'r') as f:
            manifest = yaml.safe_load(f)
        
        # Load results if available
        results = None
        if result_files and result_files[0].exists():
            with open(result_files[0], 'r') as f:
                results = json.load(f)
        
        return {
            'manifest': manifest,
            'results': results
        }
    
    def list_experiments(self, from_archive: bool = False) -> List[Dict[str, Any]]:
        """
        List all experiments.
        
        Args:
            from_archive: Whether to list archived experiments
            
        Returns:
            List of experiment summaries
        """
        if from_archive:
            manifest_files = list(self.archives_dir.glob("*/manifests/experiment_*.yaml"))
        else:
            manifest_files = list((self.latest_dir / "manifests").glob("experiment_*.yaml"))
        
        experiments = []
        for manifest_file in manifest_files:
            with open(manifest_file, 'r') as f:
                manifest = yaml.safe_load(f)
            
            experiments.append({
                'experiment_id': manifest['experiment_id'],
                'timestamp': manifest['timestamp'],
                'dataset': manifest['configuration'].get('dataset'),
                'binning': manifest['configuration'].get('binning_strategy'),
                'mitigation': manifest['configuration'].get('mitigation_technique'),
                'training_method': manifest['configuration'].get('training_method')
            })
        
        return sorted(experiments, key=lambda x: x['timestamp'], reverse=True)
    
    def create_summary(self) -> Dict[str, Any]:
        """
        Create summary of latest run.
        
        Returns:
            Summary dictionary with counts and metadata
        """
        experiments = self.list_experiments(from_archive=False)
        
        # Count by configuration
        datasets = set(e['dataset'] for e in experiments)
        binning_strategies = set(e['binning'] for e in experiments)
        mitigation_techniques = set(e['mitigation'] for e in experiments)
        training_methods = set(e['training_method'] for e in experiments)
        
        summary = {
            'total_experiments': len(experiments),
            'datasets': list(datasets),
            'binning_strategies': list(binning_strategies),
            'mitigation_techniques': list(mitigation_techniques),
            'training_methods': list(training_methods),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = self.latest_dir / "run_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"✓ Created run summary: {summary_path}")
        return summary
