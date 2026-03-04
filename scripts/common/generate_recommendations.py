#!/usr/bin/env python3
"""
Generate fairness triage recommendations for pipeline datasets (pre-model).

This script:
1. Loads standardized datasets (or a custom CSV)
2. Profiles each dataset using DataProfiler
3. Runs the triage rule engine (categories A–F)
4. Outputs a JSON payload and/or Markdown report

Usage:
    # For registered pipeline datasets:
    python scripts/common/generate_recommendations.py --pipeline cardiac

    # For an ad-hoc CSV:
    python scripts/common/generate_recommendations.py \
        --csv /path/to/data.csv \
        --label target \
        --sensitive sex age_group \
        --format both
"""

import sys
import logging
import argparse
import json
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from fairxai.cli.runner_base import get_project_root, load_pipeline_config, setup_phase_logging
from fairxai.cli.runner_utils import resolve_run_id, get_run_root
from fairxai.recommendations.engine import RecommendationEngine
from fairxai.recommendations.ingestion import ingestion_from_schema, confirm_ingestion
from fairxai.recommendations.output import to_json_string, to_markdown


def main():
    parser = argparse.ArgumentParser(
        description='Generate fairness triage recommendations'
    )
    parser.add_argument(
        '--pipeline',
        type=str,
        default='cardiac',
        choices=['cardiac', 'dermatology'],
        help='Pipeline name (e.g., cardiac, dermatology)',
    )
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Verbosity: -v=info, -vv=debug')
    parser.add_argument(
        '--run-id',
        type=str,
        default=os.getenv('RUN_ID'),
        help='Run identifier (optional, enables run-scoped outputs)',
    )

    # Ad-hoc CSV mode
    parser.add_argument('--csv', type=str, help='Path to a CSV file (ad-hoc mode)')
    parser.add_argument('--label', type=str, help='Label/target column name')
    parser.add_argument(
        '--sensitive',
        type=str,
        nargs='+',
        help='Sensitive attribute column names',
    )
    parser.add_argument(
        '--identifier',
        type=str,
        nargs='*',
        help='Identifier column names (to exclude from features)',
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        help='Human-readable dataset name',
    )

    # Schema-based mode
    parser.add_argument(
        '--schema',
        type=str,
        help='Path to schema JSON (e.g., configs/schema/cardiac.json)',
    )
    parser.add_argument(
        '--dataset-key',
        type=str,
        help='Key within schema JSON (e.g., cleveland)',
    )

    # Output config
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: output/{pipeline}/recommendations/)',
    )
    parser.add_argument(
        '--format',
        type=str,
        default='both',
        choices=['json', 'markdown', 'both'],
        help='Output format',
    )
    parser.add_argument(
        '--overrides',
        type=str,
        help='JSON string of column overrides, e.g. \'{"col": {"role": "sensitive"}}\'',
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to recommendation thresholds YAML',
    )
    parser.add_argument(
        '--history-path',
        type=str,
        help='Path to results directory for historical reference',
    )

    args = parser.parse_args()
    pipeline = args.pipeline

    # Paths
    project_root = get_project_root(Path(__file__))
    run_id = resolve_run_id(args.run_id) if args.run_id else None
    setup_phase_logging(
        project_root, 'recommendations.log', verbose=args.verbose,
        run_id=run_id, stage_name='recommend',
    )

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif run_id:
        output_dir = get_run_root(project_root / f'output/{pipeline}', run_id) / 'recommendations'
    else:
        output_dir = project_root / f'output/{pipeline}/recommendations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize engine
    config_path = args.config or str(project_root / 'configs/recommendations/thresholds.yaml')
    history_path = args.history_path or str(project_root / f'output/{pipeline}')

    engine = RecommendationEngine(
        config_path=config_path,
        project_root=str(project_root),
        history_base_path=history_path,
    )

    logging.info("[PHASE] Generating fairness triage recommendations")

    # Parse overrides
    overrides = {}
    if args.overrides:
        try:
            overrides = json.loads(args.overrides)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid --overrides JSON: {e}")
            sys.exit(1)

    # Determine datasets to process
    if args.csv:
        # --- Ad-hoc CSV mode ---
        _process_adhoc(engine, args, output_dir, overrides)
    elif args.schema and args.dataset_key:
        # --- Single dataset from schema ---
        _process_schema_dataset(engine, project_root, args, output_dir)
    else:
        # --- Pipeline mode: process all registered datasets ---
        _process_pipeline_datasets(engine, project_root, pipeline, args, output_dir)

    logging.info(f"\n{'='*60}")
    logging.info("[PHASE] Recommendation generation complete")
    logging.info(f"{'='*60}")
    logging.info(f"Output saved to: {output_dir}")


def _process_adhoc(engine, args, output_dir, overrides):
    """Process a standalone CSV file."""
    logging.info(f"Processing ad-hoc CSV: {args.csv}")

    ingestion = engine.ingest(
        args.csv,
        label_column=args.label,
        sensitive_columns=args.sensitive,
        identifier_columns=args.identifier,
        dataset_name=args.dataset_name,
    )

    if overrides:
        ingestion = confirm_ingestion(ingestion, overrides)

    report = engine.generate(ingestion)
    name = ingestion.dataset_name or 'dataset'
    _save_report(report, output_dir, name, args.format)


def _process_schema_dataset(engine, project_root, args, output_dir):
    """Process a single dataset referenced by schema + key."""
    schema_path = args.schema
    if not Path(schema_path).is_absolute():
        schema_path = str(project_root / schema_path)

    logging.info(f"Processing dataset '{args.dataset_key}' from schema: {schema_path}")

    ingestion = engine.ingest_from_schema(
        schema_path, args.dataset_key,
        data_dir=str(Path(schema_path).parent.parent.parent / 'data' / 'raw' / 'cardiac'),
    )
    report = engine.generate(ingestion)
    _save_report(report, output_dir, args.dataset_key, args.format)


def _process_pipeline_datasets(engine, project_root, pipeline, args, output_dir):
    """Process all standardized datasets in the pipeline's raw directory."""
    pipeline_cfg = load_pipeline_config(project_root, pipeline)
    data_raw = project_root / pipeline_cfg['paths']['raw_dir']
    schema_path = str(project_root / pipeline_cfg['runtime']['schema_mapping_json'])
    sensitive_attrs = pipeline_cfg.get('fairness', {}).get('sensitive_attributes', ['age_group', 'sex'])

    dataset_files = sorted(data_raw.glob('*_standardized.csv'))

    if not dataset_files:
        logging.error(f"No standardized datasets found in {data_raw}")
        logging.error("Run the load_data phase first (scripts/common/load_data.py --pipeline %s)." % pipeline)
        return

    logging.info(f"Found {len(dataset_files)} standardized dataset(s)")

    for filepath in dataset_files:
        dataset_name = filepath.stem.replace('_standardized', '')
        logging.info(f"\n{'='*60}")
        logging.info(f"Generating recommendations: {dataset_name}")
        logging.info(f"{'='*60}")

        # Build ingestion from the CSV with pipeline-level sensitive attrs
        ingestion = engine.ingest(
            str(filepath),
            sensitive_columns=sensitive_attrs,
            dataset_name=dataset_name,
        )

        report = engine.generate(ingestion)

        _log_summary(report, dataset_name)
        _save_report(report, output_dir, dataset_name, args.format)


def _log_summary(report, dataset_name):
    """Log a concise summary of the triage result."""
    logging.info(f"\n--- Triage Summary: {dataset_name} ---")
    logging.info(f"  Readiness: {report.readiness_status.value}")
    logging.info(f"  P0 (critical): {report.critical_count}")
    logging.info(f"  P1 (high): {report.high_count}")
    logging.info(f"  Total recommendations: {len(report.recommendations)}")

    for rec in report.recommendations:
        icon = {"P0": "🔴", "P1": "🟠", "P2": "🟡", "P3": "🔵"}.get(rec.priority.value, "⚪")
        logging.info(f"  {icon} [{rec.priority.value}][{rec.category.value}] {rec.title}")

    if report.limitations:
        logging.info(f"  Limitations:")
        for lim in report.limitations:
            logging.info(f"    - {lim}")


def _save_report(report, output_dir, name, fmt):
    """Write report to disk in requested format(s)."""
    if fmt in ('json', 'both'):
        json_path = output_dir / f'{name}_triage.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(to_json_string(report))
        logging.info(f"[SUCCESS] JSON saved: {json_path}")

    if fmt in ('markdown', 'both'):
        md_path = output_dir / f'{name}_triage.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(to_markdown(report))
        logging.info(f"[SUCCESS] Markdown saved: {md_path}")


if __name__ == "__main__":
    main()
