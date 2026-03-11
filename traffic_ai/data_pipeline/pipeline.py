from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from traffic_ai.config.settings import Settings
from traffic_ai.data_pipeline.cleaning import DataCleaner
from traffic_ai.data_pipeline.features import FeatureEngineer
from traffic_ai.data_pipeline.ingestion import DataIngestor, SourceResult
from traffic_ai.data_pipeline.preprocessing import DatasetPreprocessor, PreparedDataset
from traffic_ai.utils.io_utils import write_json


@dataclass(slots=True)
class DataPipelineResult:
    source_results: list[SourceResult]
    cleaned_files: list[Path]
    modeling_table_path: Path
    prepared_dataset: PreparedDataset


class TrafficDataPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ingestor = DataIngestor(settings)
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.preprocessor = DatasetPreprocessor(
            timestamp_col=settings.get("data.timestamp_column", "timestamp"),
            target_col=settings.get("data.target_column", "optimal_phase"),
            normalize=bool(settings.get("data.normalize", True)),
        )

    def run(
        self,
        include_kaggle: bool = True,
        include_public: bool = True,
        include_local_csv: bool = True,
    ) -> DataPipelineResult:
        sources = self.ingestor.ingest_all(
            include_kaggle=include_kaggle,
            include_public=include_public,
            include_local_csv=include_local_csv,
        )

        raw_paths = [item.path for item in sources]
        cleaned_files = self.cleaner.clean_files(raw_paths, self.settings.processed_data_dir)
        combined = self._combine_cleaned(cleaned_files)
        modeling_table = self.engineer.build_modeling_table(combined)

        modeling_table_path = self.settings.processed_data_dir / "modeling_table.csv"
        modeling_table.to_csv(modeling_table_path, index=False)

        prepared = self.preprocessor.prepare(
            modeling_table,
            train_ratio=float(self.settings.get("data.train_ratio", 0.7)),
            val_ratio=float(self.settings.get("data.val_ratio", 0.15)),
            test_ratio=float(self.settings.get("data.test_ratio", 0.15)),
        )
        self.preprocessor.save_scaler(self.settings.output_dir / "models" / "feature_scaler.joblib")
        write_json(prepared.metadata, self.settings.output_dir / "results" / "dataset_metadata.json")

        return DataPipelineResult(
            source_results=sources,
            cleaned_files=cleaned_files,
            modeling_table_path=modeling_table_path,
            prepared_dataset=prepared,
        )

    @staticmethod
    def _combine_cleaned(cleaned_files: list[Path]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for file in cleaned_files:
            try:
                frames.append(pd.read_csv(file, low_memory=False))
            except Exception:
                continue
        if not frames:
            raise RuntimeError("No cleaned dataset could be loaded.")
        combined = pd.concat(frames, ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        combined = combined.dropna(subset=["timestamp"])
        return combined.sort_values("timestamp").reset_index(drop=True)

