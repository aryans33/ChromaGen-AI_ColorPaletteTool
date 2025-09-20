import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


def setup_logging():
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
	)


def find_csv_files(data_dir: Path) -> List[Path]:
	if not data_dir.exists():
		raise FileNotFoundError(f"Data directory not found: {data_dir}")
	csvs = sorted([p for p in data_dir.rglob("*.csv") if p.is_file()])
	return csvs


def read_csv_auto(fp: Path) -> pd.DataFrame:
	# Try robust read with auto delimiter and BOM handling
	try:
		df = pd.read_csv(fp, sep=None, engine="python", encoding="utf-8-sig", on_bad_lines="skip")
	except Exception:
		df = pd.read_csv(fp, sep=None, engine="python", encoding="latin1", on_bad_lines="skip")
	return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
	def to_snake(s: str) -> str:
		s = s.strip().lower()
		for ch in ["/", "\\", "-", ".", "(", ")", "[", "]", "{", "}", ":", ";", ",", "#", "@", "!", "?", "%", "&", "*", "+", "=", "|", "<", ">", "'","\""]:
			s = s.replace(ch, " ")
		s = "_".join([part for part in s.split() if part])
		while "__" in s:
			s = s.replace("__", "_")
		return s
	cols = []
	seen = set()
	for c in df.columns:
		base = to_snake(str(c))
		name = base or "col"
		i = 1
		while name in seen:
			name = f"{base}_{i}"
			i += 1
		seen.add(name)
		cols.append(name)
	df.columns = cols
	return df


def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
	# Trim strings and normalize common NA tokens
	if df.empty:
		return df
	obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
	if obj_cols:
		df[obj_cols] = df[obj_cols].apply(lambda s: s.astype(str).str.strip())
	na_tokens = {"", "na", "n/a", "null", "none", "nan", "-", "—", "?", "missing"}
	for col in df.columns:
		if df[col].dtype == object:
			df[col] = df[col].replace(to_replace=list(na_tokens), value=np.nan, regex=False)
	return df


def try_parse_dates(df: pd.DataFrame, threshold: float = 0.8) -> Tuple[pd.DataFrame, List[str]]:
	date_cols = []
	for col in df.columns:
		if df[col].dtype == object:
			ser = df[col]
			non_null = ser.dropna()
			if non_null.empty:
				continue
			try:
				# removed deprecated infer_datetime_format
				parsed = pd.to_datetime(non_null, errors="coerce")
				success_rate = parsed.notna().mean()
				if success_rate >= threshold:
					df[col] = pd.to_datetime(df[col], errors="coerce")
					date_cols.append(col)
			except Exception:
				continue
	return df, date_cols


def expand_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
	for col in date_cols:
		df[f"{col}_year"] = df[col].dt.year
		df[f"{col}_month"] = df[col].dt.month
		df[f"{col}_day"] = df[col].dt.day
		df.drop(columns=[col], inplace=True)
	return df


def coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
	for col in df.columns:
		if df[col].dtype == object:
			# Attempt numeric coercion for mixed-type columns
			coerced = pd.to_numeric(df[col].str.replace(",", "", regex=False), errors="coerce")
			# If sufficient conversion, adopt numeric
			if coerced.notna().mean() >= 0.8:
				df[col] = coerced
	return df


def split_features_target(df: pd.DataFrame, target: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
	if target is None:
		return df, None
	if target not in df.columns:
		raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")
	y = df[target]
	X = df.drop(columns=[target])
	return X, y


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
	cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
	num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

	numeric_pipeline = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler()),
	])

	# Version-compatible OneHotEncoder: prefer sparse_output (>=1.2), fallback to sparse (older)
	try:
		ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
	except TypeError:
		ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

	categorical_pipeline = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="most_frequent")),
		("onehot", ohe),
	])

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_pipeline, num_cols),
			("cat", categorical_pipeline, cat_cols),
		],
		remainder="drop",
	)
	return preprocessor, num_cols, cat_cols


def process(data_dir: Path, output_dir: Path, target: Optional[str], test_size: float, random_state: int):
	output_dir.mkdir(parents=True, exist_ok=True)

	csv_files = find_csv_files(data_dir)
	if not csv_files:
		raise FileNotFoundError(f"No CSV files found in {data_dir}")

	logging.info(f"Found {len(csv_files)} CSV file(s). Using all by concatenation.")
	dfs = []
	for fp in csv_files:
		logging.info(f"Loading: {fp}")
		df = read_csv_auto(fp)
		dfs.append(df)
	raw = pd.concat(dfs, ignore_index=True, sort=False)

	logging.info(f"Raw shape: {raw.shape}")
	df = raw.copy()
	df = standardize_column_names(df)
	df = normalize_values(df)
	# Drop duplicate rows
	before = len(df)
	df = df.drop_duplicates()
	logging.info(f"Dropped {before - len(df)} duplicate rows")

	# Parse dates and expand
	df, date_cols = try_parse_dates(df)
	if date_cols:
		logging.info(f"Detected date columns: {date_cols}")
		df = expand_dates(df, date_cols)

	# Coerce numerics when possible
	df = coerce_numerics(df)

	# Split features/target
	X, y = split_features_target(df, target)

	# Build and fit preprocessor
	preprocessor, num_cols, cat_cols = build_preprocessor(X)
	X_processed = preprocessor.fit_transform(X)
	feature_names = preprocessor.get_feature_names_out().tolist()

	Xp = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

	artifacts = {
		"numeric_columns": num_cols,
		"categorical_columns": cat_cols,
		"generated_features": feature_names,
		"date_columns_expanded": date_cols,
		"rows_raw": int(len(raw)),
		"rows_deduped": int(len(df)),
		"columns_input": list(X.columns),
		"target": target,
	}

	# Save outputs
	if y is not None:
		X_train, X_test, y_train, y_test = train_test_split(
			Xp, y, test_size=test_size, random_state=random_state, stratify=None
		)
		X_train.to_csv(output_dir / "X_train.csv", index=False)
		X_test.to_csv(output_dir / "X_test.csv", index=False)
		y_train.to_csv(output_dir / "y_train.csv", index=False)
		y_test.to_csv(output_dir / "y_test.csv", index=False)
		logging.info(f"Saved train/test splits to {output_dir}")
	else:
		Xp.to_csv(output_dir / "X_processed.csv", index=False)
		logging.info(f"Saved processed features to {output_dir / 'X_processed.csv'}")

	# Save preprocessor and metadata
	joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
	with open(output_dir / "preprocess_report.json", "w", encoding="utf-8") as f:
		json.dump(artifacts, f, indent=2)

	logging.info("Preprocessing complete.")


def main():
	setup_logging()
	parser = argparse.ArgumentParser(description="Preprocess CSV dataset(s) in the data folder.")
	parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent / "data"),
	                    help="Directory containing CSV files. Defaults to ./data")
	parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "data" / "processed"),
	                    help="Directory to write processed outputs. Defaults to ./data/processed")
	parser.add_argument("--target", type=str, default=None, help="Name of target column, if supervised.")
	parser.add_argument("--test-size", type=float, default=0.2, help="Test split size if target is provided.")
	parser.add_argument("--random-state", type=int, default=42, help="Random state for splitting.")
	args = parser.parse_args()

	data_dir = Path(args.data_dir)
	output_dir = Path(args.output_dir)

	process(data_dir=data_dir, output_dir=output_dir, target=args.target, test_size=args.test_size, random_state=args.random_state)


if __name__ == "__main__":
	main()
