import pandas as pd
import pytest
from sqlalchemy import text
from heart_risk.config import Settings
from heart_risk.data.profile import get_missing_tokens, parse_in_memory, EDAProfiler, run_profiling
from heart_risk.db.repository import read_raw, write_raw, count_rows

@pytest.fixture
def sample_df():
    # Inject missing tokens to test detection
    return pd.DataFrame({
        "id": ["1", "2", "3", "4", "5"],
        "age": ["40", "50", "?", "NA", "60"],
        "sex": ["1", "0", "1", "1", "0"],
        "dataset": ["A", "A", "B", "B", "C"],
        "chol": ["200", "0", "", "250", " "],  # space and empty string
        "num": ["0", "1", "2", "0", "4"]
    })

@pytest.fixture
def populated_engine(sqlite_engine, sample_df):
    write_raw(sqlite_engine, sample_df, "dummy_sha256")
    return sqlite_engine


class TestMissingTokens:
    def test_finds_blanks_and_special_chars(self, sample_df):
        cols = ["age", "chol"]
        missing = get_missing_tokens(sample_df, cols)
        
        # age has "?" and "NA"
        assert "'?'" in missing["age"]
        assert "'NA'" in missing["age"]
        
        # chol has "" and " "
        assert "''" in missing["chol"]
        assert "' '" in missing["chol"]


class TestInMemoryParsing:
    def test_parses_numerics_without_changing_original(self, sample_df):
        cols = list(sample_df.columns)
        parsed_df = parse_in_memory(sample_df, cols)
        
        # Original should still be strings
        assert pd.api.types.is_string_dtype(sample_df["age"])
        assert pd.api.types.is_string_dtype(sample_df["chol"])
        
        # Parsed should convert missing tokens to NaN and cast numerics to float/int
        assert pd.api.types.is_numeric_dtype(parsed_df["age"])
        assert pd.api.types.is_numeric_dtype(parsed_df["chol"])
        
        # Check NaNs are correctly placed
        assert pd.isna(parsed_df["age"].iloc[2]) # "?"
        assert pd.isna(parsed_df["age"].iloc[3]) # "NA"
        assert pd.isna(parsed_df["chol"].iloc[2]) # ""
        assert pd.isna(parsed_df["chol"].iloc[4]) # " "


class TestDatabaseReadOnly:
    def test_profiling_does_not_modify_db(self, populated_engine, mock_settings):
        # 1. Capture baseline state
        df_before = read_raw(populated_engine)
        count_before = count_rows(populated_engine)
        
        import hashlib
        df_sorted_before = df_before.sort_values(by='row_id').reset_index(drop=True)
        checksum_before = hashlib.sha256(pd.util.hash_pandas_object(df_sorted_before).values).hexdigest()
        
        # 2. Run Profiling
        run_profiling(populated_engine, mock_settings)
        
        # 3. Capture post-run state
        df_after = read_raw(populated_engine)
        count_after = count_rows(populated_engine)
        df_sorted_after = df_after.sort_values(by='row_id').reset_index(drop=True)
        checksum_after = hashlib.sha256(pd.util.hash_pandas_object(df_sorted_after).values).hexdigest()
        
        # 4. Compare
        assert count_before == count_after, "Row count changed during EDA!"
        assert checksum_before == checksum_after, "Database contents changed during EDA!"
