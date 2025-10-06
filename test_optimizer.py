"""
Basic Test Suite for NFL DFS Optimizer
Version: 1.0.0

Run with: pytest test_optimizer.py -v

NICE TO HAVE: Provides basic test coverage for critical functions
"""

import pytest
import pandas as pd
import numpy as np
from nfl_dfs_optimizer import (
    # Utility functions
    normalize_position,
    normalize_ownership,
    validate_and_normalize_dataframe,
    calculate_lineup_similarity,
    format_lineup_for_export,
    validate_export_format,
    
    # Classes
    ValidationLevel,
    ExportFormat,
    
    # Configuration
    DraftKingsRules,
)


# ============================================================================
# POSITION NORMALIZATION TESTS
# ============================================================================

class TestPositionNormalization:
    """Test position normalization handles various formats"""
    
    def test_standard_positions(self):
        """Test standard position abbreviations"""
        assert normalize_position("QB") == "QB"
        assert normalize_position("RB") == "RB"
        assert normalize_position("WR") == "WR"
        assert normalize_position("TE") == "TE"
        assert normalize_position("K") == "K"
        assert normalize_position("DST") == "DST"
    
    def test_lowercase_positions(self):
        """Test lowercase inputs"""
        assert normalize_position("qb") == "QB"
        assert normalize_position("wr") == "WR"
        assert normalize_position("te") == "TE"
    
    def test_full_position_names(self):
        """Test full position names"""
        assert normalize_position("QUARTERBACK") == "QB"
        assert normalize_position("Wide Receiver") == "WR"
        assert normalize_position("TIGHT END") == "TE"
        assert normalize_position("RUNNING BACK") == "RB"
    
    def test_defense_variations(self):
        """Test various defense notations"""
        assert normalize_position("DST") == "DST"
        assert normalize_position("D/ST") == "DST"
        assert normalize_position("DEF") == "DST"
        assert normalize_position("DEFENSE") == "DST"
    
    def test_invalid_position_raises_error(self):
        """Test that invalid positions raise ValueError"""
        with pytest.raises(ValueError):
            normalize_position("INVALID")
        
        with pytest.raises(ValueError):
            normalize_position("XYZ")


# ============================================================================
# OWNERSHIP NORMALIZATION TESTS
# ============================================================================

class TestOwnershipNormalization:
    """Test ownership handles both decimal and percentage formats"""
    
    def test_decimal_format_conversion(self):
        """Test decimal format (0-1) converts to percentage"""
        df = pd.DataFrame({
            'Player': ['PlayerA', 'PlayerB', 'PlayerC'],
            'Ownership': [0.15, 0.8, 0.05]
        })
        result = normalize_ownership(df)
        
        assert result['Ownership'].iloc[0] == 15.0
        assert result['Ownership'].iloc[1] == 80.0
        assert result['Ownership'].iloc[2] == 5.0
    
    def test_percentage_format_unchanged(self):
        """Test percentage format (0-100) stays the same"""
        df = pd.DataFrame({
            'Player': ['PlayerA', 'PlayerB'],
            'Ownership': [15.0, 80.0]
        })
        result = normalize_ownership(df)
        
        assert result['Ownership'].iloc[0] == 15.0
        assert result['Ownership'].iloc[1] == 80.0
    
    def test_missing_ownership_defaults(self):
        """Test missing ownership gets default value"""
        df = pd.DataFrame({
            'Player': ['PlayerA', 'PlayerB'],
        })
        result = normalize_ownership(df)
        
        assert 'Ownership' in result.columns
        assert result['Ownership'].iloc[0] == 10.0
    
    def test_invalid_ownership_clamped(self):
        """Test values > 100 get clamped"""
        df = pd.DataFrame({
            'Player': ['PlayerA'],
            'Ownership': [150.0]
        })
        result = normalize_ownership(df)
        
        assert result['Ownership'].iloc[0] == 100.0
    
    def test_negative_ownership_defaults(self):
        """Test negative ownership gets default"""
        df = pd.DataFrame({
            'Player': ['PlayerA'],
            'Ownership': [-5.0]
        })
        result = normalize_ownership(df)
        
        assert result['Ownership'].iloc[0] == 10.0


# ============================================================================
# LINEUP SIMILARITY TESTS
# ============================================================================

class TestLineupSimilarity:
    """Test lineup similarity calculation"""
    
    def test_identical_lineups(self):
        """Test identical lineups return 1.0"""
        lineup1 = {
            'Captain': 'PlayerA',
            'FLEX': ['PlayerB', 'PlayerC', 'PlayerD', 'PlayerE', 'PlayerF']
        }
        
        lineup2 = {
            'Captain': 'PlayerA',
            'FLEX': ['PlayerB', 'PlayerC', 'PlayerD', 'PlayerE', 'PlayerF']
        }
        
        similarity = calculate_lineup_similarity(lineup1, lineup2)
        assert similarity == 1.0
    
    def test_completely_different_lineups(self):
        """Test completely different lineups return 0.0"""
        lineup1 = {
            'Captain': 'PlayerA',
            'FLEX': ['PlayerB', 'PlayerC', 'PlayerD', 'PlayerE', 'PlayerF']
        }
        
        lineup2 = {
            'Captain': 'PlayerX',
            'FLEX': ['PlayerY', 'PlayerZ', 'Player1', 'Player2', 'Player3']
        }
        
        similarity = calculate_lineup_similarity(lineup1, lineup2)
        assert similarity == 0.0
    
    def test_partial_similarity(self):
        """Test partially similar lineups"""
        lineup1 = {
            'Captain': 'PlayerA',
            'FLEX': ['PlayerB', 'PlayerC', 'PlayerD', 'PlayerE', 'PlayerF']
        }
        
        lineup2 = {
            'Captain': 'PlayerA',  # Same captain
            'FLEX': ['PlayerB', 'PlayerC', 'PlayerX', 'PlayerY', 'PlayerZ']  # 2 same, 3 diff
        }
        
        similarity = calculate_lineup_similarity(lineup1, lineup2)
        # 3 players in common out of 6 total = 0.5
        assert similarity == 0.5
    
    def test_string_flex_format(self):
        """Test similarity works with string FLEX format"""
        lineup1 = {
            'Captain': 'PlayerA',
            'FLEX': 'PlayerB, PlayerC, PlayerD, PlayerE, PlayerF'
        }
        
        lineup2 = {
            'Captain': 'PlayerA',
            'FLEX': ['PlayerB', 'PlayerC', 'PlayerD', 'PlayerE', 'PlayerF']
        }
        
        similarity = calculate_lineup_similarity(lineup1, lineup2)
        assert similarity == 1.0


# ============================================================================
# EXPORT FORMAT TESTS
# ============================================================================

class TestExportFormat:
    """Test lineup export formatting"""
    
    def test_draftkings_format_basic(self):
        """Test DraftKings format export"""
        lineups = [
            {
                'Captain': 'QB Smith',
                'FLEX': ['WR Jones', 'RB Brown', 'TE Davis', 'WR Miller', 'K Taylor']
            }
        ]
        
        df = format_lineup_for_export(lineups, ExportFormat.DRAFTKINGS)
        
        assert 'CPT' in df.columns
        assert 'FLEX1' in df.columns
        assert 'FLEX5' in df.columns
        assert df.iloc[0]['CPT'] == 'QB Smith'
        assert df.iloc[0]['FLEX1'] == 'WR Jones'
    
    def test_standard_format(self):
        """Test standard format export"""
        lineups = [
            {
                'Captain': 'QB Smith',
                'FLEX': ['WR Jones', 'RB Brown', 'TE Davis', 'WR Miller', 'K Taylor'],
                'Total_Salary': 49500,
                'Projected': 125.5,
                'Total_Ownership': 85.0
            }
        ]
        
        df = format_lineup_for_export(lineups, ExportFormat.STANDARD)
        
        assert 'Lineup' in df.columns
        assert 'Captain' in df.columns
        assert 'FLEX' in df.columns
        assert df.iloc[0]['Lineup'] == 1
        assert df.iloc[0]['Captain'] == 'QB Smith'
    
    def test_export_validation_valid(self):
        """Test export validation passes for valid lineups"""
        lineups = [
            {
                'Captain': 'QB Smith',
                'FLEX': ['WR Jones', 'RB Brown', 'TE Davis', 'WR Miller', 'K Taylor']
            }
        ]
        
        is_valid, error_msg = validate_export_format(lineups, ExportFormat.DRAFTKINGS)
        
        assert is_valid
        assert error_msg == ""
    
    def test_export_validation_invalid_flex_count(self):
        """Test export validation fails for wrong FLEX count"""
        lineups = [
            {
                'Captain': 'QB Smith',
                'FLEX': ['WR Jones', 'RB Brown']  # Only 2 instead of 5
            }
        ]
        
        is_valid, error_msg = validate_export_format(lineups, ExportFormat.DRAFTKINGS)
        
        assert not is_valid
        assert "5 FLEX players" in error_msg
    
    def test_empty_lineups_export(self):
        """Test export handles empty lineup list"""
        lineups = []
        
        df = format_lineup_for_export(lineups, ExportFormat.STANDARD)
        
        assert df.empty


# ============================================================================
# DATA VALIDATION TESTS
# ============================================================================

class TestDataValidation:
    """Test data validation and normalization"""
    
    def test_valid_dataframe_passes(self):
        """Test that valid DataFrame passes validation"""
        df = pd.DataFrame({
            'Player': ['PlayerA', 'PlayerB', 'PlayerC', 'PlayerD', 'PlayerE', 'PlayerF'],
            'Position': ['QB', 'RB', 'WR', 'TE', 'K', 'DST'],
            'Team': ['TeamA', 'TeamA', 'TeamB', 'TeamB', 'TeamA', 'TeamB'],
            'Salary': [8000, 6500, 5000, 4000, 3500, 3000],
            'Projected_Points': [25.5, 18.0, 15.5, 12.0, 8.5, 7.0],
            'Ownership': [20.0, 15.0, 10.0, 8.0, 5.0, 3.0]
        })
        
        result_df, warnings = validate_and_normalize_dataframe(df, ValidationLevel.MODERATE)
        
        assert len(result_df) == 6
        assert 'Player' in result_df.columns
        assert 'Position' in result_df.columns
    
    def test_position_normalization_in_validation(self):
        """Test that positions get normalized during validation"""
        df = pd.DataFrame({
            'Player': ['PlayerA'],
            'Position': ['quarterback'],  # Lowercase full name
            'Team': ['TeamA'],
            'Salary': [8000],
            'Projected_Points': [25.5],
            'Ownership': [20.0]
        })
        
        result_df, warnings = validate_and_normalize_dataframe(df, ValidationLevel.MODERATE)
        
        assert result_df.iloc[0]['Position'] == 'QB'


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestDraftKingsRules:
    """Test DraftKings rules constants"""
    
    def test_salary_cap(self):
        """Test salary cap is correct"""
        assert DraftKingsRules.SALARY_CAP == 50000
    
    def test_roster_size(self):
        """Test roster size is correct"""
        assert DraftKingsRules.ROSTER_SIZE == 6
    
    def test_captain_multiplier(self):
        """Test captain multiplier is correct"""
        assert DraftKingsRules.CAPTAIN_MULTIPLIER == 1.5
    
    def test_team_requirements(self):
        """Test team diversity requirements"""
        assert DraftKingsRules.MIN_TEAMS_REQUIRED == 2
        assert DraftKingsRules.MAX_PLAYERS_PER_TEAM == 5


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
