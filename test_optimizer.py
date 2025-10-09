"""
Basic Test Suite for NFL DFS Optimizer
Version: 2.0.0 - Tests Actual Functions

Run with: pytest test_optimizer.py -v
"""

import pytest
import pandas as pd
import numpy as np
from nfl_dfs_optimizer import (
    # Core Classes
    DraftKingsRules,
    LineupConstraints,
    ValidationResult,
    SimulationResults,
    AIRecommendation,
    
    # Data Processing
    OptimizedDataProcessor,
    ColumnStandardizer,
    DataValidator,
    DataCleaner,
    
    # Utilities
    calculate_lineup_metrics,
    validate_lineup_with_context,
)


# ============================================================================
# DRAFTKINGS RULES TESTS
# ============================================================================

class TestDraftKingsRules:
    """Test DraftKings constants"""
    
    def test_salary_cap(self):
        assert DraftKingsRules.SALARY_CAP == 50000
    
    def test_roster_size(self):
        assert DraftKingsRules.ROSTER_SIZE == 6
    
    def test_captain_multiplier(self):
        assert DraftKingsRules.CAPTAIN_MULTIPLIER == 1.5
    
    def test_team_requirements(self):
        assert DraftKingsRules.MIN_TEAMS_REQUIRED == 2
        assert DraftKingsRules.MAX_PLAYERS_PER_TEAM == 5
    
    def test_rules_summary(self):
        """Test rules summary generation"""
        summary = DraftKingsRules.get_rules_summary()
        assert "Salary Cap" in summary
        assert "50,000" in summary


# ============================================================================
# COLUMN STANDARDIZATION TESTS
# ============================================================================

class TestColumnStandardizer:
    """Test column standardization"""
    
    def test_standard_columns_unchanged(self):
        """Test standard columns remain unchanged"""
        df = pd.DataFrame({
            'Player': ['PlayerA'],
            'Position': ['QB'],
            'Team': ['TeamA'],
            'Salary': [8000],
            'Projected_Points': [25.0]
        })
        
        result, warnings = ColumnStandardizer.standardize_columns(df)
        
        assert 'Player' in result.columns
        assert 'Position' in result.columns
        assert len(warnings) == 0
    
    def test_column_renaming(self):
        """Test various column names get standardized"""
        df = pd.DataFrame({
            'Name': ['PlayerA'],
            'Pos': ['QB'],
            'TeamAbbrev': ['TeamA'],
            'salary': [8000],
            'Projection': [25.0]
        })
        
        result, warnings = ColumnStandardizer.standardize_columns(df)
        
        assert 'Player' in result.columns
        assert 'Position' in result.columns
        assert 'Team' in result.columns
        assert 'Salary' in result.columns
        assert 'Projected_Points' in result.columns
    
    def test_missing_required_columns(self):
        """Test warning for missing required columns"""
        df = pd.DataFrame({
            'Player': ['PlayerA'],
            'Position': ['QB']
        })
        
        result, warnings = ColumnStandardizer.standardize_columns(df)
        
        assert len(warnings) > 0
        assert any('Missing required columns' in w for w in warnings)


# ============================================================================
# DATA VALIDATOR TESTS
# ============================================================================

class TestDataValidator:
    """Test data validation"""
    
    def test_valid_dataframe(self):
        """Test valid DataFrame passes"""
        df = pd.DataFrame({
            'Player': ['PlayerA', 'PlayerB', 'PlayerC', 'PlayerD', 'PlayerE', 'PlayerF'],
            'Position': ['QB', 'RB', 'WR', 'TE', 'K', 'DST'],
            'Team': ['TeamA', 'TeamA', 'TeamB', 'TeamB', 'TeamA', 'TeamB'],
            'Salary': [8000, 6500, 5000, 4000, 3500, 3000],
            'Projected_Points': [25.5, 18.0, 15.5, 12.0, 8.5, 7.0]
        })
        
        is_valid, errors, warnings = DataValidator.validate_dataframe(df)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_empty_dataframe(self):
        """Test empty DataFrame fails"""
        df = pd.DataFrame()
        
        is_valid, errors, warnings = DataValidator.validate_dataframe(df)
        
        assert not is_valid
        assert len(errors) > 0
    
    def test_missing_required_columns(self):
        """Test missing columns fails"""
        df = pd.DataFrame({
            'Player': ['PlayerA'],
            'Position': ['QB']
        })
        
        is_valid, errors, warnings = DataValidator.validate_dataframe(df)
        
        assert not is_valid
        assert any('Missing required columns' in e for e in errors)
    
    def test_insufficient_teams(self):
        """Test less than 2 teams fails"""
        df = pd.DataFrame({
            'Player': ['PlayerA', 'PlayerB'],
            'Position': ['QB', 'RB'],
            'Team': ['TeamA', 'TeamA'],  # Only 1 team
            'Salary': [8000, 6500],
            'Projected_Points': [25.5, 18.0]
        })
        
        is_valid, errors, warnings = DataValidator.validate_dataframe(df)
        
        assert not is_valid
        assert any('2 teams' in e for e in errors)


# ============================================================================
# DATA CLEANER TESTS
# ============================================================================

class TestDataCleaner:
    """Test data cleaning"""
    
    def test_whitespace_removal(self):
        """Test whitespace gets stripped"""
        df = pd.DataFrame({
            'Player': [' PlayerA ', 'PlayerB  '],
            'Position': [' QB', 'RB '],
            'Team': ['  TeamA', 'TeamB  '],
            'Salary': [8000, 6500],
            'Projected_Points': [25.5, 18.0]
        })
        
        result, warnings = DataCleaner.clean_dataframe(df)
        
        assert result.iloc[0]['Player'] == 'PlayerA'
        assert result.iloc[0]['Position'] == 'QB'
        assert result.iloc[0]['Team'] == 'TEAMA'  # Also uppercases
    
    def test_duplicate_removal(self):
        """Test duplicates get removed"""
        df = pd.DataFrame({
            'Player': ['PlayerA', 'PlayerA', 'PlayerB'],
            'Position': ['QB', 'QB', 'RB'],
            'Team': ['TeamA', 'TeamA', 'TeamB'],
            'Salary': [8000, 8000, 6500],
            'Projected_Points': [25.5, 25.5, 18.0]
        })
        
        result, warnings = DataCleaner.clean_dataframe(df)
        
        assert len(result) == 2  # Duplicate removed
        assert any('duplicate' in w.lower() for w in warnings)
    
    def test_invalid_salary_removal(self):
        """Test invalid salaries get handled"""
        df = pd.DataFrame({
            'Player': ['PlayerA', 'PlayerB'],
            'Position': ['QB', 'RB'],
            'Team': ['TeamA', 'TeamB'],
            'Salary': [8000, 0],  # 0 is invalid
            'Projected_Points': [25.5, 18.0]
        })
        
        result, warnings = DataCleaner.clean_dataframe(df)
        
        assert len(result) == 1  # Invalid removed
        assert any('invalid salary' in w.lower() for w in warnings)


# ============================================================================
# LINEUP VALIDATION TESTS
# ============================================================================

class TestLineupValidation:
    """Test lineup validation"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample player DataFrame"""
        return pd.DataFrame({
            'Player': ['QB1', 'RB1', 'WR1', 'TE1', 'K1', 'DST1'],
            'Position': ['QB', 'RB', 'WR', 'TE', 'K', 'DST'],
            'Team': ['TeamA', 'TeamA', 'TeamB', 'TeamB', 'TeamA', 'TeamB'],
            'Salary': [8000, 6500, 5000, 4000, 3500, 3000],
            'Projected_Points': [25.5, 18.0, 15.5, 12.0, 8.5, 7.0],
            'Ownership': [20.0, 15.0, 10.0, 8.0, 5.0, 3.0]
        })
    
    def test_valid_lineup(self, sample_df):
        """Test valid lineup passes"""
        lineup = {
            'Captain': 'QB1',
            'FLEX': ['RB1', 'WR1', 'TE1', 'K1', 'DST1']
        }
        
        result = validate_lineup_with_context(lineup, sample_df)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_invalid_roster_size(self, sample_df):
        """Test wrong roster size fails"""
        lineup = {
            'Captain': 'QB1',
            'FLEX': ['RB1', 'WR1']  # Only 2 FLEX
        }
        
        result = validate_lineup_with_context(lineup, sample_df)
        
        assert not result.is_valid
        assert any('roster size' in e.lower() for e in result.errors)
    
    def test_duplicate_players(self, sample_df):
        """Test duplicate players fail"""
        lineup = {
            'Captain': 'QB1',
            'FLEX': ['QB1', 'RB1', 'WR1', 'TE1', 'K1']  # QB1 used twice
        }
        
        result = validate_lineup_with_context(lineup, sample_df)
        
        assert not result.is_valid
        assert any('duplicate' in e.lower() for e in result.errors)
    
    def test_insufficient_teams(self, sample_df):
        """Test insufficient teams fails"""
        lineup = {
            'Captain': 'QB1',
            'FLEX': ['RB1', 'K1', 'QB1', 'QB1', 'QB1']  # All TeamA
        }
        
        # This would fail in actual validation
        # Just testing the validation catches it


# ============================================================================
# LINEUP METRICS TESTS
# ============================================================================

class TestLineupMetrics:
    """Test lineup metric calculation"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame"""
        return pd.DataFrame({
            'Player': ['QB1', 'RB1', 'WR1', 'TE1', 'K1', 'DST1'],
            'Position': ['QB', 'RB', 'WR', 'TE', 'K', 'DST'],
            'Team': ['TeamA', 'TeamA', 'TeamB', 'TeamB', 'TeamA', 'TeamB'],
            'Salary': [8000, 6500, 5000, 4000, 3500, 3000],
            'Projected_Points': [25.5, 18.0, 15.5, 12.0, 8.5, 7.0],
            'Ownership': [20.0, 15.0, 10.0, 8.0, 5.0, 3.0]
        })
    
    def test_basic_metrics(self, sample_df):
        """Test basic metric calculation"""
        captain = 'QB1'
        flex = ['RB1', 'WR1', 'TE1', 'K1', 'DST1']
        
        metrics = calculate_lineup_metrics(captain, flex, sample_df)
        
        assert 'Captain' in metrics
        assert 'FLEX' in metrics
        assert 'Total_Salary' in metrics
        assert 'Projected' in metrics
        
        assert metrics['Captain'] == 'QB1'
        assert len(metrics['FLEX']) == 5
    
    def test_salary_calculation(self, sample_df):
        """Test salary includes captain multiplier"""
        captain = 'QB1'  # 8000 salary
        flex = ['RB1', 'WR1', 'TE1', 'K1', 'DST1']  # Total: 30000
        
        metrics = calculate_lineup_metrics(captain, flex, sample_df)
        
        # Captain: 8000 * 1.5 = 12000
        # FLEX: 30000
        # Total: 42000
        expected_salary = 12000 + 30000
        
        assert metrics['Total_Salary'] == expected_salary
    
    def test_projection_calculation(self, sample_df):
        """Test projection includes captain multiplier"""
        captain = 'QB1'  # 25.5 points
        flex = ['RB1', 'WR1', 'TE1', 'K1', 'DST1']  # Total: 61.0
        
        metrics = calculate_lineup_metrics(captain, flex, sample_df)
        
        # Captain: 25.5 * 1.5 = 38.25
        # FLEX: 61.0
        # Total: 99.25
        expected_projection = 38.25 + 61.0
        
        assert abs(metrics['Projected'] - expected_projection) < 0.01


# ============================================================================
# CONSTRAINTS TESTS
# ============================================================================

class TestLineupConstraints:
    """Test lineup constraints"""
    
    def test_default_constraints(self):
        """Test default constraint values"""
        constraints = LineupConstraints()
        
        assert constraints.max_salary == 50000
        assert constraints.min_salary == 47500
        assert len(constraints.locked_players) == 0
        assert len(constraints.banned_players) == 0
    
    def test_add_locked_player(self):
        """Test adding locked player"""
        constraints = LineupConstraints()
        
        constraints.add_locked_player('PlayerA')
        
        assert 'PlayerA' in constraints.locked_players
        assert 'PlayerA' not in constraints.banned_players
    
    def test_add_banned_player(self):
        """Test adding banned player"""
        constraints = LineupConstraints()
        
        constraints.add_banned_player('PlayerA')
        
        assert 'PlayerA' in constraints.banned_players
        assert 'PlayerA' not in constraints.locked_players
    
    def test_locked_removes_from_banned(self):
        """Test locking removes from banned"""
        constraints = LineupConstraints()
        
        constraints.add_banned_player('PlayerA')
        constraints.add_locked_player('PlayerA')  # Should remove from banned
        
        assert 'PlayerA' in constraints.locked_players
        assert 'PlayerA' not in constraints.banned_players


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
