# -*- coding: utf-8 -*-

"""
DFS Showdown Optimizer - Configuration, Data Models, and Core Infrastructure
Version: 2.0 - Triple AI Enhanced
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard Library
import os
import sys
import json
import hashlib
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

# Data Science & Optimization
import pandas as pd
import numpy as np
import pulp
from scipy import stats
from scipy.optimize import minimize

# Streamlit (UI)
import streamlit as st

# API (conditional import handled in api_manager)
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None

# ============================================================================
# VERSION AND METADATA
# ============================================================================

VERSION = "2.0.0"
BUILD_DATE = "2025-01-15"
AUTHOR = "DFS Optimization Team"

APP_METADATA = {
    'version': VERSION,
    'build_date': BUILD_DATE,
    'author': AUTHOR,
    'features': [
        'Triple AI Strategist System',
        'AI-as-Chef Optimization',
        'Dynamic Ownership Bucketing',
        'Multi-Dimensional Enforcement',
        'Advanced Correlation Analysis',
        'Contrarian Narrative Detection',
        'Parallel Lineup Generation',
        'Comprehensive Logging & Monitoring'
    ]
}

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class AIStrategistType(Enum):
    """AI Strategist Types"""
    GAME_THEORY = "Game Theory"
    CORRELATION = "Correlation"
    CONTRARIAN_NARRATIVE = "Contrarian Narrative"

    @classmethod
    def all_types(cls) -> List['AIStrategistType']:
        return [cls.GAME_THEORY, cls.CORRELATION, cls.CONTRARIAN_NARRATIVE]

    def get_description(self) -> str:
        descriptions = {
            self.GAME_THEORY: "Analyzes ownership leverage and field tendencies to exploit market inefficiencies",
            self.CORRELATION: "Identifies positive and negative player correlations based on game script",
            self.CONTRARIAN_NARRATIVE: "Finds low-owned narratives and pivot opportunities the field misses"
        }
        return descriptions.get(self, "Unknown strategist type")

    def get_icon(self) -> str:
        icons = {
            self.GAME_THEORY: "🎯",
            self.CORRELATION: "🔗",
            self.CONTRARIAN_NARRATIVE: "💡"
        }
        return icons.get(self, "🤖")


class AIEnforcementLevel(Enum):
    """AI Enforcement Levels - How strictly to follow AI recommendations"""
    STRICT = "Strict"  # All AI rules enforced as hard constraints
    STRONG = "Strong"  # Most AI rules enforced, some flexibility
    BALANCED = "Balanced"  # Mix of hard and soft constraints
    GUIDANCE = "Guidance"  # AI recommendations as soft preferences
    MINIMAL = "Minimal"  # AI used for analysis only, minimal enforcement

    def get_hard_constraint_ratio(self) -> float:
        """Get ratio of rules that should be hard constraints"""
        ratios = {
            self.STRICT: 0.9,
            self.STRONG: 0.7,
            self.BALANCED: 0.5,
            self.GUIDANCE: 0.3,
            self.MINIMAL: 0.1
        }
        return ratios.get(self, 0.5)

    def get_soft_constraint_weight(self) -> float:
        """Get weight for soft constraints"""
        weights = {
            self.STRICT: 0.95,
            self.STRONG: 0.85,
            self.BALANCED: 0.7,
            self.GUIDANCE: 0.5,
            self.MINIMAL: 0.3
        }
        return weights.get(self, 0.7)


class ConstraintType(Enum):
    """Optimization Constraint Types"""
    HARD = "hard"
    SOFT = "soft"
    GUIDANCE = "guidance"

    def get_violation_penalty(self) -> float:
        """Get penalty for violating this constraint type"""
        penalties = {
            self.HARD: float('inf'),  # Cannot violate
            self.SOFT: 100.0,  # High penalty
            self.GUIDANCE: 10.0  # Low penalty
        }
        return penalties.get(self, 50.0)


class OptimizationStatus(Enum):
    """Optimization Status Codes"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    CANCELLED = "cancelled"


class LogLevel(Enum):
    """Logging Levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def get_priority(self) -> int:
        priorities = {
            self.DEBUG: 1,
            self.INFO: 2,
            self.WARNING: 3,
            self.ERROR: 4,
            self.CRITICAL: 5
        }
        return priorities.get(self, 2)


# ============================================================================
# OPTIMIZER CONFIGURATION
# ============================================================================

class OptimizerConfig:
    """Comprehensive Optimizer Configuration"""

    # ========== DraftKings Showdown Rules ==========
    LINEUP_SIZE = 6
    CAPTAIN_MULTIPLIER = 1.5
    SALARY_CAP = 50000
    MIN_SALARY = 2500
    MAX_SALARY = 15000

    # Team diversity
    MIN_TEAMS_REPRESENTED = 2  # Must have players from both teams
    MAX_PLAYERS_PER_TEAM = 5
    MIN_PLAYERS_PER_TEAM = 1

    # ========== Optimization Settings ==========
    MAX_PARALLEL_THREADS = 4
    OPTIMIZATION_TIMEOUT = 30  # seconds per lineup
    MAX_RETRIES_PER_LINEUP = 3
    SOLVER_TIME_LIMIT = 15  # seconds for individual solve

    # Constraint relaxation factors for retry attempts
    RELAXATION_FACTORS = [1.0, 0.85, 0.7, 0.5]  # Progressive relaxation

    # ========== AI Configuration ==========

    # AI weights for synthesis (must sum to 1.0)
    AI_WEIGHTS = {
        'game_theory': 0.35,
        'correlation': 0.35,
        'contrarian': 0.30
    }

    # Confidence thresholds
    MIN_AI_CONFIDENCE = 0.4  # Below this, use fallback
    HIGH_CONFIDENCE_THRESHOLD = 0.75

    # API settings
    API_RATE_LIMIT_PER_MINUTE = 50
    API_MAX_TOKENS = 2000
    API_TEMPERATURE = 0.7
    API_RETRY_ATTEMPTS = 2
    API_RETRY_DELAY = 1.0  # seconds

    # Cache settings
    MAX_CACHE_SIZE = 100
    CACHE_EXPIRY_HOURS = 24

    # ========== Field Size Configurations ==========
    FIELD_SIZE_CONFIGS = {
        'small_field': {
            'name': 'Small Field (<500)',
            'min_ownership': 80,
            'max_ownership': 120,
            'max_chalk_players': 3,
            'min_leverage_players': 0,
            'ai_enforcement': 'Strong',
            'diversity_weight': 0.3,
            'description': 'Smaller fields allow for chalkier plays with less contrarian need'
        },
        'medium_field': {
            'name': 'Medium Field (500-2000)',
            'min_ownership': 70,
            'max_ownership': 100,
            'max_chalk_players': 2,
            'min_leverage_players': 1,
            'ai_enforcement': 'Strong',
            'diversity_weight': 0.5,
            'description': 'Balance between chalk and leverage plays'
        },
        'large_field': {
            'name': 'Large Field (2K-10K)',
            'min_ownership': 60,
            'max_ownership': 90,
            'max_chalk_players': 2,
            'min_leverage_players': 2,
            'ai_enforcement': 'Strict',
            'diversity_weight': 0.7,
            'description': 'Larger fields require more contrarian plays and leverage'
        },
        'large_multi': {
            'name': 'Large Multi-Entry (10K+)',
            'min_ownership': 50,
            'max_ownership': 85,
            'max_chalk_players': 1,
            'min_leverage_players': 2,
            'ai_enforcement': 'Strict',
            'diversity_weight': 0.8,
            'description': 'Very large fields demand maximum differentiation'
        },
        'milly_maker': {
            'name': 'Milly Maker (100K+)',
            'min_ownership': 40,
            'max_ownership': 75,
            'max_chalk_players': 1,
            'min_leverage_players': 3,
            'ai_enforcement': 'Strict',
            'diversity_weight': 0.9,
            'description': 'Massive fields require extreme leverage and uniqueness'
        }
    }

    # ========== Ownership Configuration ==========

    # GPP ownership targets by field size
    GPP_OWNERSHIP_TARGETS = {
        'small_field': (80, 120),
        'medium_field': (70, 100),
        'large_field': (60, 90),
        'large_multi': (50, 85),
        'milly_maker': (40, 75)
    }

    # Ownership bucketing thresholds
    OWNERSHIP_BUCKETS = {
        'mega_chalk': 35,  # >35% owned
        'chalk': 20,  # 20-35%
        'moderate': 15,  # 15-20%
        'pivot': 10,  # 10-15%
        'leverage': 5,  # 5-10%
        'super_leverage': 2  # <5%
    }

    # Default ownership by position (used when ownership data missing)
    DEFAULT_OWNERSHIP = 10.0

    @staticmethod
    def get_default_ownership(position: str, salary: float) -> float:
        """Calculate default ownership based on position and salary"""

        # Base ownership by position
        position_base = {
            'QB': 12.0,
            'RB': 10.0,
            'WR': 8.0,
            'TE': 7.0,
            'DST': 6.0,
            'FLEX': 8.0
        }

        base = position_base.get(position, 8.0)

        # Salary adjustment (higher salary = higher ownership usually)
        if salary >= 10000:
            salary_factor = 1.5
        elif salary >= 8000:
            salary_factor = 1.2
        elif salary >= 6000:
            salary_factor = 1.0
        elif salary >= 4000:
            salary_factor = 0.8
        else:
            salary_factor = 0.6

        return min(base * salary_factor, 40.0)

    # ========== Lineup Diversity Settings ==========

    DIVERSITY_SETTINGS = {
        'min_unique_captains': 0.8,  # 80% of lineups should have unique captains
        'max_player_exposure': 0.4,  # No player in more than 40% of lineups
        'min_jaccard_distance': 0.3,  # Lineups should be at least 30% different
        'position_balance_tolerance': 0.2  # Allow 20% variance in position distribution
    }

    # ========== Stack Configuration ==========

    STACK_SETTINGS = {
        'qb_primary_correlation': 0.7,
        'qb_te_correlation': 0.65,
        'qb_rb_correlation': 0.2,
        'same_team_wr_correlation': -0.3,
        'rb_rb_correlation': -0.5,
        'opposing_dst_correlation': -0.4,
        'bring_back_min_total': 48,  # Minimum game total for bring-back
        'onslaught_min_spread': 7,  # Minimum spread for onslaught stacks
        'max_correlation_stack_size': 4
    }

    # ========== Performance Settings ==========

    PERFORMANCE_SETTINGS = {
        'enable_monitoring': True,
        'enable_detailed_logging': True,
        'log_retention_days': 7,
        'max_log_file_size_mb': 50,
        'enable_metrics_export': False,
        'metrics_export_interval': 300  # seconds
    }

    # ========== Validation Settings ==========

    VALIDATION_SETTINGS = {
        'strict_mode': True,
        'allow_auto_correction': True,
        'max_correction_attempts': 3,
        'validation_timeout': 5  # seconds
    }

    # ========== Export Settings ==========

    EXPORT_SETTINGS = {
        'default_format': 'draftkings',
        'include_metadata': True,
        'timestamp_files': True,
        'compression': None  # or 'gzip', 'zip'
    }

    @classmethod
    def get_field_config(cls, field_size: str) -> Dict:
        """Get configuration for specific field size"""
        return cls.FIELD_SIZE_CONFIGS.get(
            field_size,
            cls.FIELD_SIZE_CONFIGS['large_field']
        )

    @classmethod
    def validate_config(cls) -> Tuple[bool, List[str]]:
        """Validate configuration consistency"""
        errors = []

        # Check AI weights sum to 1.0
        weight_sum = sum(cls.AI_WEIGHTS.values())
        if not (0.99 <= weight_sum <= 1.01):
            errors.append(f"AI weights sum to {weight_sum}, should be 1.0")

        # Check salary constraints
        if cls.MIN_SALARY * cls.LINEUP_SIZE > cls.SALARY_CAP:
            errors.append("Minimum salary configuration exceeds salary cap")

        # Check ownership targets
        for field_size, (min_own, max_own) in cls.GPP_OWNERSHIP_TARGETS.items():
            if min_own >= max_own:
                errors.append(f"Invalid ownership range for {field_size}")

        # Check threads
        if cls.MAX_PARALLEL_THREADS < 1:
            errors.append("MAX_PARALLEL_THREADS must be at least 1")

        return len(errors) == 0, errors


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class AIRecommendation:
    """AI Strategist Recommendation with comprehensive validation"""

    captain_targets: List[str] = field(default_factory=list)
    must_play: List[str] = field(default_factory=list)
    never_play: List[str] = field(default_factory=list)
    stacks: List[Dict] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)
    confidence: float = 0.7
    enforcement_rules: List[Dict] = field(default_factory=list)
    narrative: str = ""
    source_ai: Optional[AIStrategistType] = None

    # Optional attributes for specific AI types
    ownership_leverage: Optional[Dict] = None  # Game theory
    correlation_matrix: Optional[Dict] = None  # Correlation
    contrarian_angles: Optional[List[str]] = None  # Contrarian
    ceiling_plays: Optional[List[str]] = None  # Contrarian

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate and normalize after initialization"""

        # Ensure confidence is in valid range
        self.confidence = max(0.0, min(1.0, self.confidence))

        # Remove duplicates from lists
        self.captain_targets = list(dict.fromkeys(self.captain_targets))
        self.must_play = list(dict.fromkeys(self.must_play))
        self.never_play = list(dict.fromkeys(self.never_play))

        # Limit sizes
        self.captain_targets = self.captain_targets[:10]
        self.must_play = self.must_play[:8]
        self.never_play = self.never_play[:5]
        self.stacks = self.stacks[:15]
        self.key_insights = self.key_insights[:5]

    def validate(self, auto_correct: bool = True) -> Tuple[bool, List[str], Dict]:
        """
        Comprehensive validation with optional auto-correction

        Returns:
            (is_valid, errors, corrections_applied)
        """
        errors = []
        corrections = {}

        # Check for empty recommendations
        if not self.captain_targets and not self.must_play:
            errors.append("No recommendations provided")

            if auto_correct:
                # This would need actual player data
                corrections['added_default_captains'] = True

        # Check confidence range
        if not 0.0 <= self.confidence <= 1.0:
            errors.append(f"Confidence {self.confidence} out of range")

            if auto_correct:
                self.confidence = max(0.0, min(1.0, self.confidence))
                corrections['normalized_confidence'] = self.confidence

        # Check for conflicts (player in both must_play and never_play)
        conflicts = set(self.must_play) & set(self.never_play)
        if conflicts:
            errors.append(f"Conflicting recommendations: {conflicts}")

            if auto_correct:
                # Remove from never_play (must_play takes precedence)
                self.never_play = [p for p in self.never_play if p not in conflicts]
                corrections['resolved_conflicts'] = list(conflicts)

        # Validate stack structures
        for i, stack in enumerate(self.stacks):
            if not isinstance(stack, dict):
                errors.append(f"Stack {i} is not a dictionary")
                continue

            if 'type' not in stack:
                if auto_correct:
                    stack['type'] = 'standard'
                    corrections[f'stack_{i}_added_type'] = True

        # Update validation status
        self.validation_passed = len(errors) == 0
        self.validation_errors = errors

        return self.validation_passed, errors, corrections

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'captain_targets': self.captain_targets,
            'must_play': self.must_play,
            'never_play': self.never_play,
            'stacks': self.stacks,
            'key_insights': self.key_insights,
            'confidence': self.confidence,
            'enforcement_rules': self.enforcement_rules,
            'narrative': self.narrative,
            'source_ai': self.source_ai.value if self.source_ai else None,
            'ownership_leverage': self.ownership_leverage,
            'correlation_matrix': self.correlation_matrix,
            'contrarian_angles': self.contrarian_angles,
            'ceiling_plays': self.ceiling_plays,
            'timestamp': self.timestamp.isoformat(),
            'validation_passed': self.validation_passed,
            'validation_errors': self.validation_errors
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AIRecommendation':
        """Create from dictionary"""

        # Handle source_ai
        source_ai = None
        if data.get('source_ai'):
            for ai_type in AIStrategistType:
                if ai_type.value == data['source_ai']:
                    source_ai = ai_type
                    break

        # Handle timestamp
        timestamp = datetime.now()
        if data.get('timestamp'):
            try:
                timestamp = datetime.fromisoformat(data['timestamp'])
            except:
                pass

        return cls(
            captain_targets=data.get('captain_targets', []),
            must_play=data.get('must_play', []),
            never_play=data.get('never_play', []),
            stacks=data.get('stacks', []),
            key_insights=data.get('key_insights', []),
            confidence=data.get('confidence', 0.7),
            enforcement_rules=data.get('enforcement_rules', []),
            narrative=data.get('narrative', ''),
            source_ai=source_ai,
            ownership_leverage=data.get('ownership_leverage'),
            correlation_matrix=data.get('correlation_matrix'),
            contrarian_angles=data.get('contrarian_angles'),
            ceiling_plays=data.get('ceiling_plays'),
            timestamp=timestamp,
            validation_passed=data.get('validation_passed', True),
            validation_errors=data.get('validation_errors', [])
        )

    def get_summary(self) -> str:
        """Get human-readable summary"""
        summary_parts = [
            f"{self.source_ai.value if self.source_ai else 'Unknown'} Strategy",
            f"Confidence: {self.confidence:.0%}",
            f"Captains: {len(self.captain_targets)}",
            f"Must-Play: {len(self.must_play)}",
            f"Stacks: {len(self.stacks)}"
        ]

        return " | ".join(summary_parts)


@dataclass
class OptimizationResult:
    """Results from optimization run"""

    lineups: pd.DataFrame
    status: OptimizationStatus
    num_requested: int
    num_generated: int
    success_rate: float
    total_time: float
    avg_time_per_lineup: float

    # AI-specific metrics
    ai_synthesis: Optional[Dict] = None
    ai_enforcement_level: Optional[AIEnforcementLevel] = None
    enforcement_success_rate: float = 0.0

    # Diversity metrics
    unique_captains: int = 0
    avg_ownership: float = 0.0
    ownership_std: float = 0.0
    avg_leverage_score: float = 0.0

    # Performance metrics
    attempts_per_lineup: float = 0.0
    constraint_violations: List[Dict] = field(default_factory=list)

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.now)

    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        return {
            'status': self.status.value,
            'success_rate': f"{self.success_rate:.1%}",
            'total_time': f"{self.total_time:.2f}s",
            'avg_time': f"{self.avg_time_per_lineup:.2f}s",
            'lineups_generated': f"{self.num_generated}/{self.num_requested}",
            'unique_captains': self.unique_captains,
            'avg_ownership': f"{self.avg_ownership:.1f}%",
            'avg_leverage': f"{self.avg_leverage_score:.2f}"
        }

    def is_successful(self) -> bool:
        """Check if optimization was successful"""
        return self.status in [OptimizationStatus.SUCCESS, OptimizationStatus.PARTIAL_SUCCESS]


@dataclass
class PlayerData:
    """Enhanced player data model"""

    name: str
    position: str
    team: str
    salary: int
    projected_points: float
    ownership: float = 10.0

    # Advanced metrics
    ceiling: Optional[float] = None
    floor: Optional[float] = None
    value: Optional[float] = None
    gpp_score: Optional[float] = None

    # Optional attributes
    opponent: Optional[str] = None
    game_total: Optional[float] = None
    team_implied_total: Optional[float] = None
    weather: Optional[str] = None

    def __post_init__(self):
        """Calculate derived metrics"""

        # Calculate value
        if self.value is None:
            self.value = self.projected_points / (self.salary / 1000)

        # Estimate ceiling/floor if not provided
        if self.ceiling is None:
            self.ceiling = self.projected_points * 1.4

        if self.floor is None:
            self.floor = self.projected_points * 0.7

        # Calculate GPP score
        if self.gpp_score is None:
            self.gpp_score = self.value * (30 / (self.ownership + 10))

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_series(cls, series: pd.Series) -> 'PlayerData':
        """Create from pandas Series"""
        return cls(
            name=series.get('Player', ''),
            position=series.get('Position', 'FLEX'),
            team=series.get('Team', ''),
            salary=int(series.get('Salary', 0)),
            projected_points=float(series.get('Projected_Points', 0)),
            ownership=float(series.get('Ownership', 10)),
            ceiling=series.get('Ceiling'),
            floor=series.get('Floor'),
            opponent=series.get('Opponent'),
            game_total=series.get('Game_Total'),
            team_implied_total=series.get('Team_Implied'),
            weather=series.get('Weather')
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_timestamp() -> str:
    """Get formatted timestamp"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_file_timestamp() -> str:
    """Get timestamp suitable for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_currency(amount: float) -> str:
    """Format currency with proper formatting"""
    return f"${amount:,.0f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage"""
    return f"{value:.{decimals}f}%"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))


def normalize_list(values: List[float]) -> List[float]:
    """Normalize list of values to 0-1 range"""
    if not values:
        return []

    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        return [0.5] * len(values)

    return [(v - min_val) / (max_val - min_val) for v in values]


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """
    Comprehensive DataFrame validation

    Returns:
        (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    if df is None or df.empty:
        errors.append("DataFrame is empty")
        return False, errors, warnings

    # Check required columns
    required_columns = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    # Check for duplicates
    if 'Player' in df.columns and df['Player'].duplicated().any():
        duplicates = df[df['Player'].duplicated()]['Player'].tolist()
        errors.append(f"Duplicate players found: {duplicates}")

    # Check data types and ranges
    if 'Salary' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Salary']):
            errors.append("Salary column must be numeric")
        else:
            if df['Salary'].min() < OptimizerConfig.MIN_SALARY:
                warnings.append(f"Some salaries below minimum: {df['Salary'].min()}")
            if df['Salary'].max() > OptimizerConfig.MAX_SALARY:
                warnings.append(f"Some salaries above maximum: {df['Salary'].max()}")

    if 'Projected_Points' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Projected_Points']):
            errors.append("Projected_Points column must be numeric")
        else:
            if df['Projected_Points'].min() < 0:
                errors.append("Negative projections found")

    # Check team count
    if 'Team' in df.columns:
        unique_teams = df['Team'].nunique()
        if unique_teams < 2:
            errors.append(f"Need at least 2 teams, found {unique_teams}")
        elif unique_teams > 2:
            warnings.append(f"More than 2 teams found ({unique_teams})")

    # Check minimum player count
    min_players = OptimizerConfig.LINEUP_SIZE * 2
    if len(df) < min_players:
        warnings.append(f"Only {len(df)} players (recommended minimum: {min_players})")

    is_valid = len(errors) == 0

    return is_valid, errors, warnings


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

def initialize_config() -> bool:
    """Initialize and validate configuration on module load"""
    is_valid, errors = OptimizerConfig.validate_config()

    if not is_valid:
        print(f"Configuration validation failed: {errors}")
        return False

    return True


# Validate config on import
_CONFIG_VALID = initialize_config()

if not _CONFIG_VALID:
    print("WARNING: Configuration validation failed. Some features may not work correctly.")


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Version
    'VERSION', 'BUILD_DATE', 'APP_METADATA',

    # Enums
    'AIStrategistType', 'AIEnforcementLevel', 'ConstraintType',
    'OptimizationStatus', 'LogLevel',

    # Configuration
    'OptimizerConfig',

    # Data Models
    'AIRecommendation', 'OptimizationResult', 'PlayerData',

    # Utilities
    'get_timestamp', 'get_file_timestamp', 'format_currency',
    'format_percentage', 'format_duration', 'safe_divide', 'clamp',
    'normalize_list', 'validate_dataframe'
]

"""
Advanced Logging and Performance Monitoring System
Features: Structured logging, performance tracking, AI decision logging, metrics aggregation
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
import traceback

from config_and_models import (
    LogLevel, AIStrategistType, OptimizationStatus,
    get_timestamp, format_duration, safe_divide
)

# ============================================================================
# LOG ENTRY DATA MODEL
# ============================================================================

@dataclass
class LogEntry:
    """Structured log entry"""

    timestamp: datetime
    level: LogLevel
    message: str
    module: str = "main"
    function: Optional[str] = None

    # Contextual data
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Performance data
    duration_ms: Optional[float] = None
    memory_mb: Optional[float] = None

    # Error tracking
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    stack_trace: Optional[str] = None

    # AI-specific
    ai_strategist: Optional[str] = None
    ai_confidence: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'message': self.message,
            'module': self.module,
            'function': self.function,
            'context': self.context,
            'tags': self.tags,
            'duration_ms': self.duration_ms,
            'memory_mb': self.memory_mb,
            'exception_type': self.exception_type,
            'exception_message': self.exception_message,
            'stack_trace': self.stack_trace,
            'ai_strategist': self.ai_strategist,
            'ai_confidence': self.ai_confidence
        }

    def to_string(self, include_context: bool = False) -> str:
        """Convert to human-readable string"""
        parts = [
            f"[{self.timestamp.strftime('%H:%M:%S')}]",
            f"[{self.level.value}]",
            f"[{self.module}]",
        ]

        if self.function:
            parts.append(f"[{self.function}]")

        parts.append(self.message)

        if self.duration_ms:
            parts.append(f"({self.duration_ms:.2f}ms)")

        if include_context and self.context:
            parts.append(f"Context: {self.context}")

        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")

        return " ".join(parts)


# ============================================================================
# LOGGER CLASS
# ============================================================================

class AdvancedLogger:
    """
    Advanced logging system with multiple output targets and filtering
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logger"""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True

        # Log storage
        self.logs = deque(maxlen=1000)  # Keep last 1000 logs in memory
        self._log_lock = threading.Lock()

        # Log file settings
        self.log_dir = "logs"
        self.log_file = None
        self.enable_file_logging = True
        self.enable_console_logging = True

        # Filtering
        self.min_level = LogLevel.DEBUG
        self.enabled_tags = set()  # Empty = all tags enabled
        self.disabled_tags = set()

        # Statistics
        self.stats = {
            'total_logs': 0,
            'by_level': defaultdict(int),
            'by_module': defaultdict(int),
            'by_tag': defaultdict(int),
            'errors': 0,
            'warnings': 0
        }

        # Create log directory
        self._setup_log_directory()

    def _setup_log_directory(self):
        """Setup log directory and file"""
        try:
            os.makedirs(self.log_dir, exist_ok=True)

            # Create new log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(self.log_dir, f"optimizer_{timestamp}.log")

            # Write header
            with open(self.log_file, 'w') as f:
                f.write(f"=== DFS Optimizer Log - Started {get_timestamp()} ===\n")
                f.write(f"Log Level: {self.min_level.value}\n")
                f.write("=" * 70 + "\n\n")

        except Exception as e:
            print(f"Failed to setup log directory: {e}")
            self.enable_file_logging = False

    def log(self, message: str, level: str = "INFO", module: str = "main",
            function: Optional[str] = None, context: Optional[Dict] = None,
            tags: Optional[List[str]] = None, duration_ms: Optional[float] = None):
        """
        Log a message

        Args:
            message: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            module: Module name
            function: Function name
            context: Additional context data
            tags: List of tags for filtering
            duration_ms: Duration in milliseconds (for performance logs)
        """

        # Parse level
        try:
            log_level = LogLevel[level.upper()]
        except KeyError:
            log_level = LogLevel.INFO

        # Check if should log
        if not self._should_log(log_level, tags or []):
            return

        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now(),
            level=log_level,
            message=message,
            module=module,
            function=function,
            context=context or {},
            tags=tags or [],
            duration_ms=duration_ms
        )

        # Store and output
        self._store_log(entry)
        self._output_log(entry)

        # Update statistics
        self._update_stats(entry)

    def log_exception(self, exception: Exception, context: str = "",
                     module: str = "main", critical: bool = False):
        """
        Log an exception with full traceback

        Args:
            exception: The exception object
            context: Context about where the exception occurred
            module: Module name
            critical: Whether this is a critical error
        """

        level = LogLevel.CRITICAL if critical else LogLevel.ERROR

        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=f"Exception in {context}: {str(exception)}",
            module=module,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stack_trace=traceback.format_exc(),
            tags=['exception', 'error']
        )

        self._store_log(entry)
        self._output_log(entry)
        self._update_stats(entry)

    def log_ai_decision(self, decision_type: str, strategist: str,
                       success: bool, details: Optional[Dict] = None,
                       confidence: Optional[float] = None):
        """
        Log an AI strategist decision

        Args:
            decision_type: Type of decision (recommendation, validation, etc.)
            strategist: AI strategist name
            success: Whether the decision was successful
            details: Additional details about the decision
            confidence: AI confidence score
        """

        message = f"AI Decision: {decision_type} by {strategist} - {'Success' if success else 'Failed'}"

        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message=message,
            module="ai",
            context=details or {},
            tags=['ai', 'decision', strategist.lower().replace(' ', '_')],
            ai_strategist=strategist,
            ai_confidence=confidence
        )

        self._store_log(entry)
        self._output_log(entry)
        self._update_stats(entry)

    def log_optimization_start(self, num_lineups: int, field_size: str,
                              config: Optional[Dict] = None):
        """Log optimization start"""

        message = f"Starting optimization: {num_lineups} lineups for {field_size}"

        context = config or {}
        context.update({
            'num_lineups': num_lineups,
            'field_size': field_size
        })

        self.log(
            message,
            level="INFO",
            module="optimizer",
            function="generate_lineups",
            context=context,
            tags=['optimization', 'start']
        )

    def log_optimization_end(self, num_generated: int, total_time: float,
                            success_rate: float, avg_ownership: float):
        """Log optimization end"""

        message = (f"Optimization complete: {num_generated} lineups generated "
                  f"in {format_duration(total_time)}")

        context = {
            'num_generated': num_generated,
            'total_time': total_time,
            'success_rate': success_rate,
            'avg_ownership': avg_ownership
        }

        self.log(
            message,
            level="INFO",
            module="optimizer",
            function="generate_lineups",
            context=context,
            tags=['optimization', 'end'],
            duration_ms=total_time * 1000
        )

    def log_lineup_generation(self, strategy: str, lineup_num: int,
                             status: str, num_constraints: int):
        """Log individual lineup generation"""

        message = f"Lineup {lineup_num} ({strategy}): {status}"

        context = {
            'strategy': strategy,
            'lineup_num': lineup_num,
            'status': status,
            'num_constraints': num_constraints
        }

        level = "INFO" if status == "SUCCESS" else "WARNING"

        self.log(
            message,
            level=level,
            module="optimizer",
            function="build_lineup",
            context=context,
            tags=['lineup', 'generation', strategy.lower()]
        )

    def _should_log(self, level: LogLevel, tags: List[str]) -> bool:
        """Check if message should be logged based on filters"""

        # Check level
        if level.get_priority() < self.min_level.get_priority():
            return False

        # Check tags
        if tags:
            # If specific tags are enabled, message must have at least one
            if self.enabled_tags and not any(tag in self.enabled_tags for tag in tags):
                return False

            # Check if any tags are disabled
            if any(tag in self.disabled_tags for tag in tags):
                return False

        return True

    def _store_log(self, entry: LogEntry):
        """Store log entry in memory"""
        with self._log_lock:
            self.logs.append(entry)

    def _output_log(self, entry: LogEntry):
        """Output log to console and file"""

        log_string = entry.to_string(include_context=False)

        # Console output
        if self.enable_console_logging:
            # Color coding for different levels (if terminal supports it)
            if entry.level == LogLevel.ERROR or entry.level == LogLevel.CRITICAL:
                print(f"\033[91m{log_string}\033[0m")  # Red
            elif entry.level == LogLevel.WARNING:
                print(f"\033[93m{log_string}\033[0m")  # Yellow
            else:
                print(log_string)

        # File output
        if self.enable_file_logging and self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(log_string + "\n")

                    # Add full context for errors
                    if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                        if entry.context:
                            f.write(f"  Context: {json.dumps(entry.context, indent=2)}\n")
                        if entry.stack_trace:
                            f.write(f"  Stack Trace:\n{entry.stack_trace}\n")

            except Exception as e:
                print(f"Failed to write to log file: {e}")

    def _update_stats(self, entry: LogEntry):
        """Update logging statistics"""
        self.stats['total_logs'] += 1
        self.stats['by_level'][entry.level.value] += 1
        self.stats['by_module'][entry.module] += 1

        for tag in entry.tags:
            self.stats['by_tag'][tag] += 1

        if entry.level == LogLevel.ERROR or entry.level == LogLevel.CRITICAL:
            self.stats['errors'] += 1
        elif entry.level == LogLevel.WARNING:
            self.stats['warnings'] += 1

    def get_logs(self, level: Optional[LogLevel] = None,
                module: Optional[str] = None,
                tags: Optional[List[str]] = None,
                limit: int = 100) -> List[LogEntry]:
        """
        Retrieve filtered logs

        Args:
            level: Filter by log level
            module: Filter by module
            tags: Filter by tags (any match)
            limit: Maximum number of logs to return

        Returns:
            List of matching log entries
        """

        with self._log_lock:
            filtered = list(self.logs)

        # Apply filters
        if level:
            filtered = [log for log in filtered if log.level == level]

        if module:
            filtered = [log for log in filtered if log.module == module]

        if tags:
            filtered = [log for log in filtered
                       if any(tag in log.tags for tag in tags)]

        # Return most recent first
        filtered.reverse()

        return filtered[:limit]

    def get_stats(self) -> Dict:
        """Get logging statistics"""
        return dict(self.stats)

    def export_logs(self, filepath: str, format: str = 'json'):
        """
        Export logs to file

        Args:
            filepath: Output file path
            format: Export format ('json' or 'csv')
        """

        with self._log_lock:
            logs_to_export = list(self.logs)

        try:
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump([log.to_dict() for log in logs_to_export], f, indent=2)

            elif format == 'csv':
                # Convert to DataFrame and export
                import pandas as pd

                log_dicts = [log.to_dict() for log in logs_to_export]
                df = pd.DataFrame(log_dicts)
                df.to_csv(filepath, index=False)

            self.log(f"Exported {len(logs_to_export)} logs to {filepath}", "INFO")

        except Exception as e:
            self.log_exception(e, f"export_logs to {filepath}")

    def clear_logs(self):
        """Clear in-memory logs"""
        with self._lock:
            self.logs.clear()

        self.log("Logs cleared", "INFO", tags=['maintenance'])

    def set_level(self, level: str):
        """Set minimum log level"""
        try:
            self.min_level = LogLevel[level.upper()]
            self.log(f"Log level set to {self.min_level.value}", "INFO")
        except KeyError:
            self.log(f"Invalid log level: {level}", "WARNING")


# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """
    Performance monitoring and timing system
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize performance monitor"""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True

        # Timer storage
        self.active_timers = {}
        self.timer_history = defaultdict(list)
        self._timer_lock = threading.Lock()

        # Metrics
        self.metrics = {
            'api_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'lineups_generated': 0,
            'lineups_failed': 0,
            'constraint_violations': 0,
            'total_optimization_time': 0.0
        }
        self._metrics_lock = threading.Lock()

        # Performance thresholds
        self.thresholds = {
            'api_call': 5.0,  # seconds
            'lineup_generation': 10.0,  # seconds
            'optimization_total': 300.0  # seconds
        }

        self.logger = get_logger()

    def start_timer(self, timer_name: str):
        """Start a named timer"""
        with self._timer_lock:
            self.active_timers[timer_name] = time.time()

    def stop_timer(self, timer_name: str) -> float:
        """
        Stop a named timer and return elapsed time

        Returns:
            Elapsed time in seconds
        """

        with self._timer_lock:
            if timer_name not in self.active_timers:
                self.logger.log(
                    f"Timer '{timer_name}' not found",
                    "WARNING",
                    tags=['performance', 'timer']
                )
                return 0.0

            start_time = self.active_timers.pop(timer_name)
            elapsed = time.time() - start_time

            # Store in history
            self.timer_history[timer_name].append(elapsed)

            # Check threshold
            if timer_name in self.thresholds:
                if elapsed > self.thresholds[timer_name]:
                    self.logger.log(
                        f"Timer '{timer_name}' exceeded threshold: {elapsed:.2f}s > {self.thresholds[timer_name]}s",
                        "WARNING",
                        tags=['performance', 'threshold']
                    )

            return elapsed

    def record_metric(self, metric_name: str, value: Any = 1):
        """Record a metric value"""
        with self._metrics_lock:
            if metric_name in self.metrics:
                if isinstance(self.metrics[metric_name], (int, float)):
                    self.metrics[metric_name] += value
                else:
                    self.metrics[metric_name] = value
            else:
                self.metrics[metric_name] = value

    def increment_metric(self, metric_name: str, amount: float = 1):
        """Increment a metric"""
        with self._metrics_lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = 0
            self.metrics[metric_name] += amount

    def get_timer_stats(self, timer_name: str) -> Dict:
        """Get statistics for a timer"""

        with self._timer_lock:
            history = self.timer_history.get(timer_name, [])

        if not history:
            return {
                'count': 0,
                'total': 0,
                'avg': 0,
                'min': 0,
                'max': 0
            }

        return {
            'count': len(history),
            'total': sum(history),
            'avg': sum(history) / len(history),
            'min': min(history),
            'max': max(history),
            'recent': history[-10:]  # Last 10 timings
        }

    def get_all_metrics(self) -> Dict:
        """Get all recorded metrics"""
        with self._metrics_lock:
            return dict(self.metrics)

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""

        summary = {
            'metrics': self.get_all_metrics(),
            'timers': {},
            'cache_hit_rate': 0.0,
            'lineup_success_rate': 0.0
        }

        # Timer statistics
        with self._timer_lock:
            for timer_name in self.timer_history:
                summary['timers'][timer_name] = self.get_timer_stats(timer_name)

        # Calculated rates
        total_cache = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_cache > 0:
            summary['cache_hit_rate'] = self.metrics['cache_hits'] / total_cache

        total_lineups = self.metrics['lineups_generated'] + self.metrics['lineups_failed']
        if total_lineups > 0:
            summary['lineup_success_rate'] = self.metrics['lineups_generated'] / total_lineups

        return summary

    def reset_metrics(self):
        """Reset all metrics and timers"""
        with self._metrics_lock:
            self.metrics = {key: 0 for key in self.metrics}

        with self._timer_lock:
            self.active_timers.clear()
            self.timer_history.clear()

        self.logger.log("Performance metrics reset", "INFO", tags=['performance', 'reset'])


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

_logger_instance = None
_perf_monitor_instance = None

def get_logger() -> AdvancedLogger:
    """Get global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AdvancedLogger()
    return _logger_instance

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _perf_monitor_instance
    if _perf_monitor_instance is None:
        _perf_monitor_instance = PerformanceMonitor()
    return _perf_monitor_instance


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'LogEntry',
    'AdvancedLogger',
    'PerformanceMonitor',
    'get_logger',
    'get_performance_monitor'
]

"""
Enhanced logging and monitoring system
Improvements: Structured logging, persistence, analytics, alerting, performance profiling
"""

class GlobalLogger:
    """Enhanced logger with structured logging, persistence, and analytics"""

    def __init__(self, log_dir: Optional[str] = None):
        self.logs = deque(maxlen=200)
        self.error_logs = deque(maxlen=50)
        self.ai_decisions = deque(maxlen=100)
        self.optimization_events = deque(maxlen=50)
        self.performance_metrics = defaultdict(list)
        self._lock = threading.RLock()

        # Enhanced tracking
        self.error_patterns = defaultdict(int)
        self.last_cleanup = datetime.now()
        self.session_start = datetime.now()
        self.log_counts_by_level = defaultdict(int)
        self.context_tags = defaultdict(list)

        # Persistence
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.current_log_file = self.log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        # Alert system
        self.alert_thresholds = {
            'error_rate': 0.1,
            'critical_count': 5,
            'memory_warnings': 3,
            'consecutive_failures': 10
        }
        self.alerts_triggered = []
        self.consecutive_errors = 0

        # Performance profiling
        self.operation_profiles = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'min_time': float('inf'),
            'max_time': 0,
            'errors': 0
        })

    def log(self, message: str, level: str = "INFO", context: Dict = None,
            tags: List[str] = None, persist: bool = True):
        """Enhanced logging with tags and context"""
        with self._lock:
            entry = {
                'timestamp': datetime.now(),
                'level': level.upper(),
                'message': message,
                'context': context or {},
                'tags': tags or [],
                'thread': threading.current_thread().name,
                'session_age': (datetime.now() - self.session_start).total_seconds()
            }

            self.logs.append(entry)
            self.log_counts_by_level[level.upper()] += 1

            # Track by tags
            for tag in (tags or []):
                self.context_tags[tag].append(entry)

            if level.upper() in ["ERROR", "CRITICAL"]:
                self.error_logs.append(entry)
                error_key = self._extract_error_pattern(message)
                self.error_patterns[error_key] += 1
                self.consecutive_errors += 1
                self._check_alerts()
            else:
                self.consecutive_errors = 0

            if persist and level.upper() in ["ERROR", "CRITICAL", "WARNING"]:
                self._persist_log(entry)

            if (datetime.now() - self.last_cleanup).seconds > 300:
                self._cleanup()

            if level.upper() in ["ERROR", "CRITICAL"]:
                timestamp = entry['timestamp'].strftime('%H:%M:%S')
                print(f"[{timestamp}] {level}: {message}")

    def _extract_error_pattern(self, message: str) -> str:
        """Extract error pattern for grouping"""
        pattern = message.lower()
        # Remove numbers
        import re
        pattern = re.sub(r'\d+', 'N', pattern)
        pattern = re.sub(r'"[^"]*"', '"X"', pattern)
        pattern = re.sub(r"'[^']*'", "'X'", pattern)
        return pattern[:100]

    def _persist_log(self, entry: Dict):
        """Persist log entry to JSONL file"""
        try:
            serializable_entry = {
                'timestamp': entry['timestamp'].isoformat(),
                'level': entry['level'],
                'message': entry['message'],
                'context': entry.get('context', {}),
                'tags': entry.get('tags', []),
                'thread': entry['thread']
            }

            with open(self.current_log_file, 'a') as f:
                f.write(json.dumps(serializable_entry) + '\n')
        except Exception as e:
            print(f"Failed to persist log: {e}")

    def _check_alerts(self):
        """Check and trigger alerts"""
        total_logs = sum(self.log_counts_by_level.values())
        if total_logs < 10:
            return

        # Error rate alert
        error_count = self.log_counts_by_level['ERROR'] + self.log_counts_by_level['CRITICAL']
        error_rate = error_count / total_logs

        if error_rate > self.alert_thresholds['error_rate']:
            alert = {
                'type': 'high_error_rate',
                'timestamp': datetime.now(),
                'value': error_rate,
                'threshold': self.alert_thresholds['error_rate'],
                'message': f"Error rate {error_rate:.1%} exceeds threshold {self.alert_thresholds['error_rate']:.1%}"
            }
            if alert not in self.alerts_triggered:
                self.alerts_triggered.append(alert)

        # Critical count alert
        if self.log_counts_by_level['CRITICAL'] >= self.alert_thresholds['critical_count']:
            alert = {
                'type': 'critical_threshold',
                'timestamp': datetime.now(),
                'value': self.log_counts_by_level['CRITICAL'],
                'message': f"{self.log_counts_by_level['CRITICAL']} critical errors detected"
            }
            if alert not in self.alerts_triggered:
                self.alerts_triggered.append(alert)

        # Consecutive failures
        if self.consecutive_errors >= self.alert_thresholds['consecutive_failures']:
            alert = {
                'type': 'consecutive_failures',
                'timestamp': datetime.now(),
                'value': self.consecutive_errors,
                'message': f"{self.consecutive_errors} consecutive errors detected"
            }
            if alert not in self.alerts_triggered:
                self.alerts_triggered.append(alert)

    def _cleanup(self):
        """Memory cleanup and log rotation"""
        cutoff = datetime.now() - timedelta(hours=1)

        # Clean old performance metrics
        for key in list(self.performance_metrics.keys()):
            self.performance_metrics[key] = [
                m for m in self.performance_metrics[key]
                if m.get('timestamp', datetime.now()) > cutoff
            ]
            if not self.performance_metrics[key]:
                del self.performance_metrics[key]

        # Clean context tags
        for tag in list(self.context_tags.keys()):
            self.context_tags[tag] = [
                entry for entry in self.context_tags[tag]
                if entry['timestamp'] > cutoff
            ]
            if not self.context_tags[tag]:
                del self.context_tags[tag]

        self.last_cleanup = datetime.now()

    def log_exception(self, exception: Exception, context: str = "",
                     critical: bool = False, suggestions: List[str] = None):
        """Enhanced exception logging with auto-suggestions"""
        with self._lock:
            level = "CRITICAL" if critical else "ERROR"
            error_msg = f"{context}: {str(exception)}" if context else str(exception)

            # Get or generate suggestions
            if suggestions is None:
                suggestions = self._get_error_suggestions(exception, context)

            entry = {
                'timestamp': datetime.now(),
                'level': level,
                'message': error_msg,
                'exception_type': type(exception).__name__,
                'traceback': traceback.format_exc(),
                'suggestions': suggestions,
                'context': context,
                'thread': threading.current_thread().name
            }

            self.error_logs.append(entry)
            self.consecutive_errors += 1

            # Display helpful error message
            if suggestions:
                self.log(f"Suggestion: {suggestions[0]}", "INFO")

            self._persist_log(entry)

    def _get_error_suggestions(self, exception: Exception, context: str) -> List[str]:
        """Generate context-aware suggestions"""
        suggestions = []
        error_str = str(exception).lower()

        if isinstance(exception, KeyError):
            suggestions.append("Check that all required columns are present in the CSV")
            suggestions.append("Verify player names match exactly between datasets")
        elif isinstance(exception, ValueError):
            if "salary" in error_str:
                suggestions.append("Check salary cap constraints - may be too restrictive")
                suggestions.append("Verify player salaries are within valid range (3000-12000)")
            elif "ownership" in error_str:
                suggestions.append("Verify ownership projections are between 0-100")
        elif "timeout" in error_str:
            suggestions.append("Reduce number of lineups or increase timeout setting")
            suggestions.append("Simplify AI constraints")
        elif "memory" in error_str:
            suggestions.append("Clear cache and restart session")
            suggestions.append("Reduce MAX_HISTORY_ENTRIES in config")
        elif "infeasible" in error_str or "no solution" in error_str:
            suggestions.append("AI constraints may be too strict - try Moderate enforcement")
            suggestions.append("Check for conflicting requirements in AI recommendations")
        elif "api" in error_str or "connection" in error_str:
            suggestions.append("Check API key and internet connection")
            suggestions.append("Try manual mode if API continues to fail")

        return suggestions[:3]

    def log_ai_decision(self, decision_type: str, ai_source: str,
                       success: bool, details: Dict = None, confidence: float = 0):
        """Log AI decision with metadata"""
        with self._lock:
            entry = {
                'timestamp': datetime.now(),
                'type': decision_type,
                'source': ai_source,
                'success': success,
                'confidence': confidence,
                'details': details or {}
            }
            self.ai_decisions.append(entry)

    def log_optimization_start(self, num_lineups: int, field_size: str, settings: Dict):
        """Log optimization start"""
        with self._lock:
            self.optimization_events.append({
                'timestamp': datetime.now(),
                'event': 'start',
                'num_lineups': num_lineups,
                'field_size': field_size,
                'settings': settings
            })

    def log_optimization_end(self, lineups_generated: int, time_taken: float,
                            success_rate: float, avg_ownership: float = 0):
        """Log optimization completion"""
        with self._lock:
            self.optimization_events.append({
                'timestamp': datetime.now(),
                'event': 'complete',
                'lineups_generated': lineups_generated,
                'time_taken': time_taken,
                'success_rate': success_rate,
                'avg_ownership': avg_ownership
            })

    def profile_operation(self, operation_name: str, time_taken: float,
                         success: bool = True, metadata: Dict = None):
        """Profile an operation for performance analysis"""
        with self._lock:
            profile = self.operation_profiles[operation_name]
            profile['count'] += 1
            profile['total_time'] += time_taken
            profile['min_time'] = min(profile['min_time'], time_taken)
            profile['max_time'] = max(profile['max_time'], time_taken)

            if not success:
                profile['errors'] += 1

            if metadata:
                profile.setdefault('metadata', []).append(metadata)

    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict:
        """Get performance statistics for operations"""
        with self._lock:
            if operation_name:
                if operation_name in self.operation_profiles:
                    profile = self.operation_profiles[operation_name]
                    return {
                        'name': operation_name,
                        'count': profile['count'],
                        'avg_time': profile['total_time'] / max(profile['count'], 1),
                        'min_time': profile['min_time'],
                        'max_time': profile['max_time'],
                        'total_time': profile['total_time'],
                        'error_rate': profile['errors'] / max(profile['count'], 1)
                    }
                return {}

            # Return all operations
            return {
                name: {
                    'count': profile['count'],
                    'avg_time': profile['total_time'] / max(profile['count'], 1),
                    'error_rate': profile['errors'] / max(profile['count'], 1)
                }
                for name, profile in self.operation_profiles.items()
            }

    def get_logs_by_tag(self, tag: str, limit: int = 50) -> List[Dict]:
        """Retrieve logs by tag"""
        with self._lock:
            return list(self.context_tags.get(tag, []))[-limit:]

    def get_error_summary(self) -> Dict:
        """Get comprehensive error summary"""
        with self._lock:
            total_logs = sum(self.log_counts_by_level.values())

            return {
                'total_errors': len(self.error_logs),
                'error_rate': (self.log_counts_by_level['ERROR'] +
                              self.log_counts_by_level['CRITICAL']) / max(total_logs, 1),
                'top_patterns': sorted(
                    self.error_patterns.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                'recent_errors': [
                    {
                        'timestamp': e['timestamp'].strftime('%H:%M:%S'),
                        'message': e['message'][:100],
                        'type': e.get('exception_type', 'Unknown')
                    }
                    for e in list(self.error_logs)[-5:]
                ],
                'consecutive_errors': self.consecutive_errors,
                'alerts': len(self.alerts_triggered)
            }

    def get_session_summary(self) -> Dict:
        """Get complete session summary"""
        with self._lock:
            session_duration = (datetime.now() - self.session_start).total_seconds()
            total_logs = sum(self.log_counts_by_level.values())

            return {
                'session_duration_seconds': session_duration,
                'session_start': self.session_start.isoformat(),
                'total_logs': total_logs,
                'logs_by_level': dict(self.log_counts_by_level),
                'logs_per_minute': total_logs / max(session_duration / 60, 1),
                'error_summary': self.get_error_summary(),
                'ai_decisions': len(self.ai_decisions),
                'optimization_runs': len([e for e in self.optimization_events if e['event'] == 'start']),
                'alerts_triggered': len(self.alerts_triggered),
                'operations_profiled': len(self.operation_profiles)
            }

    def export_logs(self, filepath: str, include_context: bool = True):
        """Export all logs to file"""
        with self._lock:
            export_data = {
                'session_summary': self.get_session_summary(),
                'logs': [
                    {
                        'timestamp': log['timestamp'].isoformat(),
                        'level': log['level'],
                        'message': log['message'],
                        'context': log.get('context', {}) if include_context else {},
                        'tags': log.get('tags', [])
                    }
                    for log in self.logs
                ],
                'errors': [
                    {
                        'timestamp': err['timestamp'].isoformat(),
                        'message': err['message'],
                        'exception_type': err.get('exception_type'),
                        'suggestions': err.get('suggestions', [])
                    }
                    for err in self.error_logs
                ]
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)


class PerformanceMonitor:
    """Enhanced performance monitoring with profiling and recommendations"""

    def __init__(self):
        self.timers = {}
        self.metrics = defaultdict(list)
        self._lock = threading.RLock()
        self.start_times = {}

        # Enhanced tracking
        self.operation_counts = defaultdict(int)
        self.operation_times = defaultdict(list)
        self.memory_snapshots = deque(maxlen=20)
        self.bottlenecks = []

        # Performance baselines
        self.baselines = {
            'lineup_generation': 2.0,  # seconds
            'ai_api_call': 5.0,  # seconds
            'optimization_iteration': 0.5  # seconds
        }

    def start_timer(self, operation: str, metadata: Dict = None):
        """Start timing with metadata"""
        with self._lock:
            self.start_times[operation] = {
                'start': time.time(),
                'metadata': metadata or {}
            }
            self.operation_counts[operation] += 1

    def stop_timer(self, operation: str, check_baseline: bool = True) -> float:
        """Stop timing and check against baseline"""
        with self._lock:
            if operation not in self.start_times:
                return 0

            timer_data = self.start_times[operation]
            elapsed = time.time() - timer_data['start']
            del self.start_times[operation]

            # Store timing
            self.operation_times[operation].append(elapsed)

            # Keep only recent timings
            if len(self.operation_times[operation]) > 100:
                self.operation_times[operation] = self.operation_times[operation][-50:]

            # Check against baseline
            if check_baseline and operation in self.baselines:
                baseline = self.baselines[operation]
                if elapsed > baseline * 1.5:
                    self.bottlenecks.append({
                        'operation': operation,
                        'timestamp': datetime.now(),
                        'elapsed': elapsed,
                        'baseline': baseline,
                        'slowdown_factor': elapsed / baseline
                    })

            return elapsed

    def record_metric(self, metric_name: str, value: float, tags: Dict = None):
        """Record a metric with tags"""
        with self._lock:
            self.metrics[metric_name].append({
                'value': value,
                'timestamp': datetime.now(),
                'tags': tags or {}
            })

            # Cleanup old metrics
            cutoff = datetime.now() - timedelta(hours=1)
            self.metrics[metric_name] = [
                m for m in self.metrics[metric_name]
                if m['timestamp'] > cutoff
            ]

    def take_memory_snapshot(self):
        """Take memory snapshot if psutil available"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            snapshot = {
                'timestamp': datetime.now(),
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }

            self.memory_snapshots.append(snapshot)
            return snapshot
        except ImportError:
            return None

    def get_operation_stats(self, operation: str) -> Dict:
        """Get comprehensive statistics"""
        with self._lock:
            times = self.operation_times.get(operation, [])
            if not times:
                return {}

            times_array = np.array(times)

            stats = {
                'count': self.operation_counts[operation],
                'avg_time': np.mean(times),
                'median_time': np.median(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'std_time': np.std(times),
                'total_time': np.sum(times),
                'p95_time': np.percentile(times, 95),
                'p99_time': np.percentile(times, 99)
            }

            # Check against baseline
            if operation in self.baselines:
                stats['baseline'] = self.baselines[operation]
                stats['vs_baseline'] = stats['avg_time'] / self.baselines[operation]
                stats['exceeds_baseline'] = stats['avg_time'] > self.baselines[operation]

            return stats

    def get_bottlenecks(self, top_n: int = 5) -> List[Dict]:
        """Get top performance bottlenecks"""
        with self._lock:
            # Sort by slowdown factor
            sorted_bottlenecks = sorted(
                self.bottlenecks,
                key=lambda x: x['slowdown_factor'],
                reverse=True
            )
            return sorted_bottlenecks[:top_n]

    def get_recommendations(self) -> List[str]:
        """Get performance improvement recommendations"""
        recommendations = []

        # Check bottlenecks
        bottlenecks = self.get_bottlenecks(3)
        if bottlenecks:
            worst = bottlenecks[0]
            recommendations.append(
                f"Optimize '{worst['operation']}' - running {worst['slowdown_factor']:.1f}x slower than baseline"
            )

        # Check memory
        if self.memory_snapshots:
            recent_memory = [s['percent'] for s in list(self.memory_snapshots)[-5:]]
            if recent_memory and np.mean(recent_memory) > 80:
                recommendations.append(
                    "High memory usage detected - consider reducing cache sizes"
                )

        # Check operation frequency
        for op, count in self.operation_counts.items():
            if count > 1000:
                stats = self.get_operation_stats(op)
                if stats and stats.get('total_time', 0) > 60:
                    recommendations.append(
                        f"High frequency operation '{op}' - consider caching or optimization"
                    )

        return recommendations


class AIDecisionTracker:
    """Enhanced AI decision tracker with learning and pattern recognition"""

    def __init__(self):
        self.decisions = deque(maxlen=200)
        self.performance_feedback = deque(maxlen=200)
        self.decision_patterns = defaultdict(list)
        self._lock = threading.RLock()

        # Learning components
        self.successful_patterns = defaultdict(float)
        self.failed_patterns = defaultdict(float)
        self.confidence_calibration = defaultdict(list)

        # Pattern analysis
        self.pattern_evolution = []
        self.strategy_effectiveness = defaultdict(lambda: {
            'total': 0, 'success': 0, 'avg_score': 0
        })

    def track_decision(self, ai_type: Any, decision: Any, context: Dict = None):
        """Track AI decision with context"""
        with self._lock:
            entry = {
                'timestamp': datetime.now(),
                'ai_type': ai_type,
                'captain_count': len(decision.captain_targets) if hasattr(decision, 'captain_targets') else 0,
                'confidence': decision.confidence if hasattr(decision, 'confidence') else 0,
                'enforcement_rules': len(decision.enforcement_rules) if hasattr(decision, 'enforcement_rules') else 0,
                'context': context or {}
            }

            self.decisions.append(entry)

            # Track patterns
            pattern_key = self._extract_pattern(decision)
            self.decision_patterns[pattern_key].append(entry)

    def _extract_pattern(self, decision: Any) -> str:
        """Extract decision pattern"""
        if not hasattr(decision, 'confidence'):
            return "unknown"

        elements = [
            f"conf_{int(decision.confidence*10)}",
            f"capt_{len(decision.captain_targets) if hasattr(decision, 'captain_targets') else 0}",
            f"must_{len(decision.must_play) if hasattr(decision, 'must_play') else 0}",
            f"stack_{len(decision.stacks) if hasattr(decision, 'stacks') else 0}"
        ]
        return "_".join(elements)

    def record_performance(self, lineup: Dict, actual_score: Optional[float] = None,
                          won_tournament: bool = False):
        """Record lineup performance for learning"""
        with self._lock:
            if actual_score is not None:
                projected = lineup.get('Projected', 0)
                accuracy = 1 - abs(actual_score - projected) / max(actual_score, 1)

                entry = {
                    'timestamp': datetime.now(),
                    'strategy': lineup.get('AI_Strategy', 'unknown'),
                    'projected': projected,
                    'actual': actual_score,
                    'accuracy': accuracy,
                    'success': actual_score > projected * 1.1,
                    'won': won_tournament,
                    'ownership_tier': lineup.get('Ownership_Tier', 'unknown')
                }

                self.performance_feedback.append(entry)

                # Update strategy effectiveness
                strategy = entry['strategy']
                self.strategy_effectiveness[strategy]['total'] += 1
                if entry['success']:
                    self.strategy_effectiveness[strategy]['success'] += 1

                # Update pattern success
                pattern_key = f"{strategy}_{entry['ownership_tier']}"
                if entry['success']:
                    self.successful_patterns[pattern_key] += 1
                else:
                    self.failed_patterns[pattern_key] += 1

                # Confidence calibration
                confidence = lineup.get('Confidence', 0.5)
                self.confidence_calibration[int(confidence * 10)].append(accuracy)

    def get_learning_insights(self) -> Dict:
        """Get comprehensive learning insights"""
        with self._lock:
            insights = {
                'total_decisions': len(self.decisions),
                'avg_confidence': np.mean([d['confidence'] for d in self.decisions]) if self.decisions else 0,
                'pattern_performance': {},
                'confidence_calibration': {},
                'strategy_rankings': []
            }

            # Pattern success rates
            for pattern in set(list(self.successful_patterns.keys()) + list(self.failed_patterns.keys())):
                successes = self.successful_patterns.get(pattern, 0)
                failures = self.failed_patterns.get(pattern, 0)
                total = successes + failures

                if total >= 5:  # Minimum sample size
                    insights['pattern_performance'][pattern] = {
                        'success_rate': successes / total,
                        'total': total,
                        'confidence': 'high' if total >= 20 else 'medium' if total >= 10 else 'low'
                    }

            # Confidence calibration
            for conf_level, accuracies in self.confidence_calibration.items():
                if accuracies:
                    insights['confidence_calibration'][conf_level / 10] = {
                        'avg_accuracy': np.mean(accuracies),
                        'sample_size': len(accuracies),
                        'well_calibrated': abs(conf_level / 10 - np.mean(accuracies)) < 0.15
                    }

            # Strategy rankings
            for strategy, data in self.strategy_effectiveness.items():
                if data['total'] >= 5:
                    insights['strategy_rankings'].append({
                        'strategy': strategy,
                        'success_rate': data['success'] / data['total'],
                        'total_uses': data['total']
                    })

            insights['strategy_rankings'].sort(key=lambda x: x['success_rate'], reverse=True)

            return insights

    def get_recommended_adjustments(self) -> Dict:
        """Get AI-driven recommendations for adjustment"""
        insights = self.get_learning_insights()
        adjustments = {}

        # Confidence adjustments
        calibration = insights.get('confidence_calibration', {})
        for conf_level, data in calibration.items():
            if not data['well_calibrated'] and data['sample_size'] >= 10:
                actual_accuracy = data['avg_accuracy']
                adjustments[f'confidence_{conf_level}'] = {
                    'current': conf_level,
                    'recommended': actual_accuracy,
                    'reason': 'Confidence miscalibration detected'
                }

        # Strategy adjustments
        pattern_perf = insights.get('pattern_performance', {})
        for pattern, stats in pattern_perf.items():
            if stats['total'] >= 10:
                if stats['success_rate'] > 0.7:
                    adjustments[f'boost_{pattern}'] = {
                        'multiplier': 1.2,
                        'reason': f"High success rate: {stats['success_rate']:.1%}"
                    }
                elif stats['success_rate'] < 0.3:
                    adjustments[f'reduce_{pattern}'] = {
                        'multiplier': 0.7,
                        'reason': f"Low success rate: {stats['success_rate']:.1%}"
                    }

        return adjustments


# Singleton accessors
_global_logger = None
_performance_monitor = None
_ai_tracker = None

def get_logger():
    """Get or create global logger"""
    global _global_logger
    if _global_logger is None:
        _global_logger = GlobalLogger()
    return _global_logger

def get_performance_monitor():
    """Get or create performance monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def get_ai_tracker():
    """Get or create AI tracker"""
    global _ai_tracker
    if _ai_tracker is None:
        _ai_tracker = AIDecisionTracker()
    return _ai_tracker

"""
AI enforcement engine and validation
Improvements: Multi-tier enforcement, constraint learning, feasibility checking
"""

class AIEnforcementEngine:
    """Enhanced enforcement with adaptive learning and feasibility analysis"""

    def __init__(self, enforcement_level=None):
        from config import AIEnforcementLevel
        self.enforcement_level = enforcement_level or AIEnforcementLevel.STRONG
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()

        # Enhanced tracking
        self.applied_rules = deque(maxlen=200)
        self.rule_success_rate = defaultdict(lambda: {'applied': 0, 'successful': 0})
        self.violation_patterns = defaultdict(int)
        self.constraint_relaxations = []

        # Learning system
        self.rule_effectiveness = defaultdict(float)
        self.conflicting_rules = []

    def create_enforcement_rules(self, recommendations: Dict) -> Dict:
        """Create tiered enforcement rules with conflict resolution"""
        self.logger.log(
            f"Creating enforcement rules at {self.enforcement_level.value} level",
            "INFO",
            tags=['enforcement', 'ai']
        )

        rules = {
            'hard_constraints': [],
            'soft_constraints': [],
            'stacking_rules': [],
            'ownership_rules': [],
            'correlation_rules': [],
            'meta': {
                'enforcement_level': self.enforcement_level.value,
                'total_recommendations': len(recommendations),
                'timestamp': datetime.now()
            }
        }

        # Create rules based on enforcement level
        if self.enforcement_level.value == "Mandatory":
            rules = self._create_mandatory_rules(recommendations)
        elif self.enforcement_level.value == "Strong":
            rules = self._create_strong_rules(recommendations)
        elif self.enforcement_level.value == "Moderate":
            rules = self._create_moderate_rules(recommendations)
        else:  # Advisory
            rules = self._create_advisory_rules(recommendations)

        # Add advanced stacking rules
        rules['stacking_rules'].extend(
            self._create_stacking_rules(recommendations)
        )

        # Detect and resolve conflicts
        rules = self._resolve_rule_conflicts(rules)

        # Sort by priority
        for rule_type in ['hard_constraints', 'soft_constraints', 'stacking_rules']:
            if rule_type in rules:
                rules[rule_type].sort(
                    key=lambda x: x.get('priority', 0),
                    reverse=True
                )

        total_rules = sum(
            len(v) for k, v in rules.items()
            if isinstance(v, list)
        )

        self.logger.log(
            f"Created {total_rules} enforcement rules",
            "INFO",
            context={'by_type': {k: len(v) for k, v in rules.items() if isinstance(v, list)}}
        )

        return rules

    def _resolve_rule_conflicts(self, rules: Dict) -> Dict:
        """Detect and resolve conflicting rules"""
        conflicts_found = []

        # Check captain conflicts
        captain_rules = [
            r for r in rules.get('hard_constraints', [])
            if 'captain' in r.get('rule', '')
        ]

        must_include = set()
        must_exclude = set()

        for rule in rules.get('hard_constraints', []):
            if rule.get('rule') == 'must_include':
                must_include.add(rule.get('player'))
            elif rule.get('rule') == 'must_exclude':
                must_exclude.add(rule.get('player'))

        # Find conflicts
        conflicts = must_include & must_exclude
        if conflicts:
            conflicts_found.append({
                'type': 'must_include_exclude',
                'players': list(conflicts)
            })

            # Resolve: prioritize must_include
            rules['hard_constraints'] = [
                r for r in rules['hard_constraints']
                if not (r.get('rule') == 'must_exclude' and r.get('player') in conflicts)
            ]

            self.logger.log(
                f"Resolved conflicts for players: {conflicts}",
                "WARNING",
                tags=['conflict_resolution']
            )

        if conflicts_found:
            self.conflicting_rules.extend(conflicts_found)

        return rules

    def _create_mandatory_rules(self, recommendations: Dict) -> Dict:
        """Strict enforcement - all AI decisions become hard constraints"""
        from config import OptimizerConfig, AIStrategistType

        rules = {
            'hard_constraints': [],
            'soft_constraints': [],
            'stacking_rules': [],
            'ownership_rules': [],
            'correlation_rules': []
        }

        for ai_type, rec in recommendations.items():
            weight = OptimizerConfig.AI_WEIGHTS.get(
                ai_type.value.lower().replace(' ', '_'),
                0.33
            )

            # Captain constraints (highest priority)
            if rec.captain_targets:
                rules['hard_constraints'].append({
                    'rule': 'captain_from_list',
                    'players': rec.captain_targets[:7],
                    'source': ai_type.value,
                    'priority': int(100 * weight * rec.confidence),
                    'type': 'hard',
                    'ai_weight': weight,
                    'confidence': rec.confidence
                })

            # Must play (top 3 only for feasibility)
            for i, player in enumerate(rec.must_play[:3]):
                rules['hard_constraints'].append({
                    'rule': 'must_include',
                    'player': player,
                    'source': ai_type.value,
                    'priority': int(90 * weight * rec.confidence) - i * 5,
                    'type': 'hard'
                })

            # Never play (top 3)
            for i, player in enumerate(rec.never_play[:3]):
                rules['hard_constraints'].append({
                    'rule': 'must_exclude',
                    'player': player,
                    'source': ai_type.value,
                    'priority': int(85 * weight * rec.confidence) - i * 5,
                    'type': 'hard'
                })

            # Stack constraints
            for i, stack in enumerate(rec.stacks[:3]):
                rules['stacking_rules'].append({
                    'rule': 'must_stack',
                    'stack': stack,
                    'source': ai_type.value,
                    'priority': int(80 * weight * rec.confidence) - i * 5,
                    'type': 'hard'
                })

        return rules

    def _create_strong_rules(self, recommendations: Dict) -> Dict:
        """Strong enforcement with high-confidence becoming hard constraints"""
        rules = self._create_moderate_rules(recommendations)

        from config import OptimizerConfig

        # Upgrade high-confidence recommendations to hard constraints
        for ai_type, rec in recommendations.items():
            if rec.confidence > 0.7:
                weight = OptimizerConfig.AI_WEIGHTS.get(
                    ai_type.value.lower().replace(' ', '_'),
                    0.33
                )

                if rec.captain_targets:
                    rules['hard_constraints'].append({
                        'rule': 'captain_selection',
                        'players': rec.captain_targets[:5],
                        'source': ai_type.value,
                        'priority': int(95 * weight * rec.confidence),
                        'type': 'hard',
                        'reason': 'high_confidence'
                    })

        return rules

    def _create_moderate_rules(self, recommendations: Dict) -> Dict:
        """Balanced enforcement - consensus becomes hard, rest soft"""
        from config import OptimizerConfig

        rules = {
            'hard_constraints': [],
            'soft_constraints': [],
            'stacking_rules': [],
            'ownership_rules': [],
            'correlation_rules': []
        }

        # Find consensus recommendations
        captain_counts = defaultdict(int)
        must_play_counts = defaultdict(int)

        for rec in recommendations.values():
            for captain in rec.captain_targets:
                captain_counts[captain] += 1
            for player in rec.must_play:
                must_play_counts[player] += 1

        # Consensus captains become hard constraints
        for captain, count in captain_counts.items():
            if count >= 2:
                rules['hard_constraints'].append({
                    'rule': 'consensus_captain',
                    'player': captain,
                    'agreement': count,
                    'priority': 90 + count * 5,
                    'type': 'hard',
                    'reason': 'multi_ai_consensus'
                })

        # Single AI recommendations become soft constraints
        for ai_type, rec in recommendations.items():
            weight = OptimizerConfig.AI_WEIGHTS.get(
                ai_type.value.lower().replace(' ', '_'),
                0.33
            )

            # Non-consensus captains
            for captain in rec.captain_targets[:5]:
                if captain_counts[captain] == 1:
                    rules['soft_constraints'].append({
                        'rule': 'prefer_captain',
                        'player': captain,
                        'source': ai_type.value,
                        'weight': weight * rec.confidence,
                        'priority': int(70 * weight * rec.confidence),
                        'type': 'soft'
                    })

            # Must play players
            for i, player in enumerate(rec.must_play[:3]):
                if must_play_counts[player] == 1:
                    rules['soft_constraints'].append({
                        'rule': 'prefer_player',
                        'player': player,
                        'source': ai_type.value,
                        'weight': weight * rec.confidence * (1 - i * 0.1),
                        'priority': int(70 * weight * rec.confidence) - i * 5,
                        'type': 'soft'
                    })

        return rules

    def _create_advisory_rules(self, recommendations: Dict) -> Dict:
        """All recommendations are soft suggestions"""
        from config import OptimizerConfig

        rules = {
            'hard_constraints': [],
            'soft_constraints': [],
            'stacking_rules': [],
            'ownership_rules': [],
            'correlation_rules': []
        }

        for ai_type, rec in recommendations.items():
            weight = OptimizerConfig.AI_WEIGHTS.get(
                ai_type.value.lower().replace(' ', '_'),
                0.33
            )

            # All as soft constraints
            for i, captain in enumerate(rec.captain_targets[:5]):
                rules['soft_constraints'].append({
                    'rule': 'prefer_captain',
                    'player': captain,
                    'source': ai_type.value,
                    'weight': weight * rec.confidence * (1 - i * 0.1),
                    'priority': int(60 * weight * rec.confidence),
                    'type': 'soft'
                })

        return rules

    def _create_stacking_rules(self, recommendations: Dict) -> List[Dict]:
        """Create comprehensive stacking rules"""
        stacking_rules = []

        from config import StackType

        for ai_type, rec in recommendations.items():
            for stack in rec.stacks:
                stack_type = stack.get('type', 'standard')

                # Onslaught stacks
                if stack_type == 'onslaught':
                    stacking_rules.append({
                        'rule': 'onslaught_stack',
                        'players': stack.get('players', []),
                        'team': stack.get('team'),
                        'min_players': 3,
                        'max_players': 5,
                        'scenario': stack.get('scenario', 'blowout'),
                        'priority': 85,
                        'source': ai_type.value,
                        'correlation_strength': 0.6
                    })

                # Bring-back stacks
                elif stack_type == 'bring_back':
                    stacking_rules.append({
                        'rule': 'bring_back_stack',
                        'primary_players': stack.get('primary_stack', []),
                        'bring_back_player': stack.get('bring_back'),
                        'game_total': stack.get('game_total', 45),
                        'priority': 80,
                        'source': ai_type.value,
                        'correlation_strength': 0.5
                    })

                # Leverage stacks
                elif stack_type == 'leverage':
                    stacking_rules.append({
                        'rule': 'leverage_stack',
                        'players': [stack.get('player1'), stack.get('player2')],
                        'combined_ownership_max': stack.get('combined_ownership', 20),
                        'leverage_score_min': 3.0,
                        'priority': 75,
                        'source': ai_type.value,
                        'correlation_strength': 0.4
                    })

                # Standard QB stacks
                elif 'player1' in stack and 'player2' in stack:
                    stacking_rules.append({
                        'rule': 'standard_stack',
                        'players': [stack['player1'], stack['player2']],
                        'correlation': stack.get('correlation', 0.5),
                        'priority': 70,
                        'source': ai_type.value
                    })

        # Deduplicate
        seen = set()
        unique_stacks = []

        for stack in stacking_rules:
            players = stack.get('players', [])
            if players:
                stack_id = "_".join(sorted(players[:2]))
                if stack_id not in seen:
                    seen.add(stack_id)
                    unique_stacks.append(stack)
            else:
                unique_stacks.append(stack)

        return unique_stacks

    def validate_lineup_against_ai(self, lineup: Dict,
                                   enforcement_rules: Dict) -> Tuple[bool, List[str], Dict]:
        """
        Validate with detailed feedback
        Returns: (is_valid, violations, validation_details)
        """
        violations = []
        validation_details = {
            'hard_constraint_checks': 0,
            'hard_constraint_passes': 0,
            'soft_constraint_checks': 0,
            'soft_constraint_score': 0.0
        }

        captain = lineup.get('Captain')
        flex = lineup.get('FLEX', [])
        all_players = [captain] + flex

        # Check hard constraints
        for rule in enforcement_rules.get('hard_constraints', []):
            validation_details['hard_constraint_checks'] += 1

            if rule['rule'] == 'captain_from_list':
                if captain not in rule['players']:
                    violations.append({
                        'type': 'captain_constraint',
                        'message': f"Captain {captain} not in AI list",
                        'rule': rule,
                        'severity': 'high'
                    })
                else:
                    validation_details['hard_constraint_passes'] += 1

            elif rule['rule'] == 'must_include':
                if rule['player'] not in all_players:
                    violations.append({
                        'type': 'missing_player',
                        'message': f"Missing required player: {rule['player']}",
                        'rule': rule,
                        'severity': 'high'
                    })
                else:
                    validation_details['hard_constraint_passes'] += 1

            elif rule['rule'] == 'must_exclude':
                if rule['player'] in all_players:
                    violations.append({
                        'type': 'excluded_player',
                        'message': f"Included banned player: {rule['player']}",
                        'rule': rule,
                        'severity': 'high'
                    })
                else:
                    validation_details['hard_constraint_passes'] += 1

        # Check soft constraints (accumulate score)
        for rule in enforcement_rules.get('soft_constraints', []):
            validation_details['soft_constraint_checks'] += 1

            weight = rule.get('weight', 0.5)

            if rule['rule'] == 'prefer_captain' and captain == rule['player']:
                validation_details['soft_constraint_score'] += weight
            elif rule['rule'] == 'prefer_player' and rule['player'] in all_players:
                validation_details['soft_constraint_score'] += weight

        # Check stacking rules
        for stack_rule in enforcement_rules.get('stacking_rules', []):
            if stack_rule.get('type') == 'hard':
                if not self._validate_stack_rule(all_players, stack_rule):
                    violations.append({
                        'type': 'stack_violation',
                        'message': f"Stack rule violated: {stack_rule.get('rule')}",
                        'rule': stack_rule,
                        'severity': 'medium'
                    })

        # Track violations for learning
        for violation in violations:
            self.violation_patterns[violation['type']] += 1

        is_valid = len([v for v in violations if v['severity'] == 'high']) == 0

        # Record for learning
        self.applied_rules.append({
            'timestamp': datetime.now(),
            'lineup_num': lineup.get('Lineup', 0),
            'valid': is_valid,
            'violations': len(violations),
            'details': validation_details
        })

        return is_valid, [v['message'] for v in violations], validation_details

    def _validate_stack_rule(self, players: List[str], stack_rule: Dict) -> bool:
        """Validate specific stack rule"""
        rule_type = stack_rule.get('rule')

        if rule_type == 'onslaught_stack':
            required_players = stack_rule.get('players', [])
            min_required = stack_rule.get('min_players', 3)
            count = sum(1 for p in required_players if p in players)
            return count >= min_required

        elif rule_type == 'bring_back_stack':
            primary = stack_rule.get('primary_players', [])
            bring_back = stack_rule.get('bring_back_player')
            has_primary = any(p in players for p in primary)
            has_bring_back = bring_back in players
            return has_primary and has_bring_back

        elif rule_type == 'standard_stack':
            required = stack_rule.get('players', [])
            return all(p in players for p in required)

        return True

    def get_enforcement_stats(self) -> Dict:
        """Get enforcement statistics"""
        with self._lock if hasattr(self, '_lock') else nullcontext():
            total_applied = len(self.applied_rules)
            if total_applied == 0:
                return {}

            valid_lineups = sum(1 for r in self.applied_rules if r['valid'])

            return {
                'total_validations': total_applied,
                'valid_lineups': valid_lineups,
                'validation_rate': valid_lineups / total_applied,
                'avg_violations': np.mean([r['violations'] for r in self.applied_rules]),
                'violation_patterns': dict(self.violation_patterns),
                'enforcement_level': self.enforcement_level.value,
                'conflicts_resolved': len(self.conflicting_rules)
            }

    def relax_enforcement(self):
        """Relax enforcement level by one step"""
        from config import AIEnforcementLevel

        levels = [
            AIEnforcementLevel.MANDATORY,
            AIEnforcementLevel.STRONG,
            AIEnforcementLevel.MODERATE,
            AIEnforcementLevel.ADVISORY
        ]

        current_idx = levels.index(self.enforcement_level)
        if current_idx < len(levels) - 1:
            old_level = self.enforcement_level.value
            self.enforcement_level = levels[current_idx + 1]

            self.constraint_relaxations.append({
                'timestamp': datetime.now(),
                'from': old_level,
                'to': self.enforcement_level.value,
                'reason': 'manual_relaxation'
            })

            self.logger.log(
                f"Enforcement relaxed from {old_level} to {self.enforcement_level.value}",
                "WARNING",
                tags=['enforcement', 'relaxation']
            )


from contextlib import contextmanager

@contextmanager
def nullcontext():
    """Null context manager for when no lock is needed"""
    yield

"""
Base AI Strategist with enhanced learning and adaptation
Improvements: Adaptive prompts, performance tracking, multi-slate optimization, caching
"""

class BaseAIStrategist:
    """Enhanced base strategist with learning and adaptive capabilities"""

    def __init__(self, api_manager=None, strategist_type=None):
        self.api_manager = api_manager
        self.strategist_type = strategist_type
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()

        # Enhanced caching
        self.response_cache = {}
        self.prompt_cache = {}
        self.max_cache_size = 50
        self._cache_lock = threading.RLock()

        # Performance tracking
        self.performance_history = deque(maxlen=200)
        self.successful_patterns = defaultdict(float)
        self.failed_patterns = defaultdict(float)

        # Adaptive learning
        self.adaptive_confidence_modifier = 1.0
        self.slate_performance = defaultdict(list)
        self.prompt_versions = []
        self.best_prompt_version = None

        # Fallback confidence by strategist type
        from config import AIStrategistType
        self.fallback_confidence = {
            AIStrategistType.GAME_THEORY: 0.5,
            AIStrategistType.CORRELATION: 0.55,
            AIStrategistType.CONTRARIAN_NARRATIVE: 0.45
        }

    def get_recommendation(self, df, game_info: Dict, field_size: str,
                          use_api: bool = True) -> Any:
        """Enhanced recommendation with comprehensive error handling"""

        self.perf_monitor.start_timer(f"{self.strategist_type.value}_recommendation")

        try:
            # Validate inputs
            if df.empty:
                self.logger.log(
                    f"{self.strategist_type.value}: Empty DataFrame",
                    "ERROR",
                    tags=['ai', 'validation']
                )
                return self._get_fallback_recommendation(df, field_size)

            # Analyze slate for context
            slate_profile = self._analyze_slate_profile(df, game_info)

            # Generate cache key
            cache_key = self._generate_cache_key(df, game_info, field_size)

            # Check cache
            cached_rec = self._get_from_cache(cache_key)
            if cached_rec:
                self.logger.log(
                    f"{self.strategist_type.value}: Cache hit",
                    "DEBUG",
                    tags=['cache', 'performance']
                )
                cached_rec.confidence *= self.adaptive_confidence_modifier
                return cached_rec

            # Generate prompt with slate context
            prompt = self.generate_prompt(df, game_info, field_size, slate_profile)

            # Get response
            if use_api and self.api_manager and self.api_manager.client:
                response = self._get_api_response(prompt)
            else:
                response = self._get_fallback_response(df, game_info, field_size, slate_profile)

            # Parse response
            recommendation = self.parse_response(response, df, field_size)

            # Apply learned adjustments
            recommendation = self._apply_learned_adjustments(recommendation, slate_profile)

            # Validate
            is_valid, errors, corrections = recommendation.validate(auto_correct=True)

            if not is_valid:
                self.logger.log(
                    f"{self.strategist_type.value} validation errors: {errors}",
                    "WARNING",
                    context={'corrections': corrections},
                    tags=['ai', 'validation']
                )

            # Add enforcement rules
            recommendation.enforcement_rules = self.create_enforcement_rules(
                recommendation, df, field_size, slate_profile
            )

            # Cache result
            self._add_to_cache(cache_key, recommendation)

            elapsed = self.perf_monitor.stop_timer(f"{self.strategist_type.value}_recommendation")

            self.logger.log(
                f"{self.strategist_type.value}: Recommendation generated in {elapsed:.2f}s",
                "INFO",
                context={'confidence': recommendation.confidence},
                tags=['ai', 'success']
            )

            return recommendation

        except Exception as e:
            self.logger.log_exception(
                e,
                f"{self.strategist_type.value}.get_recommendation",
                critical=False
            )
            self.perf_monitor.stop_timer(f"{self.strategist_type.value}_recommendation")
            return self._get_fallback_recommendation(df, field_size)

    def _analyze_slate_profile(self, df, game_info: Dict) -> Dict:
        """Enhanced slate analysis with pattern detection"""
        profile = {
            'player_count': len(df),
            'avg_salary': df['Salary'].mean(),
            'salary_std': df['Salary'].std(),
            'salary_range': df['Salary'].max() - df['Salary'].min(),
            'avg_ownership': df.get('Ownership', pd.Series([10])).mean(),
            'ownership_std': df.get('Ownership', pd.Series([10])).std(),
            'ownership_concentration': self._calculate_concentration(df.get('Ownership', pd.Series([10]))),
            'total': game_info.get('total', 45),
            'spread': abs(game_info.get('spread', 0)),
            'weather': game_info.get('weather', 'Clear'),
            'teams': df['Team'].nunique(),
            'positions': df['Position'].value_counts().to_dict(),
            'value_distribution': df['Projected_Points'].std() / df['Projected_Points'].mean(),
            'is_primetime': game_info.get('primetime', False),
            'injuries': game_info.get('injury_count', 0)
        }

        # Determine slate characteristics
        profile['slate_type'] = self._determine_slate_type(df, game_info)
        profile['pricing_efficiency'] = self._calculate_pricing_efficiency(df)
        profile['chalk_concentration'] = self._calculate_chalk_concentration(df)
        profile['leverage_opportunities'] = self._identify_leverage_opportunities(df)

        return profile

    def _calculate_concentration(self, series) -> float:
        """Calculate Herfindahl concentration index"""
        if len(series) == 0:
            return 0
        shares = series / series.sum()
        return (shares ** 2).sum()

    def _determine_slate_type(self, df, game_info: Dict) -> str:
        """Enhanced slate type detection"""
        total = game_info.get('total', 45)
        spread = abs(game_info.get('spread', 0))
        weather = game_info.get('weather', 'Clear')

        # Multiple conditions
        conditions = []

        if total > 50 and spread < 3:
            conditions.append('shootout')
        if total < 40:
            conditions.append('low_scoring')
        if spread > 10:
            conditions.append('blowout_risk')
        if weather in ['Rain', 'Snow', 'Wind']:
            conditions.append('weather_game')
        if df['Salary'].std() < 1000:
            conditions.append('flat_pricing')

        # Return primary condition or default
        return conditions[0] if conditions else 'standard'

    def _calculate_pricing_efficiency(self, df) -> float:
        """Calculate how efficiently players are priced"""
        df = df.copy()
        df['value'] = df['Projected_Points'] / (df['Salary'] / 1000)
        value_std = df['value'].std()
        value_mean = df['value'].mean()

        # Lower coefficient of variation = more efficient pricing
        return 1 - (value_std / value_mean) if value_mean > 0 else 0.5

    def _calculate_chalk_concentration(self, df) -> float:
        """Calculate how concentrated ownership is in top players"""
        if 'Ownership' not in df.columns:
            return 0.5

        top_5_ownership = df.nlargest(5, 'Ownership')['Ownership'].sum()
        total_ownership = df['Ownership'].sum()

        return top_5_ownership / total_ownership if total_ownership > 0 else 0.5

    def _identify_leverage_opportunities(self, df) -> int:
        """Count number of high-leverage opportunities"""
        if 'Ownership' not in df.columns:
            return 0

        # High projection + low ownership = leverage
        df = df.copy()
        df['leverage_score'] = df['Projected_Points'] / (df['Ownership'] + 1)

        # Top 20% leverage scores
        threshold = df['leverage_score'].quantile(0.8)
        return len(df[df['leverage_score'] >= threshold])

    def _apply_learned_adjustments(self, recommendation: Any, slate_profile: Dict) -> Any:
        """Apply pattern learning to adjust recommendations"""

        slate_type = slate_profile.get('slate_type', 'standard')

        # Boost confidence if this slate type has been successful
        if slate_type in self.successful_patterns:
            success_rate = self.successful_patterns[slate_type]
            if success_rate > 0.7:
                confidence_boost = min(0.2, (success_rate - 0.7) * 0.5)
                recommendation.confidence = min(0.95, recommendation.confidence + confidence_boost)

        # Apply adaptive confidence modifier
        recommendation.confidence *= self.adaptive_confidence_modifier

        # Adjust captain targets based on slate type
        if slate_type == 'shootout':
            # Prioritize pass catchers
            qbs_receivers = []
            for player in recommendation.captain_targets:
                # This would need access to player data - simplified here
                qbs_receivers.append(player)

            # Reorder to prioritize these
            recommendation.captain_targets = qbs_receivers[:5] + [
                p for p in recommendation.captain_targets if p not in qbs_receivers
            ]

        elif slate_type == 'blowout_risk':
            # Favor game script leverage
            # This is where you'd adjust based on expected blowout
            pass

        # Adjust based on chalk concentration
        chalk_conc = slate_profile.get('chalk_concentration', 0.5)
        if chalk_conc > 0.6:  # High chalk concentration
            # Increase contrarian plays
            recommendation.contrarian_angles.append(
                f"High chalk concentration ({chalk_conc:.1%}) detected - increasing contrarian weight"
            )

        return recommendation

    def track_performance(self, lineup: Dict, actual_points: Optional[float] = None,
                         slate_profile: Optional[Dict] = None):
        """Enhanced performance tracking with slate context"""

        if actual_points is not None:
            projected = lineup.get('Projected', 0)
            accuracy = 1 - abs(actual_points - projected) / max(actual_points, 1)

            performance_data = {
                'strategy': self.strategist_type.value,
                'projected': projected,
                'actual': actual_points,
                'accuracy': accuracy,
                'success': actual_points > projected * 1.1,
                'timestamp': datetime.now(),
                'slate_type': slate_profile.get('slate_type') if slate_profile else 'unknown'
            }

            self.performance_history.append(performance_data)

            # Update slate-specific performance
            if slate_profile:
                slate_type = slate_profile.get('slate_type', 'unknown')
                self.slate_performance[slate_type].append(accuracy)

            # Update adaptive confidence
            if len(self.performance_history) >= 20:
                recent_accuracy = np.mean([
                    p['accuracy'] for p in list(self.performance_history)[-20:]
                ])
                self.adaptive_confidence_modifier = 0.5 + recent_accuracy

            # Update pattern success
            slate_type = performance_data['slate_type']
            if performance_data['success']:
                self.successful_patterns[slate_type] += 1
            else:
                self.failed_patterns[slate_type] += 1

    def get_performance_insights(self) -> Dict:
        """Get detailed performance insights"""
        if not self.performance_history:
            return {}

        recent = list(self.performance_history)[-50:]

        insights = {
            'total_tracked': len(self.performance_history),
            'avg_accuracy': np.mean([p['accuracy'] for p in recent]),
            'success_rate': np.mean([p['success'] for p in recent]),
            'adaptive_modifier': self.adaptive_confidence_modifier,
            'slate_performance': {},
            'trend': 'unknown'
        }

        # Per-slate performance
        for slate_type, accuracies in self.slate_performance.items():
            if accuracies:
                insights['slate_performance'][slate_type] = {
                    'avg_accuracy': np.mean(accuracies),
                    'sample_size': len(accuracies),
                    'trend': self._calculate_trend(accuracies)
                }

        # Overall trend
        if len(recent) >= 10:
            accuracies = [p['accuracy'] for p in recent]
            insights['trend'] = self._calculate_trend(accuracies)

        return insights

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from time series"""
        if len(values) < 10:
            return 'insufficient_data'

        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        return 'stable'

    def create_enforcement_rules(self, recommendation: Any, df,
                                 field_size: str, slate_profile: Dict) -> List[Dict]:
        """Create strategist-specific enforcement rules"""

        rules = []
        available_players = set(df['Player'].values)

        # Validate and filter
        valid_captains = [c for c in recommendation.captain_targets if c in available_players]

        if valid_captains:
            base_priority = int(recommendation.confidence * 100)
            slate_type = slate_profile.get('slate_type', 'standard')

            # Adjust priority based on slate
            priority_modifier = {
                'shootout': 1.2,
                'low_scoring': 0.8,
                'blowout_risk': 1.1,
                'flat_pricing': 0.9,
                'standard': 1.0
            }.get(slate_type, 1.0)

            adjusted_priority = int(base_priority * priority_modifier)

            rule_type = 'hard' if recommendation.confidence > 0.8 else 'soft'

            rules.append({
                'type': rule_type,
                'constraint': f'captain_{self.strategist_type.value}',
                'players': valid_captains[:5],
                'priority': adjusted_priority,
                'description': f'{self.strategist_type.value}: Priority captains',
                'slate_context': slate_type,
                'confidence': recommendation.confidence
            })

        # Must play enforcement
        valid_must_play = [p for p in recommendation.must_play[:5] if p in available_players]

        for i, player in enumerate(valid_must_play):
            rule_type = 'hard' if recommendation.confidence > 0.7 and i < 2 else 'soft'
            rules.append({
                'type': rule_type,
                'constraint': f'include_{player}',
                'player': player,
                'weight': recommendation.confidence if rule_type == 'soft' else 1.0,
                'priority': int((recommendation.confidence - i * 0.1) * 50),
                'description': f'{self.strategist_type.value}: Include {player}'
            })

        return rules

    def _get_fallback_recommendation(self, df, field_size: str) -> Any:
        """Enhanced fallback with statistical analysis"""

        from data_models import AIRecommendation

        self.logger.log(
            f"{self.strategist_type.value}: Using fallback recommendation",
            "WARNING",
            tags=['fallback', 'ai']
        )

        # Statistical captain selection
        if 'Ownership' in df.columns:
            # Balance projection and ownership
            df_copy = df.copy()
            df_copy['score'] = (
                df_copy['Projected_Points'] / df_copy['Projected_Points'].max() * 0.6 +
                (100 - df_copy['Ownership']) / 100 * 0.4
            )
            captains = df_copy.nlargest(7, 'score')['Player'].tolist()
        else:
            captains = df.nlargest(7, 'Projected_Points')['Player'].tolist()

        # Statistical must-play
        must_play = df.nlargest(3, 'Projected_Points')['Player'].tolist()

        # Avoid extreme chalk
        if 'Ownership' in df.columns:
            never_play = df[df['Ownership'] > 40].nlargest(2, 'Ownership')['Player'].tolist()
        else:
            never_play = []

        return AIRecommendation(
            captain_targets=captains,
            must_play=must_play,
            never_play=never_play,
            stacks=[],
            key_insights=["Fallback recommendation - statistical analysis only"],
            confidence=self.fallback_confidence.get(self.strategist_type, 0.4),
            narrative=f"Statistical {self.strategist_type.value} fallback",
            source_ai=self.strategist_type
        )

    def _generate_cache_key(self, df, game_info: Dict, field_size: str) -> str:
        """Generate cache key from inputs"""
        # Create hash from key elements
        key_elements = [
            str(len(df)),
            str(df['Player'].tolist()),
            str(game_info.get('total', 0)),
            str(game_info.get('spread', 0)),
            field_size,
            self.strategist_type.value
        ]

        key_string = "|".join(key_elements)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get from cache with lock"""
        with self._cache_lock:
            return self.response_cache.get(cache_key)

    def _add_to_cache(self, cache_key: str, recommendation: Any):
        """Add to cache with size management"""
        with self._cache_lock:
            self.response_cache[cache_key] = recommendation

            # Manage cache size (LRU-like)
            if len(self.response_cache) > self.max_cache_size:
                # Remove oldest 20%
                keys_to_remove = list(self.response_cache.keys())[:int(self.max_cache_size * 0.2)]
                for key in keys_to_remove:
                    del self.response_cache[key]

    def _get_api_response(self, prompt: str) -> str:
        """Get API response with retry logic"""
        max_retries = 2

        for attempt in range(max_retries):
            try:
                response = self.api_manager.get_ai_response(prompt, self.strategist_type)
                if response and response != '{}':
                    return response
            except Exception as e:
                self.logger.log(
                    f"API attempt {attempt + 1} failed: {e}",
                    "WARNING",
                    tags=['api', 'retry']
                )
                if attempt < max_retries - 1:
                    time.sleep(1)

        return "{}"

    def _get_fallback_response(self, df, game_info: Dict,
                               field_size: str, slate_profile: Dict) -> str:
        """Generate fallback response without API"""
        # This would be implemented by subclasses
        return "{}"

    # Abstract methods for subclasses
    def generate_prompt(self, df, game_info: Dict, field_size: str,
                       slate_profile: Dict) -> str:
        """Generate prompt - implemented by subclasses"""
        raise NotImplementedError

    def parse_response(self, response: str, df, field_size: str) -> Any:
        """Parse response - implemented by subclasses"""
        raise NotImplementedError

"""
Game Theory AI Strategist with enhanced ownership analysis
Improvements: Dynamic leverage calculation, field simulation, historical patterns
"""

class GPPGameTheoryStrategist(BaseAIStrategist):
    """Enhanced game theory strategist with ownership leverage optimization"""

    def __init__(self, api_manager=None):
        from config import AIStrategistType
        super().__init__(api_manager, AIStrategistType.GAME_THEORY)

        # Game theory specific
        self.ownership_models = {}
        self.leverage_cache = {}
        self.field_simulations = deque(maxlen=20)
        self.exploit_patterns = defaultdict(list)

    def generate_prompt(self, df, game_info: Dict, field_size: str,
                       slate_profile: Dict) -> str:
        """Generate game theory focused prompt with ownership analysis"""

        self.logger.log(
            f"Generating Game Theory prompt for {field_size}",
            "DEBUG",
            tags=['prompt', 'game_theory']
        )

        if df.empty:
            return "Error: Empty player pool"

        # Enhanced ownership analysis
        ownership_analysis = self._analyze_ownership_landscape(df, field_size)
        leverage_plays = self._calculate_leverage_plays(df, field_size)
        field_tendencies = self._predict_field_tendencies(df, slate_profile)

        # Get field-specific strategy
        from config import OptimizerConfig
        field_config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size, {})
        target_ownership = OptimizerConfig.GPP_OWNERSHIP_TARGETS.get(field_size, (50, 80))

        prompt = f"""You are an elite DFS game theory strategist optimizing for {field_size} GPP tournaments.

GAME CONTEXT:
Teams: {game_info.get('teams', 'Unknown')}
Total: {game_info.get('total', 45)} | Spread: {game_info.get('spread', 0)}
Weather: {game_info.get('weather', 'Clear')}
Slate Type: {slate_profile.get('slate_type', 'standard')}
Pricing Efficiency: {slate_profile.get('pricing_efficiency', 0.5):.2%}

OWNERSHIP LANDSCAPE ANALYSIS:
{ownership_analysis['summary']}

Mega Chalk (>35%): {len(ownership_analysis['mega_chalk'])} players
Chalk (20-35%): {len(ownership_analysis['chalk'])} players
Pivot Zone (10-20%): {len(ownership_analysis['pivot'])} players
Leverage (<10%): {len(ownership_analysis['leverage'])} players

HIGH LEVERAGE OPPORTUNITIES:
{self._format_leverage_plays(leverage_plays)}

FIELD TENDENCY PREDICTIONS:
{field_tendencies['summary']}
Expected Chalk Exposure: {field_tendencies['chalk_exposure']:.0%}
Expected Captain Concentration: {field_tendencies['captain_concentration']}

CONTRARIAN TARGETS:
{self._format_contrarian_targets(df, leverage_plays)}

TARGET OWNERSHIP RANGE: {target_ownership[0]}-{target_ownership[1]}%
Max Chalk Players: {field_config.get('max_chalk_players', 2)}
Min Leverage Players: {field_config.get('min_leverage_players', 2)}

PROVIDE ACTIONABLE GAME THEORY RULES IN JSON:
{{
    "captain_rules": {{
        "must_be_one_of": ["exact_player_names"],
        "ownership_ceiling": {target_ownership[1] * 0.2},
        "min_projection": 15,
        "leverage_score_min": {3.0 if 'large' in field_size else 2.0},
        "game_theory_rationale": "Specific ownership arbitrage explanation"
    }},
    "lineup_rules": {{
        "must_include": ["exact_player_names"],
        "never_include": ["exact_player_names"],
        "ownership_sum_range": {list(target_ownership)},
        "min_leverage_players": {field_config.get('min_leverage_players', 2)},
        "max_chalk_players": {field_config.get('max_chalk_players', 2)},
        "ownership_distribution": "How to distribute ownership across lineup"
    }},
    "exploitation_strategy": {{
        "field_blind_spot": "What the field is missing",
        "ownership_arbitrage": "Specific players where ownership doesn't match equity",
        "captain_leverage": "Why your captain beats chalk captains",
        "correlation_edge": "How your construction differs from field"
    }},
    "game_theory_model": {{
        "field_stack_rate": 0.75,
        "field_captain_pool": ["top_5_most_likely"],
        "your_captain_edge": "Your captain vs field captain comparison",
        "win_condition": "Specific scenario where your lineup wins"
    }},
    "confidence": 0.85,
    "key_insight": "The ONE ownership exploit that separates from field"
}}

Focus on quantifiable ownership edges and specific game theory advantages.
"""

        return prompt

    def _analyze_ownership_landscape(self, df, field_size: str) -> Dict:
        """Deep ownership analysis with market inefficiencies"""

        analysis = {
            'mega_chalk': [],
            'chalk': [],
            'pivot': [],
            'leverage': [],
            'summary': "",
            'inefficiencies': []
        }

        # Categorize by ownership
        for _, row in df.iterrows():
            player = row['Player']
            ownership = row.get('Ownership', 10)
            projection = row['Projected_Points']
            salary = row['Salary']

            # Calculate value
            value = projection / (salary / 1000)

            # Categorize
            if ownership >= 35:
                analysis['mega_chalk'].append(player)
            elif ownership >= 20:
                analysis['chalk'].append(player)
            elif ownership >= 10:
                analysis['pivot'].append(player)
            else:
                analysis['leverage'].append(player)

            # Find inefficiencies (high value, low ownership)
            if ownership < 15 and value > 3.0:
                analysis['inefficiencies'].append({
                    'player': player,
                    'ownership': ownership,
                    'value': value,
                    'projection': projection
                })

        # Create summary
        total_players = len(df)
        chalk_pct = (len(analysis['mega_chalk']) + len(analysis['chalk'])) / total_players * 100

        analysis['summary'] = f"""Ownership Structure: {chalk_pct:.0f}% of pool is chalk (>20% owned)
Market Inefficiencies: {len(analysis['inefficiencies'])} underpriced low-owned plays found
Leverage Opportunity Score: {len(analysis['leverage']) / total_players * 100:.0f}%"""

        return analysis

    def _calculate_leverage_plays(self, df, field_size: str) -> List[Dict]:
        """Calculate leverage score for each player"""

        leverage_plays = []

        for _, row in df.iterrows():
            player = row['Player']
            ownership = row.get('Ownership', 10)
            projection = row['Projected_Points']
            salary = row['Salary']

            # Leverage score: projection potential / ownership exposure
            leverage_score = (projection / df['Projected_Points'].max() * 100) / (ownership + 1)

            # Value component
            value = projection / (salary / 1000)

            # Combined leverage metric
            combined_leverage = leverage_score * 0.7 + value * 0.3

            if ownership < 15 and projection >= df['Projected_Points'].median():
                leverage_plays.append({
                    'player': player,
                    'ownership': ownership,
                    'projection': projection,
                    'leverage_score': combined_leverage,
                    'value': value,
                    'position': row['Position']
                })

        # Sort by leverage score
        leverage_plays.sort(key=lambda x: x['leverage_score'], reverse=True)

        return leverage_plays[:15]

    def _predict_field_tendencies(self, df, slate_profile: Dict) -> Dict:
        """Predict what the field will do"""

        tendencies = {
            'chalk_exposure': 0.0,
            'captain_concentration': 'high',
            'stack_preference': 'QB+WR1',
            'summary': ""
        }

        # Estimate chalk exposure based on slate type
        slate_type = slate_profile.get('slate_type', 'standard')

        if slate_type == 'shootout':
            tendencies['chalk_exposure'] = 0.75
            tendencies['captain_concentration'] = 'very_high'
            tendencies['stack_preference'] = 'QB+multiple_WR'
        elif slate_type == 'low_scoring':
            tendencies['chalk_exposure'] = 0.60
            tendencies['captain_concentration'] = 'moderate'
        else:
            tendencies['chalk_exposure'] = 0.65
            tendencies['captain_concentration'] = 'high'

        # Calculate captain concentration
        if 'Ownership' in df.columns:
            top_3_captain_own = df.nlargest(3, 'Projected_Points')['Ownership'].mean()
            if top_3_captain_own > 50:
                tendencies['captain_concentration'] = 'very_high'
            elif top_3_captain_own > 35:
                tendencies['captain_concentration'] = 'high'
            else:
                tendencies['captain_concentration'] = 'moderate'

        tendencies['summary'] = f"""Field will likely:
- Play {tendencies['chalk_exposure']:.0%} chalk overall
- Heavily concentrate captains ({tendencies['captain_concentration']} concentration)
- Favor {tendencies['stack_preference']} stacking patterns
- Overlook {slate_type} specific leverage plays"""

        return tendencies

    def _format_leverage_plays(self, leverage_plays: List[Dict]) -> str:
        """Format leverage plays for prompt"""
        if not leverage_plays:
            return "No significant leverage plays identified"

        lines = []
        for i, play in enumerate(leverage_plays[:10], 1):
            lines.append(
                f"{i}. {play['player']} ({play['position']}) - "
                f"{play['ownership']:.1f}% owned, "
                f"{play['projection']:.1f} proj, "
                f"Leverage: {play['leverage_score']:.2f}"
            )

        return "\n".join(lines)

    def _format_contrarian_targets(self, df, leverage_plays: List[Dict]) -> str:
        """Identify specific contrarian angles"""

        targets = []

        # Low-owned high-ceiling players
        for play in leverage_plays[:5]:
            if play['ownership'] < 10:
                targets.append(
                    f"• {play['player']}: Only {play['ownership']:.1f}% owned "
                    f"despite {play['projection']:.1f} point ceiling"
                )

        # Salary-based values
        if 'Value' not in df.columns:
            df['Value'] = df['Projected_Points'] / (df['Salary'] / 1000)

        high_value_low_own = df[
            (df.get('Ownership', 10) < 12) &
            (df['Value'] > df['Value'].quantile(0.75))
        ]

        for _, row in high_value_low_own.head(3).iterrows():
            targets.append(
                f"• {row['Player']}: {row['Value']:.2f} value "
                f"at only {row.get('Ownership', 10):.1f}% ownership"
            )

        return "\n".join(targets) if targets else "Standard leverage plays only"

    def parse_response(self, response: str, df, field_size: str) -> Any:
        """Parse game theory response with enhanced validation"""

        from data_models import AIRecommendation

        try:
            # Parse JSON
            if response and response != '{}':
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    self.logger.log(
                        "Failed to parse JSON, using text extraction",
                        "WARNING",
                        tags=['parse', 'fallback']
                    )
                    data = self._extract_from_text_response(response, df)
            else:
                data = {}

            available_players = set(df['Player'].values)

            # Extract and validate captain rules
            captain_rules = data.get('captain_rules', {})
            captain_targets = captain_rules.get('must_be_one_of', [])
            valid_captains = [c for c in captain_targets if c in available_players]

            # If insufficient captains, use game theory selection
            if len(valid_captains) < 3:
                valid_captains = self._select_game_theory_captains(df, field_size)

            # Extract lineup rules
            lineup_rules = data.get('lineup_rules', {})
            must_include = [p for p in lineup_rules.get('must_include', []) if p in available_players]
            never_include = [p for p in lineup_rules.get('never_include', []) if p in available_players]
            ownership_range = lineup_rules.get('ownership_sum_range', [60, 90])

            # Extract exploitation strategy
            exploitation = data.get('exploitation_strategy', {})

            # Build insights
            key_insights = [
                data.get('key_insight', 'Ownership leverage strategy'),
                exploitation.get('ownership_arbitrage', ''),
                exploitation.get('captain_leverage', '')
            ]
            key_insights = [i for i in key_insights if i][:3]

            # Build enforcement rules
            enforcement_rules = self._build_game_theory_enforcement(
                valid_captains, must_include, never_include,
                ownership_range, lineup_rules, field_size
            )

            confidence = data.get('confidence', 0.75)
            confidence = max(0.0, min(1.0, confidence))

            # Create recommendation
            recommendation = AIRecommendation(
                captain_targets=valid_captains[:7],
                must_play=must_include[:5],
                never_play=never_include[:5],
                stacks=[],  # Game theory doesn't focus on stacks
                key_insights=key_insights,
                confidence=confidence,
                enforcement_rules=enforcement_rules,
                narrative=exploitation.get('field_blind_spot', 'Game theory optimization'),
                source_ai=self.strategist_type,
                ownership_leverage={
                    'ownership_range': ownership_range,
                    'ownership_ceiling': captain_rules.get('ownership_ceiling', 15),
                    'min_leverage': lineup_rules.get('min_leverage_players', 2),
                    'max_chalk': lineup_rules.get('max_chalk_players', 2),
                    'leverage_score_min': captain_rules.get('leverage_score_min', 3.0)
                }
            )

            return recommendation

        except Exception as e:
            self.logger.log_exception(e, "parse_game_theory_response")
            return self._get_fallback_recommendation(df, field_size)

    def _select_game_theory_captains(self, df, field_size: str) -> List[str]:
        """Select captains using game theory principles"""

        from config import OptimizerConfig

        target_ownership = OptimizerConfig.GPP_OWNERSHIP_TARGETS.get(field_size, (50, 80))
        max_captain_own = target_ownership[1] * 0.25  # 25% of max total ownership

        # Calculate leverage score for each potential captain
        df_copy = df.copy()
        df_copy['leverage'] = (
            df_copy['Projected_Points'] / df_copy['Projected_Points'].max() * 100 /
            (df_copy.get('Ownership', 10) + 5)
        )

        # Filter by ownership ceiling
        eligible = df_copy[df_copy.get('Ownership', 10) <= max_captain_own]

        if len(eligible) < 5:
            eligible = df_copy[df_copy.get('Ownership', 10) <= max_captain_own * 1.5]

        # Sort by leverage
        captains = eligible.nlargest(7, 'leverage')['Player'].tolist()

        return captains

    def _build_game_theory_enforcement(self, captains: List[str], must_include: List[str],
                                      never_include: List[str], ownership_range: Tuple,
                                      lineup_rules: Dict, field_size: str) -> List[Dict]:
        """Build game theory specific rules"""

        rules = []

        # Captain constraint (highest priority)
        if captains:
            rules.append({
                'type': 'hard',
                'constraint': 'game_theory_captain',
                'players': captains[:5],
                'priority': 100,
                'description': 'Game theory optimal captains',
                'rationale': 'Ownership leverage maximization'
            })

        # Ownership constraints
        rules.append({
            'type': 'hard',
            'constraint': 'ownership_sum',
            'min': ownership_range[0],
            'max': ownership_range[1],
            'priority': 95,
            'description': f'Total ownership {ownership_range[0]}-{ownership_range[1]}%'
        })

        # Leverage requirements
        min_leverage = lineup_rules.get('min_leverage_players', 2)
        if min_leverage > 0:
            rules.append({
                'type': 'hard',
                'constraint': 'min_leverage',
                'count': min_leverage,
                'ownership_threshold': 10,
                'priority': 85,
                'description': f'Minimum {min_leverage} leverage plays'
            })

        # Chalk limitations
        max_chalk = lineup_rules.get('max_chalk_players', 2)
        rules.append({
            'type': 'hard',
            'constraint': 'max_chalk',
            'count': max_chalk,
            'ownership_threshold': 25,
            'priority': 80,
            'description': f'Maximum {max_chalk} chalk players'
        })

        # Must include
        for i, player in enumerate(must_include[:3]):
            rules.append({
                'type': 'hard',
                'constraint': 'must_include',
                'player': player,
                'priority': 90 - i * 5,
                'description': f'Must include {player} (game theory edge)'
            })

        # Never include (fades)
        for i, player in enumerate(never_include[:3]):
            rules.append({
                'type': 'hard',
                'constraint': 'must_exclude',
                'player': player,
                'priority': 85 - i * 5,
                'description': f'Fade chalk: {player}'
            })

        return rules

    def _extract_from_text_response(self, response: str, df) -> Dict:
        """Extract game theory data from text"""

        data = {
            'captain_rules': {'must_be_one_of': []},
            'lineup_rules': {'ownership_sum_range': [60, 90]},
            'confidence': 0.6
        }

        lines = response.lower().split('\n')

        # Extract ownership numbers
        for line in lines:
            if 'captain' in line and any(char.isdigit() for char in line):
                numbers = [int(s) for s in line.split() if s.isdigit()]
                if numbers:
                    data['captain_rules']['ownership_ceiling'] = min(numbers)

            if 'leverage' in line and 'players' in line:
                numbers = [int(s) for s in line.split() if s.isdigit()]
                if numbers:
                    data['lineup_rules']['min_leverage_players'] = max(1, min(numbers))

        return data

"""
Correlation AI Strategist with advanced stacking analysis
Improvements: Game script correlation, multi-way stacks, bring-back optimization, negative correlation detection
"""

class GPPCorrelationStrategist(BaseAIStrategist):
    """Enhanced correlation strategist with game script analysis"""

    def __init__(self, api_manager=None):
        from config import AIStrategistType
        super().__init__(api_manager, AIStrategistType.CORRELATION)

        # Correlation specific
        self.correlation_matrix = {}
        self.stack_history = deque(maxlen=100)
        self.game_script_models = {}
        self.position_correlations = self._initialize_position_correlations()

    def _initialize_position_correlations(self) -> Dict:
        """Initialize known position correlations"""
        return {
            ('QB', 'WR'): 0.7,
            ('QB', 'TE'): 0.65,
            ('QB', 'RB'): 0.15,  # Pass-catching RB
            ('WR', 'WR'): -0.3,  # Same team WRs compete
            ('RB', 'RB'): -0.5,  # Same backfield
            ('DST', 'QB'): -0.4,  # Opposing QB
            ('DST', 'WR'): -0.35  # Opposing pass catchers
        }

    def generate_prompt(self, df, game_info: Dict, field_size: str,
                       slate_profile: Dict) -> str:
        """Generate correlation-focused prompt with game script analysis"""

        self.logger.log(
            f"Generating Correlation prompt for {field_size}",
            "DEBUG",
            tags=['prompt', 'correlation']
        )

        if df.empty:
            return "Error: Empty player pool"

        # Team analysis
        teams = df['Team'].unique()[:2]
        team1 = teams[0] if len(teams) > 0 else "Team1"
        team2 = teams[1] if len(teams) > 1 else "Team2"

        team1_df = df[df['Team'] == team1]
        team2_df = df[df['Team'] == team2]

        # Analyze correlation opportunities
        qb_stacks = self._identify_qb_stacks(df, team1, team2)
        onslaught_potential = self._analyze_onslaught_stacks(df, game_info)
        bring_back_opportunities = self._identify_bring_back_stacks(df, game_info)
        negative_correlations = self._identify_negative_correlations(df)

        # Game script analysis
        game_script = self._analyze_game_script(game_info, slate_profile)

        # Determine game flow expectations
        total = game_info.get('total', 45)
        spread = game_info.get('spread', 0)
        favorite = team1 if spread < 0 else team2
        underdog = team2 if favorite == team1 else team1

        prompt = f"""You are an expert DFS correlation strategist. Create SPECIFIC stacking rules for {field_size} GPP.

GAME ENVIRONMENT:
Total: {total} | Spread: {spread}
Favorite: {favorite} by {abs(spread)} points
Underdog: {underdog}
Slate Type: {slate_profile.get('slate_type', 'standard')}
Expected Game Script: {game_script['script_type']}

{game_script['analysis']}

TEAM 1 - {team1} ({'Favorite' if team1 == favorite else 'Underdog'}):
{self._format_team_data(team1_df)}

TEAM 2 - {team2} ({'Favorite' if team2 == favorite else 'Underdog'}):
{self._format_team_data(team2_df)}

PRIMARY STACK OPPORTUNITIES:
{self._format_stack_opportunities(qb_stacks)}

ONSLAUGHT STACK ANALYSIS (3-4+ same team):
{onslaught_potential['summary']}

BRING-BACK OPPORTUNITIES:
{bring_back_opportunities['summary']}

NEGATIVE CORRELATIONS TO AVOID:
{self._format_negative_correlations(negative_correlations)}

CREATE ADVANCED CORRELATION RULES IN JSON:
{{
    "primary_stacks": [
        {{
            "type": "QB_WR1",
            "player1": "exact_qb_name",
            "player2": "exact_wr_name",
            "correlation": 0.7,
            "game_script_dependency": "standard|shootout|blowout",
            "narrative": "Why this stack wins in this game environment"
        }}
    ],
    "onslaught_stacks": [
        {{
            "team": "winning_team",
            "players": ["qb", "wr1", "wr2", "rb"],
            "min_required": 3,
            "scenario": "blowout_correlation",
            "total_requirement": {total},
            "spread_requirement": {abs(spread)}
        }}
    ],
    "bring_back_stacks": [
        {{
            "primary": ["favored_qb", "favored_wr1"],
            "bring_back": "underdog_wr1",
            "game_total_min": {total - 5},
            "correlation_strength": 0.5,
            "narrative": "High-scoring game theory"
        }}
    ],
    "leverage_stacks": [
        {{
            "type": "contrarian",
            "player1": "low_own_qb",
            "player2": "low_own_wr",
            "combined_ownership_max": 20,
            "leverage_rationale": "Why field misses this correlation"
        }}
    ],
    "negative_correlation": [
        {{
            "avoid_together": ["player_a", "player_b"],
            "reason": "target_competition|backfield_share|opposing_offense"
        }}
    ],
    "game_script_stacks": {{
        "shootout": ["qb1", "wr1", "opp_wr1"],
        "blowout": ["fav_qb", "fav_wr1", "fav_rb", "fav_wr2"],
        "defensive": ["dst", "opposing_players_limited"],
        "balanced": ["balanced_attack_both_teams"]
    }},
    "captain_correlation": {{
        "best_captains_for_stacking": ["player_names"],
        "captain_stack_pairs": [
            {{"captain": "player", "pair_with": ["teammates"]}}
        ],
        "bring_back_captains": ["opposing_team_players"]
    }},
    "advanced_correlations": {{
        "three_way_stacks": [
            {{"players": ["qb", "wr1", "wr2"], "strength": 0.6}}
        ],
        "rb_passing_game": [
            {{"qb": "qb_name", "pass_catching_rb": "rb_name", "correlation": 0.5}}
        ],
        "te_correlation": [
            {{"qb": "qb_name", "te": "te_name", "game_script": "trailing"}}
        ]
    }},
    "confidence": 0.8,
    "stack_narrative": "Primary correlation thesis for this slate"
}}

Provide exact player names and focus on correlations that maximize ceiling through game script.
"""

        return prompt

    def _analyze_game_script(self, game_info: Dict, slate_profile: Dict) -> Dict:
        """Analyze expected game script and implications"""

        total = game_info.get('total', 45)
        spread = abs(game_info.get('spread', 0))
        slate_type = slate_profile.get('slate_type', 'standard')

        script = {
            'script_type': 'balanced',
            'analysis': '',
            'favorite_script': 'standard',
            'underdog_script': 'standard'
        }

        # Determine script
        if total > 50 and spread < 3:
            script['script_type'] = 'shootout'
            script['favorite_script'] = 'passing_attack'
            script['underdog_script'] = 'passing_attack'
            script['analysis'] = f"""SHOOTOUT EXPECTED (O/U {total}, Spread {spread}):
- Both teams will throw extensively
- High pass attempt correlation
- Bring-back stacks highly viable
- Multiple WR correlation opportunities
- RB usage may be limited"""

        elif spread > 10:
            script['script_type'] = 'blowout_risk'
            script['favorite_script'] = 'run_heavy_late'
            script['underdog_script'] = 'pass_heavy_catchup'
            script['analysis'] = f"""BLOWOUT RISK (Spread {spread}):
- Favorite: Early passing, late game run-heavy
- Underdog: Pass-heavy throughout (negative game script)
- Onslaught stack potential for favorite
- Underdog pass catchers in garbage time
- Favorite RB value in 2nd half"""

        elif total < 42:
            script['script_type'] = 'low_scoring'
            script['favorite_script'] = 'clock_control'
            script['underdog_script'] = 'efficient_passing'
            script['analysis'] = f"""LOW-SCORING GAME (O/U {total}):
- Defensive game or weather impact
- TD-dependent players have leverage
- RB correlation increases
- Lower pass attempt volume
- Target dominant pieces only"""

        else:
            script['analysis'] = f"""BALANCED GAME SCRIPT (O/U {total}, Spread {spread}):
- Standard offensive approaches expected
- Both pass and run viable
- Traditional QB+WR1 stacks optimal
- Bring-back with top WR from opposing team
- Flex spots for high-upside pieces"""

        return script

    def _identify_qb_stacks(self, df, team1: str, team2: str) -> List[Dict]:
        """Identify primary QB stacking opportunities"""

        qb_stacks = []

        # For each QB, find top pass catchers
        qbs = df[df['Position'] == 'QB']

        for _, qb_row in qbs.iterrows():
            qb = qb_row['Player']
            team = qb_row['Team']
            qb_proj = qb_row['Projected_Points']

            # Find teammates who are pass catchers
            teammates = df[
                (df['Team'] == team) &
                (df['Position'].isin(['WR', 'TE']))
            ].nlargest(4, 'Projected_Points')

            for _, teammate_row in teammates.iterrows():
                correlation_strength = self._calculate_correlation_strength(
                    qb_row, teammate_row
                )

                qb_stacks.append({
                    'qb': qb,
                    'receiver': teammate_row['Player'],
                    'qb_projection': qb_proj,
                    'receiver_projection': teammate_row['Projected_Points'],
                    'combined_projection': qb_proj + teammate_row['Projected_Points'],
                    'correlation': correlation_strength,
                    'team': team,
                    'position': teammate_row['Position']
                })

        # Sort by combined projection and correlation
        qb_stacks.sort(
            key=lambda x: x['combined_projection'] * x['correlation'],
            reverse=True
        )

        return qb_stacks[:8]

    def _calculate_correlation_strength(self, player1_row, player2_row) -> float:
        """Calculate correlation strength between two players"""

        pos1 = player1_row['Position']
        pos2 = player2_row['Position']
        team1 = player1_row['Team']
        team2 = player2_row['Team']

        # Same team base correlation
        if team1 == team2:
            base_corr = self.position_correlations.get((pos1, pos2), 0.3)
        else:
            # Opposing teams have negative correlation generally
            base_corr = self.position_correlations.get((pos1, pos2), -0.2)

        # Adjust for salary (higher salary = more correlation for positive stacks)
        if base_corr > 0:
            salary_factor = (
                (player1_row['Salary'] + player2_row['Salary']) / 20000
            )
            base_corr *= (0.8 + salary_factor * 0.2)

        return max(-1.0, min(1.0, base_corr))

    def _analyze_onslaught_stacks(self, df, game_info: Dict) -> Dict:
        """Analyze 3-4 player same-team stack potential"""

        analysis = {
            'viable': False,
            'summary': '',
            'opportunities': []
        }

        spread = abs(game_info.get('spread', 0))
        total = game_info.get('total', 45)

        # Onslaught stacks work best in blowouts or high totals
        if spread > 7 or total > 52:
            analysis['viable'] = True

            # Analyze each team
            for team in df['Team'].unique():
                team_df = df[df['Team'] == team].nlargest(5, 'Projected_Points')

                if len(team_df) >= 3:
                    players = team_df['Player'].tolist()
                    combined_proj = team_df['Projected_Points'].sum()

                    analysis['opportunities'].append({
                        'team': team,
                        'players': players,
                        'combined_projection': combined_proj,
                        'scenario': 'blowout' if spread > 7 else 'shootout'
                    })

            analysis['summary'] = f"""Onslaught stacks VIABLE:
- Game script supports concentrated roster (spread {spread}, total {total})
- {len(analysis['opportunities'])} teams with 3-4 stack potential
- Best for: {"Favorite in blowout" if spread > 7 else "Either team in shootout"}
- Correlation strength: ~0.55-0.60 for 3-way, ~0.50 for 4-way"""
        else:
            analysis['summary'] = "Onslaught stacks RISKY: Game script doesn't support heavy concentration"

        return analysis

    def _identify_bring_back_stacks(self, df, game_info: Dict) -> Dict:
        """Identify bring-back stack opportunities"""

        opportunities = {
            'viable': False,
            'summary': '',
            'stacks': []
        }

        total = game_info.get('total', 45)

        # Bring-backs work in high-scoring games
        if total >= 48:
            opportunities['viable'] = True

            teams = df['Team'].unique()
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]

                # Primary stack candidates from team 1
                team1_qbs = df[(df['Team'] == team1) & (df['Position'] == 'QB')]
                team1_pass_catchers = df[
                    (df['Team'] == team1) &
                    (df['Position'].isin(['WR', 'TE']))
                ].nlargest(3, 'Projected_Points')

                # Bring-back candidates from team 2
                team2_pass_catchers = df[
                    (df['Team'] == team2) &
                    (df['Position'].isin(['WR', 'TE']))
                ].nlargest(3, 'Projected_Points')

                # Create bring-back combinations
                for _, qb_row in team1_qbs.iterrows():
                    for _, primary_row in team1_pass_catchers.iterrows():
                        for _, bringback_row in team2_pass_catchers.iterrows():
                            opportunities['stacks'].append({
                                'primary_qb': qb_row['Player'],
                                'primary_receiver': primary_row['Player'],
                                'bring_back': bringback_row['Player'],
                                'combined_projection': (
                                    qb_row['Projected_Points'] +
                                    primary_row['Projected_Points'] +
                                    bringback_row['Projected_Points']
                                )
                            })

            opportunities['summary'] = f"""Bring-back stacks RECOMMENDED (Total: {total}):
- High game total supports bring-back correlation
- {len(opportunities['stacks'])} viable combinations identified
- Correlation strength: ~0.45-0.50
- Protects against one-sided blowout
- Increases ceiling if both offenses produce"""
        else:
            opportunities['summary'] = f"Bring-back stacks QUESTIONABLE: Total {total} may not support dual correlation"

        return opportunities

    def _identify_negative_correlations(self, df) -> List[Dict]:
        """Identify player pairs with negative correlation"""

        negative_pairs = []

        # Same position, same team (especially RB and WR)
        for team in df['Team'].unique():
            team_df = df[df['Team'] == team]

            # Same backfield RBs
            rbs = team_df[team_df['Position'] == 'RB']
            if len(rbs) >= 2:
                for i, rb1 in rbs.iterrows():
                    for j, rb2 in rbs.iterrows():
                        if i < j:
                            negative_pairs.append({
                                'player1': rb1['Player'],
                                'player2': rb2['Player'],
                                'reason': 'same_backfield',
                                'correlation': -0.5,
                                'avoid_together': True
                            })

            # Same team WRs (target competition)
            wrs = team_df[team_df['Position'] == 'WR']
            if len(wrs) >= 3:
                top_wrs = wrs.nlargest(3, 'Projected_Points')
                wr_list = top_wrs['Player'].tolist()
                if len(wr_list) >= 3:
                    negative_pairs.append({
                        'player1': wr_list[0],
                        'player2': wr_list[2],  # WR1 and WR3
                        'reason': 'target_competition',
                        'correlation': -0.25,
                        'avoid_together': False  # Can stack but not ideal
                    })

        # Opposing QB and DST
        for team in df['Team'].unique():
            opposing_team = [t for t in df['Team'].unique() if t != team][0] if len(df['Team'].unique()) > 1 else None

            if opposing_team:
                team_qb = df[(df['Team'] == team) & (df['Position'] == 'QB')]
                opp_dst = df[(df['Team'] == opposing_team) & (df['Position'] == 'DST')]

                if not team_qb.empty and not opp_dst.empty:
                    negative_pairs.append({
                        'player1': team_qb.iloc[0]['Player'],
                        'player2': opp_dst.iloc[0]['Player'],
                        'reason': 'opposing_dst',
                        'correlation': -0.4,
                        'avoid_together': True
                    })

        return negative_pairs[:10]

    def _format_team_data(self, team_df) -> str:
        """Format team data for prompt"""
        if team_df.empty:
            return "No data"

        top_players = team_df.nlargest(8, 'Projected_Points')
        lines = []

        for _, row in top_players.iterrows():
            lines.append(
                f"{row['Player']} ({row['Position']}) - "
                f"${row['Salary']:,} - {row['Projected_Points']:.1f} pts"
            )

        return "\n".join(lines)

    def _format_stack_opportunities(self, qb_stacks: List[Dict]) -> str:
        """Format QB stacking opportunities"""
        if not qb_stacks:
            return "No clear QB stacks identified"

        lines = []
        for i, stack in enumerate(qb_stacks[:5], 1):
            lines.append(
                f"{i}. {stack['qb']} + {stack['receiver']} ({stack['position']}) - "
                f"Combined: {stack['combined_projection']:.1f} pts, "
                f"Correlation: {stack['correlation']:.2f}"
            )

        return "\n".join(lines)

    def _format_negative_correlations(self, negative_pairs: List[Dict]) -> str:
        """Format negative correlations"""
        if not negative_pairs:
            return "No significant negative correlations"

        lines = []
        for pair in negative_pairs[:5]:
            lines.append(
                f"• Avoid: {pair['player1']} + {pair['player2']} "
                f"({pair['reason']}, correlation: {pair['correlation']:.2f})"
            )

        return "\n".join(lines)

    def parse_response(self, response: str, df, field_size: str) -> Any:
        """Parse correlation response with stack validation"""

        from data_models import AIRecommendation

        try:
            if response and response != '{}':
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    self.logger.log(
                        "Failed to parse JSON, using correlation extraction",
                        "WARNING",
                        tags=['parse', 'fallback']
                    )
                    data = self._extract_correlation_from_text(response, df)
            else:
                data = {}

            available_players = set(df['Player'].values)
            all_stacks = []

            # Process primary stacks
            for stack in data.get('primary_stacks', []):
                if self._validate_stack_players(stack, available_players):
                    stack['priority'] = 'high'
                    stack['enforced'] = True
                    all_stacks.append(stack)

            # Process onslaught stacks
            for onslaught in data.get('onslaught_stacks', []):
                players = onslaught.get('players', [])
                valid_players = [p for p in players if p in available_players]

                if len(valid_players) >= 3:
                    all_stacks.append({
                        'type': 'onslaught',
                        'players': valid_players,
                        'team': onslaught.get('team', ''),
                        'min_required': onslaught.get('min_required', 3),
                        'scenario': onslaught.get('scenario', 'Blowout correlation'),
                        'priority': 'high',
                        'correlation': 0.6
                    })

            # Process bring-back stacks
            for bring_back in data.get('bring_back_stacks', []):
                primary = bring_back.get('primary', [])
                opponent = bring_back.get('bring_back')

                valid_primary = [p for p in primary if p in available_players]

                if valid_primary and opponent in available_players:
                    all_stacks.append({
                        'type': 'bring_back',
                        'primary_stack': valid_primary,
                        'bring_back': opponent,
                        'game_total': bring_back.get('game_total_min', 45),
                        'priority': 'high',
                        'correlation': 0.5
                    })

            # Process leverage stacks
            for stack in data.get('leverage_stacks', []):
                if self._validate_stack_players(stack, available_players):
                    stack['priority'] = 'medium'
                    stack['leverage'] = True
                    all_stacks.append(stack)

            # If no valid stacks, create statistical ones
            if len(all_stacks) < 2:
                all_stacks.extend(self._create_statistical_stacks(df))

            # Extract captain correlation
            captain_rules = data.get('captain_correlation', {})
            captain_targets = captain_rules.get('best_captains_for_stacking', [])
            valid_captains = [c for c in captain_targets if c in available_players]

            if len(valid_captains) < 3:
                valid_captains.extend(self._get_correlation_captains(df, all_stacks))
                valid_captains = list(dict.fromkeys(valid_captains))[:7]

            # Process negative correlations
            avoid_pairs = []
            for neg_corr in data.get('negative_correlation', []):
                players = neg_corr.get('avoid_together', [])
                if len(players) >= 2 and all(p in available_players for p in players[:2]):
                    avoid_pairs.append({
                        'players': players[:2],
                        'reason': neg_corr.get('reason', 'negative correlation')
                    })

            # Build enforcement rules
            enforcement_rules = self._build_correlation_enforcement_rules(
                all_stacks, avoid_pairs, valid_captains
            )

            # Build correlation matrix
            self.correlation_matrix = self._build_correlation_matrix(all_stacks, avoid_pairs)

            confidence = data.get('confidence', 0.75)
            confidence = max(0.0, min(1.0, confidence))

            # Extract insights
            key_insights = [
                data.get('stack_narrative', 'Correlation-based construction'),
                f"Primary focus: {all_stacks[0]['type'] if all_stacks else 'standard'} stacks",
                f"{len(all_stacks)} correlation plays identified"
            ]

            recommendation = AIRecommendation(
                captain_targets=valid_captains,
                must_play=[],
                never_play=[],
                stacks=all_stacks[:10],
                key_insights=key_insights,
                confidence=confidence,
                enforcement_rules=enforcement_rules,
                narrative=data.get('stack_narrative', 'Correlation optimization'),
                source_ai=self.strategist_type,
                correlation_matrix=self.correlation_matrix
            )

            return recommendation

        except Exception as e:
            self.logger.log_exception(e, "parse_correlation_response")
            return self._get_fallback_recommendation(df, field_size)

    def _validate_stack_players(self, stack: Dict, available_players: Set[str]) -> bool:
        """Validate stack has available players"""
        player1 = stack.get('player1')
        player2 = stack.get('player2')

        return (player1 and player2 and
                player1 in available_players and
                player2 in available_players)

    def _create_statistical_stacks(self, df) -> List[Dict]:
        """Create correlation stacks using statistical analysis"""
        stacks = []

        try:
            qbs = df[df['Position'] == 'QB']

            for _, qb in qbs.iterrows():
                team = qb['Team']
                teammates = df[
                    (df['Team'] == team) &
                    (df['Position'].isin(['WR', 'TE']))
                ]

                if not teammates.empty:
                    top_teammates = teammates.nlargest(3, 'Projected_Points')

                    for i, (_, teammate) in enumerate(top_teammates.iterrows()):
                        correlation_strength = 0.7 - (i * 0.1)
                        stacks.append({
                            'player1': qb['Player'],
                            'player2': teammate['Player'],
                            'type': f"QB_{teammate['Position']}",
                            'correlation': correlation_strength,
                            'priority': 'high' if i == 0 else 'medium'
                        })

                # Bring-back stacks
                opponents = df[
                    (df['Team'] != team) &
                    (df['Position'].isin(['WR', 'TE']))
                ]
                if not opponents.empty:
                    top_opponent = opponents.nlargest(1, 'Projected_Points').iloc[0]
                    stacks.append({
                        'type': 'bring_back',
                        'primary_stack': [qb['Player'], top_teammates.iloc[0]['Player']] if not teammates.empty else [qb['Player']],
                        'bring_back': top_opponent['Player'],
                        'correlation': 0.5,
                        'priority': 'medium'
                    })

        except Exception as e:
            self.logger.log(f"Error creating statistical stacks: {e}", "WARNING")

        return stacks[:8]

    def _get_correlation_captains(self, df, stacks: List[Dict]) -> List[str]:
        """Get captain targets based on correlation analysis"""
        captains = []

        # Prioritize QBs in stacks
        for stack in stacks:
            if stack.get('type') in ['QB_WR', 'QB_TE', 'primary']:
                player1 = stack.get('player1')
                if player1:
                    player_data = df[df['Player'] == player1]
                    if not player_data.empty and player_data.iloc[0]['Position'] == 'QB':
                        if player1 not in captains:
                            captains.append(player1)

        # Add top pass catchers
        for stack in stacks[:5]:
            if 'player2' in stack:
                if stack['player2'] not in captains:
                    captains.append(stack['player2'])

        # Add bring-back targets
        for stack in stacks:
            if stack.get('type') == 'bring_back':
                bring_back = stack.get('bring_back')
                if bring_back and bring_back not in captains:
                    captains.append(bring_back)

        return captains[:7]

    def _build_correlation_enforcement_rules(self, stacks: List[Dict],
                                            avoid_pairs: List[Dict],
                                            captains: List[str]) -> List[Dict]:
        """Build correlation enforcement rules"""
        rules = []

        # Captain rules
        if captains:
            rules.append({
                'type': 'hard',
                'constraint': 'correlation_captain',
                'players': captains[:5],
                'priority': 95,
                'description': 'Correlation-optimized captains'
            })

        # High priority stacks
        high_priority_stacks = [s for s in stacks if s.get('priority') == 'high'][:3]

        for i, stack in enumerate(high_priority_stacks):
            if stack.get('type') == 'onslaught':
                rules.append({
                    'type': 'hard' if i == 0 else 'soft',
                    'constraint': 'onslaught_stack',
                    'players': stack['players'][:4],
                    'min_players': stack.get('min_required', 3),
                    'weight': 0.9 if i > 0 else 1.0,
                    'priority': 90 - (i * 5),
                    'description': f"Onslaught: {stack.get('team', 'team')} correlation"
                })

            elif stack.get('type') == 'bring_back':
                rules.append({
                    'type': 'hard' if i == 0 else 'soft',
                    'constraint': 'bring_back_stack',
                    'primary': stack.get('primary_stack', []),
                    'bring_back': stack.get('bring_back'),
                    'weight': 0.8 if i > 0 else 1.0,
                    'priority': 85 - (i * 5),
                    'description': 'Bring-back correlation'
                })

            else:
                players = []
                if 'player1' in stack and 'player2' in stack:
                    players = [stack['player1'], stack['player2']]

                if players:
                    rules.append({
                        'type': 'hard' if i == 0 else 'soft',
                        'constraint': 'correlation_stack',
                        'players': players,
                        'correlation': stack.get('correlation', 0.5),
                        'weight': 0.8 if i > 0 else 1.0,
                        'priority': 85 - (i * 5),
                        'description': f"Stack: {stack.get('type', 'correlation')}"
                    })

        # Negative correlations
        for avoid in avoid_pairs[:3]:
            rules.append({
                'type': 'soft',
                'constraint': 'avoid_together',
                'players': avoid['players'],
                'weight': 0.7,
                'priority': 50,
                'description': avoid['reason']
            })

        return rules

    def _build_correlation_matrix(self, stacks: List[Dict], avoid_pairs: List[Dict]) -> Dict:
        """Build correlation matrix"""
        matrix = {}

        # Positive correlations from stacks
        for stack in stacks:
            if 'player1' in stack and 'player2' in stack:
                key = f"{stack['player1']}_{stack['player2']}"
                matrix[key] = stack.get('correlation', 0.5)

            elif stack.get('type') == 'onslaught' and 'players' in stack:
                players = stack['players']
                for i in range(len(players)):
                    for j in range(i+1, len(players)):
                        key = f"{players[i]}_{players[j]}"
                        matrix[key] = 0.4

        # Negative correlations
        for avoid in avoid_pairs:
            players = avoid['players']
            if len(players) >= 2:
                key = f"{players[0]}_{players[1]}"
                matrix[key] = -0.5

        return matrix

    def _extract_correlation_from_text(self, response: str, df) -> Dict:
        """Extract correlation data from text"""
        data = {
            'primary_stacks': [],
            'confidence': 0.6
        }

        # Look for QB-receiver mentions
        qbs = df[df['Position'] == 'QB']['Player'].tolist()
        receivers = df[df['Position'].isin(['WR', 'TE'])]['Player'].tolist()

        for qb in qbs:
            for receiver in receivers:
                if qb in response and receiver in response:
                    qb_team = df[df['Player'] == qb]['Team'].values
                    rec_team = df[df['Player'] == receiver]['Team'].values

                    if len(qb_team) > 0 and len(rec_team) > 0 and qb_team[0] == rec_team[0]:
                        data['primary_stacks'].append({
                            'player1': qb,
                            'player2': receiver,
                            'type': 'QB_stack',
                            'correlation': 0.6
                        })
                        break

        return data

"""
Contrarian Narrative AI Strategist with pattern recognition
Improvements: Narrative detection, pivot analysis, chalk fade strategies, hidden value identification
"""

class GPPContrarianNarrativeStrategist(BaseAIStrategist):
    """Enhanced contrarian strategist with narrative pattern recognition"""

    def __init__(self, api_manager=None):
        from config import AIStrategistType
        super().__init__(api_manager, AIStrategistType.CONTRARIAN_NARRATIVE)

        # Contrarian specific
        self.narrative_patterns = defaultdict(list)
        self.chalk_fade_history = deque(maxlen=100)
        self.pivot_success_rate = {}
        self.hidden_correlations = []

    def generate_prompt(self, df, game_info: Dict, field_size: str,
                       slate_profile: Dict) -> str:
        """Generate contrarian narrative focused prompt"""

        self.logger.log(
            f"Generating Contrarian Narrative prompt for {field_size}",
            "DEBUG",
            tags=['prompt', 'contrarian']
        )

        if df.empty:
            return "Error: Empty player pool"

        # Calculate contrarian opportunities
        df_copy = df.copy()
        df_copy['Value'] = df_copy['Projected_Points'] / (df_copy['Salary'] / 1000)
        df_copy['Contrarian_Score'] = self._calculate_contrarian_score(df_copy)

        # Find specific angles
        low_owned_ceiling = self._identify_ceiling_plays(df_copy)
        hidden_value = self._identify_hidden_value(df_copy)
        contrarian_captains = df_copy.nlargest(10, 'Contrarian_Score')
        chalk_to_fade = self._identify_chalk_fades(df_copy, slate_profile)
        leverage_scenarios = self._identify_leverage_scenarios(df_copy, game_info)
        pivot_opportunities = self._identify_pivot_opportunities(df_copy)

        teams = df['Team'].unique()[:2]
        total = game_info.get('total', 45)
        spread = game_info.get('spread', 0)

        prompt = f"""You are a contrarian DFS strategist who finds NON-OBVIOUS narratives that win GPP tournaments.

GAME SETUP:
{teams[0] if len(teams) > 0 else 'Team1'} vs {teams[1] if len(teams) > 1 else 'Team2'}
Total: {total} | Spread: {spread}
Slate Type: {slate_profile.get('slate_type', 'standard')}
Chalk Concentration: {slate_profile.get('chalk_concentration', 0.5):.0%}

CONTRARIAN OPPORTUNITIES:

LOW-OWNED HIGH CEILING (<10% owned):
{self._format_ceiling_plays(low_owned_ceiling)}

HIDDEN VALUE PLAYS:
{self._format_hidden_value(hidden_value)}

CONTRARIAN CAPTAIN CANDIDATES:
{self._format_contrarian_captains(contrarian_captains)}

CHALK TO FADE (>30% owned):
{self._format_chalk_fades(chalk_to_fade)}

LEVERAGE GAME SCRIPTS:
{self._format_leverage_scenarios(leverage_scenarios)}

PIVOT OPPORTUNITIES:
{self._format_pivot_opportunities(pivot_opportunities)}

CREATE CONTRARIAN TOURNAMENT-WINNING NARRATIVES IN JSON:
{{
    "primary_narrative": "The ONE scenario that creates a unique winning lineup",
    "contrarian_captains": [
        {{
            "player": "exact_name",
            "ownership": 5.0,
            "narrative": "Specific reason this 5% captain wins tournaments",
            "ceiling_path": "Exact scenario for 30+ point game",
            "game_script_trigger": "What needs to happen"
        }}
    ],
    "hidden_correlations": [
        {{
            "player1": "name1",
            "player2": "name2",
            "narrative": "Non-obvious connection the field misses",
            "game_script": "When this correlation pays off"
        }}
    ],
    "fade_the_chalk": [
        {{
            "player": "chalk_name",
            "ownership": 35,
            "fade_reason": "Specific quantifiable bust risk",
            "pivot_to": "alternative_player",
            "edge_narrative": "Why pivot has better expectation"
        }}
    ],
    "leverage_scenarios": [
        {{
            "scenario": "Specific game script",
            "probability": "Field assigns 10%, actually 25%",
            "beneficiaries": ["player1", "player2"],
            "counter_narrative": "Why field underrates this",
            "win_condition": "If this happens, these players smash"
        }}
    ],
    "contrarian_game_theory": {{
        "what_field_expects": "The chalk narrative",
        "fatal_flaw": "Why the chalk narrative is wrong",
        "exploit_angle": "Your specific edge",
        "unique_construction": "How your roster beats the field",
        "field_concentration": "Where 70% of field concentrates"
    }},
    "boom_paths": [
        {{
            "player": "name",
            "path_to_ceiling": "Specific statistical path to 3x+ value",
            "indicators": "In-game indicators to watch",
            "probability": "Realistic % chance",
            "field_discount": "What ownership should be vs what it is"
        }}
    ],
    "tournament_winner": {{
        "captain": "exact_contrarian_captain",
        "core": ["player1", "player2", "player3"],
        "differentiators": ["unique1", "unique2"],
        "total_ownership": 65,
        "win_condition": "The exact scenario that makes this lineup win",
        "field_miss": "What specific thing does the field not see"
    }},
    "anti_chalk_strategy": {{
        "max_chalk_in_lineup": 1,
        "chalk_players_to_avoid": ["player_names"],
        "contrarian_correlation": "Stack field avoids",
        "leverage_captain": "Captain field won't play"
    }},
    "confidence": 0.75,
    "key_edge": "The ONE angle that separates from field"
}}

Find the narrative that makes sub-5% plays optimal. Think about game flow scenarios the field ignores.
Identify SPECIFIC contrarian angles, not generic "low owned = good".
"""

        return prompt

    def _calculate_contrarian_score(self, df) -> pd.Series:
        """Calculate contrarian score combining projection and ownership"""
        return (
            df['Projected_Points'] / df['Projected_Points'].max() * 100 /
            (df.get('Ownership', 10) + 1)
        )

    def _identify_ceiling_plays(self, df) -> pd.DataFrame:
        """Identify low-owned players with high ceilings"""
        low_owned = df[df.get('Ownership', 10) < 10]

        if 'Ceiling' in df.columns:
            return low_owned.nlargest(10, 'Ceiling')
        else:
            # Estimate ceiling as projection * 1.4
            low_owned = low_owned.copy()
            low_owned['Ceiling'] = low_owned['Projected_Points'] * 1.4
            return low_owned.nlargest(10, 'Ceiling')

    def _identify_hidden_value(self, df) -> pd.DataFrame:
        """Identify underpriced players field is missing"""
        df = df.copy()

        # High value + medium-low ownership
        hidden = df[
            (df['Value'] > df['Value'].quantile(0.7)) &
            (df.get('Ownership', 10) < 15)
        ]

        return hidden.nlargest(10, 'Value')

    def _identify_chalk_fades(self, df, slate_profile: Dict) -> pd.DataFrame:
        """Identify high-owned players to fade"""
        chalk_threshold = 25

        if slate_profile.get('chalk_concentration', 0.5) > 0.6:
            chalk_threshold = 20  # Lower threshold in high-chalk slates

        chalk = df[df.get('Ownership', 10) > chalk_threshold]

        # Add fade scores (look for bust risk indicators)
        if not chalk.empty:
            chalk = chalk.copy()

            # Factors that suggest fade:
            # - Very high ownership relative to projection
            # - High salary but mediocre value
            chalk['Fade_Score'] = (
                chalk['Ownership'] / (chalk['Projected_Points'] + 5) *
                (chalk['Salary'] / 10000)
            )

            return chalk.nlargest(5, 'Fade_Score')

        return pd.DataFrame()

    def _identify_leverage_scenarios(self, df, game_info: Dict) -> List[Dict]:
        """Identify game scripts that create leverage"""
        scenarios = []

        total = game_info.get('total', 45)
        spread = abs(game_info.get('spread', 0))

        # Blowout scenario
        if spread > 7:
            teams = df['Team'].unique()
            if len(teams) >= 2:
                underdog = teams[1] if spread > 0 else teams[0]

                underdog_players = df[df['Team'] == underdog].nlargest(3, 'Projected_Points')

                scenarios.append({
                    'scenario': 'Underdog keeps it close / wins',
                    'probability': 'Field: 15%, Reality: 30%',
                    'beneficiaries': underdog_players['Player'].tolist(),
                    'narrative': f'{underdog} players have leverage if game stays competitive'
                })

        # Low total surprise
        if total < 43:
            high_td_players = df.nlargest(5, 'Projected_Points')['Player'].tolist()

            scenarios.append({
                'scenario': 'Game goes over total',
                'probability': 'Field: 30%, Reality: 45%',
                'beneficiaries': high_td_players[:3],
                'narrative': 'Low total creates TD-dependent leverage'
            })

        # Weather narrative
        weather = game_info.get('weather', 'Clear')
        if weather in ['Rain', 'Wind']:
            rbs = df[df['Position'] == 'RB'].nlargest(4, 'Projected_Points')['Player'].tolist()

            scenarios.append({
                'scenario': 'Weather less impactful than expected',
                'probability': 'Field overreacts',
                'beneficiaries': rbs,
                'narrative': 'RB ownership increases, but passing still viable'
            })

        return scenarios

    def _identify_pivot_opportunities(self, df) -> List[Dict]:
        """Identify pivot opportunities from chalk"""
        pivots = []

        # Group by position and find chalk with similar alternatives
        for position in df['Position'].unique():
            pos_df = df[df['Position'] == position]

            if len(pos_df) < 2:
                continue

            # Find chalk in this position
            chalk = pos_df[pos_df.get('Ownership', 10) > 25]

            for _, chalk_player in chalk.iterrows():
                # Find similar players with much lower ownership
                similar_salary = pos_df[
                    abs(pos_df['Salary'] - chalk_player['Salary']) < 1500
                ]

                similar_proj = similar_salary[
                    abs(similar_salary['Projected_Points'] - chalk_player['Projected_Points']) < 3
                ]

                low_owned_similar = similar_proj[
                    similar_proj.get('Ownership', 10) < chalk_player.get('Ownership', 30) / 3
                ]

                for _, pivot_player in low_owned_similar.iterrows():
                    if pivot_player['Player'] != chalk_player['Player']:
                        pivots.append({
                            'from': chalk_player['Player'],
                            'to': pivot_player['Player'],
                            'ownership_diff': chalk_player.get('Ownership', 30) - pivot_player.get('Ownership', 10),
                            'proj_diff': chalk_player['Projected_Points'] - pivot_player['Projected_Points'],
                            'leverage_gain': chalk_player.get('Ownership', 30) / max(pivot_player.get('Ownership', 10), 1)
                        })

        # Sort by leverage gain
        pivots.sort(key=lambda x: x['leverage_gain'], reverse=True)
        return pivots[:5]

    def _format_ceiling_plays(self, ceiling_df) -> str:
        """Format ceiling plays"""
        if ceiling_df.empty:
            return "No significant low-owned ceiling plays"

        lines = []
        for _, row in ceiling_df.head(10).iterrows():
            lines.append(
                f"• {row['Player']} ({row['Position']}) - "
                f"{row.get('Ownership', 10):.1f}% owned, "
                f"{row.get('Ceiling', row['Projected_Points'] * 1.4):.1f} ceiling"
            )

        return "\n".join(lines)

    def _format_hidden_value(self, hidden_df) -> str:
        """Format hidden value"""
        if hidden_df.empty:
            return "No hidden value plays identified"

        lines = []
        for _, row in hidden_df.head(8).iterrows():
            lines.append(
                f"• {row['Player']} - "
                f"{row['Value']:.2f} value, "
                f"{row.get('Ownership', 10):.1f}% owned"
            )

        return "\n".join(lines)

    def _format_contrarian_captains(self, contrarian_df) -> str:
        """Format contrarian captain candidates"""
        if contrarian_df.empty:
            return "No contrarian captains identified"

        lines = []
        for _, row in contrarian_df.head(8).iterrows():
            lines.append(
                f"• {row['Player']} - "
                f"Score: {row['Contrarian_Score']:.2f}, "
                f"{row.get('Ownership', 10):.1f}% owned"
            )

        return "\n".join(lines)

    def _format_chalk_fades(self, chalk_df) -> str:
        """Format chalk to fade"""
        if chalk_df.empty:
            return "No significant chalk to fade"

        lines = []
        for _, row in chalk_df.head(5).iterrows():
            lines.append(
                f"• {row['Player']} - "
                f"{row.get('Ownership', 30):.1f}% owned, "
                f"Fade score: {row.get('Fade_Score', 0):.2f}"
            )

        return "\n".join(lines)

    def _format_leverage_scenarios(self, scenarios: List[Dict]) -> str:
        """Format leverage scenarios"""
        if not scenarios:
            return "No specific leverage scenarios identified"

        lines = []
        for scenario in scenarios:
            lines.append(
                f"• {scenario['scenario']}\n"
                f"  Beneficiaries: {', '.join(scenario['beneficiaries'][:3])}\n"
                f"  Narrative: {scenario['narrative']}"
            )

        return "\n".join(lines)

    def _format_pivot_opportunities(self, pivots: List[Dict]) -> str:
        """Format pivot opportunities"""
        if not pivots:
            return "No clear pivot opportunities"

        lines = []
        for pivot in pivots[:5]:
            lines.append(
                f"• Pivot from {pivot['from']} to {pivot['to']}\n"
                f"  Ownership diff: {pivot['ownership_diff']:.1f}%, "
                f"Leverage gain: {pivot['leverage_gain']:.1f}x"
            )

        return "\n".join(lines)

    def parse_response(self, response: str, df, field_size: str) -> Any:
        """Parse contrarian narrative response"""

        from data_models import AIRecommendation

        try:
            if response and response != '{}':
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    self.logger.log(
                        "Failed to parse JSON, using narrative extraction",
                        "WARNING",
                        tags=['parse', 'fallback']
                    )
                    data = self._extract_narrative_from_text(response, df)
            else:
                data = {}

            available_players = set(df['Player'].values)

            # Extract contrarian captains
            contrarian_captains = []
            captain_narratives = {}

            for captain_data in data.get('contrarian_captains', []):
                player = captain_data.get('player')
                if player and player in available_players:
                    contrarian_captains.append(player)
                    captain_narratives[player] = {
                        'narrative': captain_data.get('narrative', ''),
                        'ceiling_path': captain_data.get('ceiling_path', ''),
                        'ownership': captain_data.get('ownership', 5)
                    }

            # Fallback to statistical contrarian captains
            if len(contrarian_captains) < 3:
                contrarian_captains.extend(
                    self._find_statistical_contrarian_captains(df, contrarian_captains)
                )
                contrarian_captains = contrarian_captains[:7]

            # Extract tournament winner lineup
            tournament_winner = data.get('tournament_winner', {})
            tw_captain = tournament_winner.get('captain')
            tw_core = tournament_winner.get('core', [])
            tw_differentiators = tournament_winner.get('differentiators', [])

            must_play = []

            if tw_captain and tw_captain in available_players:
                if tw_captain not in contrarian_captains:
                    contrarian_captains.insert(0, tw_captain)

            for player in tw_core + tw_differentiators:
                if player in available_players and player not in must_play:
                    must_play.append(player)

            # Extract fades
            fades = []
            pivots = {}

            for fade_data in data.get('fade_the_chalk', []):
                fade_player = fade_data.get('player')
                pivot_player = fade_data.get('pivot_to')

                if fade_player and fade_player in available_players:
                    player_ownership = df[df['Player'] == fade_player]['Ownership'].values
                    if len(player_ownership) > 0 and player_ownership[0] > 20:
                        fades.append(fade_player)

                        if pivot_player and pivot_player in available_players:
                            pivots[fade_player] = pivot_player
                            if pivot_player not in must_play:
                                must_play.append(pivot_player)

            # Extract hidden correlations
            hidden_stacks = []

            for corr in data.get('hidden_correlations', []):
                p1 = corr.get('player1')
                p2 = corr.get('player2')

                if p1 in available_players and p2 in available_players:
                    hidden_stacks.append({
                        'player1': p1,
                        'player2': p2,
                        'type': 'hidden',
                        'narrative': corr.get('narrative', 'Non-obvious correlation'),
                        'correlation': 0.4
                    })

            # Build contrarian angles
            contrarian_theory = data.get('contrarian_game_theory', {})
            contrarian_angles = [
                contrarian_theory.get('fatal_flaw', ''),
                contrarian_theory.get('exploit_angle', ''),
                contrarian_theory.get('unique_construction', '')
            ]
            contrarian_angles = [a for a in contrarian_angles if a]

            # Build enforcement rules
            enforcement_rules = self._build_contrarian_enforcement_rules(
                contrarian_captains, must_play, fades, hidden_stacks, captain_narratives
            )

            # Extract insights
            key_insights = [
                data.get('primary_narrative', 'Contrarian approach'),
                f"Fade {len(fades)} chalk plays",
                f"{len(contrarian_captains)} contrarian captains identified"
            ]

            confidence = data.get('confidence', 0.7)
            confidence = max(0.0, min(1.0, confidence))

            recommendation = AIRecommendation(
                captain_targets=contrarian_captains,
                must_play=must_play[:5],
                never_play=fades[:5],
                stacks=hidden_stacks[:5],
                key_insights=key_insights[:3],
                confidence=confidence,
                enforcement_rules=enforcement_rules,
                narrative=data.get('primary_narrative', 'Contrarian strategy'),
                source_ai=self.strategist_type,
                contrarian_angles=contrarian_angles[:3],
                ceiling_plays=[b.get('player') for b in data.get('boom_paths', [])][:3]
            )

            return recommendation

        except Exception as e:
            self.logger.log_exception(e, "parse_contrarian_response")
            return self._get_fallback_recommendation(df, field_size)

    def _find_statistical_contrarian_captains(self, df, existing: List[str]) -> List[str]:
        """Find contrarian captains statistically"""
        df_copy = df.copy()
        df_copy['Contrarian_Score'] = (
            df_copy['Projected_Points'] / df_copy['Projected_Points'].max() * 100 /
            (df_copy.get('Ownership', 10) + 1)
        )

        eligible = df_copy[
            (~df_copy['Player'].isin(existing)) &
            (df_copy.get('Ownership', 10) < 15)
        ]

        contrarian_plays = eligible.nlargest(5, 'Contrarian_Score')
        return contrarian_plays['Player'].tolist()

    def _build_contrarian_enforcement_rules(self, captains: List[str], must_play: List[str],
                                           fades: List[str], hidden_stacks: List[Dict],
                                           captain_narratives: Dict) -> List[Dict]:
        """Build contrarian enforcement rules"""
        rules = []

        # Ultra-contrarian captain rule
        ultra_contrarian = []
        moderate_contrarian = []

        for captain in captains[:5]:
            ownership = captain_narratives.get(captain, {}).get('ownership', 10)
            if ownership < 5:
                ultra_contrarian.append(captain)
            else:
                moderate_contrarian.append(captain)

        if ultra_contrarian:
            rules.append({
                'type': 'hard',
                'constraint': 'ultra_contrarian_captain',
                'players': ultra_contrarian,
                'priority': 100,
                'description': 'Ultra-contrarian captain (<5% owned)'
            })

        if moderate_contrarian:
            rules.append({
                'type': 'soft',
                'constraint': 'contrarian_captain',
                'players': moderate_contrarian,
                'weight': 0.8,
                'priority': 90,
                'description': 'Contrarian captain options'
            })

        # Tournament core
        for i, player in enumerate(must_play[:3]):
            rules.append({
                'type': 'hard' if i == 0 else 'soft',
                'constraint': f'tournament_core_{player}',
                'player': player,
                'weight': 0.9 - (i * 0.1) if i > 0 else 1.0,
                'priority': 85 - (i * 5),
                'description': f'Tournament core: {player}'
            })

        # Fade rules
        for i, fade in enumerate(fades[:3]):
            rules.append({
                'type': 'hard' if i == 0 else 'soft',
                'constraint': f'fade_chalk_{fade}',
                'player': fade,
                'exclude': True,
                'weight': 0.8 - (i * 0.1) if i > 0 else 1.0,
                'priority': 80 - (i * 5),
                'description': f'Fade chalk: {fade}'
            })

        # Hidden correlations
        for i, stack in enumerate(hidden_stacks[:2]):
            rules.append({
                'type': 'soft',
                'constraint': 'hidden_correlation',
                'players': [stack['player1'], stack['player2']],
                'weight': 0.7 - (i * 0.1),
                'priority': 60 - (i * 5),
                'description': stack.get('narrative', 'Hidden correlation')
            })

        return rules

    def _extract_narrative_from_text(self, response: str, df) -> Dict:
        """Extract contrarian narrative from text"""
        data = {
            'contrarian_captains': [],
            'fade_the_chalk': [],
            'confidence': 0.6
        }

        low_owned = df[df.get('Ownership', 10) < 10]['Player'].tolist()
        high_owned = df[df.get('Ownership', 10) > 30]['Player'].tolist()

        for player in low_owned:
            if player.lower() in response.lower():
                data['contrarian_captains'].append({
                    'player': player,
                    'narrative': 'Low ownership leverage'
                })

        for player in high_owned:
            if f"fade {player.lower()}" in response.lower() or f"avoid {player.lower()}" in response.lower():
                data['fade_the_chalk'].append({
                    'player': player,
                    'ownership': df[df['Player'] == player]['Ownership'].values[0]
                })

        return data

"""
AI Synthesis Engine and Ownership Bucket Management
Improvements: Weighted consensus, conflict resolution, dynamic bucket thresholds, multi-dimensional analysis
"""

class AISynthesisEngine:
    """Enhanced synthesis with weighted voting and pattern analysis"""

    def __init__(self):
        self.logger = get_logger()
        self.synthesis_history = deque(maxlen=50)
        self.consensus_patterns = defaultdict(list)
        self.disagreement_analysis = []

    def synthesize_recommendations(self, game_theory, correlation, contrarian) -> Dict:
        """
        Enhanced synthesis with weighted voting and conflict resolution

        Args:
            game_theory: Game theory AI recommendation
            correlation: Correlation AI recommendation
            contrarian: Contrarian AI recommendation

        Returns:
            Comprehensive synthesis with consensus analysis
        """

        self.logger.log(
            "Synthesizing triple AI recommendations",
            "INFO",
            tags=['synthesis', 'ai']
        )

        from config import OptimizerConfig, AIStrategistType

        synthesis = {
            'captain_strategy': {},
            'player_rankings': {},
            'stacking_rules': [],
            'avoidance_rules': [],
            'enforcement_rules': [],
            'confidence': 0,
            'narrative': "",
            'patterns': [],
            'consensus_analysis': {},
            'disagreement_analysis': {}
        }

        # Get AI weights
        weights = {
            AIStrategistType.GAME_THEORY: OptimizerConfig.AI_WEIGHTS.get('game_theory', 0.35),
            AIStrategistType.CORRELATION: OptimizerConfig.AI_WEIGHTS.get('correlation', 0.35),
            AIStrategistType.CONTRARIAN_NARRATIVE: OptimizerConfig.AI_WEIGHTS.get('contrarian', 0.30)
        }

        # Captain synthesis with weighted voting
        captain_votes = defaultdict(lambda: {'votes': [], 'weighted_score': 0, 'positions': []})

        for ai_type, rec in [
            (AIStrategistType.GAME_THEORY, game_theory),
            (AIStrategistType.CORRELATION, correlation),
            (AIStrategistType.CONTRARIAN_NARRATIVE, contrarian)
        ]:
            weight = weights[ai_type]

            for i, captain in enumerate(rec.captain_targets[:7]):
                position_weight = 1.0 - (i * 0.08)  # Decay by position
                weighted_vote = weight * rec.confidence * position_weight

                captain_votes[captain]['votes'].append(ai_type.value)
                captain_votes[captain]['weighted_score'] += weighted_vote
                captain_votes[captain]['positions'].append(i + 1)

        # Classify captains by consensus level
        for captain, vote_data in captain_votes.items():
            vote_count = len(vote_data['votes'])

            if vote_count == 3:
                synthesis['captain_strategy'][captain] = {
                    'level': 'unanimous',
                    'score': vote_data['weighted_score'],
                    'sources': vote_data['votes'],
                    'avg_position': np.mean(vote_data['positions'])
                }
            elif vote_count == 2:
                synthesis['captain_strategy'][captain] = {
                    'level': 'majority',
                    'score': vote_data['weighted_score'],
                    'sources': vote_data['votes'],
                    'avg_position': np.mean(vote_data['positions'])
                }
            else:
                synthesis['captain_strategy'][captain] = {
                    'level': vote_data['votes'][0],
                    'score': vote_data['weighted_score'],
                    'sources': vote_data['votes'],
                    'avg_position': vote_data['positions'][0]
                }

        # Sort captains by weighted score
        sorted_captains = sorted(
            synthesis['captain_strategy'].items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        synthesis['captain_strategy'] = dict(sorted_captains)

        # Player rankings with multi-factor scoring
        player_scores = defaultdict(lambda: {
            'total_score': 0,
            'factors': {},
            'sources': []
        })

        for ai_type, rec in [
            (AIStrategistType.GAME_THEORY, game_theory),
            (AIStrategistType.CORRELATION, correlation),
            (AIStrategistType.CONTRARIAN_NARRATIVE, contrarian)
        ]:
            weight = weights[ai_type]

            # Score must-play players
            for i, player in enumerate(rec.must_play[:5]):
                position_weight = 1.0 - (i * 0.1)
                score = weight * rec.confidence * position_weight

                player_scores[player]['total_score'] += score
                player_scores[player]['factors'][ai_type.value] = score
                player_scores[player]['sources'].append(ai_type.value)

            # Negative scores for fades
            for player in rec.never_play[:3]:
                score = -weight * rec.confidence * 0.5
                player_scores[player]['total_score'] += score
                player_scores[player]['factors'][f'{ai_type.value}_fade'] = score

        # Normalize scores
        if player_scores:
            max_score = max(abs(data['total_score']) for data in player_scores.values())
            if max_score > 0:
                for player, data in player_scores.items():
                    data['normalized_score'] = data['total_score'] / max_score

                synthesis['player_rankings'] = dict(sorted(
                    player_scores.items(),
                    key=lambda x: x[1]['total_score'],
                    reverse=True
                ))

        # Stack synthesis with deduplication and prioritization
        all_stacks = []
        stack_votes = defaultdict(lambda: {'count': 0, 'sources': [], 'data': []})

        for ai_type, rec in [
            (AIStrategistType.GAME_THEORY, game_theory),
            (AIStrategistType.CORRELATION, correlation),
            (AIStrategistType.CONTRARIAN_NARRATIVE, contrarian)
        ]:
            for stack in rec.stacks:
                stack_signature = self._get_stack_signature(stack)

                stack_votes[stack_signature]['count'] += 1
                stack_votes[stack_signature]['sources'].append(ai_type.value)
                stack_votes[stack_signature]['data'].append(stack)

        # Process stacks by consensus
        for signature, vote_data in stack_votes.items():
            # Use the stack with highest priority or correlation
            best_stack = max(
                vote_data['data'],
                key=lambda s: s.get('correlation', 0.5)
            )

            # Add consensus metadata
            best_stack['consensus'] = vote_data['count'] >= 2
            best_stack['vote_count'] = vote_data['count']
            best_stack['sources'] = vote_data['sources']

            # Boost priority for consensus stacks
            if best_stack['consensus']:
                best_stack['priority'] = best_stack.get('priority', 50) + (vote_data['count'] * 10)

            all_stacks.append(best_stack)

        synthesis['stacking_rules'] = sorted(
            all_stacks,
            key=lambda s: (s.get('vote_count', 0), s.get('priority', 50)),
            reverse=True
        )[:15]

        # Analyze consensus and disagreement
        synthesis['consensus_analysis'] = self._analyze_consensus(
            game_theory, correlation, contrarian, captain_votes, player_scores
        )

        synthesis['disagreement_analysis'] = self._analyze_disagreements(
            game_theory, correlation, contrarian
        )

        # Pattern analysis
        synthesis['patterns'] = self._analyze_patterns(
            game_theory, correlation, contrarian, synthesis['consensus_analysis']
        )

        # Calculate overall confidence
        confidences = [
            game_theory.confidence * weights[AIStrategistType.GAME_THEORY],
            correlation.confidence * weights[AIStrategistType.CORRELATION],
            contrarian.confidence * weights[AIStrategistType.CONTRARIAN_NARRATIVE]
        ]

        # Adjust confidence based on consensus
        base_confidence = sum(confidences)
        consensus_boost = synthesis['consensus_analysis'].get('consensus_score', 0) * 0.1
        synthesis['confidence'] = min(0.95, base_confidence + consensus_boost)

        # Create unified narrative
        narratives = []
        if game_theory.narrative:
            narratives.append(f"Game Theory: {game_theory.narrative[:120]}")
        if correlation.narrative:
            narratives.append(f"Correlation: {correlation.narrative[:120]}")
        if contrarian.narrative:
            narratives.append(f"Contrarian: {contrarian.narrative[:120]}")

        synthesis['narrative'] = " | ".join(narratives)

        # Synthesize enforcement rules
        synthesis['enforcement_rules'] = self._synthesize_enforcement_rules(
            game_theory, correlation, contrarian, weights
        )

        # Store in history
        self.synthesis_history.append({
            'timestamp': datetime.now(),
            'confidence': synthesis['confidence'],
            'captain_count': len(synthesis['captain_strategy']),
            'consensus_score': synthesis['consensus_analysis'].get('consensus_score', 0),
            'patterns': synthesis['patterns']
        })

        self.logger.log(
            f"Synthesis complete: {synthesis['confidence']:.2%} confidence, "
            f"{len(synthesis['captain_strategy'])} captains, "
            f"{len(synthesis['stacking_rules'])} stacks",
            "INFO",
            context={'consensus': synthesis['consensus_analysis']},
            tags=['synthesis', 'complete']
        )

        return synthesis

    def _get_stack_signature(self, stack: Dict) -> str:
        """Create unique signature for stack deduplication"""
        if 'players' in stack:
            return tuple(sorted(stack['players'][:3]))
        elif 'player1' in stack and 'player2' in stack:
            return tuple(sorted([stack['player1'], stack['player2']]))
        else:
            return str(stack.get('type', 'unknown'))

    def _analyze_consensus(self, game_theory, correlation, contrarian,
                          captain_votes: Dict, player_scores: Dict) -> Dict:
        """Analyze level of consensus between AIs"""

        analysis = {
            'consensus_score': 0.0,
            'captain_consensus': 0,
            'player_consensus': 0,
            'unanimous_captains': [],
            'majority_captains': [],
            'divergent_captains': []
        }

        # Captain consensus analysis
        for captain, vote_data in captain_votes.items():
            vote_count = len(vote_data['votes'])

            if vote_count == 3:
                analysis['unanimous_captains'].append(captain)
                analysis['captain_consensus'] += 1.0
            elif vote_count == 2:
                analysis['majority_captains'].append(captain)
                analysis['captain_consensus'] += 0.6
            else:
                analysis['divergent_captains'].append({
                    'captain': captain,
                    'source': vote_data['votes'][0]
                })

        # Normalize captain consensus
        total_captains = len(captain_votes)
        if total_captains > 0:
            analysis['captain_consensus'] /= total_captains

        # Player consensus analysis
        multi_source_players = sum(
            1 for data in player_scores.values()
            if len(data['sources']) >= 2
        )

        total_players = len(player_scores)
        if total_players > 0:
            analysis['player_consensus'] = multi_source_players / total_players

        # Overall consensus score
        analysis['consensus_score'] = (
            analysis['captain_consensus'] * 0.6 +
            analysis['player_consensus'] * 0.4
        )

        return analysis

    def _analyze_disagreements(self, game_theory, correlation, contrarian) -> Dict:
        """Analyze where AIs disagree"""

        disagreements = {
            'conflicting_fades': [],
            'strategy_divergence': [],
            'confidence_variance': 0
        }

        # Find players that one AI likes and another fades
        all_must_play = set(game_theory.must_play + correlation.must_play + contrarian.must_play)
        all_never_play = set(game_theory.never_play + correlation.never_play + contrarian.never_play)

        conflicts = all_must_play & all_never_play
        if conflicts:
            disagreements['conflicting_fades'] = list(conflicts)

        # Analyze confidence variance
        confidences = [game_theory.confidence, correlation.confidence, contrarian.confidence]
        disagreements['confidence_variance'] = np.std(confidences)

        # Strategy divergence
        strategies = []
        for rec in [game_theory, correlation, contrarian]:
            if hasattr(rec, 'ownership_leverage') and rec.ownership_leverage:
                strategies.append('ownership_focused')
            elif len(rec.stacks) > 3:
                strategies.append('correlation_focused')
            else:
                strategies.append('balanced')

        if len(set(strategies)) == 3:
            disagreements['strategy_divergence'] = 'high'
        elif len(set(strategies)) == 2:
            disagreements['strategy_divergence'] = 'moderate'
        else:
            disagreements['strategy_divergence'] = 'low'

        return disagreements

    def _analyze_patterns(self, game_theory, correlation, contrarian, consensus_analysis: Dict) -> List[str]:
        """Identify patterns in recommendations"""
        patterns = []

        # Consensus patterns
        if consensus_analysis.get('consensus_score', 0) > 0.7:
            patterns.append("High AI consensus - strong conviction plays")
        elif consensus_analysis.get('consensus_score', 0) < 0.3:
            patterns.append("Low consensus - diverse perspectives, use balanced approach")

        unanimous = consensus_analysis.get('unanimous_captains', [])
        if unanimous:
            patterns.append(f"Strong consensus on {len(unanimous)} captains")

        # Stack patterns
        stack_types = defaultdict(int)
        for rec in [game_theory, correlation, contrarian]:
            for stack in rec.stacks:
                stack_type = stack.get('type', 'standard')
                stack_types[stack_type] += 1

        if stack_types.get('onslaught', 0) > 1:
            patterns.append("Multiple AIs recommend onslaught stacks")

        if stack_types.get('bring_back', 0) > 1:
            patterns.append("Bring-back correlation identified by multiple AIs")

        # Ownership patterns
        avg_ownership_emphasis = []
        if hasattr(game_theory, 'ownership_leverage') and game_theory.ownership_leverage:
            target = game_theory.ownership_leverage.get('ownership_ceiling', 100)
            avg_ownership_emphasis.append(target)

        if avg_ownership_emphasis:
            avg_target = np.mean(avg_ownership_emphasis)
            if avg_target < 15:
                patterns.append("Ultra-contrarian ownership profile recommended")
            elif avg_target < 25:
                patterns.append("Leverage-focused ownership approach")

        # Confidence patterns
        confidences = [game_theory.confidence, correlation.confidence, contrarian.confidence]
        if all(c > 0.7 for c in confidences):
            patterns.append("All AIs highly confident - strong slate read")
        elif any(c < 0.5 for c in confidences):
            patterns.append("Mixed confidence levels - uncertain slate")

        return patterns

    def _synthesize_enforcement_rules(self, game_theory, correlation, contrarian, weights: Dict) -> List[Dict]:
        """Synthesize enforcement rules from all AIs"""
        rules = []

        all_rules = (
            game_theory.enforcement_rules +
            correlation.enforcement_rules +
            contrarian.enforcement_rules
        )

        # Group similar rules
        rule_groups = defaultdict(list)

        for rule in all_rules:
            # Create grouping key
            key = f"{rule.get('type')}_{rule.get('constraint')}_{rule.get('player', '')}"
            rule_groups[key].append(rule)

        # Consolidate groups
        for group in rule_groups.values():
            if len(group) > 1:
                # Multiple AIs suggest similar rule
                consolidated = group[0].copy()

                # Increase priority based on consensus
                base_priority = max(r.get('priority', 50) for r in group)
                consensus_boost = len(group) * 8
                consolidated['priority'] = base_priority + consensus_boost
                consolidated['consensus_count'] = len(group)
                consolidated['sources'] = [r.get('source', 'unknown') for r in group]

                rules.append(consolidated)
            else:
                rules.append(group[0])

        # Sort by priority
        rules.sort(key=lambda r: r.get('priority', 50), reverse=True)

        return rules[:25]

    def get_synthesis_quality_score(self) -> float:
        """Calculate quality score of recent syntheses"""
        if not self.synthesis_history:
            return 0.5

        recent = list(self.synthesis_history)[-10:]

        # Average confidence
        avg_confidence = np.mean([s['confidence'] for s in recent])

        # Consensus stability
        consensus_scores = [s.get('consensus_score', 0.5) for s in recent]
        consensus_stability = 1 - np.std(consensus_scores)

        # Captain diversity
        captain_counts = [s['captain_count'] for s in recent]
        captain_diversity = min(np.std(captain_counts) / np.mean(captain_counts), 1.0) if np.mean(captain_counts) > 0 else 0

        # Pattern consistency
        all_patterns = []
        for s in recent:
            all_patterns.extend(s.get('patterns', []))

        unique_patterns = len(set(all_patterns))
        pattern_score = min(unique_patterns / max(len(all_patterns), 1), 1.0)

        # Combine scores
        quality = (
            avg_confidence * 0.4 +
            consensus_stability * 0.3 +
            captain_diversity * 0.15 +
            pattern_score * 0.15
        )

        return min(quality, 1.0)


class AIOwnershipBucketManager:
    """Enhanced ownership bucket management with dynamic thresholds"""

    def __init__(self, enforcement_engine=None):
        self.enforcement_engine = enforcement_engine
        self.logger = get_logger()

        # Dynamic bucket thresholds (will adjust based on slate)
        self.bucket_thresholds = {
            'mega_chalk': 35,
            'chalk': 20,
            'moderate': 15,
            'pivot': 10,
            'leverage': 5,
            'super_leverage': 2
        }

        # Bucket performance history
        self.bucket_performance = defaultdict(lambda: {'uses': 0, 'success': 0})

    def categorize_players(self, df, field_size: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Categorize players with dynamic thresholds

        Args:
            df: Player DataFrame
            field_size: Optional field size for threshold adjustment

        Returns:
            Dictionary of ownership buckets
        """

        # Adjust thresholds based on field size and ownership distribution
        adjusted_thresholds = self._adjust_thresholds(df, field_size)

        buckets = {
            'mega_chalk': [],
            'chalk': [],
            'moderate': [],
            'pivot': [],
            'leverage': [],
            'super_leverage': [],
            'stats': {}
        }

        for _, row in df.iterrows():
            player = row['Player']
            ownership = row.get('Ownership', 10)

            # Categorize
            if ownership >= adjusted_thresholds['mega_chalk']:
                buckets['mega_chalk'].append(player)
            elif ownership >= adjusted_thresholds['chalk']:
                buckets['chalk'].append(player)
            elif ownership >= adjusted_thresholds['moderate']:
                buckets['moderate'].append(player)
            elif ownership >= adjusted_thresholds['pivot']:
                buckets['pivot'].append(player)
            elif ownership >= adjusted_thresholds['leverage']:
                buckets['leverage'].append(player)
            else:
                buckets['super_leverage'].append(player)

        # Calculate statistics
        buckets['stats'] = {
            'total_players': len(df),
            'chalk_concentration': (len(buckets['mega_chalk']) + len(buckets['chalk'])) / len(df),
            'leverage_opportunity': len(buckets['leverage']) + len(buckets['super_leverage']),
            'thresholds_used': adjusted_thresholds
        }

        self.logger.log(
            f"Categorized {len(df)} players into ownership buckets",
            "DEBUG",
            context={'stats': buckets['stats']},
            tags=['ownership', 'buckets']
        )

        return buckets

    def _adjust_thresholds(self, df, field_size: Optional[str]) -> Dict[str, float]:
        """Dynamically adjust thresholds based on slate characteristics"""

        thresholds = self.bucket_thresholds.copy()

        if 'Ownership' not in df.columns:
            return thresholds

        # Analyze ownership distribution
        ownership_mean = df['Ownership'].mean()
        ownership_std = df['Ownership'].std()
        ownership_max = df['Ownership'].max()

        # If ownership is highly concentrated, adjust thresholds down
        if ownership_max > 50:
            # Very chalky slate
            factor = 0.85
            thresholds = {k: v * factor for k, v in thresholds.items()}
        elif ownership_std < 8:
            # Flat ownership distribution
            factor = 1.1
            thresholds = {k: v * factor for k, v in thresholds.items()}

        # Adjust based on field size
        if field_size:
            if 'large' in field_size or 'milly' in field_size:
                # Lower thresholds for large fields (more contrarian)
                factor = 0.9
                thresholds = {k: v * factor for k, v in thresholds.items()}
            elif 'small' in field_size:
                # Raise thresholds for small fields
                factor = 1.1
                thresholds = {k: v * factor for k, v in thresholds.items()}

        return thresholds

    def calculate_gpp_leverage(self, players: List[str], df) -> float:
        """
        Enhanced leverage score calculation

        Args:
            players: List of players in lineup
            df: Player DataFrame

        Returns:
            Leverage score (higher = more leverage)
        """

        if not players:
            return 0

        total_projection = 0
        total_ownership = 0
        leverage_bonus = 0
        position_diversity = set()

        for player in players:
            player_data = df[df['Player'] == player]
            if not player_data.empty:
                row = player_data.iloc[0]
                projection = row.get('Projected_Points', 0)
                ownership = row.get('Ownership', 10)
                position = row.get('Position', 'FLEX')

                # Captain gets multiplier
                if player == players[0]:
                    projection *= 1.5
                    ownership *= 1.5

                total_projection += projection
                total_ownership += ownership
                position_diversity.add(position)

                # Bonus for leverage plays
                if ownership < 5:
                    leverage_bonus += 15
                elif ownership < 10:
                    leverage_bonus += 10
                elif ownership < 15:
                    leverage_bonus += 5

        # Base leverage: projection strength vs ownership exposure
        if total_ownership > 0:
            base_leverage = (total_projection / len(players)) / (total_ownership / len(players) + 1)
        else:
            base_leverage = 0

        # Position diversity bonus (more positions = potentially more leverage)
        diversity_bonus = len(position_diversity) * 2

        return base_leverage * 10 + leverage_bonus + diversity_bonus

    def get_bucket_recommendations(self, field_size: str, num_lineups: int) -> Dict:
        """Get recommended bucket usage for field size"""

        from config import OptimizerConfig

        recommendations = {
            'mega_chalk_limit': 1,
            'chalk_limit': 2,
            'min_leverage': 2,
            'target_ownership': (60, 90),
            'bucket_distribution': {}
        }

        # Get field config
        field_config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size, {})

        recommendations['mega_chalk_limit'] = field_config.get('max_chalk_players', 2)
        recommendations['min_leverage'] = field_config.get('min_leverage_players', 2)
        recommendations['target_ownership'] = (
            field_config.get('min_total_ownership', 60),
            field_config.get('max_total_ownership', 90)
        )

        # Recommended distribution across buckets
        if 'large' in field_size or 'milly' in field_size:
            recommendations['bucket_distribution'] = {
                'mega_chalk': 0,
                'chalk': '0-1',
                'moderate': '1-2',
                'pivot': '2-3',
                'leverage': '2-3',
                'super_leverage': '0-1'
            }
        elif 'small' in field_size:
            recommendations['bucket_distribution'] = {
                'mega_chalk': '0-2',
                'chalk': '2-3',
                'moderate': '1-2',
                'pivot': '1-2',
                'leverage': '0-1',
                'super_leverage': '0'
            }
        else:
            recommendations['bucket_distribution'] = {
                'mega_chalk': '0-1',
                'chalk': '1-2',
                'moderate': '2',
                'pivot': '1-2',
                'leverage': '1-2',
                'super_leverage': '0-1'
            }

        return recommendations

    def track_bucket_performance(self, lineup: Dict, buckets: Dict, success: bool):
        """Track performance by bucket usage"""

        all_players = [lineup.get('Captain')] + lineup.get('FLEX', [])

        for player in all_players:
            # Find which bucket this player was in
            for bucket_name, players in buckets.items():
                if bucket_name != 'stats' and player in players:
                    self.bucket_performance[bucket_name]['uses'] += 1
                    if success:
                        self.bucket_performance[bucket_name]['success'] += 1
                    break

    def get_bucket_effectiveness(self) -> Dict:
        """Get effectiveness statistics by bucket"""

        effectiveness = {}

        for bucket, data in self.bucket_performance.items():
            if data['uses'] > 0:
                effectiveness[bucket] = {
                    'uses': data['uses'],
                    'success_rate': data['success'] / data['uses'],
                    'sample_size': 'sufficient' if data['uses'] >= 20 else 'limited'
                }

        return effectiveness

"""
Claude API Manager with enhanced caching and error handling
Improvements: Rate limiting, retry logic, response validation, cost tracking
"""

class ClaudeAPIManager:
    """Enhanced Claude API manager with robust error handling"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None

        # Response cache
        self.cache = {}
        self.max_cache_size = 50
        self._cache_lock = threading.RLock()

        # Statistics
        self.stats = {
            'requests': 0,
            'errors': 0,
            'cache_hits': 0,
            'cache_size': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'avg_response_time': 0,
            'by_ai': {}
        }

        # Response time tracking
        self.response_times = deque(maxlen=100)

        # Rate limiting
        self.rate_limit = {
            'max_per_minute': 50,
            'requests_this_minute': 0,
            'minute_start': time.time()
        }

        # Cost tracking (approximate)
        self.cost_per_1k_tokens = {
            'input': 0.003,  # $3 per million input tokens
            'output': 0.015   # $15 per million output tokens
        }

        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()

        self.initialize_client()

    def initialize_client(self):
        """Initialize Claude client with validation"""
        try:
            if not self.api_key or not self.api_key.startswith('sk-'):
                raise ValueError("Invalid API key format")

            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)

                # Validate connection
                if self.validate_connection():
                    self.logger.log(
                        "Claude API client initialized successfully",
                        "INFO",
                        tags=['api', 'init']
                    )
                else:
                    raise Exception("Connection validation failed")

            except ImportError:
                self.logger.log(
                    "Anthropic library not installed. Install with: pip install anthropic",
                    "ERROR",
                    tags=['api', 'error']
                )
                self.client = None

        except Exception as e:
            self.logger.log_exception(e, "initialize_claude_api")
            self.client = None

    def get_ai_response(self, prompt: str, ai_type=None) -> str:
        """
        Get response from Claude with comprehensive error handling

        Args:
            prompt: The prompt to send
            ai_type: Optional AI strategist type for tracking

        Returns:
            Response string (JSON or text)
        """

        # Generate cache key
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        # Check cache
        with self._cache_lock:
            if prompt_hash in self.cache:
                self.stats['cache_hits'] += 1
                if ai_type:
                    self._update_ai_stats(ai_type, cache_hit=True)

                self.logger.log(
                    f"Cache hit for {ai_type.value if ai_type else 'unknown'}",
                    "DEBUG",
                    tags=['cache', 'hit']
                )
                return self.cache[prompt_hash]

        # Check rate limit
        self._check_rate_limit()

        # Update statistics
        self.stats['requests'] += 1
        if ai_type:
            self._update_ai_stats(ai_type, request=True)

        try:
            if not self.client:
                raise Exception("API client not initialized")

            self.perf_monitor.start_timer("claude_api_call")
            start_time = time.time()

            # Make API call
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.7,
                system=self._get_system_prompt(ai_type),
                messages=[{"role": "user", "content": prompt}]
            )

            elapsed = self.perf_monitor.stop_timer("claude_api_call")
            response_time = time.time() - start_time

            # Extract response
            response = message.content[0].text if message.content else "{}"

            # Update statistics
            self.response_times.append(response_time)
            self.stats['avg_response_time'] = np.mean(list(self.response_times))

            # Estimate tokens and cost
            input_tokens = len(prompt) // 4
            output_tokens = len(response) // 4

            cost = (
                input_tokens / 1000 * self.cost_per_1k_tokens['input'] +
                output_tokens / 1000 * self.cost_per_1k_tokens['output']
            )

            self.stats['total_tokens'] += (input_tokens + output_tokens)
            self.stats['total_cost'] += cost

            if ai_type:
                self._update_ai_stats(
                    ai_type,
                    tokens=input_tokens + output_tokens,
                    time=response_time,
                    cost=cost
                )

            # Cache response
            with self._cache_lock:
                self.cache[prompt_hash] = response
                self.stats['cache_size'] = len(self.cache)

                # Manage cache size (LRU)
                if len(self.cache) > self.max_cache_size:
                    keys_to_remove = list(self.cache.keys())[:int(self.max_cache_size * 0.3)]
                    for key in keys_to_remove:
                        del self.cache[key]

            self.logger.log(
                f"API response received for {ai_type.value if ai_type else 'unknown'} "
                f"({len(response)} chars, {elapsed:.2f}s, ${cost:.4f})",
                "DEBUG",
                context={'tokens': input_tokens + output_tokens, 'cost': cost},
                tags=['api', 'success']
            )

            return response

        except Exception as e:
            self.stats['errors'] += 1
            if ai_type:
                self._update_ai_stats(ai_type, error=True)

            self.logger.log_exception(
                e,
                f"API error for {ai_type.value if ai_type else 'unknown'}",
                critical=False
            )

            self.perf_monitor.stop_timer("claude_api_call")

            return "{}"

    def _get_system_prompt(self, ai_type) -> str:
        """Get appropriate system prompt based on AI type"""

        base_prompt = """You are an expert Daily Fantasy Sports (DFS) strategist specializing in NFL tournament optimization.
You provide specific, actionable recommendations using exact player names and clear reasoning.
Always respond with valid JSON containing specific player recommendations.
Your recommendations must be enforceable as optimization constraints."""

        if ai_type is None:
            return base_prompt

        from config import AIStrategistType

        if ai_type == AIStrategistType.GAME_THEORY:
            return base_prompt + """
Focus on ownership leverage, field tendencies, and exploitable patterns.
Identify specific ownership arbitrage opportunities where projected performance exceeds public perception."""

        elif ai_type == AIStrategistType.CORRELATION:
            return base_prompt + """
Focus on player correlations, stacking patterns, and game script dependencies.
Identify both positive correlations (stacks) and negative correlations (avoid together)."""

        elif ai_type == AIStrategistType.CONTRARIAN_NARRATIVE:
            return base_prompt + """
Focus on contrarian angles, low-owned narratives, and pivot opportunities.
Identify specific scenarios where the field is wrong and leverage can be gained."""

        return base_prompt

    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        current_time = time.time()

        # Reset counter if minute has passed
        if current_time - self.rate_limit['minute_start'] >= 60:
            self.rate_limit['requests_this_minute'] = 0
            self.rate_limit['minute_start'] = current_time

        # Check if at limit
        if self.rate_limit['requests_this_minute'] >= self.rate_limit['max_per_minute']:
            wait_time = 60 - (current_time - self.rate_limit['minute_start'])
            if wait_time > 0:
                self.logger.log(
                    f"Rate limit reached, waiting {wait_time:.1f}s",
                    "WARNING",
                    tags=['api', 'rate_limit']
                )
                time.sleep(wait_time)
                self.rate_limit['requests_this_minute'] = 0
                self.rate_limit['minute_start'] = time.time()

        self.rate_limit['requests_this_minute'] += 1

    def _update_ai_stats(self, ai_type, request=False, cache_hit=False,
                        error=False, tokens=0, time=0, cost=0):
        """Update AI-specific statistics"""

        ai_name = ai_type.value if hasattr(ai_type, 'value') else str(ai_type)

        if ai_name not in self.stats['by_ai']:
            self.stats['by_ai'][ai_name] = {
                'requests': 0,
                'errors': 0,
                'cache_hits': 0,
                'tokens': 0,
                'total_time': 0,
                'avg_time': 0,
                'cost': 0
            }

        ai_stats = self.stats['by_ai'][ai_name]

        if request:
            ai_stats['requests'] += 1
        if cache_hit:
            ai_stats['cache_hits'] += 1
        if error:
            ai_stats['errors'] += 1
        if tokens:
            ai_stats['tokens'] += tokens
        if time:
            ai_stats['total_time'] += time
            ai_stats['avg_time'] = ai_stats['total_time'] / max(ai_stats['requests'], 1)
        if cost:
            ai_stats['cost'] += cost

    def get_stats(self) -> Dict:
        """Get comprehensive API usage statistics"""
        with self._cache_lock:
            return {
                'requests': self.stats['requests'],
                'errors': self.stats['errors'],
                'error_rate': self.stats['errors'] / max(self.stats['requests'], 1),
                'cache_hits': self.stats['cache_hits'],
                'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['requests'], 1),
                'cache_size': len(self.cache),
                'total_tokens': self.stats['total_tokens'],
                'total_cost': self.stats['total_cost'],
                'avg_response_time': self.stats['avg_response_time'],
                'by_ai': dict(self.stats['by_ai']),
                'rate_limit': {
                    'current_minute_usage': self.rate_limit['requests_this_minute'],
                    'max_per_minute': self.rate_limit['max_per_minute']
                }
            }

    def clear_cache(self):
        """Clear response cache"""
        with self._cache_lock:
            self.cache.clear()
            self.stats['cache_size'] = 0

        self.logger.log("API cache cleared", "INFO", tags=['cache', 'clear'])

    def validate_connection(self) -> bool:
        """Validate API connection"""
        try:
            if not self.client:
                return False

            test_prompt = "Respond with only: OK"

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=10,
                temperature=0,
                messages=[{"role": "user", "content": test_prompt}]
            )

            return bool(message.content)

        except Exception as e:
            self.logger.log(f"API validation failed: {e}", "ERROR", tags=['api', 'validation'])
            return False

    def estimate_cost(self, num_lineups: int, use_api: bool = True) -> Dict:
        """Estimate API cost for lineup generation"""

        if not use_api:
            return {'total_cost': 0, 'per_lineup': 0, 'breakdown': {}}

        # Estimate based on typical usage
        avg_prompt_tokens = 1500  # Typical prompt size
        avg_response_tokens = 500  # Typical response size

        # 3 AI calls per generation
        total_calls = 3

        total_tokens = (avg_prompt_tokens + avg_response_tokens) * total_calls

        cost_per_generation = (
            avg_prompt_tokens * total_calls / 1000 * self.cost_per_1k_tokens['input'] +
            avg_response_tokens * total_calls / 1000 * self.cost_per_1k_tokens['output']
        )

        # Account for potential retries (10% overhead)
        cost_per_generation *= 1.1

        return {
            'total_cost': cost_per_generation,
            'per_lineup': cost_per_generation / max(num_lineups, 1),
            'breakdown': {
                'ai_calls': total_calls,
                'estimated_tokens': total_tokens,
                'with_cache': cost_per_generation * 0.3  # Assumes 70% cache hit rate
            }
        }

"""
Core Optimizer Engine with AI-driven lineup generation
Improvements: Multi-threaded generation, adaptive constraints, intelligent retry logic, lineup diversity tracking
"""

class AIChefGPPOptimizer:
    """Main optimizer where AI is the chef and optimization executes the strategy"""

    def __init__(self, df, game_info: Dict, field_size: str = 'large_field', api_manager=None):

        self._validate_inputs(df, game_info, field_size)

        self.df = df.copy()
        self.game_info = game_info
        self.field_size = field_size
        self.api_manager = api_manager

        # Initialize AI strategists
        self.game_theory_ai = GPPGameTheoryStrategist(api_manager)
        self.correlation_ai = GPPCorrelationStrategist(api_manager)
        self.contrarian_ai = GPPContrarianNarrativeStrategist(api_manager)

        # Core components
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()

        # Get field configuration
        from config import OptimizerConfig, AIEnforcementLevel
        field_config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(
            field_size,
            OptimizerConfig.FIELD_SIZE_CONFIGS['large_field']
        )

        # Initialize enforcement engine
        enforcement_level_str = field_config.get('ai_enforcement', 'Strong')
        enforcement_level = AIEnforcementLevel[enforcement_level_str.upper().replace(' ', '_')]
        self.enforcement_engine = AIEnforcementEngine(enforcement_level)

        # Supporting components
        self.synthesis_engine = AISynthesisEngine()
        self.bucket_manager = AIOwnershipBucketManager(self.enforcement_engine)

        # Tracking
        self.ai_decisions_log = []
        self.optimization_log = []
        self.generated_lineups = []

        # Lineup generation statistics
        self.lineup_generation_stats = {
            'attempts': 0,
            'successes': 0,
            'failures_by_reason': defaultdict(int),
            'unique_captains': set(),
            'ownership_distribution': [],
            'diversity_metrics': {}
        }

        # Threading
        from config import OptimizerConfig
        self.max_workers = min(OptimizerConfig.MAX_PARALLEL_THREADS, 4)
        self.generation_timeout = OptimizerConfig.OPTIMIZATION_TIMEOUT

        # Prepare data
        self._prepare_data()

    def _validate_inputs(self, df, game_info: Dict, field_size: str):
        """Comprehensive input validation"""
        if df is None or df.empty:
            raise ValueError("Player pool DataFrame cannot be empty")

        required_columns = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate data types
        if not pd.api.types.is_numeric_dtype(df['Salary']):
            raise ValueError("Salary column must be numeric")

        if not pd.api.types.is_numeric_dtype(df['Projected_Points']):
            raise ValueError("Projected_Points column must be numeric")

        # Validate salary range
        from config import OptimizerConfig
        if df['Salary'].min() < OptimizerConfig.MIN_SALARY:
            raise ValueError(f"Player salary below minimum: {df['Salary'].min()}")

        if df['Salary'].max() > OptimizerConfig.MAX_SALARY:
            raise ValueError(f"Player salary above maximum: {df['Salary'].max()}")

        # Check for duplicates
        if df['Player'].duplicated().any():
            raise ValueError("Duplicate player names found in pool")

        from config import OptimizerConfig
        if field_size not in OptimizerConfig.FIELD_SIZE_CONFIGS:
            self.logger.log(f"Unknown field size {field_size}, using large_field", "WARNING")

    def _prepare_data(self):
        """Prepare data with additional calculations"""
        from config import OptimizerConfig

        # Add ownership if missing
        if 'Ownership' not in self.df.columns:
            self.df['Ownership'] = self.df.apply(
                lambda row: OptimizerConfig.get_default_ownership(
                    row['Position'], row['Salary']
                ), axis=1
            )

        # Calculate value metrics
        self.df['Value'] = self.df['Projected_Points'] / (self.df['Salary'] / 1000)
        self.df['GPP_Score'] = self.df['Value'] * (30 / (self.df['Ownership'] + 10))

        # Add ceiling and floor estimates if not present
        if 'Ceiling' not in self.df.columns:
            self.df['Ceiling'] = self.df['Projected_Points'] * 1.4

        if 'Floor' not in self.df.columns:
            self.df['Floor'] = self.df['Projected_Points'] * 0.7

        # Team counts for validation
        self.team_counts = self.df['Team'].value_counts().to_dict()

        self.logger.log(
            f"Data prepared: {len(self.df)} players, {len(self.team_counts)} teams",
            "INFO",
            context={'teams': list(self.team_counts.keys())},
            tags=['data', 'preparation']
        )

    def get_triple_ai_strategies(self, use_api: bool = True) -> Dict:
        """Get strategies from all three AIs with parallel execution"""

        self.logger.log("Getting strategies from three AI strategists", "INFO", tags=['ai', 'strategies'])
        self.perf_monitor.start_timer("get_ai_strategies")

        from config import AIStrategistType
        recommendations = {}

        if use_api and self.api_manager and self.api_manager.client:
            # API mode - parallel execution
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(
                        self.game_theory_ai.get_recommendation,
                        self.df, self.game_info, self.field_size, use_api
                    ): AIStrategistType.GAME_THEORY,
                    executor.submit(
                        self.correlation_ai.get_recommendation,
                        self.df, self.game_info, self.field_size, use_api
                    ): AIStrategistType.CORRELATION,
                    executor.submit(
                        self.contrarian_ai.get_recommendation,
                        self.df, self.game_info, self.field_size, use_api
                    ): AIStrategistType.CONTRARIAN_NARRATIVE
                }

                for future in as_completed(futures):
                    ai_type = futures[future]
                    try:
                        recommendation = future.result(timeout=30)
                        recommendations[ai_type] = recommendation
                        self.logger.log(
                            f"{ai_type.value} recommendation received",
                            "INFO",
                            tags=['ai', ai_type.value.lower()]
                        )
                    except Exception as e:
                        self.logger.log_exception(e, f"{ai_type.value} failed")
                        recommendations[ai_type] = self._get_fallback_recommendation(ai_type)
        else:
            # Manual mode - sequential
            recommendations = self._get_manual_ai_strategies()

        # Ensure all recommendations present
        for ai_type in AIStrategistType:
            if ai_type not in recommendations:
                self.logger.log(f"Missing recommendation for {ai_type.value}", "WARNING")
                recommendations[ai_type] = self._get_fallback_recommendation(ai_type)

        # Log AI decisions
        for ai_type, rec in recommendations.items():
            self.logger.log_ai_decision(
                "strategy_received",
                ai_type.value,
                True,
                {
                    'captains': len(rec.captain_targets),
                    'confidence': rec.confidence,
                    'stacks': len(rec.stacks)
                },
                rec.confidence
            )

        elapsed = self.perf_monitor.stop_timer("get_ai_strategies")
        self.logger.log(f"AI strategies obtained in {elapsed:.2f}s", "INFO")

        return recommendations

    def _get_fallback_recommendation(self, ai_type):
        """Get fallback recommendation for specific AI type"""
        if ai_type.value == "Game Theory":
            return self.game_theory_ai._get_fallback_recommendation(self.df, self.field_size)
        elif ai_type.value == "Correlation":
            return self.correlation_ai._get_fallback_recommendation(self.df, self.field_size)
        else:
            return self.contrarian_ai._get_fallback_recommendation(self.df, self.field_size)

    def _get_manual_ai_strategies(self) -> Dict:
        """Get AI strategies through manual input - placeholder for UI integration"""
        # This would be called from the Streamlit UI
        # For now, return fallbacks
        from config import AIStrategistType

        return {
            AIStrategistType.GAME_THEORY: self.game_theory_ai._get_fallback_recommendation(self.df, self.field_size),
            AIStrategistType.CORRELATION: self.correlation_ai._get_fallback_recommendation(self.df, self.field_size),
            AIStrategistType.CONTRARIAN_NARRATIVE: self.contrarian_ai._get_fallback_recommendation(self.df, self.field_size)
        }

    def synthesize_ai_strategies(self, recommendations: Dict) -> Dict:
        """Synthesize three AI perspectives into unified strategy"""

        self.logger.log("Synthesizing triple AI strategies", "INFO", tags=['synthesis'])

        try:
            # Use synthesis engine
            from config import AIStrategistType
            synthesis = self.synthesis_engine.synthesize_recommendations(
                recommendations.get(AIStrategistType.GAME_THEORY),
                recommendations.get(AIStrategistType.CORRELATION),
                recommendations.get(AIStrategistType.CONTRARIAN_NARRATIVE)
            )

            # Create enforcement rules
            enforcement_rules = self.enforcement_engine.create_enforcement_rules(recommendations)

            # Validate rules
            from enforcement_validation import AIConfigValidator
            validation = AIConfigValidator.validate_ai_requirements(enforcement_rules, self.df)

            if not validation['is_valid']:
                self.logger.log(
                    f"AI requirements validation failed: {validation['errors']}",
                    "WARNING",
                    context={'warnings': validation['warnings']},
                    tags=['validation', 'warning']
                )

            return {
                'synthesis': synthesis,
                'enforcement_rules': enforcement_rules,
                'validation': validation,
                'recommendations': recommendations
            }

        except Exception as e:
            self.logger.log_exception(e, "synthesize_ai_strategies")
            return self._get_fallback_synthesis(recommendations)

    def _get_fallback_synthesis(self, recommendations: Dict) -> Dict:
        """Create fallback synthesis"""
        return {
            'synthesis': {
                'captain_strategy': {},
                'player_rankings': {},
                'stacking_rules': [],
                'enforcement_rules': [],
                'confidence': 0.4,
                'narrative': "Using fallback synthesis"
            },
            'enforcement_rules': {'hard_constraints': [], 'soft_constraints': []},
            'validation': {'is_valid': True, 'errors': [], 'warnings': []},
            'recommendations': recommendations
        }

    def generate_ai_driven_lineups(self, num_lineups: int, ai_strategy: Dict) -> pd.DataFrame:
        """Generate lineups with AI enforcement and intelligent parallelization"""

        self.perf_monitor.start_timer("total_optimization")
        start_time = time.time()

        self.logger.log_optimization_start(num_lineups, self.field_size, {
            'mode': 'AI-as-Chef',
            'enforcement': self.enforcement_engine.enforcement_level.value,
            'parallel': self.max_workers > 1
        })

        # Extract components
        synthesis = ai_strategy.get('synthesis', {})
        enforcement_rules = ai_strategy.get('enforcement_rules', {})

        # Pre-generation validation
        min_salary_lineup = self.df.nsmallest(6, 'Salary')['Salary'].sum()
        from config import OptimizerConfig
        if min_salary_lineup > OptimizerConfig.SALARY_CAP:
            self.logger.log(
                "Cannot create valid lineup - minimum salary exceeds cap!",
                "CRITICAL",
                tags=['validation', 'error']
            )
            return pd.DataFrame()

        # Prepare data structures
        players = self.df['Player'].tolist()
        positions = self.df.set_index('Player')['Position'].to_dict()
        teams = self.df.set_index('Player')['Team'].to_dict()
        salaries = self.df.set_index('Player')['Salary'].to_dict()
        points = self.df.set_index('Player')['Projected_Points'].to_dict()
        ownership = self.df.set_index('Player')['Ownership'].to_dict()

        # Apply AI modifications
        ai_adjusted_points = self._apply_ai_adjustments(points, synthesis)

        # Get strategy distribution
        from enforcement_validation import AIConfigValidator
        consensus_level = self._determine_consensus_level(synthesis)
        strategy_distribution = AIConfigValidator.get_ai_strategy_distribution(
            self.field_size, num_lineups, consensus_level
        )

        self.logger.log(
            f"Strategy distribution: {strategy_distribution}",
            "INFO",
            context={'consensus': consensus_level},
            tags=['strategy']
        )

        all_lineups = []
        used_captains = set()

        # Create lineup tasks
        lineup_tasks = []
        for strategy, count in strategy_distribution.items():
            strategy_name = strategy if isinstance(strategy, str) else strategy.value
            for i in range(count):
                lineup_tasks.append((len(lineup_tasks) + 1, strategy_name))

        # Use parallel or sequential generation
        if self.max_workers > 1 and len(lineup_tasks) > 5:
            all_lineups = self._generate_lineups_parallel(
                lineup_tasks, players, salaries, ai_adjusted_points,
                ownership, positions, teams, enforcement_rules,
                synthesis, used_captains
            )
        else:
            all_lineups = self._generate_lineups_sequential(
                lineup_tasks, players, salaries, ai_adjusted_points,
                ownership, positions, teams, enforcement_rules,
                synthesis, used_captains
            )

        # Calculate final metrics
        total_time = time.time() - start_time
        success_rate = len(all_lineups) / max(num_lineups, 1)

        avg_ownership = np.mean([lu.get('Total_Ownership', 0) for lu in all_lineups]) if all_lineups else 0

        self.logger.log_optimization_end(len(all_lineups), total_time, success_rate, avg_ownership)

        # Calculate diversity metrics
        if all_lineups:
            self._calculate_diversity_metrics(all_lineups)

        # Store generated lineups
        self.generated_lineups = all_lineups

        return pd.DataFrame(all_lineups) if all_lineups else pd.DataFrame()

    def _generate_lineups_parallel(self, lineup_tasks, players, salaries, points,
                                   ownership, positions, teams, enforcement_rules,
                                   synthesis, used_captains):
        """Parallel lineup generation with thread safety"""

        all_lineups = []
        captain_lock = threading.Lock()
        lineup_lock = threading.Lock()

        def generate_single_lineup(task_data):
            lineup_num, strategy_name = task_data

            # Thread-safe captain tracking
            with captain_lock:
                local_used_captains = used_captains.copy()

            lineup = self._build_ai_enforced_lineup(
                lineup_num=lineup_num,
                strategy=strategy_name,
                players=players,
                salaries=salaries,
                points=points,
                ownership=ownership,
                positions=positions,
                teams=teams,
                enforcement_rules=enforcement_rules,
                synthesis=synthesis,
                used_captains=local_used_captains
            )

            if lineup:
                is_valid, violations, details = self.enforcement_engine.validate_lineup_against_ai(
                    lineup, enforcement_rules
                )

                if is_valid:
                    with captain_lock:
                        if lineup['Captain'] not in used_captains:
                            used_captains.add(lineup['Captain'])
                            return lineup

            return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(generate_single_lineup, task) for task in lineup_tasks]

            for future in as_completed(futures):
                try:
                    lineup = future.result(timeout=self.generation_timeout)
                    if lineup:
                        with lineup_lock:
                            all_lineups.append(lineup)
                except Exception as e:
                    self.logger.log(f"Parallel generation error: {e}", "DEBUG", tags=['parallel', 'error'])

        return all_lineups

    def _generate_lineups_sequential(self, lineup_tasks, players, salaries, points,
                                    ownership, positions, teams, enforcement_rules,
                                    synthesis, used_captains):
        """Sequential lineup generation"""

        all_lineups = []

        for lineup_num, strategy_name in lineup_tasks:
            lineup = self._build_ai_enforced_lineup(
                lineup_num=lineup_num,
                strategy=strategy_name,
                players=players,
                salaries=salaries,
                points=points,
                ownership=ownership,
                positions=positions,
                teams=teams,
                enforcement_rules=enforcement_rules,
                synthesis=synthesis,
                used_captains=used_captains
            )

            if lineup:
                is_valid, violations, details = self.enforcement_engine.validate_lineup_against_ai(
                    lineup, enforcement_rules
                )

                if is_valid:
                    all_lineups.append(lineup)
                    used_captains.add(lineup['Captain'])
                    self.logger.log_lineup_generation(
                        strategy_name, lineup_num, "SUCCESS",
                        len(enforcement_rules.get('hard_constraints', []))
                    )
                else:
                    self.lineup_generation_stats['failures_by_reason']['validation'] += 1

        return all_lineups

    def _build_ai_enforced_lineup(self, lineup_num: int, strategy: str, players: List[str],
                                  salaries: Dict, points: Dict, ownership: Dict,
                                  positions: Dict, teams: Dict, enforcement_rules: Dict,
                                  synthesis: Dict, used_captains: Set[str]) -> Optional[Dict]:
        """Build lineup with three-tier constraint relaxation"""

        max_attempts = 3
        constraint_relaxation = [1.0, 0.8, 0.6]

        for attempt in range(max_attempts):
            try:
                self.lineup_generation_stats['attempts'] += 1

                model = pulp.LpProblem(f"AI_Lineup_{lineup_num}_{strategy}_a{attempt}", pulp.LpMaximize)

                # Decision variables
                flex = pulp.LpVariable.dicts("Flex", players, cat='Binary')
                captain = pulp.LpVariable.dicts("Captain", players, cat='Binary')

                # AI-modified objective
                player_weights = synthesis.get('player_rankings', {})

                objective = pulp.lpSum([
                    points[p] * player_weights.get(p, {}).get('normalized_score', 1.0) * flex[p] +
                    1.5 * points[p] * player_weights.get(p, {}).get('normalized_score', 1.0) * captain[p]
                    for p in players
                ])

                model += objective

                # Basic DraftKings constraints
                model += pulp.lpSum(captain.values()) == 1
                model += pulp.lpSum(flex.values()) == 5

                for p in players:
                    model += flex[p] + captain[p] <= 1

                # Salary constraint with relaxation
                from config import OptimizerConfig
                salary_cap = OptimizerConfig.SALARY_CAP + (500 * attempt if attempt > 0 else 0)

                model += pulp.lpSum([
                    salaries[p] * flex[p] + 1.5 * salaries[p] * captain[p]
                    for p in players
                ]) <= salary_cap

                # DraftKings team diversity
                unique_teams = list(set(teams.values()))

                for team in unique_teams:
                    team_players = [p for p in players if teams.get(p) == team]
                    if team_players:
                        model += pulp.lpSum([flex[p] + captain[p] for p in team_players]) >= 1

                max_from_team = 5 if attempt > 1 else OptimizerConfig.MAX_PLAYERS_PER_TEAM
                for team in unique_teams:
                    team_players = [p for p in players if teams.get(p) == team]
                    if team_players:
                        model += pulp.lpSum([flex[p] + captain[p] for p in team_players]) <= max_from_team

                # Apply AI constraints based on attempt
                if attempt == 0:
                    self._apply_strict_ai_constraints(
                        model, flex, captain, enforcement_rules, players,
                        used_captains, synthesis, strategy, teams
                    )
                elif attempt == 1:
                    self._apply_relaxed_ai_constraints(
                        model, flex, captain, enforcement_rules, players,
                        used_captains, constraint_relaxation[attempt]
                    )
                else:
                    self._apply_minimal_constraints(
                        model, captain, players, used_captains, ownership
                    )

                # Solve
                timeout = 5 + (attempt * 5)
                model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout))

                if pulp.LpStatus[model.status] == 'Optimal':
                    lineup = self._extract_lineup_from_solution(
                        flex, captain, players, salaries, points, ownership,
                        lineup_num, strategy, synthesis
                    )

                    if lineup and self._verify_dk_requirements(lineup, teams):
                        self.lineup_generation_stats['successes'] += 1
                        self.lineup_generation_stats['unique_captains'].add(lineup['Captain'])
                        self.lineup_generation_stats['ownership_distribution'].append(
                            lineup.get('Total_Ownership', 0)
                        )

                        if attempt > 0:
                            self.logger.log(
                                f"Lineup {lineup_num} succeeded on attempt {attempt + 1}",
                                "DEBUG",
                                tags=['optimization', 'retry']
                            )
                        return lineup
                    elif lineup:
                        self.lineup_generation_stats['failures_by_reason']['dk_requirements'] += 1
                else:
                    self.lineup_generation_stats['failures_by_reason']['no_solution'] += 1

            except Exception as e:
                self.lineup_generation_stats['failures_by_reason']['exception'] += 1
                self.logger.log(
                    f"Lineup {lineup_num} attempt {attempt + 1} error: {str(e)}",
                    "DEBUG",
                    tags=['optimization', 'error']
                )

        return None

    def _apply_strict_ai_constraints(self, model, flex, captain, enforcement_rules,
                                    players, used_captains, synthesis, strategy, teams):
        """Apply strict AI constraints (first attempt)"""

        # Captain constraints from AI
        valid_captains = []

        for constraint in enforcement_rules.get('hard_constraints', []):
            if constraint.get('rule') in ['captain_selection', 'captain_from_list']:
                rule_captains = [p for p in constraint.get('players', []) if p in players]
                valid_captains.extend(rule_captains)

        # Remove used captains
        valid_captains = list(set([c for c in valid_captains if c not in used_captains]))

        if valid_captains:
            model += pulp.lpSum([captain[c] for c in valid_captains]) == 1

        # Must include/exclude constraints
        for constraint in enforcement_rules.get('hard_constraints', []):
            rule = constraint.get('rule')

            if rule == 'must_include':
                player = constraint.get('player')
                if player and player in players:
                    model += flex[player] + captain[player] >= 1

            elif rule == 'must_exclude':
                player = constraint.get('player')
                if player and player in players:
                    model += flex[player] + captain[player] == 0

        # Stacking constraints
        for constraint in enforcement_rules.get('stacking_rules', []):
            if constraint.get('type') == 'hard' and 'players' in constraint:
                stack_players = [p for p in constraint['players'] if p in players]
                if len(stack_players) >= 2:
                    min_stack = constraint.get('min_players', 2)
                    model += pulp.lpSum([flex[p] + captain[p] for p in stack_players]) >= min(min_stack, len(stack_players))

    def _apply_relaxed_ai_constraints(self, model, flex, captain, enforcement_rules,
                                     players, used_captains, relaxation_factor):
        """Apply relaxed constraints (second attempt)"""

        # Expand captain pool
        top_players = self.df.nlargest(15, 'Projected_Points')['Player'].tolist()
        valid_captains = [p for p in top_players if p in players and p not in used_captains]

        if valid_captains:
            model += pulp.lpSum([captain[c] for c in valid_captains]) == 1

    def _apply_minimal_constraints(self, model, captain, players, used_captains, ownership):
        """Apply minimal constraints (final attempt)"""

        available_captains = [
            p for p in players
            if p not in used_captains and ownership.get(p, 10) < 60
        ]

        if available_captains:
            model += pulp.lpSum([captain[c] for c in available_captains[:20]]) == 1

    def _verify_dk_requirements(self, lineup: Dict, teams: Dict) -> bool:
        """Verify DraftKings Showdown requirements"""

        captain = lineup.get('Captain')
        flex_players = lineup.get('FLEX', [])

        if not captain or len(flex_players) != 5:
            return False

        all_players = [captain] + flex_players

        # Check team representation
        team_counts = defaultdict(int)
        for player in all_players:
            team = teams.get(player)
            if team:
                team_counts[team] += 1

        # Must have at least 1 from each team
        unique_teams = set(teams.values())
        if len(team_counts) < len(unique_teams):
            return False

        # Max 5 from one team
        for count in team_counts.values():
            if count > 5:
                return False

        return True

    def _extract_lineup_from_solution(self, flex, captain, players, salaries,
                                     points, ownership, lineup_num, strategy, synthesis):
        """Extract lineup from solved model"""

        captain_pick = None
        flex_picks = []

        for p in players:
            if captain[p].value() == 1:
                captain_pick = p
            if flex[p].value() == 1:
                flex_picks.append(p)

        if captain_pick and len(flex_picks) == 5:
            total_salary = sum(salaries[p] for p in flex_picks) + 1.5 * salaries[captain_pick]
            total_proj = sum(points[p] for p in flex_picks) + 1.5 * points[captain_pick]
            total_ownership = ownership.get(captain_pick, 5) * 1.5 + sum(
                ownership.get(p, 5) for p in flex_picks
            )

            # Calculate leverage score
            all_players = [captain_pick] + flex_picks
            leverage_score = self.bucket_manager.calculate_gpp_leverage(all_players, self.df)

            # Determine ownership tier
            if total_ownership < 60:
                ownership_tier = 'Elite Contrarian'
            elif total_ownership < 80:
                ownership_tier = 'Optimal'
            elif total_ownership < 100:
                ownership_tier = 'Balanced'
            else:
                ownership_tier = 'Chalky'

            from config import OptimizerConfig

            return {
                'Lineup': lineup_num,
                'Strategy': strategy,
                'Captain': captain_pick,
                'Captain_Own%': ownership.get(captain_pick, 5),
                'FLEX': flex_picks,
                'Projected': round(total_proj, 2),
                'Salary': int(total_salary),
                'Salary_Remaining': int(OptimizerConfig.SALARY_CAP - total_salary),
                'Total_Ownership': round(total_ownership, 1),
                'Ownership_Tier': ownership_tier,
                'AI_Strategy': strategy,
                'AI_Enforced': True,
                'Confidence': synthesis.get('confidence', 0.5),
                'Leverage_Score': round(leverage_score, 2)
            }

        return None

    def _apply_ai_adjustments(self, points: Dict, synthesis: Dict) -> Dict:
        """Apply AI-recommended adjustments to projections"""
        adjusted = points.copy()

        # Apply player rankings as multipliers
        rankings = synthesis.get('player_rankings', {})

        for player, ranking_data in rankings.items():
            if player in adjusted:
                normalized_score = ranking_data.get('normalized_score', 0)

                if normalized_score > 0:
                    multiplier = 1.0 + min(normalized_score * 0.2, 0.4)
                else:
                    multiplier = max(0.7, 1.0 + normalized_score * 0.3)

                adjusted[player] *= multiplier

        return adjusted

    def _determine_consensus_level(self, synthesis: Dict) -> str:
        """Determine consensus level from synthesis"""
        consensus_analysis = synthesis.get('consensus_analysis', {})
        consensus_score = consensus_analysis.get('consensus_score', 0.5)

        if consensus_score > 0.7:
            return 'high'
        elif consensus_score > 0.4:
            return 'mixed'
        else:
            return 'low'

    def _calculate_diversity_metrics(self, lineups: List[Dict]):
        """Calculate lineup diversity metrics"""

        if not lineups:
            return

        # Captain diversity
        captain_diversity = len(set(lu['Captain'] for lu in lineups)) / len(lineups)

        # Average unique players per lineup
        all_player_sets = []
        for lu in lineups:
            players = set([lu['Captain']] + lu['FLEX'])
            all_player_sets.append(players)

        # Jaccard similarity between lineups
        similarities = []
        for i in range(len(all_player_sets)):
            for j in range(i + 1, len(all_player_sets)):
                intersection = len(all_player_sets[i] & all_player_sets[j])
                union = len(all_player_sets[i] | all_player_sets[j])
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)

        avg_similarity = np.mean(similarities) if similarities else 0
        diversity_score = 1 - avg_similarity

        self.lineup_generation_stats['diversity_metrics'] = {
            'captain_diversity': captain_diversity,
            'avg_jaccard_similarity': avg_similarity,
            'diversity_score': diversity_score,
            'unique_captains': len(self.lineup_generation_stats['unique_captains'])
        }

        self.logger.log(
            f"Diversity metrics: {diversity_score:.2%} score, "
            f"{captain_diversity:.2%} captain diversity",
            "INFO",
            context=self.lineup_generation_stats['diversity_metrics'],
            tags=['diversity', 'metrics']
        )

"""
Streamlit UI utility functions
Improvements: Enhanced visualizations, export functions, session management, user feedback
"""

def init_ai_session_state():
    """Initialize session state with memory management"""
    if 'ai_recommendations' not in st.session_state:
        st.session_state['ai_recommendations'] = {}
    if 'ai_synthesis' not in st.session_state:
        st.session_state['ai_synthesis'] = None
    if 'ai_enforcement_history' not in st.session_state:
        st.session_state['ai_enforcement_history'] = deque(maxlen=100)
    if 'optimization_history' not in st.session_state:
        st.session_state['optimization_history'] = deque(maxlen=10)
    if 'ai_mode' not in st.session_state:
        st.session_state['ai_mode'] = 'enforced'
    if 'api_manager' not in st.session_state:
        st.session_state['api_manager'] = None
    if 'lineups_df' not in st.session_state:
        st.session_state['lineups_df'] = pd.DataFrame()
    if 'df' not in st.session_state:
        st.session_state['df'] = pd.DataFrame()
    if 'last_optimization_time' not in st.session_state:
        st.session_state['last_optimization_time'] = None
    if 'optimization_count' not in st.session_state:
        st.session_state['optimization_count'] = 0
    if 'user_preferences' not in st.session_state:
        st.session_state['user_preferences'] = {}


def validate_and_process_dataframe(df) -> Tuple[pd.DataFrame, Dict]:
    """Enhanced validation and processing with auto-correction"""

    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'fixes_applied': [],
        'stats': {}
    }

    try:
        # Column mapping
        column_mappings = {
            'first_name': 'First_Name',
            'last_name': 'Last_Name',
            'position': 'Position',
            'team': 'Team',
            'salary': 'Salary',
            'ppg_projection': 'Projected_Points',
            'ownership_projection': 'Ownership',
            'name': 'Player',
            'proj': 'Projected_Points',
            'own': 'Ownership',
            'sal': 'Salary',
            'pos': 'Position'
        }

        # Rename columns
        df = df.rename(columns={k.lower(): v for k, v in column_mappings.items()})

        # Create Player column
        if 'Player' not in df.columns:
            if 'First_Name' in df.columns and 'Last_Name' in df.columns:
                df['Player'] = df['First_Name'].fillna('') + ' ' + df['Last_Name'].fillna('')
                df['Player'] = df['Player'].str.strip()
                validation['fixes_applied'].append("Created Player names from first/last")
            else:
                validation['errors'].append("Cannot determine player names")
                validation['is_valid'] = False
                return df, validation

        # Ensure numeric columns
        numeric_columns = ['Salary', 'Projected_Points', 'Ownership']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                na_count = df[col].isna().sum()
                if na_count > 0:
                    validation['warnings'].append(f"{na_count} {col} values couldn't be converted")

                    if col == 'Salary':
                        from config import OptimizerConfig
                        df[col] = df[col].fillna(OptimizerConfig.MIN_SALARY)
                    elif col == 'Projected_Points':
                        df[col] = df[col].fillna(df[col].median())
                    elif col == 'Ownership':
                        from config import OptimizerConfig
                        df[col] = df[col].fillna(OptimizerConfig.DEFAULT_OWNERSHIP)

        # Add ownership if missing
        if 'Ownership' not in df.columns:
            from config import OptimizerConfig
            df['Ownership'] = df.apply(
                lambda row: OptimizerConfig.get_default_ownership(
                    row.get('Position', 'FLEX'),
                    row.get('Salary', 5000)
                ), axis=1
            )
            validation['fixes_applied'].append("Added projected ownership")

        # Validate ownership range
        if 'Ownership' in df.columns:
            df.loc[df['Ownership'] < 0, 'Ownership'] = 0
            df.loc[df['Ownership'] > 100, 'Ownership'] = 100

        # Remove duplicates
        if df.duplicated(subset=['Player']).any():
            dup_count = df.duplicated(subset=['Player']).sum()
            df = df.drop_duplicates(subset=['Player'], keep='first')
            validation['warnings'].append(f"Removed {dup_count} duplicate players")

        # Statistics
        validation['stats'] = {
            'total_players': len(df),
            'teams': len(df['Team'].unique()),
            'positions': df['Position'].value_counts().to_dict(),
            'min_salary': df['Salary'].min(),
            'max_salary': df['Salary'].max(),
            'avg_projection': df['Projected_Points'].mean()
        }

        # Validate minimums
        if len(df) < 6:
            validation['errors'].append(f"Only {len(df)} players (minimum 6)")
            validation['is_valid'] = False

        # Validate teams
        if len(df['Team'].unique()) != 2:
            validation['warnings'].append(f"Expected 2 teams, found {len(df['Team'].unique())}")

        # Validate salary feasibility
        from config import OptimizerConfig
        min_lineup_salary = df.nsmallest(6, 'Salary')['Salary'].sum()
        if min_lineup_salary > OptimizerConfig.SALARY_CAP:
            validation['errors'].append("Minimum salary exceeds cap")
            validation['is_valid'] = False

    except Exception as e:
        validation['errors'].append(f"Processing error: {str(e)}")
        validation['is_valid'] = False

    return df, validation


def display_ai_recommendations(recommendations: Dict):
    """Enhanced AI recommendation display"""

    st.markdown("### Triple AI Strategic Analysis")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_confidence = np.mean([rec.confidence for rec in recommendations.values()])
        st.metric("Avg Confidence", f"{avg_confidence:.0%}")

    with col2:
        total_captains = len(set(
            captain for rec in recommendations.values()
            for captain in rec.captain_targets
        ))
        st.metric("Unique Captains", total_captains)

    with col3:
        total_stacks = sum(len(rec.stacks) for rec in recommendations.values())
        st.metric("Total Stacks", total_stacks)

    with col4:
        total_rules = sum(len(rec.enforcement_rules) for rec in recommendations.values())
        st.metric("Enforcement Rules", total_rules)

    # Detailed tabs
    from config import AIStrategistType

    tab1, tab2, tab3 = st.tabs(["Game Theory", "Correlation", "Contrarian"])

    with tab1:
        display_single_ai_recommendation(
            recommendations.get(AIStrategistType.GAME_THEORY),
            "Game Theory"
        )

    with tab2:
        display_single_ai_recommendation(
            recommendations.get(AIStrategistType.CORRELATION),
            "Correlation"
        )

    with tab3:
        display_single_ai_recommendation(
            recommendations.get(AIStrategistType.CONTRARIAN_NARRATIVE),
            "Contrarian"
        )


def display_single_ai_recommendation(rec, name: str):
    """Display single AI recommendation with enhanced formatting"""

    if not rec:
        st.warning(f"No {name} recommendation available")
        return

    try:
        # Confidence with color
        if rec.confidence > 0.7:
            st.success(f"**{name} Strategy** - High Confidence: {rec.confidence:.0%}")
        elif rec.confidence > 0.5:
            st.info(f"**{name} Strategy** - Moderate Confidence: {rec.confidence:.0%}")
        else:
            st.warning(f"**{name} Strategy** - Low Confidence: {rec.confidence:.0%}")

        col1, col2 = st.columns(2)

        with col1:
            if rec.narrative:
                st.markdown("**Narrative:**")
                st.write(rec.narrative[:250])

            if rec.captain_targets:
                st.markdown("**Captain Targets:**")
                for i, captain in enumerate(rec.captain_targets[:5], 1):
                    st.write(f"{i}. {captain}")

        with col2:
            if rec.must_play:
                st.markdown("**Must Play:**")
                for player in rec.must_play[:5]:
                    st.write(f"✓ {player}")

            if rec.never_play:
                st.markdown("**Fade:**")
                for player in rec.never_play[:3]:
                    st.write(f"✗ {player}")

        if rec.stacks:
            with st.expander(f"Stacks ({len(rec.stacks)})"):
                for stack in rec.stacks[:5]:
                    if isinstance(stack, dict):
                        stack_type = stack.get('type', 'standard')
                        st.write(f"• **{stack_type}**: {stack}")

    except Exception as e:
        st.error(f"Error displaying {name}: {str(e)}")


def display_ai_synthesis(synthesis: Dict):
    """Enhanced synthesis display"""

    st.markdown("### AI Synthesis & Consensus")

    consensus_score = synthesis.get('confidence', 0) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Consensus")
        if consensus_score > 70:
            st.success(f"HIGH ({consensus_score:.0f}%)")
        elif consensus_score > 50:
            st.warning(f"MODERATE ({consensus_score:.0f}%)")
        else:
            st.info(f"LOW ({consensus_score:.0f}%)")

    with col2:
        captain_strategy = synthesis.get('captain_strategy', {})
        st.markdown("#### Captain Agreement")
        unanimous = len([c for c, data in captain_strategy.items()
                        if data.get('level') == 'unanimous'])
        majority = len([c for c, data in captain_strategy.items()
                       if data.get('level') == 'majority'])
        st.write(f"Unanimous: {unanimous}")
        st.write(f"Majority: {majority}")

    with col3:
        st.markdown("#### Enforcement")
        st.write(f"Rules: {len(synthesis.get('enforcement_rules', []))}")
        st.write(f"Stacks: {len(synthesis.get('stacking_rules', []))}")

    # Patterns
    if synthesis.get('patterns'):
        with st.expander("Analysis Patterns"):
            for pattern in synthesis['patterns']:
                st.write(f"• {pattern}")


def export_lineups_draftkings(lineups_df) -> str:
    """Export in DraftKings format"""
    try:
        dk_lineups = []

        for _, row in lineups_df.iterrows():
            flex_players = row['FLEX'] if isinstance(row['FLEX'], list) else []

            while len(flex_players) < 5:
                flex_players.append('')

            dk_lineups.append({
                'CPT': row.get('Captain', ''),
                'FLEX 1': flex_players[0],
                'FLEX 2': flex_players[1],
                'FLEX 3': flex_players[2],
                'FLEX 4': flex_players[3],
                'FLEX 5': flex_players[4]
            })

        return pd.DataFrame(dk_lineups).to_csv(index=False)

    except Exception as e:
        get_logger().log_exception(e, "export_draftkings")
        return ""


def export_detailed_lineups(lineups_df) -> str:
    """Export detailed lineup info"""
    try:
        detailed = []

        for idx, row in lineups_df.iterrows():
            lineup_detail = {
                'Lineup': row.get('Lineup', idx + 1),
                'Strategy': row.get('Strategy', ''),
                'Captain': row.get('Captain', ''),
                'Captain_Own%': row.get('Captain_Own%', 0),
                'Projected': row.get('Projected', 0),
                'Salary': row.get('Salary', 0),
                'Total_Ownership': row.get('Total_Ownership', 0),
                'Ownership_Tier': row.get('Ownership_Tier', ''),
                'Leverage_Score': row.get('Leverage_Score', 0),
            }

            # Add FLEX
            flex_players = row.get('FLEX', [])
            for i in range(5):
                lineup_detail[f'FLEX_{i+1}'] = flex_players[i] if i < len(flex_players) else ''

            detailed.append(lineup_detail)

        return pd.DataFrame(detailed).to_csv(index=False)

    except Exception as e:
        get_logger().log_exception(e, "export_detailed")
        return ""


def create_sample_data() -> pd.DataFrame:
    """Create sample data for testing"""

    sample_data = {
        'Player': [
            'Josh Allen', 'Stefon Diggs', 'Gabe Davis', 'Dawson Knox', 'James Cook',
            'Devin Singletary', 'Isaiah McKenzie', 'Cole Beasley',
            'Tua Tagovailoa', 'Tyreek Hill', 'Jaylen Waddle', 'Mike Gesicki',
            'Raheem Mostert', 'Jeff Wilson Jr', 'Cedrick Wilson', 'Durham Smythe'
        ],
        'Position': [
            'QB', 'WR', 'WR', 'TE', 'RB', 'RB', 'WR', 'WR',
            'QB', 'WR', 'WR', 'TE', 'RB', 'RB', 'WR', 'TE'
        ],
        'Team': [
            'BUF', 'BUF', 'BUF', 'BUF', 'BUF', 'BUF', 'BUF', 'BUF',
            'MIA', 'MIA', 'MIA', 'MIA', 'MIA', 'MIA', 'MIA', 'MIA'
        ],
        'Salary': [
            11200, 10800, 8600, 6800, 7200, 5800, 4800, 4200,
            10600, 11000, 9400, 6200, 6600, 5200, 4600, 3800
        ],
        'Projected_Points': [
            23.5, 19.2, 14.8, 11.2, 12.5, 9.8, 8.2, 7.1,
            21.8, 20.5, 16.3, 10.5, 11.8, 8.9, 7.8, 6.2
        ],
        'Ownership': [
            18.5, 22.3, 15.2, 8.5, 12.1, 6.8, 4.2, 3.1,
            16.2, 25.8, 18.9, 7.2, 10.5, 5.5, 3.8, 2.1
        ]
    }

    return pd.DataFrame(sample_data)
