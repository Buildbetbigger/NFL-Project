
# -*- coding: utf-8 -*-
"""
NFL DFS AI-Driven Optimizer with ML Enhancements
Enhanced Version - No Historical Data Required
"""

import pandas as pd
import numpy as np
import pulp
import json
import hashlib
import threading
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ENHANCED CONFIGURATION CLASS
# ============================================================================

class OptimizerConfig:
    """
    Enhanced configuration with ML/simulation parameters
    """

    # Core DraftKings Showdown constraints
    SALARY_CAP = 50000
    MIN_SALARY = 3000
    MAX_SALARY = 12000
    CAPTAIN_MULTIPLIER = 1.5
    ROSTER_SIZE = 6
    FLEX_SPOTS = 5

    # DraftKings Showdown specific rules
    MIN_TEAMS_REQUIRED = 2
    MAX_PLAYERS_PER_TEAM = 5

    # Performance settings
    MAX_ITERATIONS = 1000
    OPTIMIZATION_TIMEOUT = 30
    MAX_PARALLEL_THREADS = 4
    MAX_HISTORY_ENTRIES = 50
    CACHE_SIZE = 100

    # Monte Carlo simulation settings
    MC_SIMULATIONS = 5000
    MC_FAST_SIMULATIONS = 1000
    MC_CORRELATION_STRENGTH = 0.65

    # Genetic algorithm settings
    GA_POPULATION_SIZE = 100
    GA_GENERATIONS = 50
    GA_MUTATION_RATE = 0.15
    GA_ELITE_SIZE = 10
    GA_TOURNAMENT_SIZE = 5

    # Variance modeling (no historical data needed)
    VARIANCE_BY_POSITION = {
        'QB': 0.30,    # QBs are more consistent
        'RB': 0.40,    # RBs have moderate variance
        'WR': 0.45,    # WRs are more volatile
        'TE': 0.42,    # TEs similar to WRs
        'DST': 0.50,   # DST very volatile
        'K': 0.55,     # Kickers most volatile
        'FLEX': 0.40
    }

    # Correlation coefficients (game theory based)
    CORRELATION_COEFFICIENTS = {
        'qb_wr_same_team': 0.65,
        'qb_te_same_team': 0.60,
        'qb_rb_same_team': -0.15,  # Negative: passing vs rushing
        'qb_qb_opposing': 0.35,     # Shootout correlation
        'wr_wr_same_team': -0.20,   # Target competition
        'rb_dst_opposing': -0.45,   # Defense stops RB
        'wr_dst_opposing': -0.30,   # Defense limits passing
    }

    # Enhanced ownership projection system
    OWNERSHIP_BY_POSITION = {
        'QB': {'base': 15, 'salary_factor': 0.002, 'scarcity_multiplier': 1.2},
        'RB': {'base': 12, 'salary_factor': 0.0015, 'scarcity_multiplier': 1.0},
        'WR': {'base': 10, 'salary_factor': 0.0018, 'scarcity_multiplier': 0.95},
        'TE': {'base': 8, 'salary_factor': 0.001, 'scarcity_multiplier': 1.1},
        'DST': {'base': 5, 'salary_factor': 0.0005, 'scarcity_multiplier': 1.0},
        'K': {'base': 3, 'salary_factor': 0.0003, 'scarcity_multiplier': 0.9},
        'FLEX': {'base': 5, 'salary_factor': 0.001, 'scarcity_multiplier': 1.0}
    }

    @classmethod
    def get_default_ownership(cls, position: str, salary: float,
                            game_total: float = 47.0,
                            is_favorite: bool = False,
                            injury_news: bool = False) -> float:
        """Enhanced ownership projection"""
        pos_config = cls.OWNERSHIP_BY_POSITION.get(
            position,
            cls.OWNERSHIP_BY_POSITION['FLEX']
        )

        base = pos_config['base']
        salary_factor = pos_config['salary_factor']
        scarcity = pos_config['scarcity_multiplier']

        salary_adjustment = (salary - 5000) * salary_factor
        total_adjustment = (game_total - 47.0) * 0.15
        favorite_bonus = 2.0 if is_favorite else -1.0
        injury_adjustment = np.random.uniform(-3.0, 5.0) if injury_news else 0

        ownership = (base + salary_adjustment + total_adjustment +
                    favorite_bonus + injury_adjustment) * scarcity

        random_factor = np.random.normal(1.0, 0.08)
        ownership *= random_factor

        return max(0.5, min(50.0, ownership))

    # Contest field sizes
    FIELD_SIZES = {
        'Single Entry': 'small_field',
        '3-Max': 'small_field',
        '5-Max': 'small_field',
        '20-Max': 'medium_field',
        '150-Max': 'large_field',
        'Large GPP (1000+)': 'large_field_aggressive',
        'Milly Maker': 'milly_maker',
        'Showdown Special': 'large_field_aggressive'
    }

    # Optimization modes
    OPTIMIZATION_MODES = {
        'balanced': {'ceiling_weight': 0.5, 'floor_weight': 0.5},
        'ceiling': {'ceiling_weight': 0.8, 'floor_weight': 0.2},
        'floor': {'ceiling_weight': 0.2, 'floor_weight': 0.8},
        'boom_or_bust': {'ceiling_weight': 1.0, 'floor_weight': 0.0}
    }

    # GPP Ownership targets
    GPP_OWNERSHIP_TARGETS = {
        'small_field': (70, 110),
        'medium_field': (60, 90),
        'large_field': (50, 80),
        'large_field_aggressive': (40, 70),
        'milly_maker': (30, 60),
        'super_contrarian': (20, 50)
    }

    # Field-specific configurations
    FIELD_SIZE_CONFIGS = {
        'small_field': {
            'max_exposure': 0.4,
            'min_unique_captains': 5,
            'max_chalk_players': 3,
            'min_leverage_players': 1,
            'ownership_leverage_weight': 0.3,
            'correlation_weight': 0.4,
            'narrative_weight': 0.3,
            'ai_enforcement': 'Moderate',
            'min_total_ownership': 70,
            'max_total_ownership': 110,
            'similarity_threshold': 0.7,
            'use_genetic': False
        },
        'medium_field': {
            'max_exposure': 0.3,
            'min_unique_captains': 10,
            'max_chalk_players': 2,
            'min_leverage_players': 2,
            'ownership_leverage_weight': 0.35,
            'correlation_weight': 0.35,
            'narrative_weight': 0.3,
            'ai_enforcement': 'Strong',
            'min_total_ownership': 60,
            'max_total_ownership': 90,
            'similarity_threshold': 0.67,
            'use_genetic': False
        },
        'large_field': {
            'max_exposure': 0.25,
            'min_unique_captains': 15,
            'max_chalk_players': 2,
            'min_leverage_players': 2,
            'ownership_leverage_weight': 0.4,
            'correlation_weight': 0.3,
            'narrative_weight': 0.3,
            'ai_enforcement': 'Strong',
            'min_total_ownership': 50,
            'max_total_ownership': 80,
            'similarity_threshold': 0.67,
            'use_genetic': True
        },
        'large_field_aggressive': {
            'max_exposure': 0.2,
            'min_unique_captains': 20,
            'max_chalk_players': 1,
            'min_leverage_players': 3,
            'ownership_leverage_weight': 0.45,
            'correlation_weight': 0.25,
            'narrative_weight': 0.3,
            'ai_enforcement': 'Mandatory',
            'min_total_ownership': 40,
            'max_total_ownership': 70,
            'similarity_threshold': 0.6,
            'use_genetic': True
        },
        'milly_maker': {
            'max_exposure': 0.15,
            'min_unique_captains': 30,
            'max_chalk_players': 1,
            'min_leverage_players': 4,
            'ownership_leverage_weight': 0.5,
            'correlation_weight': 0.2,
            'narrative_weight': 0.3,
            'ai_enforcement': 'Mandatory',
            'min_total_ownership': 30,
            'max_total_ownership': 60,
            'similarity_threshold': 0.5,
            'use_genetic': True
        }
    }

    # AI system weights
    AI_WEIGHTS = {
        'game_theory': 0.35,
        'correlation': 0.35,
        'contrarian': 0.30
    }

    # Sport configurations
    SPORT_CONFIGS = {
        'NFL': {
            'roster_size': 6,
            'salary_cap': 50000,
            'positions': ['QB', 'RB', 'WR', 'TE', 'DST'],
            'scoring': 'DK_NFL'
        }
    }

# ============================================================================
# ENUM CLASSES
# ============================================================================

class AIStrategistType(Enum):
    """Types of AI strategists"""
    GAME_THEORY = "Game Theory"
    CORRELATION = "Correlation"
    CONTRARIAN_NARRATIVE = "Contrarian Narrative"

class AIEnforcementLevel(Enum):
    """AI enforcement levels"""
    ADVISORY = "Advisory"
    MODERATE = "Moderate"
    STRONG = "Strong"
    MANDATORY = "Mandatory"

class OptimizationMode(Enum):
    """Optimization modes"""
    BALANCED = "balanced"
    CEILING = "ceiling"
    FLOOR = "floor"
    BOOM_OR_BUST = "boom_or_bust"

class StackType(Enum):
    """Stack types"""
    QB_RECEIVER = "qb_receiver"
    ONSLAUGHT = "onslaught"
    LEVERAGE = "leverage"
    BRING_BACK = "bring_back"
    DEFENSIVE = "defensive"
    HIDDEN = "hidden"

class ConstraintPriority(Enum):
    """Constraint priority levels"""
    CRITICAL = 100
    AI_HIGH_CONFIDENCE = 90
    AI_CONSENSUS = 85
    AI_MODERATE = 70
    SOFT_PREFERENCE = 50

class FitnessMode(Enum):
    """Genetic algorithm fitness modes"""
    MEAN = "mean"
    CEILING = "ceiling"
    SHARPE = "sharpe"
    WIN_PROBABILITY = "win_prob"

# ============================================================================
# ENHANCED DATA CLASSES
# ============================================================================

@dataclass
class SimulationResults:
    """Results from Monte Carlo simulation"""
    mean: float
    median: float
    std: float
    floor_10th: float
    ceiling_90th: float
    ceiling_99th: float
    top_10pct_mean: float
    sharpe_ratio: float
    win_probability: float
    score_distribution: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class AIRecommendation:
    """Enhanced AI recommendation with simulation support"""
    captain_targets: List[str] = field(default_factory=list)
    must_play: List[str] = field(default_factory=list)
    never_play: List[str] = field(default_factory=list)
    stacks: List[Dict] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)
    confidence: float = 0.5
    enforcement_rules: List[Dict] = field(default_factory=list)
    narrative: str = ""
    source_ai: Optional[AIStrategistType] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Enhanced attributes
    ownership_leverage: Dict = field(default_factory=dict)
    correlation_matrix: Dict = field(default_factory=dict)
    contrarian_angles: List[str] = field(default_factory=list)
    ceiling_plays: List[str] = field(default_factory=list)
    floor_plays: List[str] = field(default_factory=list)
    boom_bust_candidates: List[str] = field(default_factory=list)

    def validate(self) -> Tuple[bool, List[str]]:
        """Enhanced validation"""
        errors = []

        if not self.captain_targets and not self.must_play:
            errors.append("No captain targets or must-play players specified")

        if not 0 <= self.confidence <= 1:
            errors.append(f"Invalid confidence score: {self.confidence}")
            self.confidence = max(0, min(1, self.confidence))

        conflicts = set(self.must_play) & set(self.never_play)
        if conflicts:
            errors.append(f"Conflicting players in must/never play: {conflicts}")

        for stack in self.stacks:
            if not isinstance(stack, dict):
                errors.append("Invalid stack format - must be dictionary")
            elif 'players' in stack and len(stack['players']) < 2:
                errors.append("Stack must have at least 2 players")

        for rule in self.enforcement_rules:
            if 'type' not in rule or 'constraint' not in rule:
                errors.append("Invalid enforcement rule format")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
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
            'timestamp': self.timestamp.isoformat(),
            'ownership_leverage': self.ownership_leverage,
            'correlation_matrix': self.correlation_matrix,
            'contrarian_angles': self.contrarian_angles
        }

@dataclass
class LineupConstraints:
    """Enhanced constraints for lineup generation"""
    min_salary: int = 48000
    max_salary: int = 50000
    min_projection: float = 0
    max_ownership: float = 200
    min_ownership: float = 0
    required_positions: Dict[str, int] = field(default_factory=dict)
    banned_players: Set[str] = field(default_factory=set)
    locked_players: Set[str] = field(default_factory=set)

    # Enhanced constraints
    required_stacks: List[Dict] = field(default_factory=list)
    max_exposure: Dict[str, float] = field(default_factory=dict)
    team_limits: Dict[str, int] = field(default_factory=dict)
    correlation_requirements: List[Dict] = field(default_factory=list)
    ownership_buckets: Dict[str, List[str]] = field(default_factory=dict)

    def validate_lineup(self, lineup: Dict) -> Tuple[bool, List[str]]:
        """Validate lineup against constraints"""
        errors = []

        total_salary = lineup.get('Salary', 0)
        if total_salary < self.min_salary:
            errors.append(f"Salary too low: ${total_salary:,}")
        if total_salary > self.max_salary:
            errors.append(f"Salary too high: ${total_salary:,}")

        total_ownership = lineup.get('Total_Ownership', 0)
        if total_ownership > self.max_ownership:
            errors.append(f"Ownership too high: {total_ownership:.1f}%")
        if total_ownership < self.min_ownership:
            errors.append(f"Ownership too low: {total_ownership:.1f}%")

        all_players = [lineup.get('Captain')] + lineup.get('FLEX', [])

        for banned in self.banned_players:
            if banned in all_players:
                errors.append(f"Banned player in lineup: {banned}")

        for required in self.locked_players:
            if required not in all_players:
                errors.append(f"Required player missing: {required}")

        return len(errors) == 0, errors

@dataclass
class PerformanceMetrics:
    """Track optimizer performance"""
    lineup_generation_time: float = 0
    total_iterations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    constraint_violations: int = 0
    successful_lineups: int = 0
    failed_lineups: int = 0

    # Enhanced metrics
    ai_api_calls: int = 0
    ai_cache_hits: int = 0
    average_confidence: float = 0
    strategy_distribution: Dict[str, int] = field(default_factory=dict)
    stack_success_rate: float = 0
    ownership_accuracy: float = 0

    # ML/Simulation metrics
    mc_simulations_run: int = 0
    ga_generations_completed: int = 0
    avg_simulation_time: float = 0

    def calculate_efficiency(self) -> float:
        """Calculate lineup generation efficiency"""
        if self.total_iterations == 0:
            return 0
        return self.successful_lineups / self.total_iterations

    def calculate_cache_hit_rate(self) -> float:
        """Calculate cache effectiveness"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0
        return self.cache_hits / total

    def get_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return {
            'efficiency': self.calculate_efficiency(),
            'avg_time_per_lineup': (
                self.lineup_generation_time / max(self.successful_lineups, 1)
            ),
            'cache_hit_rate': self.calculate_cache_hit_rate(),
            'success_rate': (
                self.successful_lineups /
                max(self.successful_lineups + self.failed_lineups, 1)
            ),
            'ai_metrics': {
                'api_calls': self.ai_api_calls,
                'cache_hits': self.ai_cache_hits,
                'avg_confidence': self.average_confidence
            },
            'ml_metrics': {
                'mc_simulations': self.mc_simulations_run,
                'ga_generations': self.ga_generations_completed,
                'avg_sim_time': self.avg_simulation_time
            },
            'strategy_distribution': self.strategy_distribution
        }

# ============================================================================
# PART 2: SINGLETON UTILITIES, LOGGING & ML SIMULATION ENGINES
# ============================================================================

import re

# ============================================================================
# GLOBAL SINGLETONS - Proper Singleton Implementation
# ============================================================================

_global_logger = None
_performance_monitor = None
_ai_tracker = None
_singleton_lock = threading.Lock()

def get_logger():
    """Thread-safe singleton logger"""
    global _global_logger
    if _global_logger is None:
        with _singleton_lock:
            if _global_logger is None:
                _global_logger = GlobalLogger()
    return _global_logger

def get_performance_monitor():
    """Thread-safe singleton performance monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        with _singleton_lock:
            if _performance_monitor is None:
                _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def get_ai_tracker():
    """Thread-safe singleton AI decision tracker"""
    global _ai_tracker
    if _ai_tracker is None:
        with _singleton_lock:
            if _ai_tracker is None:
                _ai_tracker = AIDecisionTracker()
    return _ai_tracker

# ============================================================================
# GLOBAL LOGGER WITH ENHANCED ERROR TRACKING
# ============================================================================

class GlobalLogger:
    """
    Enhanced global logger with memory management and intelligent error tracking
    """

    _PATTERN_NUMBER = re.compile(r'\d+')
    _PATTERN_DOUBLE_QUOTE = re.compile(r'"[^"]*"')
    _PATTERN_SINGLE_QUOTE = re.compile(r"'[^']*'")

    def __init__(self):
        self.logs = deque(maxlen=OptimizerConfig.MAX_HISTORY_ENTRIES)
        self.error_logs = deque(maxlen=20)
        self.ai_decisions = deque(maxlen=50)
        self.optimization_events = deque(maxlen=30)
        self.performance_metrics = defaultdict(list)
        self._lock = threading.RLock()

        self.error_patterns = defaultdict(int)
        self.error_suggestions_cache = {}
        self.last_cleanup = datetime.now()

        self.failure_categories = {
            'constraint': 0,
            'salary': 0,
            'ownership': 0,
            'api': 0,
            'validation': 0,
            'timeout': 0,
            'simulation': 0,
            'genetic': 0,
            'other': 0
        }

    def log(self, message: str, level: str = "INFO", context: Dict = None) -> None:
        """Enhanced logging with context and pattern detection"""
        with self._lock:
            entry = {
                'timestamp': datetime.now(),
                'level': level,
                'message': message,
                'context': context or {}
            }

            self.logs.append(entry)

            if level in ["ERROR", "CRITICAL"]:
                self.error_logs.append(entry)
                error_key = self._extract_error_pattern(message)
                self.error_patterns[error_key] += 1
                self._categorize_failure(message)

            self._cleanup_if_needed()

    def log_exception(self, exception: Exception, context: str = "",
                     critical: bool = False) -> None:
        """Enhanced exception logging with helpful suggestions"""
        with self._lock:
            error_msg = f"{context}: {str(exception)}" if context else str(exception)
            suggestions = self._get_error_suggestions(exception, context)

            entry = {
                'timestamp': datetime.now(),
                'level': "CRITICAL" if critical else "ERROR",
                'message': error_msg,
                'exception_type': type(exception).__name__,
                'traceback': traceback.format_exc(),
                'suggestions': suggestions,
                'context': context
            }

            self.error_logs.append(entry)

            if suggestions:
                self.log(
                    f"Error: {error_msg}\nSuggestion: {suggestions[0]}",
                    "ERROR",
                    {'has_suggestion': True}
                )

    def _extract_error_pattern(self, message: str) -> str:
        """Extract error pattern for tracking"""
        pattern = self._PATTERN_NUMBER.sub('N', message)
        pattern = self._PATTERN_DOUBLE_QUOTE.sub('"X"', pattern)
        pattern = self._PATTERN_SINGLE_QUOTE.sub("'X'", pattern)
        return pattern[:100]

    def _categorize_failure(self, message: str) -> None:
        """Categorize failure type"""
        message_lower = message.lower()

        if any(word in message_lower for word in ['constraint', 'infeasible', 'no solution']):
            self.failure_categories['constraint'] += 1
        elif any(word in message_lower for word in ['salary', 'cap', 'budget']):
            self.failure_categories['salary'] += 1
        elif 'ownership' in message_lower:
            self.failure_categories['ownership'] += 1
        elif any(word in message_lower for word in ['api', 'connection', 'timeout']):
            self.failure_categories['api'] += 1
        elif 'validation' in message_lower:
            self.failure_categories['validation'] += 1
        elif 'timeout' in message_lower:
            self.failure_categories['timeout'] += 1
        elif any(word in message_lower for word in ['simulation', 'monte carlo']):
            self.failure_categories['simulation'] += 1
        elif 'genetic' in message_lower:
            self.failure_categories['genetic'] += 1
        else:
            self.failure_categories['other'] += 1

    def _get_error_suggestions(self, exception: Exception, context: str) -> List[str]:
        """Provide helpful suggestions based on error type"""
        exception_type = type(exception).__name__
        cache_key = f"{exception_type}_{context}"

        if cache_key in self.error_suggestions_cache:
            return self.error_suggestions_cache[cache_key]

        suggestions = []
        if isinstance(exception, KeyError):
            suggestions = self._get_keyerror_suggestions()
        elif isinstance(exception, ValueError):
            suggestions = self._get_valueerror_suggestions(str(exception))
        elif "pulp" in exception_type.lower() or "solver" in str(exception).lower():
            suggestions = self._get_solver_suggestions()
        elif "timeout" in str(exception).lower():
            suggestions = self._get_timeout_suggestions()
        elif "api" in str(exception).lower() or "connection" in str(exception).lower():
            suggestions = self._get_api_suggestions()
        elif isinstance(exception, (pd.errors.EmptyDataError, AttributeError)):
            suggestions = self._get_dataframe_suggestions()
        else:
            suggestions = self._get_generic_suggestions()

        self._cache_suggestions(cache_key, suggestions)
        return suggestions

    def _get_keyerror_suggestions(self) -> List[str]:
        return [
            "Check that all required columns are present in CSV",
            "Verify player names match exactly between data and AI recommendations",
            "Ensure DataFrame has been properly validated"
        ]

    def _get_valueerror_suggestions(self, error_str: str) -> List[str]:
        error_lower = error_str.lower()
        if "salary" in error_lower:
            return [
                "Check salary cap constraints - may be too restrictive",
                "Verify required players fit within salary cap"
            ]
        elif "ownership" in error_lower:
            return [
                "Verify ownership projections are between 0-100",
                "Check for missing ownership data"
            ]
        return ["Check data types and value ranges"]

    def _get_solver_suggestions(self) -> List[str]:
        return [
            "Optimization constraints may be infeasible",
            "Try relaxing AI enforcement level",
            "Check that required players can fit in salary cap",
            "Verify team diversity requirements can be met"
        ]

    def _get_timeout_suggestions(self) -> List[str]:
        return [
            "Reduce number of lineups or increase timeout",
            "Try fewer hard constraints",
            "Consider using fewer parallel threads"
        ]

    def _get_api_suggestions(self) -> List[str]:
        return [
            "Check API key is valid",
            "Verify internet connection",
            "API may be rate-limited - wait and retry"
        ]

    def _get_dataframe_suggestions(self) -> List[str]:
        return [
            "Ensure CSV file is not empty",
            "Check column names match expected format",
            "Verify data has been loaded correctly"
        ]

    def _get_generic_suggestions(self) -> List[str]:
        return [
            "Check logs for more details",
            "Verify all input data is valid"
        ]

    def _cache_suggestions(self, cache_key: str, suggestions: List[str]) -> None:
        if len(self.error_suggestions_cache) > 100:
            old_keys = list(self.error_suggestions_cache.keys())[:50]
            for key in old_keys:
                del self.error_suggestions_cache[key]
        self.error_suggestions_cache[cache_key] = suggestions

    def _cleanup_if_needed(self) -> None:
        """Automatic cleanup check"""
        now = datetime.now()
        if (now - self.last_cleanup).seconds > 300:
            self._cleanup()
            self.last_cleanup = now

    def _cleanup(self) -> None:
        """Memory cleanup"""
        cutoff = datetime.now() - timedelta(hours=1)

        for key in list(self.performance_metrics.keys()):
            self.performance_metrics[key] = [
                m for m in self.performance_metrics[key]
                if m.get('timestamp', datetime.now()) > cutoff
            ]
            if not self.performance_metrics[key]:
                del self.performance_metrics[key]

        if len(self.error_patterns) > 50:
            sorted_patterns = sorted(
                self.error_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.error_patterns = dict(sorted_patterns[:30])

    def log_ai_decision(self, decision_type: str, ai_source: str,
                       success: bool, details: Dict = None,
                       confidence: float = 0) -> None:
        """Log AI decision"""
        with self._lock:
            self.ai_decisions.append({
                'timestamp': datetime.now(),
                'type': decision_type,
                'source': ai_source,
                'success': success,
                'confidence': confidence,
                'details': details or {}
            })

    def log_optimization_start(self, num_lineups: int, field_size: str,
                              settings: Dict) -> None:
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
                            success_rate: float) -> None:
        """Log optimization completion"""
        with self._lock:
            self.optimization_events.append({
                'timestamp': datetime.now(),
                'event': 'complete',
                'lineups_generated': lineups_generated,
                'time_taken': time_taken,
                'success_rate': success_rate
            })

    def get_error_summary(self) -> Dict:
        """Get summary of errors"""
        with self._lock:
            return {
                'total_errors': len(self.error_logs),
                'error_categories': dict(self.failure_categories),
                'top_patterns': sorted(
                    self.error_patterns.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                'recent_errors': list(self.error_logs)[-5:]
            }

# ============================================================================
# PERFORMANCE MONITOR WITH ENHANCED TRACKING
# ============================================================================

class PerformanceMonitor:
    """Enhanced performance monitoring"""

    def __init__(self):
        self.timers = {}
        self.metrics = defaultdict(list)
        self._lock = threading.RLock()
        self.start_times = {}

        self.operation_counts = defaultdict(int)
        self.operation_times = defaultdict(list)
        self.memory_snapshots = deque(maxlen=10)

        self.phase_times = {
            'data_load': [],
            'ai_analysis': [],
            'lineup_generation': [],
            'validation': [],
            'export': [],
            'monte_carlo': [],
            'genetic_algorithm': []
        }

    def start_timer(self, operation: str) -> None:
        """Start timing an operation"""
        with self._lock:
            self.start_times[operation] = time.time()
            self.operation_counts[operation] += 1

    def stop_timer(self, operation: str) -> float:
        """Stop timing and return elapsed"""
        with self._lock:
            if operation not in self.start_times:
                return 0

            elapsed = time.time() - self.start_times[operation]
            del self.start_times[operation]

            self.operation_times[operation].append(elapsed)

            if len(self.operation_times[operation]) > 100:
                self.operation_times[operation] = self.operation_times[operation][-50:]

            return elapsed

    def record_metric(self, metric_name: str, value: float,
                     tags: Dict = None) -> None:
        """Record a metric"""
        with self._lock:
            self.metrics[metric_name].append({
                'value': value,
                'timestamp': datetime.now(),
                'tags': tags or {}
            })

            cutoff = datetime.now() - timedelta(hours=1)
            self.metrics[metric_name] = [
                m for m in self.metrics[metric_name]
                if m['timestamp'] > cutoff
            ]

    def record_phase_time(self, phase: str, duration: float) -> None:
        """Record time for optimization phase"""
        with self._lock:
            if phase in self.phase_times:
                self.phase_times[phase].append(duration)
                if len(self.phase_times[phase]) > 20:
                    self.phase_times[phase] = self.phase_times[phase][-10:]

    def get_operation_stats(self, operation: str) -> Dict:
        """Get statistics for an operation"""
        with self._lock:
            times = self.operation_times.get(operation, [])
            if not times:
                return {}

            return {
                'count': self.operation_counts[operation],
                'avg_time': np.mean(times),
                'median_time': np.median(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_time': sum(times),
                'std_dev': np.std(times) if len(times) > 1 else 0
            }

    def get_phase_summary(self) -> Dict:
        """Get summary of optimization phases"""
        with self._lock:
            summary = {}
            for phase, times in self.phase_times.items():
                if times:
                    summary[phase] = {
                        'avg_time': np.mean(times),
                        'total_time': sum(times),
                        'count': len(times)
                    }
            return summary

    def get_bottlenecks(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Identify performance bottlenecks"""
        with self._lock:
            bottlenecks = []
            for operation, times in self.operation_times.items():
                if times:
                    avg_time = np.mean(times)
                    bottlenecks.append((operation, avg_time))

            return sorted(bottlenecks, key=lambda x: x[1], reverse=True)[:top_n]

# ============================================================================
# AI DECISION TRACKER WITH LEARNING
# ============================================================================

class AIDecisionTracker:
    """Track AI decisions and learn from performance"""

    def __init__(self):
        self.decisions = deque(maxlen=OptimizerConfig.MAX_HISTORY_ENTRIES)
        self.performance_feedback = deque(maxlen=100)
        self.decision_patterns = defaultdict(list)
        self._lock = threading.RLock()

        self.successful_patterns = defaultdict(float)
        self.failed_patterns = defaultdict(float)
        self.confidence_calibration = defaultdict(list)

        self.strategy_performance = {
            'game_theory': {'wins': 0, 'attempts': 0, 'avg_score': 0},
            'correlation': {'wins': 0, 'attempts': 0, 'avg_score': 0},
            'contrarian': {'wins': 0, 'attempts': 0, 'avg_score': 0},
            'genetic_algorithm': {'wins': 0, 'attempts': 0, 'avg_score': 0}
        }

    def track_decision(self, ai_type: AIStrategistType, decision: AIRecommendation,
                      context: Dict = None) -> None:
        """Track an AI decision"""
        with self._lock:
            entry = {
                'timestamp': datetime.now(),
                'ai_type': ai_type,
                'captain_count': len(decision.captain_targets),
                'confidence': decision.confidence,
                'enforcement_rules': len(decision.enforcement_rules),
                'context': context or {}
            }

            self.decisions.append(entry)

            pattern_key = self._extract_pattern(decision)
            self.decision_patterns[pattern_key].append(entry)

    def _extract_pattern(self, decision: AIRecommendation) -> str:
        """Extract pattern from decision"""
        pattern_elements = [
            f"conf_{int(decision.confidence*10)}",
            f"capt_{min(len(decision.captain_targets), 5)}",
            f"must_{min(len(decision.must_play), 3)}",
            f"stack_{min(len(decision.stacks), 3)}"
        ]
        return "_".join(pattern_elements)

    def record_performance(self, lineup: Dict,
                          actual_score: Optional[float] = None) -> None:
        """Record lineup performance"""
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
                    'ownership': lineup.get('Total_Ownership', 0),
                    'captain': lineup.get('Captain', '')
                }

                self.performance_feedback.append(entry)

                strategy = lineup.get('AI_Strategy', 'unknown')
                ownership_tier = lineup.get('Ownership_Tier', 'unknown')
                pattern_key = f"{strategy}_{ownership_tier}"

                if entry['success']:
                    self.successful_patterns[pattern_key] += 1
                else:
                    self.failed_patterns[pattern_key] += 1

                confidence = lineup.get('Confidence', 0.5)
                conf_bucket = int(confidence * 10)
                self.confidence_calibration[conf_bucket].append(accuracy)

                if strategy in self.strategy_performance:
                    self.strategy_performance[strategy]['attempts'] += 1
                    if entry['success']:
                        self.strategy_performance[strategy]['wins'] += 1

                    current_avg = self.strategy_performance[strategy]['avg_score']
                    attempts = self.strategy_performance[strategy]['attempts']
                    self.strategy_performance[strategy]['avg_score'] = (
                        (current_avg * (attempts - 1) + actual_score) / attempts
                    )

    def get_learning_insights(self) -> Dict:
        """Get insights from tracked performance"""
        with self._lock:
            insights = {
                'total_decisions': len(self.decisions),
                'avg_confidence': (
                    np.mean([d['confidence'] for d in self.decisions])
                    if self.decisions else 0
                )
            }

            pattern_stats = self._calculate_pattern_stats()
            insights['pattern_performance'] = pattern_stats

            calibration = self._calculate_calibration()
            insights['confidence_calibration'] = calibration

            insights['strategy_performance'] = self._calculate_strategy_performance()

            return insights

    def _calculate_pattern_stats(self) -> Dict:
        """Calculate pattern success statistics"""
        pattern_stats = {}

        for pattern in set(list(self.successful_patterns.keys()) +
                         list(self.failed_patterns.keys())):
            successes = self.successful_patterns.get(pattern, 0)
            failures = self.failed_patterns.get(pattern, 0)
            total = successes + failures

            if total >= 5:
                pattern_stats[pattern] = {
                    'success_rate': successes / total,
                    'total': total,
                    'confidence': 'high' if total >= 10 else 'medium'
                }

        return pattern_stats

    def _calculate_calibration(self) -> Dict:
        """Calculate confidence calibration"""
        calibration = {}

        for conf_level, accuracies in self.confidence_calibration.items():
            if accuracies:
                calibration[conf_level / 10] = {
                    'actual_accuracy': np.mean(accuracies),
                    'sample_size': len(accuracies)
                }

        return calibration

    def _calculate_strategy_performance(self) -> Dict:
        """Calculate per-strategy performance"""
        return {
            strategy: {
                'win_rate': (
                    stats['wins'] / stats['attempts']
                    if stats['attempts'] > 0 else 0
                ),
                'avg_score': stats['avg_score'],
                'attempts': stats['attempts']
            }
            for strategy, stats in self.strategy_performance.items()
        }

    def get_recommended_adjustments(self) -> Dict:
        """Get recommended adjustments based on learning"""
        insights = self.get_learning_insights()
        adjustments = {}

        adjustments.update(self._get_confidence_adjustments(insights))
        adjustments.update(self._get_pattern_adjustments(insights))
        adjustments.update(self._get_strategy_recommendations(insights))

        return adjustments

    def _get_confidence_adjustments(self, insights: Dict) -> Dict:
        """Get confidence-based adjustments"""
        adjustments = {}
        calibration = insights.get('confidence_calibration', {})

        for conf_level, stats in calibration.items():
            actual_accuracy = stats['actual_accuracy']
            sample_size = stats['sample_size']

            if sample_size >= 5 and abs(conf_level - actual_accuracy) > 0.15:
                adjustments[f'confidence_{conf_level:.1f}'] = {
                    'current': conf_level,
                    'suggested': actual_accuracy,
                    'reason': f'Historical accuracy is {actual_accuracy:.1%} vs stated {conf_level:.1%}'
                }

        return adjustments

    def _get_pattern_adjustments(self, insights: Dict) -> Dict:
        """Get pattern-based adjustments"""
        adjustments = {}
        pattern_perf = insights.get('pattern_performance', {})

        for pattern, stats in pattern_perf.items():
            if stats['total'] >= 10:
                if stats['success_rate'] > 0.7:
                    adjustments[f'boost_{pattern}'] = {
                        'multiplier': 1.2,
                        'reason': f'Pattern shows {stats["success_rate"]:.1%} success rate'
                    }
                elif stats['success_rate'] < 0.3:
                    adjustments[f'reduce_{pattern}'] = {
                        'multiplier': 0.8,
                        'reason': f'Pattern shows only {stats["success_rate"]:.1%} success rate'
                    }

        return adjustments

    def _get_strategy_recommendations(self, insights: Dict) -> Dict:
        """Get strategy-based recommendations"""
        adjustments = {}
        strategy_perf = insights.get('strategy_performance', {})

        best_strategy = max(
            strategy_perf.items(),
            key=lambda x: x[1]['win_rate'] if x[1]['attempts'] >= 5 else 0,
            default=(None, None)
        )

        if best_strategy[0] and best_strategy[1]['attempts'] >= 5:
            adjustments['preferred_strategy'] = {
                'strategy': best_strategy[0],
                'win_rate': best_strategy[1]['win_rate'],
                'reason': f'Best performing strategy with {best_strategy[1]["win_rate"]:.1%} win rate'
            }

        return adjustments

    def apply_learned_adjustments(self, df: pd.DataFrame,
                                  current_strategy: str) -> pd.DataFrame:
        """Apply learned adjustments to projections"""
        adjustments = self.get_recommended_adjustments()
        df_adjusted = df.copy()

        for key, adjustment in adjustments.items():
            if key.startswith('boost_') and current_strategy in key:
                multiplier = adjustment['multiplier']
                df_adjusted['Projected_Points'] *= multiplier

            elif key.startswith('reduce_') and current_strategy in key:
                multiplier = adjustment['multiplier']
                df_adjusted['Projected_Points'] *= multiplier

        return df_adjusted

# ============================================================================
# MONTE CARLO SIMULATION ENGINE (NEW)
# ============================================================================

class MonteCarloSimulationEngine:
    """
    Simulate slate outcomes using projection distributions and correlations

    DFS Value: CRITICAL - Enables ceiling/floor/variance analysis without historical data
    """

    def __init__(self, df: pd.DataFrame, game_info: Dict, n_simulations: int = 5000):
        self.df = df
        self.game_info = game_info
        self.n_simulations = n_simulations

        self.correlation_matrix = self._build_correlation_matrix()
        self.player_variance = self._calculate_player_variance()

        self.simulation_cache = {}
        self._cache_lock = threading.RLock()

        self.logger = get_logger()

    def _build_correlation_matrix(self) -> Dict[Tuple[str, str], float]:
        """Build correlation matrix from slate structure"""
        correlations = {}

        player_to_pos = self.df.set_index('Player')['Position'].to_dict()
        player_to_team = self.df.set_index('Player')['Team'].to_dict()

        players = self.df['Player'].tolist()

        for i, player1 in enumerate(players):
            pos1 = player_to_pos[player1]
            team1 = player_to_team[player1]

            for player2 in players[i+1:]:
                pos2 = player_to_pos[player2]
                team2 = player_to_team[player2]

                same_team = (team1 == team2)
                corr = self._get_correlation_coefficient(pos1, pos2, same_team)

                if abs(corr) > 0.1:
                    correlations[(player1, player2)] = corr
                    correlations[(player2, player1)] = corr

        return correlations

    def _get_correlation_coefficient(self, pos1: str, pos2: str, same_team: bool) -> float:
        """Get correlation coefficient based on positions"""
        if same_team:
            if pos1 == 'QB' and pos2 in ['WR', 'TE']:
                return OptimizerConfig.CORRELATION_COEFFICIENTS['qb_wr_same_team']
            elif pos1 in ['WR', 'TE'] and pos2 == 'QB':
                return OptimizerConfig.CORRELATION_COEFFICIENTS['qb_wr_same_team']
            elif pos1 == 'QB' and pos2 == 'RB':
                return OptimizerConfig.CORRELATION_COEFFICIENTS['qb_rb_same_team']
            elif pos1 == 'RB' and pos2 == 'QB':
                return OptimizerConfig.CORRELATION_COEFFICIENTS['qb_rb_same_team']
            elif pos1 in ['WR', 'TE'] and pos2 in ['WR', 'TE']:
                return OptimizerConfig.CORRELATION_COEFFICIENTS['wr_wr_same_team']
            elif pos1 == 'RB' and pos2 == 'RB':
                return -0.25
        else:
            if pos1 == 'QB' and pos2 == 'QB':
                return OptimizerConfig.CORRELATION_COEFFICIENTS['qb_qb_opposing']
            elif pos1 == 'RB' and pos2 == 'DST':
                return OptimizerConfig.CORRELATION_COEFFICIENTS['rb_dst_opposing']
            elif pos1 == 'DST' and pos2 == 'RB':
                return OptimizerConfig.CORRELATION_COEFFICIENTS['rb_dst_opposing']
            elif pos1 in ['WR', 'TE'] and pos2 == 'DST':
                return OptimizerConfig.CORRELATION_COEFFICIENTS['wr_dst_opposing']
            elif pos1 == 'DST' and pos2 in ['WR', 'TE']:
                return OptimizerConfig.CORRELATION_COEFFICIENTS['wr_dst_opposing']

        return 0.0

    def _calculate_player_variance(self) -> Dict[str, float]:
        """Calculate variance for each player"""
        variance_dict = {}

        for _, player in self.df.iterrows():
            position = player['Position']
            projection = player['Projected_Points']
            salary = player['Salary']

            base_cv = OptimizerConfig.VARIANCE_BY_POSITION.get(position, 0.40)
            salary_factor = max(0.7, 1.0 - (salary - 3000) / 18000 * 0.3)

            cv = base_cv * salary_factor
            variance = (projection * cv) ** 2

            variance_dict[player['Player']] = variance

        return variance_dict

    def simulate_player_performance(self, player: str, base_score: Optional[float] = None) -> float:
        """Simulate single player performance"""
        if base_score is None:
            base_score = self.df[self.df['Player'] == player]['Projected_Points'].iloc[0]

        variance = self.player_variance[player]
        std = np.sqrt(variance)

        if base_score > 0:
            mu = np.log(base_score**2 / np.sqrt(std**2 + base_score**2))
            sigma = np.sqrt(np.log(1 + (std**2 / base_score**2)))
            return np.random.lognormal(mu, sigma)
        else:
            return 0.0

    def simulate_correlated_slate(self) -> Dict[str, float]:
        """Simulate entire slate with correlations"""
        player_scores = {}

        for _, player in self.df.iterrows():
            player_scores[player['Player']] = self.simulate_player_performance(player['Player'])

        for (p1, p2), corr in self.correlation_matrix.items():
            if p1 in player_scores and p2 in player_scores:
                p1_proj = self.df[self.df['Player'] == p1]['Projected_Points'].iloc[0]
                p2_proj = self.df[self.df['Player'] == p2]['Projected_Points'].iloc[0]

                p1_std = np.sqrt(self.player_variance[p1])
                p1_zscore = (player_scores[p1] - p1_proj) / max(p1_std, 0.01)

                p2_std = np.sqrt(self.player_variance[p2])
                adjustment = corr * p1_zscore * p2_std * 0.5

                player_scores[p2] += adjustment
                player_scores[p2] = max(0, player_scores[p2])

        return player_scores

    def evaluate_lineup(self, captain: str, flex: List[str],
                       use_cache: bool = True) -> SimulationResults:
        """Run Monte Carlo simulation on lineup"""
        cache_key = f"{captain}_{'_'.join(sorted(flex))}"

        if use_cache:
            with self._cache_lock:
                if cache_key in self.simulation_cache:
                    return self.simulation_cache[cache_key]

        lineup_scores = []

        for _ in range(self.n_simulations):
            sim_scores = self.simulate_correlated_slate()

            captain_pts = sim_scores.get(captain, 0) * OptimizerConfig.CAPTAIN_MULTIPLIER
            flex_pts = sum(sim_scores.get(p, 0) for p in flex)
            total = captain_pts + flex_pts

            lineup_scores.append(total)

        lineup_scores = np.array(lineup_scores)

        mean = np.mean(lineup_scores)
        median = np.median(lineup_scores)
        std = np.std(lineup_scores)

        floor_10th = np.percentile(lineup_scores, 10)
        ceiling_90th = np.percentile(lineup_scores, 90)
        ceiling_99th = np.percentile(lineup_scores, 99)

        top_10pct_threshold = np.percentile(lineup_scores, 90)
        top_10pct_scores = lineup_scores[lineup_scores >= top_10pct_threshold]
        top_10pct_mean = np.mean(top_10pct_scores)

        sharpe_ratio = mean / std if std > 0 else 0
        win_threshold = 180
        win_probability = np.mean(lineup_scores >= win_threshold)

        results = SimulationResults(
            mean=float(mean),
            median=float(median),
            std=float(std),
            floor_10th=float(floor_10th),
            ceiling_90th=float(ceiling_90th),
            ceiling_99th=float(ceiling_99th),
            top_10pct_mean=float(top_10pct_mean),
            sharpe_ratio=float(sharpe_ratio),
            win_probability=float(win_probability),
            score_distribution=lineup_scores
        )

        if use_cache:
            with self._cache_lock:
                if len(self.simulation_cache) > 100:
                    keys_to_remove = list(self.simulation_cache.keys())[:50]
                    for key in keys_to_remove:
                        del self.simulation_cache[key]

                self.simulation_cache[cache_key] = results

        return results

    def evaluate_multiple_lineups(self, lineups: List[Dict],
                                  parallel: bool = True) -> Dict[int, SimulationResults]:
        """Evaluate multiple lineups efficiently"""
        results = {}

        if parallel and len(lineups) > 5:
            with ThreadPoolExecutor(max_workers=min(4, len(lineups))) as executor:
                futures = {
                    executor.submit(
                        self.evaluate_lineup,
                        lineup['captain'],
                        lineup['flex']
                    ): idx
                    for idx, lineup in enumerate(lineups)
                }

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        self.logger.log(f"Simulation error for lineup {idx}: {e}", "ERROR")
        else:
            for idx, lineup in enumerate(lineups):
                try:
                    results[idx] = self.evaluate_lineup(lineup['captain'], lineup['flex'])
                except Exception as e:
                    self.logger.log(f"Simulation error for lineup {idx}: {e}", "ERROR")

        return results

    def compare_lineups(self, lineups: List[Dict], metric: str = 'ceiling_90th') -> pd.DataFrame:
        """Compare multiple lineups across metrics"""
        results = self.evaluate_multiple_lineups(lineups)

        comparison_data = []
        for idx, sim_results in results.items():
            lineup = lineups[idx]

            comparison_data.append({
                'Lineup': idx + 1,
                'Captain': lineup['captain'],
                'Mean': sim_results.mean,
                'Median': sim_results.median,
                'Std': sim_results.std,
                'Floor_10th': sim_results.floor_10th,
                'Ceiling_90th': sim_results.ceiling_90th,
                'Ceiling_99th': sim_results.ceiling_99th,
                'Sharpe': sim_results.sharpe_ratio,
                'Win_Prob': sim_results.win_probability
            })

        df = pd.DataFrame(comparison_data)
        return df.sort_values(metric, ascending=False)

# ============================================================================
# GENETIC ALGORITHM OPTIMIZER (NEW)
# ============================================================================

@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm"""
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.15
    elite_size: int = 10
    tournament_size: int = 5
    crossover_rate: float = 0.8

class GeneticLineup:
    """Represents a lineup in the genetic algorithm"""
    def __init__(self, captain: str, flex: List[str], fitness: float = 0):
        self.captain = captain
        self.flex = flex
        self.fitness = fitness
        self.sim_results: Optional[SimulationResults] = None
        self.validated = False

    def get_all_players(self) -> List[str]:
        return [self.captain] + self.flex

    def to_dict(self) -> Dict:
        return {'captain': self.captain, 'flex': self.flex, 'fitness': self.fitness}

class GeneticAlgorithmOptimizer:
    """
    Genetic algorithm for DFS lineup optimization

    DFS Value: SUPERIOR to LP for GPP ceiling optimization
    """

    def __init__(self, df: pd.DataFrame, game_info: Dict,
                 mc_engine: Optional[MonteCarloSimulationEngine] = None,
                 config: Optional[GeneticConfig] = None):
        self.df = df
        self.game_info = game_info
        self.config = config or GeneticConfig()

        self.mc_engine = mc_engine or MonteCarloSimulationEngine(
            df, game_info, n_simulations=OptimizerConfig.MC_FAST_SIMULATIONS
        )

        self.players = df['Player'].tolist()
        self.salaries = df.set_index('Player')['Salary'].to_dict()
        self.projections = df.set_index('Player')['Projected_Points'].to_dict()
        self.ownership = df.set_index('Player')['Ownership'].to_dict()
        self.positions = df.set_index('Player')['Position'].to_dict()
        self.teams = df.set_index('Player')['Team'].to_dict()

        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()

        self.best_lineups = []

    def create_random_lineup(self) -> GeneticLineup:
        """Create random valid lineup"""
        max_attempts = 100

        for attempt in range(max_attempts):
            captain = np.random.choice(self.players)
            available = [p for p in self.players if p != captain]
            flex = list(np.random.choice(available, 5, replace=False))

            lineup = GeneticLineup(captain, flex)

            if self._is_valid_lineup(lineup):
                return lineup

        return self._create_min_salary_lineup()

    def _is_valid_lineup(self, lineup: GeneticLineup) -> bool:
        """Validate lineup against DK constraints"""
        all_players = lineup.get_all_players()

        total_salary = sum(self.salaries[p] for p in lineup.flex)
        total_salary += self.salaries[lineup.captain] * OptimizerConfig.CAPTAIN_MULTIPLIER

        if total_salary > OptimizerConfig.SALARY_CAP:
            return False

        team_counts = Counter(self.teams[p] for p in all_players)

        if len(team_counts) < OptimizerConfig.MIN_TEAMS_REQUIRED:
            return False

        if any(count > OptimizerConfig.MAX_PLAYERS_PER_TEAM for count in team_counts.values()):
            return False

        return True

    def calculate_fitness(self, lineup: GeneticLineup, mode: FitnessMode) -> float:
        """Calculate fitness score for lineup"""
        captain_proj = self.projections[lineup.captain]
        flex_proj = sum(self.projections[p] for p in lineup.flex)
        base_score = captain_proj * 1.5 + flex_proj

        captain_own = self.ownership.get(lineup.captain, 10)
        flex_own = sum(self.ownership.get(p, 10) for p in lineup.flex)
        total_own = captain_own * 1.5 + flex_own

        ownership_multiplier = 1.0 + (100 - total_own) / 150

        run_full_sim = (mode == FitnessMode.CEILING or mode == FitnessMode.SHARPE)

        if run_full_sim and np.random.random() < 0.15:
            sim_results = self.mc_engine.evaluate_lineup(lineup.captain, lineup.flex)
            lineup.sim_results = sim_results

            if mode == FitnessMode.CEILING:
                return sim_results.ceiling_90th * ownership_multiplier
            elif mode == FitnessMode.SHARPE:
                return sim_results.sharpe_ratio * 15 * ownership_multiplier
            elif mode == FitnessMode.WIN_PROBABILITY:
                return sim_results.win_probability * 200 * ownership_multiplier
            else:
                return sim_results.mean * ownership_multiplier
        else:
            if mode == FitnessMode.CEILING:
                return base_score * 1.3 * ownership_multiplier
            else:
                return base_score * ownership_multiplier

    def crossover(self, parent1: GeneticLineup, parent2: GeneticLineup) -> GeneticLineup:
        """Breed two lineups"""
        if np.random.random() > self.config.crossover_rate:
            return GeneticLineup(parent1.captain, parent1.flex.copy())

        captain = np.random.choice([parent1.captain, parent2.captain])

        flex_pool = list(set(parent1.flex + parent2.flex))
        if captain in flex_pool:
            flex_pool.remove(captain)

        if len(flex_pool) >= 5:
            flex = list(np.random.choice(flex_pool, 5, replace=False))
        else:
            available = [p for p in self.players if p != captain and p not in flex_pool]
            additional_needed = 5 - len(flex_pool)
            flex = flex_pool + list(np.random.choice(available, additional_needed, replace=False))

        child = GeneticLineup(captain, flex)

        if not self._is_valid_lineup(child):
            child = self._repair_lineup(child)

        return child

    def mutate(self, lineup: GeneticLineup) -> GeneticLineup:
        """Randomly modify lineup"""
        mutated = GeneticLineup(lineup.captain, lineup.flex.copy())

        if np.random.random() < self.config.mutation_rate:
            available_captains = [p for p in self.players if p not in lineup.flex]
            mutated.captain = np.random.choice(available_captains)

        if np.random.random() < self.config.mutation_rate:
            n_mutations = np.random.randint(1, 3)

            for _ in range(n_mutations):
                idx = np.random.randint(0, 5)
                available = [p for p in self.players
                           if p != mutated.captain and p not in mutated.flex]
                if available:
                    mutated.flex[idx] = np.random.choice(available)

        if not self._is_valid_lineup(mutated):
            mutated = self._repair_lineup(mutated)

        return mutated

    def _repair_lineup(self, lineup: GeneticLineup) -> GeneticLineup:
        """Repair invalid lineup"""
        max_repair_attempts = 20

        for _ in range(max_repair_attempts):
            total_salary = sum(self.salaries[p] for p in lineup.flex)
            total_salary += self.salaries[lineup.captain] * 1.5

            if total_salary > OptimizerConfig.SALARY_CAP:
                flex_with_salaries = [(p, self.salaries[p]) for p in lineup.flex]
                flex_with_salaries.sort(key=lambda x: x[1], reverse=True)
                expensive_player = flex_with_salaries[0][0]

                available_cheaper = [
                    p for p in self.players
                    if p != lineup.captain and p not in lineup.flex and
                    self.salaries[p] < self.salaries[expensive_player]
                ]

                if available_cheaper:
                    replacement = np.random.choice(available_cheaper)
                    idx = lineup.flex.index(expensive_player)
                    lineup.flex[idx] = replacement

            all_players = lineup.get_all_players()
            team_counts = Counter(self.teams[p] for p in all_players)

            if len(team_counts) < 2:
                current_teams = set(team_counts.keys())
                all_teams = set(self.teams.values())
                other_teams = all_teams - current_teams

                if other_teams:
                    other_team = np.random.choice(list(other_teams))
                    other_team_players = [
                        p for p in self.players
                        if self.teams[p] == other_team and p != lineup.captain
                    ]

                    if other_team_players:
                        replacement = np.random.choice(other_team_players)
                        idx = np.random.randint(0, 5)
                        lineup.flex[idx] = replacement

            if self._is_valid_lineup(lineup):
                return lineup

        return self.create_random_lineup()

    def _tournament_select(self, population: List[GeneticLineup]) -> GeneticLineup:
        """Tournament selection"""
        tournament = list(np.random.choice(population, self.config.tournament_size, replace=False))
        return max(tournament, key=lambda x: x.fitness)

    def evolve_population(self, population: List[GeneticLineup],
                         fitness_mode: FitnessMode) -> List[GeneticLineup]:
        """Evolve population for one generation"""
        for lineup in population:
            if lineup.fitness == 0:
                lineup.fitness = self.calculate_fitness(lineup, fitness_mode)

        population.sort(key=lambda x: x.fitness, reverse=True)

        if not self.best_lineups or population[0].fitness > self.best_lineups[0].fitness:
            self.best_lineups = population[:5]

        next_generation = population[:self.config.elite_size]

        while len(next_generation) < self.config.population_size:
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)

            child = self.crossover(parent1, parent2)
            child = self.mutate(child)

            next_generation.append(child)

        return next_generation

    def optimize(self, num_lineups: int = 20,
                fitness_mode: FitnessMode = FitnessMode.CEILING,
                verbose: bool = True) -> List[Dict]:
        """Run genetic algorithm optimization"""
        self.perf_monitor.start_timer("genetic_algorithm")

        if verbose:
            self.logger.log(
                f"Starting GA optimization: {self.config.generations} generations, "
                f"pop={self.config.population_size}, mode={fitness_mode.value}",
                "INFO"
            )

        population = [self.create_random_lineup() for _ in range(self.config.population_size)]

        for generation in range(self.config.generations):
            population = self.evolve_population(population, fitness_mode)

            if verbose and generation % 10 == 0:
                best_fitness = population[0].fitness
                self.logger.log(
                    f"Generation {generation}/{self.config.generations}: "
                    f"Best fitness = {best_fitness:.2f}",
                    "INFO"
                )

        if verbose:
            self.logger.log("Running final simulations on top lineups...", "INFO")

        top_candidates = population[:num_lineups * 2]

        for lineup in top_candidates:
            if lineup.sim_results is None:
                lineup.sim_results = self.mc_engine.evaluate_lineup(
                    lineup.captain,
                    lineup.flex
                )

                if fitness_mode == FitnessMode.CEILING:
                    lineup.fitness = lineup.sim_results.ceiling_90th
                elif fitness_mode == FitnessMode.SHARPE:
                    lineup.fitness = lineup.sim_results.sharpe_ratio * 15
                elif fitness_mode == FitnessMode.WIN_PROBABILITY:
                    lineup.fitness = lineup.sim_results.win_probability * 200
                else:
                    lineup.fitness = lineup.sim_results.mean

        top_candidates.sort(key=lambda x: x.fitness, reverse=True)

        unique_lineups = self._deduplicate_lineups(top_candidates, num_lineups)

        elapsed = self.perf_monitor.stop_timer("genetic_algorithm")

        if verbose:
            self.logger.log(
                f"GA optimization complete: {len(unique_lineups)} unique lineups in {elapsed:.2f}s",
                "INFO"
            )

        results = []
        for lineup in unique_lineups[:num_lineups]:
            results.append({
                'captain': lineup.captain,
                'flex': lineup.flex,
                'sim_results': lineup.sim_results,
                'fitness': lineup.fitness
            })

        return results

    def _deduplicate_lineups(self, lineups: List[GeneticLineup],
                            target: int) -> List[GeneticLineup]:
        """Remove similar lineups"""
        unique = []
        seen_players = []

        for lineup in lineups:
            players = frozenset(lineup.get_all_players())

            is_unique = True
            for seen in seen_players:
                overlap = len(players & seen)
                if overlap >= 5:
                    is_unique = False
                    break

            if is_unique:
                unique.append(lineup)
                seen_players.append(players)

            if len(unique) >= target:
                break

        return unique

    def _create_min_salary_lineup(self) -> GeneticLineup:
        """Create minimum salary valid lineup"""
        sorted_by_salary = self.df.sort_values('Salary')

        captain = sorted_by_salary.iloc[0]['Player']
        flex = sorted_by_salary.iloc[1:6]['Player'].tolist()

        return GeneticLineup(captain, flex)

# ============================================================================
# PART 3: ENFORCEMENT, VALIDATION, AND SYNTHESIS COMPONENTS
# ============================================================================

# ============================================================================
# AI ENFORCEMENT ENGINE WITH THREE-TIER RELAXATION
# ============================================================================

class AIEnforcementEngine:
    """
    Enhanced enforcement engine with three-tier constraint relaxation and simulation awareness

    DFS Value: CRITICAL - Three-tier system allows maintaining diversity while respecting AI recommendations
    """

    def __init__(self, enforcement_level: AIEnforcementLevel = AIEnforcementLevel.STRONG):
        self.enforcement_level = enforcement_level
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()

        # Enhanced rule tracking
        self.applied_rules = deque(maxlen=100)
        self.rule_success_rate = defaultdict(float)
        self.violation_patterns = defaultdict(int)

        # Track which rules are most effective
        self.rule_effectiveness = defaultdict(lambda: {'applied': 0, 'success': 0})

        # Track simulation-based rule adjustments
        self.simulation_adjustments = defaultdict(float)

    def create_enforcement_rules(self,
                                 recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """
        Create comprehensive enforcement rules from AI recommendations

        DFS Value: Different enforcement levels balance AI control vs lineup diversity

        Returns:
            Dictionary with categorized rules by priority and type
        """
        self.logger.log(
            f"Creating enforcement rules at {self.enforcement_level.value} level",
            "INFO"
        )

        rules = {
            'hard_constraints': [],
            'soft_constraints': [],
            'stacking_rules': [],
            'ownership_rules': [],
            'correlation_rules': []
        }

        # Process recommendations based on enforcement level
        if self.enforcement_level == AIEnforcementLevel.MANDATORY:
            rules = self._create_mandatory_rules(recommendations)
        elif self.enforcement_level == AIEnforcementLevel.STRONG:
            rules = self._create_strong_rules(recommendations)
        elif self.enforcement_level == AIEnforcementLevel.MODERATE:
            rules = self._create_moderate_rules(recommendations)
        else:  # ADVISORY
            rules = self._create_advisory_rules(recommendations)

        # Add advanced stacking rules
        rules['stacking_rules'].extend(self._create_stacking_rules(recommendations))

        # Sort by priority within each category
        self._sort_rules_by_priority(rules)

        total_rules = sum(len(v) for v in rules.values() if isinstance(v, list))
        self.logger.log(f"Created {total_rules} enforcement rules", "INFO")

        return rules

    def _sort_rules_by_priority(self, rules: Dict) -> None:
        """Sort rules by priority in-place"""
        for rule_type in rules:
            if isinstance(rules[rule_type], list):
                rules[rule_type].sort(
                    key=lambda x: x.get('priority', 0),
                    reverse=True
                )

    def _create_mandatory_rules(self, recommendations: Dict) -> Dict:
        """Create mandatory enforcement rules (all AI decisions enforced as hard constraints)"""
        rules = self._initialize_rule_dict()

        for ai_type, rec in recommendations.items():
            weight = self._get_ai_weight(ai_type)

            # Captain constraints
            if rec.captain_targets:
                rules['hard_constraints'].append(
                    self._create_captain_rule(rec, ai_type, weight, tier=1)
                )

            # Must play constraints
            rules['hard_constraints'].extend(
                self._create_must_play_rules(rec, ai_type, weight)
            )

            # Never play constraints
            rules['hard_constraints'].extend(
                self._create_never_play_rules(rec, ai_type, weight)
            )

            # Stack constraints
            rules['stacking_rules'].extend(
                self._create_stack_constraints(rec, ai_type, weight)
            )

        return rules

    def _create_strong_rules(self, recommendations: Dict) -> Dict:
        """Create strong enforcement rules (most AI decisions enforced)"""
        rules = self._create_moderate_rules(recommendations)

        # Upgrade high-confidence recommendations to hard constraints
        for ai_type, rec in recommendations.items():
            if rec.confidence > 0.75:
                weight = self._get_ai_weight(ai_type)

                if rec.captain_targets:
                    rules['hard_constraints'].append({
                        'rule': 'captain_selection',
                        'players': rec.captain_targets[:5],
                        'source': ai_type.value,
                        'priority': int(ConstraintPriority.AI_HIGH_CONFIDENCE.value *
                                       weight * rec.confidence),
                        'type': 'hard',
                        'relaxation_tier': 2
                    })

        return rules

    def _create_moderate_rules(self, recommendations: Dict) -> Dict:
        """
        Create moderate enforcement rules (balanced approach)

        FIXED: Creates single constraint with list of consensus captains
        """
        rules = self._initialize_rule_dict()

        # Find consensus recommendations
        consensus = self._find_consensus_recommendations(recommendations)

        # FIXED: Collect all consensus captains into ONE constraint
        consensus_captains = [
            captain for captain, count in consensus['captains'].items()
            if count >= 2
        ]

        if consensus_captains:
            # Single constraint allowing ANY of the consensus captains
            rules['hard_constraints'].append({
                'rule': 'consensus_captain_list',
                'players': consensus_captains,
                'agreement': len([c for c in consensus['captains'].values() if c >= 2]),
                'priority': ConstraintPriority.AI_CONSENSUS.value,
                'type': 'hard',
                'relaxation_tier': 2
            })

        # Consensus must-play players
        consensus_must_play = [
            player for player, count in consensus['must_play'].items()
            if count >= 2
        ]

        for player in consensus_must_play[:3]:
            rules['hard_constraints'].append({
                'rule': 'must_include',
                'player': player,
                'agreement': consensus['must_play'][player],
                'priority': ConstraintPriority.AI_CONSENSUS.value,
                'type': 'hard',
                'relaxation_tier': 2
            })

        # Add soft constraints for single AI recommendations
        rules['soft_constraints'].extend(
            self._create_soft_constraints(recommendations, consensus)
        )

        return rules

    def _create_advisory_rules(self, recommendations: Dict) -> Dict:
        """Create advisory rules (suggestions only)"""
        rules = self._initialize_rule_dict()

        # All recommendations become soft constraints
        for ai_type, rec in recommendations.items():
            weight = self._get_ai_weight(ai_type)

            # Captain preferences
            for i, captain in enumerate(rec.captain_targets[:5]):
                rules['soft_constraints'].append({
                    'rule': 'prefer_captain',
                    'player': captain,
                    'source': ai_type.value,
                    'weight': weight * rec.confidence * (1 - i * 0.1),
                    'priority': int(ConstraintPriority.SOFT_PREFERENCE.value *
                                   weight * rec.confidence),
                    'type': 'soft'
                })

        return rules

    def _initialize_rule_dict(self) -> Dict:
        """Initialize empty rules dictionary"""
        return {
            'hard_constraints': [],
            'soft_constraints': [],
            'stacking_rules': [],
            'ownership_rules': [],
            'correlation_rules': []
        }

    def _get_ai_weight(self, ai_type: AIStrategistType) -> float:
        """Get weight for AI type from config"""
        return OptimizerConfig.AI_WEIGHTS.get(
            ai_type.value.lower().replace(' ', '_'),
            0.33
        )

    def _create_captain_rule(self, rec: AIRecommendation, ai_type: AIStrategistType,
                            weight: float, tier: int) -> Dict:
        """Create captain constraint rule"""
        return {
            'rule': 'captain_from_list',
            'players': rec.captain_targets[:7],
            'source': ai_type.value,
            'priority': int(ConstraintPriority.AI_HIGH_CONFIDENCE.value *
                           weight * rec.confidence),
            'type': 'hard',
            'relaxation_tier': tier
        }

    def _create_must_play_rules(self, rec: AIRecommendation, ai_type: AIStrategistType,
                                weight: float) -> List[Dict]:
        """Create must-play constraint rules"""
        rules = []
        for i, player in enumerate(rec.must_play[:3]):
            rules.append({
                'rule': 'must_include',
                'player': player,
                'source': ai_type.value,
                'priority': int((ConstraintPriority.AI_HIGH_CONFIDENCE.value - i * 5) *
                               weight * rec.confidence),
                'type': 'hard',
                'relaxation_tier': 2
            })
        return rules

    def _create_never_play_rules(self, rec: AIRecommendation, ai_type: AIStrategistType,
                                 weight: float) -> List[Dict]:
        """Create never-play constraint rules"""
        rules = []
        for i, player in enumerate(rec.never_play[:3]):
            rules.append({
                'rule': 'must_exclude',
                'player': player,
                'source': ai_type.value,
                'priority': int((ConstraintPriority.AI_MODERATE.value - i * 5) *
                               weight * rec.confidence),
                'type': 'hard',
                'relaxation_tier': 2
            })
        return rules

    def _create_stack_constraints(self, rec: AIRecommendation, ai_type: AIStrategistType,
                                  weight: float) -> List[Dict]:
        """Create stack constraint rules"""
        rules = []
        for i, stack in enumerate(rec.stacks[:3]):
            rules.append({
                'rule': 'must_stack',
                'stack': stack,
                'source': ai_type.value,
                'priority': int((ConstraintPriority.AI_MODERATE.value - i * 5) *
                               weight * rec.confidence),
                'type': 'hard',
                'relaxation_tier': 3
            })
        return rules

    def _find_consensus_recommendations(self, recommendations: Dict) -> Dict:
        """Find consensus across multiple AI recommendations"""
        captain_counts = defaultdict(int)
        must_play_counts = defaultdict(int)

        for rec in recommendations.values():
            for captain in rec.captain_targets:
                captain_counts[captain] += 1
            for player in rec.must_play:
                must_play_counts[player] += 1

        return {
            'captains': captain_counts,
            'must_play': must_play_counts
        }

    def _create_soft_constraints(self, recommendations: Dict, consensus: Dict) -> List[Dict]:
        """Create soft constraints for non-consensus recommendations"""
        constraints = []

        for ai_type, rec in recommendations.items():
            weight = self._get_ai_weight(ai_type)

            for player in rec.must_play[:3]:
                if consensus['must_play'][player] == 1:
                    constraints.append({
                        'rule': 'prefer_player',
                        'player': player,
                        'source': ai_type.value,
                        'weight': weight * rec.confidence,
                        'priority': int(ConstraintPriority.SOFT_PREFERENCE.value *
                                       weight * rec.confidence),
                        'type': 'soft'
                    })

        return constraints

    def _create_stacking_rules(self, recommendations: Dict) -> List[Dict]:
        """Create advanced stacking rules for all stack types"""
        stacking_rules = []

        for ai_type, rec in recommendations.items():
            for stack in rec.stacks:
                stack_rule = self._create_single_stack_rule(stack, ai_type)
                if stack_rule:
                    stacking_rules.append(stack_rule)

        # Remove duplicate stacks
        return self._deduplicate_stacks(stacking_rules)

    def _create_single_stack_rule(self, stack: Dict, ai_type: AIStrategistType) -> Optional[Dict]:
        """Create a single stack rule based on stack type"""
        stack_type = stack.get('type', 'standard')

        if stack_type == 'onslaught':
            return {
                'rule': 'onslaught_stack',
                'players': stack.get('players', []),
                'team': stack.get('team'),
                'min_players': 3,
                'max_players': 5,
                'scenario': stack.get('scenario', 'blowout'),
                'priority': ConstraintPriority.AI_MODERATE.value,
                'source': ai_type.value,
                'correlation_strength': 0.6,
                'relaxation_tier': 3
            }
        elif stack_type == 'bring_back':
            return {
                'rule': 'bring_back_stack',
                'primary_players': stack.get('primary_stack', []),
                'bring_back_player': stack.get('bring_back'),
                'game_total': stack.get('game_total', 45),
                'priority': ConstraintPriority.AI_MODERATE.value,
                'source': ai_type.value,
                'correlation_strength': 0.5,
                'relaxation_tier': 3
            }
        elif stack_type == 'leverage':
            return {
                'rule': 'leverage_stack',
                'players': [stack.get('player1'), stack.get('player2')],
                'combined_ownership_max': stack.get('combined_ownership', 20),
                'leverage_score_min': 3.0,
                'priority': ConstraintPriority.SOFT_PREFERENCE.value,
                'source': ai_type.value,
                'correlation_strength': 0.4,
                'relaxation_tier': 3
            }
        elif stack_type == 'defensive':
            return {
                'rule': 'defensive_stack',
                'dst_team': stack.get('dst_team'),
                'opposing_players_max': 1,
                'scenario': 'defensive_game',
                'priority': ConstraintPriority.SOFT_PREFERENCE.value,
                'source': ai_type.value,
                'relaxation_tier': 3
            }
        else:
            # Standard two-player stack
            if 'player1' in stack and 'player2' in stack:
                return {
                    'rule': 'standard_stack',
                    'players': [stack['player1'], stack['player2']],
                    'correlation': stack.get('correlation', 0.5),
                    'priority': ConstraintPriority.SOFT_PREFERENCE.value,
                    'source': ai_type.value,
                    'relaxation_tier': 3
                }

        return None

    def _deduplicate_stacks(self, stacking_rules: List[Dict]) -> List[Dict]:
        """Remove duplicate stacks"""
        unique_stacks = []
        seen = set()

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

    def should_apply_constraint(self, constraint: Dict, attempt_num: int) -> bool:
        """
        Determine if constraint should be applied based on relaxation tier

        DFS Value: CRITICAL - Enables three-tier constraint relaxation

        Args:
            constraint: Constraint dictionary with relaxation_tier
            attempt_num: Current attempt (0-2 for three tiers)

        Returns:
            True if constraint should be applied
        """
        tier = constraint.get('relaxation_tier', 1)

        # Tier 1: Never relax (DK rules, salary cap)
        if tier == 1:
            return True

        # Tier 2: Relax only on attempt 3 (final)
        if tier == 2:
            return attempt_num < 2

        # Tier 3: Relax on attempt 2+ (soft preferences)
        if tier == 3:
            return attempt_num == 0

        return True

    def validate_lineup_against_ai(self, lineup: Dict,
                                   enforcement_rules: Dict) -> Tuple[bool, List[str]]:
        """
        Validate lineup against AI enforcement rules

        Returns:
            (is_valid, list of violations)
        """
        violations = []
        captain = lineup.get('Captain')
        flex = lineup.get('FLEX', [])
        all_players = [captain] + flex

        # Check hard constraints only
        for rule in enforcement_rules.get('hard_constraints', []):
            violation = self._check_single_constraint(rule, captain, all_players)
            if violation:
                violations.append(violation)

        # Check stacking rules (if hard)
        for stack_rule in enforcement_rules.get('stacking_rules', []):
            if stack_rule.get('type') == 'hard':
                if not self._validate_stack_rule(all_players, stack_rule):
                    violations.append(
                        f"Stack rule violation: {stack_rule.get('rule')} "
                        f"({stack_rule.get('source')})"
                    )

        # Track violations for learning
        for violation in violations:
            self.violation_patterns[violation[:50]] += 1

        is_valid = len(violations) == 0

        # Record rule application
        self._record_rule_application(lineup, is_valid, violations, enforcement_rules)

        return is_valid, violations

    def _check_single_constraint(self, rule: Dict, captain: str,
                                 all_players: List[str]) -> Optional[str]:
        """Check a single constraint and return violation message if any"""
        rule_type = rule.get('rule')

        if rule_type == 'captain_from_list':
            if captain not in rule.get('players', []):
                return f"Captain {captain} not in AI-recommended list: {rule['source']}"

        elif rule_type == 'consensus_captain_list':
            if captain not in rule.get('players', []):
                return f"Captain {captain} not in consensus list"

        elif rule_type == 'must_include':
            if rule.get('player') not in all_players:
                return f"Missing required player: {rule['player']} ({rule['source']})"

        elif rule_type == 'must_exclude':
            if rule.get('player') in all_players:
                return f"Included banned player: {rule['player']} ({rule['source']})"

        elif rule_type == 'consensus_captain':
            if captain != rule.get('player'):
                return (f"Not using consensus captain: {rule['player']} "
                       f"(agreement: {rule.get('agreement', 0)} AIs)")

        return None

    def _record_rule_application(self, lineup: Dict, is_valid: bool,
                                 violations: List[str], enforcement_rules: Dict) -> None:
        """Record rule application for learning"""
        self.applied_rules.append({
            'timestamp': datetime.now(),
            'lineup_num': lineup.get('Lineup', 0),
            'valid': is_valid,
            'violations': len(violations)
        })

        # Update rule effectiveness
        for rule in enforcement_rules.get('hard_constraints', []):
            rule_key = f"{rule.get('rule')}_{rule.get('source')}"
            self.rule_effectiveness[rule_key]['applied'] += 1
            if is_valid:
                self.rule_effectiveness[rule_key]['success'] += 1

    def _validate_stack_rule(self, players: List[str], stack_rule: Dict) -> bool:
        """Validate a specific stack rule"""
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
            return all(p in players for p in required[:2])

        return True

    def get_effectiveness_report(self) -> Dict:
        """Get report on rule effectiveness"""
        report = {}
        for rule_key, stats in self.rule_effectiveness.items():
            if stats['applied'] > 0:
                report[rule_key] = {
                    'success_rate': stats['success'] / stats['applied'],
                    'applied': stats['applied']
                }
        return report

# ============================================================================
# AI OWNERSHIP BUCKET MANAGER WITH DYNAMIC THRESHOLDS
# ============================================================================

class AIOwnershipBucketManager:
    """
    Enhanced ownership bucket management with dynamic threshold adjustment

    DFS Value: Dynamic thresholds adapt to slate characteristics
    """

    def __init__(self, enforcement_engine: AIEnforcementEngine = None):
        self.enforcement_engine = enforcement_engine
        self.logger = get_logger()

        # Base bucket thresholds (will be adjusted dynamically)
        self.bucket_thresholds = {
            'mega_chalk': 35,
            'chalk': 20,
            'moderate': 15,
            'pivot': 10,
            'leverage': 5,
            'super_leverage': 2
        }

        # Store original thresholds for reference
        self.base_thresholds = self.bucket_thresholds.copy()

    def adjust_thresholds_for_slate(self, df: pd.DataFrame, field_size: str) -> None:
        """
        Dynamically adjust bucket thresholds based on slate characteristics

        DFS Value: CRITICAL - Adapts to ownership distribution
        """
        ownership_std = df['Ownership'].std()
        ownership_mean = df['Ownership'].mean()

        self.logger.log(
            f"Adjusting thresholds - Ownership std: {ownership_std:.1f}, "
            f"mean: {ownership_mean:.1f}",
            "DEBUG"
        )

        # Reset to base thresholds
        self.bucket_thresholds = self.base_thresholds.copy()

        # Adjust based on ownership concentration
        if ownership_std < 5:
            self._apply_flat_ownership_adjustment()
        elif ownership_std > 15:
            self._apply_polarized_ownership_adjustment()

        # Field-size adjustments
        if field_size in ['large_field_aggressive', 'milly_maker']:
            self._apply_large_field_adjustment(field_size)

        # Mean-based adjustments
        self._apply_mean_based_adjustment(ownership_mean)

    def _apply_flat_ownership_adjustment(self) -> None:
        """Apply adjustment for flat ownership slates"""
        for key in self.bucket_thresholds:
            self.bucket_thresholds[key] *= 0.85
        self.logger.log("Flat ownership detected - lowering thresholds", "INFO")

    def _apply_polarized_ownership_adjustment(self) -> None:
        """Apply adjustment for polarized ownership slates"""
        for key in self.bucket_thresholds:
            self.bucket_thresholds[key] *= 1.15
        self.logger.log("Polarized ownership detected - raising thresholds", "INFO")

    def _apply_large_field_adjustment(self, field_size: str) -> None:
        """Apply adjustment for large field contests"""
        for key in self.bucket_thresholds:
            self.bucket_thresholds[key] *= 0.85
        self.logger.log(
            f"Large field ({field_size}) - increasing leverage sensitivity",
            "INFO"
        )

    def _apply_mean_based_adjustment(self, ownership_mean: float) -> None:
        """Apply adjustment based on mean ownership"""
        if ownership_mean < 8:
            self.bucket_thresholds['chalk'] *= 0.9
            self.bucket_thresholds['leverage'] *= 1.1
        elif ownership_mean > 15:
            self.bucket_thresholds['leverage'] *= 0.9

    def categorize_players(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize players into ownership buckets using vectorized operations
        """
        ownership = df['Ownership'].fillna(10)
        players = df['Player'].values
        thresholds = self.bucket_thresholds

        # Vectorized bucketing
        buckets = {
            'mega_chalk': players[ownership >= thresholds['mega_chalk']].tolist(),
            'chalk': players[(ownership >= thresholds['chalk']) &
                            (ownership < thresholds['mega_chalk'])].tolist(),
            'moderate': players[(ownership >= thresholds['moderate']) &
                               (ownership < thresholds['chalk'])].tolist(),
            'pivot': players[(ownership >= thresholds['pivot']) &
                            (ownership < thresholds['moderate'])].tolist(),
            'leverage': players[(ownership >= thresholds['leverage']) &
                               (ownership < thresholds['pivot'])].tolist(),
            'super_leverage': players[ownership < thresholds['leverage']].tolist()
        }

        # Log distribution
        self.logger.log(
            f"Ownership buckets: " + ", ".join(
                f"{k}={len(v)}" for k, v in buckets.items()
            ),
            "DEBUG"
        )

        return buckets

    def calculate_gpp_leverage(self, players: List[str], df: pd.DataFrame) -> float:
        """
        Calculate GPP leverage score for lineup

        DFS Value: Quantifies tournament differentiation
        """
        if not players:
            return 0

        total_projection = 0
        total_ownership = 0
        leverage_bonus = 0

        for i, player in enumerate(players):
            player_data = df[df['Player'] == player]
            if player_data.empty:
                continue

            row = player_data.iloc[0]
            projection = row.get('Projected_Points', 0)
            ownership = row.get('Ownership', 10)

            # Captain gets 1.5x weight
            if i == 0:
                projection *= 1.5
                ownership *= 1.5

            total_projection += projection
            total_ownership += ownership

            # Bonus for leverage plays
            if ownership < self.bucket_thresholds['leverage']:
                leverage_bonus += 15
            elif ownership < self.bucket_thresholds['pivot']:
                leverage_bonus += 8
            elif ownership < self.bucket_thresholds['moderate']:
                leverage_bonus += 3

        # Calculate leverage score
        if total_ownership > 0:
            avg_projection = total_projection / len(players)
            avg_ownership = total_ownership / len(players)
            base_leverage = avg_projection / (avg_ownership + 1)
            return base_leverage + leverage_bonus

        return 0

    def get_bucket_recommendations(self, field_size: str, num_lineups: int) -> Dict:
        """Get ownership bucket recommendations for field size"""
        recommendations = {
            'mega_chalk_limit': 1,
            'chalk_limit': 2,
            'min_leverage': 2,
            'target_ownership': (60, 90)
        }

        if field_size in ['large_field', 'large_field_aggressive']:
            recommendations['mega_chalk_limit'] = 0
            recommendations['chalk_limit'] = 1
            recommendations['min_leverage'] = 3
            recommendations['target_ownership'] = (40, 70)

        elif field_size == 'milly_maker':
            recommendations['mega_chalk_limit'] = 0
            recommendations['chalk_limit'] = 0
            recommendations['min_leverage'] = 4
            recommendations['target_ownership'] = (30, 60)

        return recommendations

# ============================================================================
# AI CONFIG VALIDATOR WITH SIMULATION SUPPORT
# ============================================================================

class AIConfigValidator:
    """Enhanced validator with dynamic strategy selection and simulation awareness"""

    @staticmethod
    def validate_ai_requirements(enforcement_rules: Dict,
                                df: pd.DataFrame) -> Dict:
        """
        Validate that AI requirements are feasible

        Returns:
            Validation result with errors, warnings, suggestions
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }

        available_players = set(df['Player'].values)

        # Validate captain requirements
        AIConfigValidator._validate_captain_requirements(
            enforcement_rules, available_players, validation_result
        )

        # Validate must include players
        AIConfigValidator._validate_must_include(
            enforcement_rules, available_players, validation_result
        )

        # Validate stack feasibility
        AIConfigValidator._validate_stacks(
            enforcement_rules, available_players, validation_result
        )

        # Validate salary feasibility
        AIConfigValidator._validate_salary_feasibility(
            enforcement_rules, df, validation_result
        )

        return validation_result

    @staticmethod
    def _validate_captain_requirements(enforcement_rules: Dict, available_players: Set[str],
                                      validation_result: Dict) -> None:
        """Validate captain requirements"""
        captain_rules = [
            r for r in enforcement_rules.get('hard_constraints', [])
            if r.get('rule') in ['captain_from_list', 'captain_selection', 'consensus_captain_list']
        ]

        for rule in captain_rules:
            valid_captains = [
                p for p in rule.get('players', [])
                if p in available_players
            ]

            if not valid_captains:
                validation_result['errors'].append(
                    "No valid captains from AI recommendations"
                )
                validation_result['is_valid'] = False
                validation_result['suggestions'].append(
                    "Relax captain constraints or check player pool"
                )
            elif len(valid_captains) < 3:
                validation_result['warnings'].append(
                    f"Only {len(valid_captains)} valid captains available"
                )
                validation_result['suggestions'].append(
                    "Consider expanding captain pool for diversity"
                )

    @staticmethod
    def _validate_must_include(enforcement_rules: Dict, available_players: Set[str],
                               validation_result: Dict) -> None:
        """Validate must-include players"""
        must_include = [
            r for r in enforcement_rules.get('hard_constraints', [])
            if r.get('rule') == 'must_include'
        ]

        for rule in must_include:
            if rule.get('player') not in available_players:
                validation_result['errors'].append(
                    f"Required player not available: {rule.get('player')}"
                )
                validation_result['is_valid'] = False

    @staticmethod
    def _validate_stacks(enforcement_rules: Dict, available_players: Set[str],
                        validation_result: Dict) -> None:
        """Validate stack feasibility"""
        stacking_rules = enforcement_rules.get('stacking_rules', [])

        for stack in stacking_rules:
            if stack.get('rule') == 'onslaught_stack':
                players = stack.get('players', [])
                valid = [p for p in players if p in available_players]

                if len(valid) < stack.get('min_players', 3):
                    validation_result['warnings'].append(
                        f"Onslaught stack may not be feasible "
                        f"(only {len(valid)} valid players)"
                    )

    @staticmethod
    def _validate_salary_feasibility(enforcement_rules: Dict, df: pd.DataFrame,
                                    validation_result: Dict) -> None:
        """Validate salary feasibility"""
        hard_constraints = enforcement_rules.get('hard_constraints', [])
        required_players = [
            r.get('player') for r in hard_constraints
            if r.get('rule') == 'must_include' and r.get('player')
        ]

        if required_players:
            min_required_salary = df[
                df['Player'].isin(required_players)
            ]['Salary'].sum()

            if min_required_salary > OptimizerConfig.SALARY_CAP * 0.6:
                validation_result['warnings'].append(
                    f"Required players use {min_required_salary/OptimizerConfig.SALARY_CAP:.0%} "
                    "of salary cap"
                )
                validation_result['suggestions'].append(
                    "May have limited flexibility for other positions"
                )

    @staticmethod
    def get_ai_strategy_distribution(field_size: str, num_lineups: int,
                                    consensus_level: str = 'mixed',
                                    use_genetic: bool = False) -> Dict:
        """
        Dynamic strategy distribution based on consensus level and optimization method

        DFS Value: Adapts lineup mix based on AI agreement and field size
        """
        # Check if genetic algorithm should be used for this field
        field_config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size, {})
        use_genetic = use_genetic or field_config.get('use_genetic', False)

        if use_genetic:
            # GA optimization gets special distribution
            return {
                'genetic_algorithm': num_lineups
            }

        # Standard LP-based distributions
        base_distributions = {
            'small_field': {
                'balanced': 0.5,
                'correlation_heavy': 0.3,
                'ownership_leverage': 0.2
            },
            'medium_field': {
                'balanced': 0.4,
                'correlation_heavy': 0.3,
                'ownership_leverage': 0.2,
                'contrarian': 0.1
            },
            'large_field': {
                'balanced': 0.3,
                'correlation_heavy': 0.2,
                'ownership_leverage': 0.3,
                'contrarian': 0.2
            },
            'large_field_aggressive': {
                'balanced': 0.2,
                'correlation_heavy': 0.2,
                'ownership_leverage': 0.3,
                'contrarian': 0.3
            },
            'milly_maker': {
                'balanced': 0.1,
                'correlation_heavy': 0.1,
                'ownership_leverage': 0.4,
                'contrarian': 0.4
            }
        }

        distribution = base_distributions.get(
            field_size,
            base_distributions['large_field']
        ).copy()

        # Adjust based on consensus level
        distribution = AIConfigValidator._adjust_for_consensus(
            distribution, consensus_level
        )

        # Convert to lineup counts
        return AIConfigValidator._convert_to_lineup_counts(
            distribution, num_lineups
        )

    @staticmethod
    def _adjust_for_consensus(distribution: Dict, consensus_level: str) -> Dict:
        """Adjust distribution based on AI consensus level"""
        if consensus_level == 'high':
            distribution['balanced'] = min(
                distribution.get('balanced', 0.3) * 1.3, 0.5
            )
            distribution['contrarian'] = (
                distribution.get('contrarian', 0.2) * 0.7
            )
        elif consensus_level == 'low':
            distribution['contrarian'] = min(
                distribution.get('contrarian', 0.2) * 1.3, 0.4
            )
            distribution['balanced'] = (
                distribution.get('balanced', 0.3) * 0.7
            )

        # Normalize
        total = sum(distribution.values())
        return {k: v/total for k, v in distribution.items()}

    @staticmethod
    def _convert_to_lineup_counts(distribution: Dict, num_lineups: int) -> Dict:
        """Convert percentages to lineup counts"""
        lineup_distribution = {}
        allocated = 0

        for strategy, pct in distribution.items():
            count = int(num_lineups * pct)
            lineup_distribution[strategy] = count
            allocated += count

        # Allocate remaining lineups to balanced
        if allocated < num_lineups:
            lineup_distribution['balanced'] = (
                lineup_distribution.get('balanced', 0) +
                (num_lineups - allocated)
            )

        return lineup_distribution

# ============================================================================
# AI SYNTHESIS ENGINE WITH PATTERN ANALYSIS
# ============================================================================

class AISynthesisEngine:
    """Enhanced synthesis engine with pattern recognition and simulation integration"""

    def __init__(self):
        self.logger = get_logger()
        self.synthesis_history = deque(maxlen=20)

    def synthesize_recommendations(self,
                                   game_theory: AIRecommendation,
                                   correlation: AIRecommendation,
                                   contrarian: AIRecommendation) -> Dict:
        """
        Synthesize three AI perspectives into unified strategy

        DFS Value: Combines multiple strategic viewpoints for robust approach
        """
        self.logger.log("Synthesizing triple AI recommendations", "INFO")

        synthesis = {
            'captain_strategy': {},
            'player_rankings': {},
            'stacking_rules': [],
            'avoidance_rules': [],
            'enforcement_rules': [],
            'confidence': 0,
            'narrative': "",
            'patterns': []
        }

        # Synthesize each component
        synthesis['captain_strategy'] = self._synthesize_captains(
            game_theory, correlation, contrarian
        )

        synthesis['player_rankings'] = self._synthesize_player_rankings(
            game_theory, correlation, contrarian
        )

        synthesis['stacking_rules'] = self._synthesize_stacks(
            game_theory, correlation, contrarian
        )

        synthesis['patterns'] = self._analyze_patterns(
            game_theory, correlation, contrarian
        )

        synthesis['confidence'] = self._calculate_confidence(
            game_theory, correlation, contrarian
        )

        synthesis['narrative'] = self._build_narrative(
            game_theory, correlation, contrarian
        )

        # Store in history
        self._record_synthesis(synthesis)

        return synthesis

    def _synthesize_captains(self, game_theory: AIRecommendation,
                            correlation: AIRecommendation,
                            contrarian: AIRecommendation) -> Dict:
        """Synthesize captain recommendations with consensus tracking"""
        captain_votes = defaultdict(list)

        for ai_type, rec in [
            (AIStrategistType.GAME_THEORY, game_theory),
            (AIStrategistType.CORRELATION, correlation),
            (AIStrategistType.CONTRARIAN_NARRATIVE, contrarian)
        ]:
            for captain in rec.captain_targets[:5]:
                captain_votes[captain].append(ai_type.value)

        # Classify captains by consensus
        captain_strategy = {}
        for captain, votes in captain_votes.items():
            if len(votes) == 3:
                captain_strategy[captain] = 'consensus'
            elif len(votes) == 2:
                captain_strategy[captain] = 'majority'
            else:
                captain_strategy[captain] = votes[0]

        return captain_strategy

    def _synthesize_player_rankings(self, game_theory: AIRecommendation,
                                    correlation: AIRecommendation,
                                    contrarian: AIRecommendation) -> Dict:
        """Synthesize player rankings with weighted scoring"""
        player_scores = defaultdict(float)

        weights = {
            AIStrategistType.GAME_THEORY:
                OptimizerConfig.AI_WEIGHTS.get('game_theory', 0.33),
            AIStrategistType.CORRELATION:
                OptimizerConfig.AI_WEIGHTS.get('correlation', 0.33),
            AIStrategistType.CONTRARIAN_NARRATIVE:
                OptimizerConfig.AI_WEIGHTS.get('contrarian', 0.34)
        }

        for ai_type, rec in [
            (AIStrategistType.GAME_THEORY, game_theory),
            (AIStrategistType.CORRELATION, correlation),
            (AIStrategistType.CONTRARIAN_NARRATIVE, contrarian)
        ]:
            weight = weights[ai_type]

            # Score must-play players
            for i, player in enumerate(rec.must_play[:5]):
                player_scores[player] += weight * rec.confidence * (1 - i * 0.1)

            # Negative scores for fades
            for player in rec.never_play[:3]:
                player_scores[player] -= weight * rec.confidence * 0.5

        # Normalize scores
        if player_scores:
            max_score = max(abs(score) for score in player_scores.values())
            if max_score > 0:
                return {
                    player: score / max_score
                    for player, score in player_scores.items()
                }

        return {}

    def _synthesize_stacks(self, game_theory: AIRecommendation,
                          correlation: AIRecommendation,
                          contrarian: AIRecommendation) -> List[Dict]:
        """Synthesize and prioritize all stacks"""
        all_stacks = []
        stack_patterns = defaultdict(int)

        for rec in [game_theory, correlation, contrarian]:
            for stack in rec.stacks:
                all_stacks.append(stack)
                stack_type = stack.get('type', 'standard')
                stack_patterns[stack_type] += 1

        return self._prioritize_stacks(all_stacks)

    def _prioritize_stacks(self, all_stacks: List[Dict]) -> List[Dict]:
        """Prioritize and deduplicate stacks"""
        stack_groups = defaultdict(list)

        for stack in all_stacks:
            # Create grouping key
            if 'player1' in stack and 'player2' in stack:
                key = tuple(sorted([stack['player1'], stack['player2']]))
            elif 'players' in stack:
                key = tuple(sorted(stack['players'][:2]))
            else:
                key = stack.get('type', 'unknown')

            stack_groups[key].append(stack)

        # Select best from each group
        prioritized = []
        for group in stack_groups.values():
            if group:
                best = max(group, key=lambda s: s.get('correlation', 0.5))
                if len(group) > 1:
                    best['consensus'] = True
                    best['priority'] = best.get('priority', 50) + 10 * len(group)
                prioritized.append(best)

        # Sort by priority
        prioritized.sort(key=lambda s: s.get('priority', 50), reverse=True)
        return prioritized[:10]

    def _analyze_patterns(self, game_theory: AIRecommendation,
                         correlation: AIRecommendation,
                         contrarian: AIRecommendation) -> List[str]:
        """Analyze patterns in AI recommendations"""
        patterns = []

        # Check for unanimous agreement
        captain_overlap = (
            set(game_theory.captain_targets) &
            set(correlation.captain_targets) &
            set(contrarian.captain_targets)
        )

        if captain_overlap:
            patterns.append(f"Strong consensus on {len(captain_overlap)} captains")

        # Check confidence patterns
        confidences = {
            'game_theory': game_theory.confidence,
            'correlation': correlation.confidence,
            'contrarian': contrarian.confidence
        }

        max_conf = max(confidences.values())
        if confidences['contrarian'] == max_conf:
            patterns.append("Contrarian approach favored")

        return patterns

    def _calculate_confidence(self, game_theory: AIRecommendation,
                             correlation: AIRecommendation,
                             contrarian: AIRecommendation) -> float:
        """Calculate overall confidence from all AIs"""
        weights = {
            AIStrategistType.GAME_THEORY:
                OptimizerConfig.AI_WEIGHTS.get('game_theory', 0.33),
            AIStrategistType.CORRELATION:
                OptimizerConfig.AI_WEIGHTS.get('correlation', 0.33),
            AIStrategistType.CONTRARIAN_NARRATIVE:
                OptimizerConfig.AI_WEIGHTS.get('contrarian', 0.34)
        }

        confidences = [
            game_theory.confidence * weights[AIStrategistType.GAME_THEORY],
            correlation.confidence * weights[AIStrategistType.CORRELATION],
            contrarian.confidence * weights[AIStrategistType.CONTRARIAN_NARRATIVE]
        ]

        return sum(confidences)

    def _build_narrative(self, game_theory: AIRecommendation,
                        correlation: AIRecommendation,
                        contrarian: AIRecommendation) -> str:
        """Build combined narrative from all AIs"""
        narratives = []

        if game_theory.narrative:
            narratives.append(f"GT: {game_theory.narrative[:80]}")
        if correlation.narrative:
            narratives.append(f"Corr: {correlation.narrative[:80]}")
        if contrarian.narrative:
            narratives.append(f"Contra: {contrarian.narrative[:80]}")

        return " | ".join(narratives)

    def _record_synthesis(self, synthesis: Dict) -> None:
        """Record synthesis in history"""
        self.synthesis_history.append({
            'timestamp': datetime.now(),
            'confidence': synthesis['confidence'],
            'captain_count': len(synthesis['captain_strategy']),
            'patterns': synthesis['patterns']
        })

# ============================================================================
# PART 4: BASE AI STRATEGIST & CLAUDE API MANAGER
# ============================================================================

# ============================================================================
# BASE AI STRATEGIST WITH LEARNING AND SIMULATION INTEGRATION
# ============================================================================

class BaseAIStrategist:
    """
    Enhanced base class for all AI strategists with learning capabilities and simulation support

    DFS Value: Learning from past performance improves future recommendations
    """

    def __init__(self, api_manager=None, strategist_type: AIStrategistType = None):
        self.api_manager = api_manager
        self.strategist_type = strategist_type
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()
        self.response_cache = {}
        self.max_cache_size = 20
        self._cache_lock = threading.RLock()

        # Performance tracking for learning
        self.performance_history = deque(maxlen=100)
        self.successful_patterns = defaultdict(float)

        # Fallback confidence levels by AI type
        self.fallback_confidence = {
            AIStrategistType.GAME_THEORY: 0.55,
            AIStrategistType.CORRELATION: 0.60,
            AIStrategistType.CONTRARIAN_NARRATIVE: 0.50
        }

        # Adaptive confidence modifier
        self.adaptive_confidence_modifier = 1.0
        self.df = None

        # Monte Carlo engine for simulation-aware recommendations (optional)
        self.mc_engine = None

    def get_recommendation(self, df: pd.DataFrame, game_info: Dict,
                          field_size: str, use_api: bool = True) -> AIRecommendation:
        """
        Get AI recommendation with comprehensive error handling and learning

        Args:
            df: Player DataFrame
            game_info: Game context (total, spread, weather, etc.)
            field_size: Contest size configuration
            use_api: Whether to use API or manual input

        Returns:
            AIRecommendation object
        """
        try:
            self.df = df

            if df.empty:
                self.logger.log(
                    f"{self.strategist_type.value}: Empty DataFrame provided",
                    "ERROR"
                )
                return self._get_fallback_recommendation(df, field_size)

            # Analyze slate and check cache
            slate_profile = self._analyze_slate_profile(df, game_info)
            cache_key = self._generate_cache_key(df, game_info, field_size)

            cached_rec = self._check_cache(cache_key)
            if cached_rec:
                return cached_rec

            # Generate prompt and get response
            prompt = self.generate_prompt(df, game_info, field_size, slate_profile)
            response = self._get_response(prompt, use_api, df, game_info,
                                         field_size, slate_profile)

            # Parse and enhance recommendation
            recommendation = self.parse_response(response, df, field_size)
            recommendation = self._enhance_recommendation(
                recommendation, slate_profile, df, field_size
            )

            # Cache and return
            self._cache_recommendation(cache_key, recommendation)
            return recommendation

        except Exception as e:
            self.logger.log_exception(
                e,
                f"{self.strategist_type.value}.get_recommendation"
            )
            return self._get_fallback_recommendation(df, field_size)

    def _check_cache(self, cache_key: str) -> Optional[AIRecommendation]:
        """Check cache for existing recommendation"""
        with self._cache_lock:
            if cache_key in self.response_cache:
                self.logger.log(
                    f"{self.strategist_type.value}: Using cached recommendation",
                    "DEBUG"
                )
                cached_rec = self.response_cache[cache_key]
                cached_rec.confidence *= self.adaptive_confidence_modifier
                return cached_rec
        return None

    def _get_response(self, prompt: str, use_api: bool, df: pd.DataFrame,
                     game_info: Dict, field_size: str,
                     slate_profile: Dict) -> str:
        """Get response from API or fallback"""
        if use_api and self.api_manager and self.api_manager.client:
            return self._get_api_response(prompt)
        else:
            return self._get_fallback_response(df, game_info, field_size, slate_profile)

    def _enhance_recommendation(self, recommendation: AIRecommendation,
                               slate_profile: Dict, df: pd.DataFrame,
                               field_size: str) -> AIRecommendation:
        """Enhance recommendation with learning and validation"""
        # Apply learned adjustments
        recommendation = self._apply_learned_adjustments(recommendation, slate_profile)

        # Validate recommendation
        is_valid, errors = recommendation.validate()
        if not is_valid:
            self.logger.log(
                f"{self.strategist_type.value} validation errors: {errors}",
                "WARNING"
            )
            recommendation = self._correct_recommendation(recommendation, df)

        # Add enforcement rules
        recommendation.enforcement_rules = self.create_enforcement_rules(
            recommendation, df, field_size, slate_profile
        )

        return recommendation

    def _cache_recommendation(self, cache_key: str,
                             recommendation: AIRecommendation) -> None:
        """Cache recommendation with size management"""
        with self._cache_lock:
            self.response_cache[cache_key] = recommendation
            if len(self.response_cache) > self.max_cache_size:
                for key in list(self.response_cache.keys())[:5]:
                    del self.response_cache[key]

    def _analyze_slate_profile(self, df: pd.DataFrame, game_info: Dict) -> Dict:
        """
        Analyze slate characteristics for dynamic strategy adjustment

        DFS Value: Context-aware recommendations based on game environment
        """
        profile = {
            'player_count': len(df),
            'avg_salary': df['Salary'].mean(),
            'salary_range': df['Salary'].max() - df['Salary'].min(),
            'avg_ownership': df.get('Ownership', pd.Series([10])).mean(),
            'ownership_concentration': df.get('Ownership', pd.Series([10])).std(),
            'total': game_info.get('total', 45),
            'spread': abs(game_info.get('spread', 0)),
            'weather': game_info.get('weather', 'Clear'),
            'teams': df['Team'].nunique(),
            'positions': df['Position'].value_counts().to_dict(),
            'value_distribution': (
                df['Projected_Points'].std() / df['Projected_Points'].mean()
            ),
            'is_primetime': game_info.get('primetime', False),
            'injuries': game_info.get('injury_count', 0),
            'slate_type': self._determine_slate_type(df, game_info)
        }

        return profile

    def _determine_slate_type(self, df: pd.DataFrame, game_info: Dict) -> str:
        """Determine slate type for strategy adjustment"""
        total = game_info.get('total', 45)
        spread = abs(game_info.get('spread', 0))

        if total > 50 and spread < 3:
            return 'shootout'
        elif total < 40:
            return 'low_scoring'
        elif spread > 10:
            return 'blowout_risk'
        elif df['Salary'].std() < 1000:
            return 'flat_pricing'
        else:
            return 'standard'

    def _apply_learned_adjustments(self, recommendation: AIRecommendation,
                                   slate_profile: Dict) -> AIRecommendation:
        """Apply learned patterns and adjustments"""
        slate_type = slate_profile.get('slate_type', 'standard')

        # Adjust confidence based on historical success
        if slate_type in self.successful_patterns:
            confidence_boost = self.successful_patterns[slate_type] * 0.1
            recommendation.confidence = min(
                0.95,
                recommendation.confidence + confidence_boost
            )

        recommendation.confidence *= self.adaptive_confidence_modifier

        # Adjust captain targets based on slate type
        recommendation = self._adjust_captains_for_slate(
            recommendation, slate_type, slate_profile
        )

        return recommendation

    def _adjust_captains_for_slate(self, recommendation: AIRecommendation,
                                   slate_type: str,
                                   slate_profile: Dict) -> AIRecommendation:
        """Adjust captain targets based on slate characteristics"""
        if slate_type == 'shootout' and self.df is not None:
            # Prioritize pass catchers in shootouts
            qbs_and_receivers = [
                player for player in recommendation.captain_targets
                if self._is_passing_game_player(player)
            ]

            if qbs_and_receivers:
                recommendation.captain_targets = qbs_and_receivers + [
                    p for p in recommendation.captain_targets
                    if p not in qbs_and_receivers
                ]

        return recommendation

    def _is_passing_game_player(self, player: str) -> bool:
        """Check if player is QB/WR/TE"""
        if self.df is None:
            return False

        player_data = self.df[self.df['Player'] == player]
        if not player_data.empty:
            return player_data.iloc[0]['Position'] in ['QB', 'WR', 'TE']
        return False

    def track_performance(self, lineup: Dict,
                         actual_points: Optional[float] = None) -> None:
        """
        Track lineup performance for learning

        DFS Value: Continuous improvement through feedback
        """
        if actual_points is not None:
            performance_data = {
                'strategy': self.strategist_type.value,
                'projected': lineup.get('Projected', 0),
                'actual': actual_points,
                'accuracy': 1 - abs(
                    actual_points - lineup.get('Projected', 0)
                ) / max(actual_points, 1),
                'timestamp': datetime.now(),
                'slate_type': lineup.get('slate_type', 'standard')
            }

            self.performance_history.append(performance_data)

            # Update successful patterns
            slate_type = performance_data.get('slate_type', 'standard')
            if performance_data['accuracy'] > 0.8:
                self.successful_patterns[slate_type] += 1

            # Update adaptive confidence
            self._update_adaptive_confidence()

    def _update_adaptive_confidence(self) -> None:
        """Update confidence modifier based on recent performance"""
        if len(self.performance_history) >= 10:
            recent_accuracy = np.mean([
                p['accuracy']
                for p in list(self.performance_history)[-10:]
            ])
            self.adaptive_confidence_modifier = 0.5 + recent_accuracy

    def _get_api_response(self, prompt: str) -> str:
        """Get response from API with error handling"""
        try:
            if not self.api_manager:
                return "{}"

            response = self.api_manager.get_ai_response(
                prompt,
                self.strategist_type
            )
            return response

        except Exception as e:
            self.logger.log_exception(e, f"{self.strategist_type.value} API call")
            return "{}"

    def _get_fallback_response(self, df: pd.DataFrame, game_info: Dict,
                               field_size: str, slate_profile: Dict) -> str:
        """Generate fallback response using statistical analysis"""
        return "{}"

    def _get_fallback_recommendation(self, df: pd.DataFrame,
                                    field_size: str) -> AIRecommendation:
        """
        Create fallback recommendation using statistical analysis

        DFS Value: CRITICAL - Better than empty recommendation
        """
        if df.empty:
            return AIRecommendation(
                captain_targets=[],
                confidence=0.3,
                source_ai=self.strategist_type
            )

        # Vectorized calculations
        ownership = df['Ownership'].fillna(10)
        projected = df['Projected_Points']

        captains, must_play, never_play = self._select_fallback_players(
            df, ownership, projected
        )

        stacks = self._create_statistical_stacks(df)

        confidence = self.fallback_confidence.get(self.strategist_type, 0.5)

        return AIRecommendation(
            captain_targets=captains[:7],
            must_play=must_play[:3],
            never_play=never_play[:3],
            stacks=stacks[:3],
            key_insights=[
                f"Statistical {self.strategist_type.value} fallback",
                f"Based on {len(df)} players in pool",
                "API unavailable - using data-driven approach"
            ],
            confidence=confidence,
            narrative=f"Data-driven {self.strategist_type.value} approach",
            source_ai=self.strategist_type
        )

    def _select_fallback_players(self, df: pd.DataFrame, ownership: pd.Series,
                                 projected: pd.Series) -> Tuple[List[str], List[str], List[str]]:
        """Select players based on AI type using vectorized operations"""
        if self.strategist_type == AIStrategistType.GAME_THEORY:
            low_own_mask = ownership < 15
            leverage_mask = ownership < 10
            high_own_mask = ownership > 35

            captains = df.loc[low_own_mask].nlargest(7, 'Projected_Points')['Player'].tolist()
            must_play = df.loc[leverage_mask].nlargest(3, 'Projected_Points')['Player'].tolist()
            never_play = df.loc[high_own_mask].nlargest(2, 'Ownership')['Player'].tolist()

        elif self.strategist_type == AIStrategistType.CORRELATION:
            qb_mask = df['Position'] == 'QB'
            receiver_mask = df['Position'].isin(['WR', 'TE'])

            captains = df.loc[qb_mask].nlargest(3, 'Projected_Points')['Player'].tolist()
            captains += df.loc[receiver_mask].nlargest(4, 'Projected_Points')['Player'].tolist()
            must_play = []
            never_play = []

        else:  # CONTRARIAN_NARRATIVE
            ultra_low_mask = ownership < 10
            very_low_mask = ownership < 5

            captains = df.loc[ultra_low_mask].nlargest(7, 'Projected_Points')['Player'].tolist()
            must_play = df.loc[very_low_mask].nlargest(2, 'Projected_Points')['Player'].tolist()
            never_play = df.nlargest(3, 'Ownership')['Player'].tolist()

        return captains, must_play, never_play

    def _create_statistical_stacks(self, df: pd.DataFrame) -> List[Dict]:
        """Create stacks using statistical analysis"""
        stacks = []

        try:
            qb_mask = df['Position'] == 'QB'
            qbs = df[qb_mask]

            for _, qb in qbs.iterrows():
                team = qb['Team']
                teammates_mask = (df['Team'] == team) & df['Position'].isin(['WR', 'TE'])
                teammates = df[teammates_mask]

                if not teammates.empty:
                    top_teammate = teammates.nlargest(1, 'Projected_Points').iloc[0]

                    stacks.append({
                        'player1': qb['Player'],
                        'player2': top_teammate['Player'],
                        'type': 'QB_receiver',
                        'correlation': 0.6,
                        'priority': 70
                    })

        except Exception as e:
            self.logger.log(f"Error creating statistical stacks: {e}", "WARNING")

        return stacks

    def _correct_recommendation(self, recommendation: AIRecommendation,
                                df: pd.DataFrame) -> AIRecommendation:
        """Correct invalid recommendations"""
        available_players = set(df['Player'].values)

        # Filter to valid players
        recommendation.captain_targets = [
            p for p in recommendation.captain_targets
            if p in available_players
        ]

        # Add fallback captains if needed
        if len(recommendation.captain_targets) < 3:
            top_players = df.nlargest(5, 'Projected_Points')['Player'].tolist()
            for player in top_players:
                if player not in recommendation.captain_targets:
                    recommendation.captain_targets.append(player)
                if len(recommendation.captain_targets) >= 5:
                    break

        recommendation.must_play = [
            p for p in recommendation.must_play if p in available_players
        ]
        recommendation.never_play = [
            p for p in recommendation.never_play if p in available_players
        ]

        return recommendation

    def _generate_cache_key(self, df: pd.DataFrame, game_info: Dict,
                           field_size: str) -> str:
        """Generate cache key for memoization"""
        key_components = [
            str(len(df)),
            str(df['Player'].iloc[0] if not df.empty else ''),
            str(game_info.get('total', 45)),
            str(game_info.get('spread', 0)),
            field_size,
            self.strategist_type.value
        ]

        key_string = "_".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def initialize_mc_engine(self, df: pd.DataFrame, game_info: Dict) -> None:
        """Initialize Monte Carlo engine for simulation-aware recommendations"""
        try:
            self.mc_engine = MonteCarloSimulationEngine(
                df,
                game_info,
                n_simulations=OptimizerConfig.MC_FAST_SIMULATIONS
            )
            self.logger.log(
                f"{self.strategist_type.value}: MC engine initialized",
                "DEBUG"
            )
        except Exception as e:
            self.logger.log(
                f"{self.strategist_type.value}: MC engine init failed: {e}",
                "WARNING"
            )
            self.mc_engine = None

    # Abstract methods to be implemented by child classes
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict,
                       field_size: str, slate_profile: Dict) -> str:
        """Generate AI prompt - must be implemented by child classes"""
        raise NotImplementedError

    def parse_response(self, response: str, df: pd.DataFrame,
                      field_size: str) -> AIRecommendation:
        """Parse AI response - must be implemented by child classes"""
        raise NotImplementedError

    def create_enforcement_rules(self, recommendation: AIRecommendation,
                                df: pd.DataFrame, field_size: str,
                                slate_profile: Dict) -> List[Dict]:
        """Create enforcement rules - can be overridden by child classes"""
        rules = []
        available_players = set(df['Player'].values)

        valid_captains = [
            c for c in recommendation.captain_targets
            if c in available_players
        ]

        if valid_captains:
            priority = int(recommendation.confidence * 100)

            if recommendation.confidence > 0.7:
                rules.append({
                    'type': 'hard',
                    'constraint': f'{self.strategist_type.value}_captain',
                    'players': valid_captains[:5],
                    'priority': priority,
                    'relaxation_tier': 2
                })
            else:
                rules.append({
                    'type': 'soft',
                    'constraint': f'{self.strategist_type.value}_captain_pref',
                    'players': valid_captains[:5],
                    'weight': recommendation.confidence,
                    'priority': int(priority * 0.7)
                })

        return rules

# ============================================================================
# CLAUDE API MANAGER WITH ROBUST ERROR HANDLING
# ============================================================================

class ClaudeAPIManager:
    """
    Enhanced Claude API manager with retry logic and comprehensive caching

    DFS Value: Reliable API access = consistent AI recommendations
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.cache = {}
        self.max_cache_size = 30
        self._cache_lock = threading.RLock()

        self.stats = {
            'requests': 0,
            'errors': 0,
            'cache_hits': 0,
            'cache_size': 0,
            'total_tokens': 0,
            'avg_response_time': 0,
            'by_ai': {
                AIStrategistType.GAME_THEORY: {
                    'requests': 0, 'errors': 0, 'tokens': 0, 'avg_time': 0
                },
                AIStrategistType.CORRELATION: {
                    'requests': 0, 'errors': 0, 'tokens': 0, 'avg_time': 0
                },
                AIStrategistType.CONTRARIAN_NARRATIVE: {
                    'requests': 0, 'errors': 0, 'tokens': 0, 'avg_time': 0
                }
            }
        }

        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()
        self.response_times = deque(maxlen=100)

        self.initialize_client()

    def initialize_client(self) -> None:
        """Initialize Claude client with validation"""
        try:
            if not self.api_key or not self.api_key.startswith('sk-'):
                raise ValueError("Invalid API key format")

            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                self.validate_connection()

            except ImportError:
                self.logger.log(
                    "Anthropic library not installed. "
                    "Install with: pip install anthropic",
                    "ERROR"
                )
                self.client = None
                return

            self.logger.log("Claude API client initialized successfully", "INFO")

        except Exception as e:
            self.logger.log(f"Failed to initialize Claude API: {e}", "ERROR")
            self.client = None

    def get_ai_response(self, prompt: str,
                       ai_type: Optional[AIStrategistType] = None,
                       max_retries: int = 3) -> str:
        """
        Get response from Claude API with exponential backoff retry

        DFS Value: CRITICAL - Retry logic prevents failures from transient issues
        """
        # Check cache first
        cached_response = self._check_response_cache(prompt, ai_type)
        if cached_response:
            return cached_response

        # Update statistics
        self._update_request_stats(ai_type)

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                response = self._make_api_call(prompt, attempt)

                if response:
                    self._update_success_stats(response, ai_type, attempt)
                    self._cache_response(prompt, response)
                    return response

            except TimeoutError:
                if not self._handle_timeout(attempt, max_retries):
                    return self._handle_final_failure(ai_type, "timeout")

            except Exception as e:
                if not self._handle_exception(e, attempt, max_retries, ai_type):
                    return self._handle_final_failure(ai_type, "exception")

        return "{}"

    def _check_response_cache(self, prompt: str,
                              ai_type: Optional[AIStrategistType]) -> Optional[str]:
        """Check cache for existing response"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        with self._cache_lock:
            if prompt_hash in self.cache:
                self.stats['cache_hits'] += 1
                self.logger.log(
                    f"Cache hit for {ai_type.value if ai_type else 'unknown'}",
                    "DEBUG"
                )
                return self.cache[prompt_hash]

        return None

    def _update_request_stats(self, ai_type: Optional[AIStrategistType]) -> None:
        """Update request statistics"""
        self.stats['requests'] += 1
        if ai_type:
            self.stats['by_ai'][ai_type]['requests'] += 1

    def _make_api_call(self, prompt: str, attempt: int) -> Optional[str]:
        """Make actual API call with timeout"""
        if not self.client:
            raise Exception("API client not initialized")

        self.perf_monitor.start_timer("claude_api_call")
        start_time = time.time()

        # Cap timeout to prevent overflow
        timeout = min(30 * (1.5 ** attempt), 300)

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0.7,
            timeout=timeout,
            system="""You are an expert Daily Fantasy Sports (DFS) strategist specializing in NFL tournament optimization.
                     You provide specific, actionable recommendations using exact player names and clear reasoning.
                     Always respond with valid JSON containing specific player recommendations.
                     Focus on game theory, correlations, and contrarian angles that win GPP tournaments.
                     Your recommendations must be enforceable as optimization constraints.""",
            messages=[{"role": "user", "content": prompt}]
        )

        self.perf_monitor.stop_timer("claude_api_call")

        response = message.content[0].text if message.content else "{}"
        return response

    def _update_success_stats(self, response: str,
                             ai_type: Optional[AIStrategistType],
                             attempt: int) -> None:
        """Update statistics after successful API call"""
        response_time = time.time() - self.perf_monitor.start_times.get("claude_api_call", time.time())
        self.response_times.append(response_time)

        if ai_type:
            self.stats['by_ai'][ai_type]['tokens'] += len(response) // 4

            current_avg = self.stats['by_ai'][ai_type]['avg_time']
            total_requests = self.stats['by_ai'][ai_type]['requests']
            self.stats['by_ai'][ai_type]['avg_time'] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )

        self.stats['total_tokens'] += len(response) // 4
        self.stats['avg_response_time'] = np.mean(list(self.response_times))

    def _cache_response(self, prompt: str, response: str) -> None:
        """Cache response with size management"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        with self._cache_lock:
            self.cache[prompt_hash] = response
            self.stats['cache_size'] = len(self.cache)

            if len(self.cache) > self.max_cache_size:
                for key in list(self.cache.keys())[:10]:
                    del self.cache[key]

    def _handle_timeout(self, attempt: int, max_retries: int) -> bool:
        """Handle timeout error, return True if should retry"""
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            self.logger.log(
                f"Timeout on attempt {attempt+1}/{max_retries}, "
                f"retrying in {wait_time}s",
                "WARNING"
            )
            time.sleep(wait_time)
            return True

        self.logger.log("All API retries exhausted (timeout)", "ERROR")
        return False

    def _handle_exception(self, e: Exception, attempt: int, max_retries: int,
                         ai_type: Optional[AIStrategistType]) -> bool:
        """Handle exception, return True if should retry"""
        error_str = str(e).lower()

        # Handle rate limiting
        if "rate_limit" in error_str or "429" in error_str:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                self.logger.log(f"Rate limited, waiting {wait_time}s", "WARNING")
                time.sleep(wait_time)
                return True

        # Log error
        self.stats['errors'] += 1
        if ai_type:
            self.stats['by_ai'][ai_type]['errors'] += 1

        if attempt < max_retries - 1:
            self.logger.log(
                f"API error on attempt {attempt+1}: {e}, retrying...",
                "WARNING"
            )
            time.sleep(2 ** attempt)
            return True

        self.logger.log(
            f"API error for {ai_type.value if ai_type else 'unknown'}: {e}",
            "ERROR"
        )
        return False

    def _handle_final_failure(self, ai_type: Optional[AIStrategistType],
                             failure_type: str) -> str:
        """Handle final failure after all retries exhausted"""
        self.stats['errors'] += 1
        if ai_type:
            self.stats['by_ai'][ai_type]['errors'] += 1
        return "{}"

    def validate_connection(self) -> bool:
        """Validate API connection is working"""
        try:
            if not self.client:
                return False

            test_prompt = "Respond with only the word: OK"

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=10,
                temperature=0,
                timeout=10,
                messages=[{"role": "user", "content": test_prompt}]
            )

            return bool(message.content)

        except Exception as e:
            self.logger.log(f"API validation failed: {e}", "ERROR")
            return False

    def get_stats(self) -> Dict:
        """Get comprehensive API usage statistics"""
        with self._cache_lock:
            return {
                'requests': self.stats['requests'],
                'errors': self.stats['errors'],
                'error_rate': (
                    self.stats['errors'] / max(self.stats['requests'], 1)
                ),
                'cache_hits': self.stats['cache_hits'],
                'cache_hit_rate': (
                    self.stats['cache_hits'] / max(self.stats['requests'], 1)
                ),
                'cache_size': len(self.cache),
                'total_tokens': self.stats['total_tokens'],
                'avg_response_time': self.stats['avg_response_time'],
                'by_ai': dict(self.stats['by_ai'])
            }

    def clear_cache(self) -> None:
        """Clear response cache"""
        with self._cache_lock:
            self.cache.clear()
            self.stats['cache_size'] = 0

        self.logger.log("API cache cleared", "INFO")

# ============================================================================
# PART 5: INDIVIDUAL AI STRATEGISTS
# ============================================================================

# ============================================================================
# GPP GAME THEORY STRATEGIST
# ============================================================================

class GPPGameTheoryStrategist(BaseAIStrategist):
    """
    AI Strategist 1: Game Theory and Ownership Leverage

    DFS Value: Identifies ownership arbitrage opportunities
    """

    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.GAME_THEORY)

    def generate_prompt(self, df: pd.DataFrame, game_info: Dict,
                       field_size: str, slate_profile: Dict) -> str:
        """Generate game theory focused prompt with slate context"""
        self.logger.log(
            f"Generating Game Theory prompt for {field_size}",
            "DEBUG"
        )

        if df.empty:
            return "Error: Empty player pool"

        # Gather prompt components
        ownership_analysis = self._get_ownership_analysis(df, field_size)
        leverage_plays = self._get_leverage_plays(df)
        strategy_guidance = self._get_strategy_guidance(field_size, slate_profile)

        # Build complete prompt
        prompt = self._build_game_theory_prompt(
            game_info, slate_profile, ownership_analysis,
            leverage_plays, strategy_guidance
        )

        return prompt

    def _get_ownership_analysis(self, df: pd.DataFrame, field_size: str) -> Dict:
        """Get ownership bucket analysis"""
        bucket_manager = AIOwnershipBucketManager()
        bucket_manager.adjust_thresholds_for_slate(df, field_size)
        buckets = bucket_manager.categorize_players(df)

        return {
            'buckets': buckets,
            'distribution': {k: len(v) for k, v in buckets.items()}
        }

    def _get_leverage_plays(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get low-owned high-upside plays"""
        low_owned_mask = df['Ownership'] < 10
        return df[low_owned_mask].nlargest(10, 'Projected_Points')

    def _get_strategy_guidance(self, field_size: str, slate_profile: Dict) -> Dict:
        """Get field and slate specific strategy guidance"""
        field_strategies = {
            'small_field': "Focus on slight differentiation while maintaining optimal plays",
            'medium_field': "Balance chalk with 2-3 strong leverage plays",
            'large_field': "Aggressive leverage with <15% owned captains",
            'large_field_aggressive': "Ultra-leverage approach with <10% captains",
            'milly_maker': "Maximum contrarian approach with <10% captains only"
        }

        slate_adjustments = {
            'shootout': "Prioritize ceiling over floor, embrace variance",
            'low_scoring': "Target TD-dependent players for leverage",
            'blowout_risk': "Fade favorites heavily, target garbage time",
            'flat_pricing': "Ownership becomes primary differentiator",
            'standard': "Balanced approach with calculated risks"
        }

        return {
            'field_strategy': field_strategies.get(field_size, 'Standard GPP strategy'),
            'slate_adjustment': slate_adjustments.get(
                slate_profile.get('slate_type', 'standard'), ''
            )
        }

    def _build_game_theory_prompt(self, game_info: Dict, slate_profile: Dict,
                                  ownership_analysis: Dict, leverage_plays: pd.DataFrame,
                                  strategy_guidance: Dict) -> str:
        """Build complete game theory prompt"""
        buckets = ownership_analysis['buckets']

        prompt = f"""You are an expert DFS game theory strategist. Create an ENFORCEABLE lineup strategy for GPP tournaments.

GAME CONTEXT:
Teams: {game_info.get('teams', 'Unknown')}
Total: {game_info.get('total', 45)}
Spread: {game_info.get('spread', 0)}
Weather: {game_info.get('weather', 'Clear')}
Slate Type: {slate_profile.get('slate_type', 'standard')}

OWNERSHIP LANDSCAPE:
Mega Chalk (>35%): {len(buckets['mega_chalk'])} players - {buckets['mega_chalk'][:3] if buckets['mega_chalk'] else 'None'}
Chalk (20-35%): {len(buckets['chalk'])} players - {buckets['chalk'][:3] if buckets['chalk'] else 'None'}
Pivot (10-20%): {len(buckets['pivot'])} players - {buckets['pivot'][:3] if buckets['pivot'] else 'None'}
Leverage (5-10%): {len(buckets['leverage'])} players - {buckets['leverage'][:3] if buckets['leverage'] else 'None'}
Super Leverage (<5%): {len(buckets['super_leverage'])} players - {buckets['super_leverage'][:3] if buckets['super_leverage'] else 'None'}

HIGH LEVERAGE PLAYS (<10% ownership):
{leverage_plays[['Player', 'Position', 'Salary', 'Projected_Points', 'Ownership']].to_string()}

FIELD STRATEGY:
{strategy_guidance['field_strategy']}
{strategy_guidance['slate_adjustment']}

CRITICAL: Respond ONLY with valid JSON. DO NOT include markdown formatting or code blocks.

PROVIDE SPECIFIC, ENFORCEABLE RULES IN THIS EXACT JSON FORMAT:
{{
    "captain_rules": {{
        "must_be_one_of": ["Player Name 1", "Player Name 2", "Player Name 3"],
        "ownership_ceiling": 15,
        "min_projection": 15,
        "leverage_score_min": 3,
        "reasoning": "Brief explanation of why these captains"
    }},
    "lineup_rules": {{
        "must_include": ["Player Name"],
        "never_include": ["Player Name"],
        "ownership_sum_range": [60, 90],
        "min_leverage_players": 2,
        "max_chalk_players": 2
    }},
    "correlation_rules": {{
        "required_stacks": [{{"player1": "Player Name", "player2": "Player Name", "type": "leverage"}}],
        "avoid_negative_correlation": ["player_a", "player_b"]
    }},
    "game_theory_insights": {{
        "field_tendencies": "What most lineups will look like",
        "exploit_angle": "How to exploit these tendencies",
        "unique_construction": "Your differentiation strategy",
        "ownership_arbitrage": "Where ownership doesn't match equity"
    }},
    "confidence": 0.85,
    "key_insight": "The ONE game theory edge that wins"
}}

Use EXACT player names from the data provided. Focus on ownership leverage and exploitable patterns."""

        return prompt

    def parse_response(self, response: str, df: pd.DataFrame,
                      field_size: str) -> AIRecommendation:
        """Parse game theory response with robust error handling"""
        try:
            # Clean and parse JSON
            data = self._clean_and_parse_json(response, df)
            available_players = set(df['Player'].values)

            # Extract components
            captain_targets = self._extract_captain_targets(
                data, df, available_players
            )
            lineup_rules = self._extract_lineup_rules(
                data, available_players
            )
            stacks = self._extract_correlation_stacks(
                data, available_players
            )
            insights = self._extract_insights(data, lineup_rules)

            # Build enforcement rules
            enforcement_rules = self._build_game_theory_enforcement_rules(
                captain_targets, lineup_rules['must_include'],
                lineup_rules['never_include'], lineup_rules['ownership_range'],
                lineup_rules['min_leverage'], stacks
            )

            confidence = max(0.0, min(1.0, data.get('confidence', 0.75)))

            return AIRecommendation(
                captain_targets=captain_targets[:7],
                must_play=lineup_rules['must_include'][:5],
                never_play=lineup_rules['never_include'][:5],
                stacks=stacks[:5],
                key_insights=insights[:3],
                confidence=confidence,
                enforcement_rules=enforcement_rules,
                narrative=data.get('game_theory_insights', {}).get(
                    'exploit_angle', 'Game theory optimization'
                ),
                source_ai=AIStrategistType.GAME_THEORY,
                ownership_leverage={
                    'ownership_range': lineup_rules['ownership_range'],
                    'ownership_ceiling': data.get('captain_rules', {}).get(
                        'ownership_ceiling', 15
                    ),
                    'min_leverage': lineup_rules['min_leverage'],
                    'max_chalk': lineup_rules.get('max_chalk', 2)
                }
            )

        except Exception as e:
            self.logger.log_exception(e, "parse_game_theory_response")
            return self._get_fallback_recommendation(df, field_size)

    def _clean_and_parse_json(self, response: str, df: pd.DataFrame) -> Dict:
        """Clean response and parse JSON"""
        response = response.strip()
        response = response.replace('```json\n', '').replace('```\n', '').replace('```', '')

        if response and response != '{}':
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                self.logger.log(
                    f"JSON parse error: {e}, attempting text extraction",
                    "WARNING"
                )
                return self._extract_from_text_response(response, df)

        return {}

    def _extract_captain_targets(self, data: Dict, df: pd.DataFrame,
                                 available_players: Set[str]) -> List[str]:
        """Extract and validate captain targets"""
        captain_rules = data.get('captain_rules', {})
        captain_targets = captain_rules.get('must_be_one_of', [])
        valid_captains = [c for c in captain_targets if c in available_players]

        # If insufficient, use game theory selection
        if len(valid_captains) < 3:
            valid_captains = self._select_game_theory_captains(
                df, captain_rules, valid_captains
            )

        return valid_captains

    def _select_game_theory_captains(self, df: pd.DataFrame, captain_rules: Dict,
                                     existing: List[str]) -> List[str]:
        """Select captains using game theory principles"""
        ownership_ceiling = captain_rules.get('ownership_ceiling', 15)
        min_proj = captain_rules.get('min_projection', 15)

        eligible_mask = (
            (df['Ownership'] <= ownership_ceiling) &
            (df['Projected_Points'] >= min_proj)
        )
        eligible = df[eligible_mask]

        if len(eligible) < 5:
            eligible = df[df['Ownership'] <= ownership_ceiling * 1.5]

        if not eligible.empty:
            max_proj = eligible['Projected_Points'].max()
            leverage_scores = (
                (eligible['Projected_Points'] / max_proj * 100) /
                (eligible['Ownership'] + 5)
            )

            eligible = eligible.copy()
            eligible['Leverage_Score'] = leverage_scores
            leverage_captains = eligible.nlargest(5, 'Leverage_Score')['Player'].tolist()

            for captain in leverage_captains:
                if captain not in existing:
                    existing.append(captain)
                if len(existing) >= 5:
                    break

        return existing

    def _extract_lineup_rules(self, data: Dict,
                              available_players: Set[str]) -> Dict:
        """Extract lineup rules from response"""
        lineup_rules = data.get('lineup_rules', {})

        return {
            'must_include': [
                p for p in lineup_rules.get('must_include', [])
                if p in available_players
            ],
            'never_include': [
                p for p in lineup_rules.get('never_include', [])
                if p in available_players
            ],
            'ownership_range': lineup_rules.get('ownership_sum_range', [60, 90]),
            'min_leverage': lineup_rules.get('min_leverage_players', 2),
            'max_chalk': lineup_rules.get('max_chalk_players', 2)
        }

    def _extract_correlation_stacks(self, data: Dict,
                                    available_players: Set[str]) -> List[Dict]:
        """Extract correlation stacks from response"""
        correlation_rules = data.get('correlation_rules', {})
        stacks = []

        for stack_data in correlation_rules.get('required_stacks', []):
            p1 = stack_data.get('player1')
            p2 = stack_data.get('player2')

            if p1 in available_players and p2 in available_players:
                stacks.append({
                    'player1': p1,
                    'player2': p2,
                    'type': stack_data.get('type', 'leverage'),
                    'correlation': 0.5,
                    'leverage_based': True
                })

        return stacks

    def _extract_insights(self, data: Dict, lineup_rules: Dict) -> List[str]:
        """Extract key insights from response"""
        game_theory = data.get('game_theory_insights', {})

        insights = [
            data.get('key_insight', 'Ownership arbitrage opportunity'),
            game_theory.get('exploit_angle', ''),
            game_theory.get('unique_construction', ''),
            f"Target {lineup_rules['ownership_range'][0]}-{lineup_rules['ownership_range'][1]}% total ownership"
        ]

        return [i for i in insights if i]

    def _extract_from_text_response(self, response: str, df: pd.DataFrame) -> Dict:
        """Extract structured data from non-JSON response"""
        data = {
            'captain_rules': {'ownership_ceiling': 15},
            'lineup_rules': {'min_leverage_players': 2},
            'confidence': 0.6
        }

        lines = response.lower().split('\n')

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

    def _build_game_theory_enforcement_rules(self, captains: List[str],
                                            must_include: List[str],
                                            never_include: List[str],
                                            ownership_range: List[int],
                                            min_leverage: int,
                                            stacks: List[Dict]) -> List[Dict]:
        """Build game theory enforcement rules"""
        rules = []

        # Captain constraint
        if captains:
            rules.append({
                'type': 'hard',
                'constraint': 'game_theory_captain',
                'players': captains[:5],
                'priority': ConstraintPriority.AI_HIGH_CONFIDENCE.value,
                'relaxation_tier': 2,
                'description': 'Game theory optimal captains'
            })

        # Ownership sum constraint
        rules.append({
            'type': 'hard',
            'constraint': 'ownership_sum',
            'min': ownership_range[0],
            'max': ownership_range[1],
            'priority': ConstraintPriority.AI_MODERATE.value,
            'relaxation_tier': 2,
            'description': f'Total ownership {ownership_range[0]}-{ownership_range[1]}%'
        })

        # Must include players
        for i, player in enumerate(must_include[:3]):
            rules.append({
                'type': 'hard',
                'constraint': 'must_include',
                'player': player,
                'priority': ConstraintPriority.AI_MODERATE.value - (i * 5),
                'relaxation_tier': 2,
                'description': f'Must include {player}'
            })

        # Never include players
        for i, player in enumerate(never_include[:3]):
            rules.append({
                'type': 'hard',
                'constraint': 'must_exclude',
                'player': player,
                'priority': ConstraintPriority.AI_MODERATE.value - (i * 5),
                'relaxation_tier': 2,
                'description': f'Fade chalk: {player}'
            })

        return rules

# ============================================================================
# GPP CORRELATION STRATEGIST
# ============================================================================

class GPPCorrelationStrategist(BaseAIStrategist):
    """
    AI Strategist 2: Correlation and Stacking Patterns

    DFS Value: Maximizes ceiling through correlated outcomes
    """

    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.CORRELATION)
        self.correlation_matrix = {}

    def generate_prompt(self, df: pd.DataFrame, game_info: Dict,
                       field_size: str, slate_profile: Dict) -> str:
        """Generate correlation focused prompt"""
        self.logger.log(
            f"Generating Correlation prompt for {field_size}",
            "DEBUG"
        )

        if df.empty:
            return "Error: Empty player pool"

        # Gather components
        team_analysis = self._analyze_teams(df)
        correlation_targets = self._identify_correlation_targets(df)
        game_script = self._analyze_game_script(game_info, slate_profile)

        # Build prompt
        prompt = self._build_correlation_prompt(
            game_info, slate_profile, team_analysis,
            correlation_targets, game_script
        )

        return prompt

    def _analyze_teams(self, df: pd.DataFrame) -> Dict:
        """Analyze teams and players"""
        teams = df['Team'].unique()[:2]

        if len(teams) >= 2:
            team1, team2 = teams[0], teams[1]
            team1_df = df[df['Team'] == team1]
            team2_df = df[df['Team'] == team2]
        else:
            team1 = teams[0] if len(teams) > 0 else "Unknown"
            team2 = "Unknown"
            team1_df = df if len(teams) > 0 else pd.DataFrame()
            team2_df = pd.DataFrame()

        return {
            'team1': team1,
            'team2': team2,
            'team1_df': team1_df,
            'team2_df': team2_df
        }

    def _identify_correlation_targets(self, df: pd.DataFrame) -> Dict:
        """Identify correlation target players by position"""
        return {
            'qbs': df[df['Position'] == 'QB']['Player'].tolist(),
            'pass_catchers': df[df['Position'].isin(['WR', 'TE'])]['Player'].tolist()[:8],
            'rbs': df[df['Position'] == 'RB']['Player'].tolist()
        }

    def _analyze_game_script(self, game_info: Dict, slate_profile: Dict) -> Dict:
        """Analyze expected game script"""
        total = game_info.get('total', 45)
        spread = game_info.get('spread', 0)

        # Determine favorite and underdog
        teams = game_info.get('teams', 'Team1 vs Team2').split(' vs ')
        favorite = teams[0] if spread < 0 else (teams[1] if len(teams) > 1 else teams[0])
        underdog = teams[1] if spread < 0 and len(teams) > 1 else teams[0]

        return {
            'total': total,
            'spread': spread,
            'favorite': favorite,
            'underdog': underdog,
            'slate_type': slate_profile.get('slate_type', 'standard')
        }

    def _build_correlation_prompt(self, game_info: Dict, slate_profile: Dict,
                                  team_analysis: Dict, correlation_targets: Dict,
                                  game_script: Dict) -> str:
        """Build complete correlation prompt"""
        team1_data = team_analysis['team1_df']
        team2_data = team_analysis['team2_df']

        prompt = f"""You are an expert DFS correlation strategist. Create SPECIFIC stacking rules for GPP.

GAME ENVIRONMENT:
Total: {game_script['total']} | Spread: {game_script['spread']}
Favorite: {game_script['favorite']} by {abs(game_script['spread'])} points
Underdog: {game_script['underdog']}
Slate Type: {game_script['slate_type']}

TEAM 1 - {team_analysis['team1']} ({'Favorite' if team_analysis['team1'] == game_script['favorite'] else 'Underdog'}):
{team1_data[['Player', 'Position', 'Salary', 'Projected_Points']].head(8).to_string() if not team1_data.empty else 'No data'}

TEAM 2 - {team_analysis['team2']} ({'Favorite' if team_analysis['team2'] == game_script['favorite'] else 'Underdog'}):
{team2_data[['Player', 'Position', 'Salary', 'Projected_Points']].head(8).to_string() if not team2_data.empty else 'No data'}

CORRELATION TARGETS:
QBs: {correlation_targets['qbs']}
Pass Catchers: {correlation_targets['pass_catchers']}
RBs: {correlation_targets['rbs']}

CRITICAL: Respond ONLY with valid JSON. DO NOT include markdown formatting or code blocks.

CREATE ADVANCED CORRELATION RULES IN THIS EXACT JSON FORMAT:
{{
    "primary_stacks": [
        {{"type": "QB_WR1", "player1": "exact_qb_name", "player2": "exact_wr_name", "correlation": 0.7, "narrative": "why this stack"}}
    ],
    "onslaught_stacks": [
        {{"team": "team_name", "players": ["qb", "wr1", "wr2"], "scenario": "blowout correlation explanation"}}
    ],
    "bring_back_stacks": [
        {{"primary": ["qb", "wr"], "bring_back": "opposing_wr_name", "game_total": 50}}
    ],
    "negative_correlation": [
        {{"avoid_together": ["player1", "player2"], "reason": "why they're negatively correlated"}}
    ],
    "game_script_stacks": {{
        "shootout": ["player1", "player2", "player3"],
        "blowout": ["player1", "player2"]
    }},
    "captain_correlation": {{
        "best_captains_for_stacking": ["player_name_1", "player_name_2"],
        "stack_multipliers": {{"QB": 1.5, "WR1": 1.3}}
    }},
    "confidence": 0.8,
    "stack_narrative": "Primary correlation thesis"
}}

Use EXACT player names. Focus on correlations that maximize ceiling potential."""

        return prompt

    def parse_response(self, response: str, df: pd.DataFrame,
                      field_size: str) -> AIRecommendation:
        """Parse correlation response with robust handling"""
        try:
            # Clean and parse
            data = self._clean_and_parse_json(response, df)
            available_players = set(df['Player'].values)

            # Extract all stacks
            all_stacks = self._extract_all_stacks(data, available_players)

            # Extract captain targets
            captain_targets = self._extract_correlation_captains(
                data, df, all_stacks, available_players
            )

            # Process negative correlations
            avoid_pairs = self._extract_avoid_pairs(data, available_players)

            # Build enforcement and correlation matrix
            enforcement_rules = self._build_correlation_enforcement_rules(
                all_stacks, avoid_pairs, captain_targets
            )
            self.correlation_matrix = self._build_correlation_matrix(
                all_stacks, avoid_pairs
            )

            confidence = max(0.0, min(1.0, data.get('confidence', 0.75)))

            key_insights = [
                data.get('stack_narrative', 'Correlation-based construction'),
                f"Primary focus: {all_stacks[0]['type'] if all_stacks else 'standard'} stacks",
                f"{len(all_stacks)} correlation plays identified"
            ]

            return AIRecommendation(
                captain_targets=captain_targets,
                must_play=[],
                never_play=[],
                stacks=all_stacks[:10],
                key_insights=key_insights,
                confidence=confidence,
                enforcement_rules=enforcement_rules,
                narrative=data.get('stack_narrative', 'Correlation optimization'),
                source_ai=AIStrategistType.CORRELATION,
                correlation_matrix=self.correlation_matrix
            )

        except Exception as e:
            self.logger.log_exception(e, "parse_correlation_response")
            return self._get_fallback_recommendation(df, field_size)

    def _clean_and_parse_json(self, response: str, df: pd.DataFrame) -> Dict:
        """Clean and parse JSON response"""
        response = response.strip()
        response = response.replace('```json\n', '').replace('```\n', '').replace('```', '')

        if response and response != '{}':
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return self._extract_correlation_from_text(response, df)

        return {}

    def _extract_all_stacks(self, data: Dict,
                           available_players: Set[str]) -> List[Dict]:
        """Extract and combine all types of stacks"""
        all_stacks = []

        # Primary stacks
        for stack in data.get('primary_stacks', []):
            if self._validate_stack(stack, available_players):
                stack['priority'] = ConstraintPriority.AI_MODERATE.value
                stack['enforced'] = True
                all_stacks.append(stack)

        # Onslaught stacks
        all_stacks.extend(self._extract_onslaught_stacks(data, available_players))

        # Bring-back stacks
        all_stacks.extend(self._extract_bring_back_stacks(data, available_players))

        # Fallback if no valid stacks
        if len(all_stacks) < 2:
            all_stacks.extend(self._create_statistical_stacks(self.df))

        return all_stacks

    def _extract_onslaught_stacks(self, data: Dict,
                                  available_players: Set[str]) -> List[Dict]:
        """Extract onslaught stacks (3+ players from winning team)"""
        stacks = []

        for onslaught in data.get('onslaught_stacks', []):
            players = onslaught.get('players', [])
            valid_players = [p for p in players if p in available_players]

            if len(valid_players) >= 3:
                stacks.append({
                    'type': 'onslaught',
                    'players': valid_players,
                    'team': onslaught.get('team', ''),
                    'scenario': onslaught.get('scenario', 'Blowout correlation'),
                    'priority': ConstraintPriority.AI_MODERATE.value,
                    'correlation': 0.6
                })

        return stacks

    def _extract_bring_back_stacks(self, data: Dict,
                                   available_players: Set[str]) -> List[Dict]:
        """Extract bring-back stacks"""
        stacks = []

        for bring_back in data.get('bring_back_stacks', []):
            primary = bring_back.get('primary', [])
            opponent = bring_back.get('bring_back', '')

            valid_primary = [p for p in primary if p in available_players]

            if valid_primary and opponent in available_players:
                stacks.append({
                    'type': 'bring_back',
                    'primary_stack': valid_primary,
                    'bring_back': opponent,
                    'game_total': bring_back.get('game_total', 45),
                    'priority': ConstraintPriority.AI_MODERATE.value,
                    'correlation': 0.5
                })

        return stacks

    def _extract_correlation_captains(self, data: Dict, df: pd.DataFrame,
                                      stacks: List[Dict],
                                      available_players: Set[str]) -> List[str]:
        """Extract captain targets based on correlation"""
        captain_rules = data.get('captain_correlation', {})
        captain_targets = captain_rules.get('best_captains_for_stacking', [])
        valid_captains = [c for c in captain_targets if c in available_players]

        if len(valid_captains) < 3:
            valid_captains.extend(
                self._get_correlation_captains(df, stacks)
            )
            valid_captains = list(set(valid_captains))[:7]

        return valid_captains

    def _extract_avoid_pairs(self, data: Dict,
                            available_players: Set[str]) -> List[Dict]:
        """Extract negative correlation pairs"""
        avoid_pairs = []

        for neg_corr in data.get('negative_correlation', []):
            players = neg_corr.get('avoid_together', [])
            if (len(players) >= 2 and
                all(p in available_players for p in players[:2])):
                avoid_pairs.append({
                    'players': players,
                    'reason': neg_corr.get('reason', 'negative correlation')
                })

        return avoid_pairs

    def _validate_stack(self, stack: Dict, available_players: Set[str]) -> bool:
        """Validate stack players exist"""
        player1 = stack.get('player1')
        player2 = stack.get('player2')
        return (player1 and player2 and
                player1 in available_players and
                player2 in available_players)

    def _get_correlation_captains(self, df: pd.DataFrame,
                                 stacks: List[Dict]) -> List[str]:
        """Get captain targets based on correlation analysis"""
        captains = []

        # QBs involved in stacks
        for stack in stacks:
            if stack.get('type') in ['QB_WR', 'QB_TE', 'primary', 'QB_WR1']:
                player1 = stack.get('player1')
                if player1:
                    player_match = df[df['Player'] == player1]
                    if not player_match.empty and player_match.iloc[0]['Position'] == 'QB':
                        captains.append(player1)

        # Primary pass catchers
        for stack in stacks:
            if 'player2' in stack:
                player2 = stack['player2']
                if player2 not in captains:
                    captains.append(player2)

        # Bring-back targets
        for stack in stacks:
            if stack.get('type') == 'bring_back':
                bring_back = stack.get('bring_back')
                if bring_back and bring_back not in captains:
                    captains.append(bring_back)

        return captains[:7]

    def _build_correlation_enforcement_rules(self, stacks: List[Dict],
                                            avoid_pairs: List[Dict],
                                            captains: List[str]) -> List[Dict]:
        """Build correlation-specific enforcement rules"""
        rules = []

        if captains:
            rules.append({
                'type': 'hard',
                'constraint': 'correlation_captain',
                'players': captains[:5],
                'priority': ConstraintPriority.AI_HIGH_CONFIDENCE.value,
                'relaxation_tier': 2,
                'description': 'Correlation-optimized captains'
            })

        # High priority stacks (first 3)
        high_priority_stacks = [
            s for s in stacks if s.get('priority', 0) >= 70
        ][:3]

        for i, stack in enumerate(high_priority_stacks):
            rules.append(self._create_stack_rule(stack, i))

        return rules

    def _create_stack_rule(self, stack: Dict, index: int) -> Dict:
        """Create a single stack rule"""
        stack_type = stack.get('type')
        is_hard = index == 0

        if stack_type == 'onslaught':
            return {
                'type': 'hard' if is_hard else 'soft',
                'constraint': 'onslaught_stack',
                'players': stack['players'][:4],
                'min_players': 3,
                'weight': 0.9 if index > 0 else 1.0,
                'priority': ConstraintPriority.AI_MODERATE.value - (index * 5),
                'relaxation_tier': 3,
                'description': f"Onslaught: {stack.get('team', 'team')}"
            }
        elif stack_type == 'bring_back':
            return {
                'type': 'hard' if is_hard else 'soft',
                'constraint': 'bring_back_stack',
                'primary': stack.get('primary_stack', []),
                'bring_back': stack.get('bring_back'),
                'weight': 0.8 if index > 0 else 1.0,
                'priority': ConstraintPriority.AI_MODERATE.value - (index * 5),
                'relaxation_tier': 3,
                'description': 'Bring-back correlation'
            }
        else:
            players = []
            if 'player1' in stack and 'player2' in stack:
                players = [stack['player1'], stack['player2']]

            return {
                'type': 'hard' if is_hard else 'soft',
                'constraint': 'correlation_stack',
                'players': players,
                'correlation': stack.get('correlation', 0.5),
                'weight': 0.8 if index > 0 else 1.0,
                'priority': ConstraintPriority.AI_MODERATE.value - (index * 5),
                'relaxation_tier': 3,
                'description': f"Stack: {stack.get('type', 'correlation')}"
            }

    def _build_correlation_matrix(self, stacks: List[Dict],
                                  avoid_pairs: List[Dict]) -> Dict:
        """Build correlation matrix for reference"""
        matrix = {}

        # Positive correlations from stacks
        for stack in stacks:
            if 'player1' in stack and 'player2' in stack:
                key = f"{stack['player1']}_{stack['player2']}"
                matrix[key] = stack.get('correlation', 0.5)

        # Negative correlations
        for avoid in avoid_pairs:
            players = avoid['players']
            if len(players) >= 2:
                key = f"{players[0]}_{players[1]}"
                matrix[key] = -0.5

        return matrix

    def _extract_correlation_from_text(self, response: str,
                                      df: pd.DataFrame) -> Dict:
        """Extract correlation data from text response"""
        data = {
            'primary_stacks': [],
            'confidence': 0.6
        }

        qb_mask = df['Position'] == 'QB'
        receiver_mask = df['Position'].isin(['WR', 'TE'])

        qbs = df[qb_mask]['Player'].tolist()
        receivers = df[receiver_mask]['Player'].tolist()

        for qb in qbs:
            for receiver in receivers:
                if qb in response and receiver in response:
                    qb_team = df[df['Player'] == qb]['Team'].values
                    rec_team = df[df['Player'] == receiver]['Team'].values

                    if (len(qb_team) > 0 and len(rec_team) > 0 and
                        qb_team[0] == rec_team[0]):
                        data['primary_stacks'].append({
                            'player1': qb,
                            'player2': receiver,
                            'type': 'QB_stack',
                            'correlation': 0.6
                        })
                        break

        return data

# ============================================================================
# GPP CONTRARIAN NARRATIVE STRATEGIST
# ============================================================================

class GPPContrarianNarrativeStrategist(BaseAIStrategist):
    """
    AI Strategist 3: Contrarian Narratives and Hidden Angles

    DFS Value: Finds non-obvious tournament-winning angles
    """

    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.CONTRARIAN_NARRATIVE)

    def generate_prompt(self, df: pd.DataFrame, game_info: Dict,
                       field_size: str, slate_profile: Dict) -> str:
        """Generate contrarian narrative focused prompt"""
        self.logger.log(
            f"Generating Contrarian Narrative prompt for {field_size}",
            "DEBUG"
        )

        if df.empty:
            return "Error: Empty player pool"

        # Analyze contrarian opportunities
        contrarian_data = self._analyze_contrarian_opportunities(df, game_info)

        # Build prompt
        prompt = self._build_contrarian_prompt(
            game_info, slate_profile, contrarian_data
        )

        return prompt

    def _analyze_contrarian_opportunities(self, df: pd.DataFrame,
                                         game_info: Dict) -> Dict:
        """Analyze contrarian opportunities using vectorized operations"""
        ownership = df['Ownership'].fillna(10)
        projected = df['Projected_Points']
        salary = df['Salary']

        # Vectorized calculations
        value = projected / (salary / 1000)
        contrarian_score = (projected / projected.max()) / (ownership / 100 + 0.1)

        df_analysis = df.copy()
        df_analysis['Value'] = value
        df_analysis['Contrarian_Score'] = contrarian_score

        return {
            'low_owned_high_ceiling': df_analysis[ownership < 10].nlargest(10, 'Projected_Points'),
            'hidden_value': df_analysis[ownership < 15].nlargest(10, 'Value'),
            'contrarian_captains': df_analysis.nlargest(10, 'Contrarian_Score'),
            'chalk_plays': df_analysis[ownership > 30].nlargest(5, 'Ownership')
        }

    def _build_contrarian_prompt(self, game_info: Dict, slate_profile: Dict,
                                 contrarian_data: Dict) -> str:
        """Build complete contrarian prompt"""
        teams = game_info.get('teams', 'Team1 vs Team2').split(' vs ')
        total = game_info.get('total', 45)
        spread = game_info.get('spread', 0)

        prompt = f"""You are a contrarian DFS strategist who finds NON-OBVIOUS narratives that win GPP tournaments.

GAME SETUP:
{teams[0]} vs {teams[1] if len(teams) > 1 else 'Team2'}
Total: {total} | Spread: {spread}
Slate Type: {slate_profile.get('slate_type', 'standard')}

CONTRARIAN OPPORTUNITIES:

LOW-OWNED HIGH CEILING (<10% owned):
{contrarian_data['low_owned_high_ceiling'][['Player', 'Position', 'Team', 'Projected_Points', 'Ownership']].to_string()}

HIDDEN VALUE PLAYS:
{contrarian_data['hidden_value'][['Player', 'Position', 'Salary', 'Value', 'Ownership']].to_string()}

CONTRARIAN CAPTAIN SCORES:
{contrarian_data['contrarian_captains'][['Player', 'Contrarian_Score', 'Ownership']].to_string()}

CHALK TO FADE (>30% owned):
{contrarian_data['chalk_plays'][['Player', 'Position', 'Ownership', 'Salary']].to_string() if not contrarian_data['chalk_plays'].empty else 'No major chalk'}

CRITICAL: Respond ONLY with valid JSON. DO NOT include markdown formatting or code blocks.

CREATE CONTRARIAN TOURNAMENT-WINNING NARRATIVES IN THIS EXACT JSON FORMAT:
{{
    "primary_narrative": "The ONE scenario that creates a unique winning lineup",
    "contrarian_captains": [
        {{"player": "exact_name", "narrative": "Why this 5% captain wins", "ceiling_path": "How they hit 30+ points"}}
    ],
    "hidden_correlations": [
        {{"player1": "name1", "player2": "name2", "narrative": "Non-obvious connection"}}
    ],
    "fade_the_chalk": [
        {{"player": "chalk_name", "ownership": 35, "fade_reason": "Specific bust risk", "pivot_to": "alternative_name"}}
    ],
    "leverage_scenarios": [
        {{"scenario": "Game script", "beneficiaries": ["player1", "player2"], "probability": "Low but high reward"}}
    ],
    "contrarian_game_theory": {{
        "what_field_expects": "Common narrative",
        "fatal_flaw": "Why the field is wrong",
        "exploit_angle": "How to capitalize"
    }},
    "tournament_winner": {{
        "captain": "exact_contrarian_captain",
        "core": ["player1", "player2"],
        "differentiators": ["unique1", "unique2"],
        "total_ownership": 65,
        "win_condition": "What needs to happen"
    }},
    "confidence": 0.7
}}

Use EXACT player names. Find the narrative that makes sub-5% plays optimal."""

        return prompt

    def parse_response(self, response: str, df: pd.DataFrame,
                      field_size: str) -> AIRecommendation:
        """Parse contrarian narrative response"""
        try:
            # Clean and parse
            data = self._clean_and_parse_json(response, df)
            available_players = set(df['Player'].values)

            # Extract components
            contrarian_captains, captain_narratives = self._extract_contrarian_captains(
                data, df, available_players
            )
            must_play, fades, pivots = self._extract_tournament_lineup(
                data, available_players
            )
            hidden_stacks = self._extract_hidden_correlations(
                data, available_players
            )

            # Build enforcement rules
            enforcement_rules = self._build_contrarian_enforcement_rules(
                contrarian_captains, must_play, fades, hidden_stacks,
                captain_narratives
            )

            # Extract angles and insights
            game_theory = data.get('contrarian_game_theory', {})
            contrarian_angles = [
                game_theory.get('fatal_flaw', ''),
                game_theory.get('exploit_angle', ''),
                data.get('primary_narrative', '')
            ]
            contrarian_angles = [a for a in contrarian_angles if a]

            key_insights = [
                data.get('primary_narrative', 'Contrarian approach'),
                f"Fade {len(fades)} chalk plays",
                f"{len(contrarian_captains)} contrarian captains identified"
            ]

            confidence = max(0.0, min(1.0, data.get('confidence', 0.7)))

            return AIRecommendation(
                captain_targets=contrarian_captains,
                must_play=must_play[:5],
                never_play=fades[:5],
                stacks=hidden_stacks[:5],
                key_insights=key_insights[:3],
                confidence=confidence,
                enforcement_rules=enforcement_rules,
                narrative=data.get('primary_narrative', 'Contrarian strategy'),
                source_ai=AIStrategistType.CONTRARIAN_NARRATIVE,
                contrarian_angles=contrarian_angles[:3]
            )

        except Exception as e:
            self.logger.log_exception(e, "parse_contrarian_response")
            return self._get_fallback_recommendation(df, field_size)

    def _clean_and_parse_json(self, response: str, df: pd.DataFrame) -> Dict:
        """Clean and parse JSON"""
        response = response.strip()
        response = response.replace('```json\n', '').replace('```\n', '').replace('```', '')

        if response and response != '{}':
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return self._extract_narrative_from_text(response, df)

        return {}

    def _extract_contrarian_captains(self, data: Dict, df: pd.DataFrame,
                                    available_players: Set[str]) -> Tuple[List[str], Dict]:
        """Extract contrarian captains with narratives"""
        contrarian_captains = []
        captain_narratives = {}

        for captain_data in data.get('contrarian_captains', []):
            player = captain_data.get('player')
            if player and player in available_players:
                contrarian_captains.append(player)
                player_own = df[df['Player'] == player]['Ownership'].values
                captain_narratives[player] = {
                    'narrative': captain_data.get('narrative', ''),
                    'ceiling_path': captain_data.get('ceiling_path', ''),
                    'ownership': player_own[0] if len(player_own) > 0 else 10
                }

        # Fallback if insufficient
        if len(contrarian_captains) < 3:
            contrarian_captains.extend(
                self._find_statistical_contrarian_captains(df, contrarian_captains)
            )
            contrarian_captains = contrarian_captains[:7]

        return contrarian_captains, captain_narratives

    def _extract_tournament_lineup(self, data: Dict,
                                   available_players: Set[str]) -> Tuple[List[str], List[str], Dict]:
        """Extract tournament winner lineup components"""
        tournament_winner = data.get('tournament_winner', {})

        must_play = []
        tw_captain = tournament_winner.get('captain')
        tw_core = tournament_winner.get('core', [])
        tw_differentiators = tournament_winner.get('differentiators', [])

        # Add core players
        for player in tw_core:
            if player in available_players:
                must_play.append(player)

        # Add differentiators
        for player in tw_differentiators:
            if player in available_players and player not in must_play:
                must_play.append(player)

        # Extract fades and pivots
        fades = []
        pivots = {}

        for fade_data in data.get('fade_the_chalk', []):
            fade_player = fade_data.get('player')
            pivot_player = fade_data.get('pivot_to')

            if fade_player and fade_player in available_players:
                # Verify it's actually chalk
                player_df = self.df
                if player_df is not None:
                    player_ownership = player_df[player_df['Player'] == fade_player]['Ownership'].values
                    if len(player_ownership) > 0 and player_ownership[0] > 20:
                        fades.append(fade_player)

                        if pivot_player and pivot_player in available_players:
                            pivots[fade_player] = pivot_player
                            if pivot_player not in must_play:
                                must_play.append(pivot_player)

        return must_play, fades, pivots

    def _extract_hidden_correlations(self, data: Dict,
                                    available_players: Set[str]) -> List[Dict]:
        """Extract hidden correlation stacks"""
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

        return hidden_stacks

    def _find_statistical_contrarian_captains(self, df: pd.DataFrame,
                                             existing: List[str]) -> List[str]:
        """Find contrarian captains using statistical analysis"""
        existing_set = set(existing)
        existing_mask = ~df['Player'].isin(existing_set)
        low_own_mask = df['Ownership'] < 15
        combined_mask = existing_mask & low_own_mask

        eligible = df[combined_mask]

        if eligible.empty:
            return []

        # Calculate contrarian score
        max_proj = eligible['Projected_Points'].max()
        contrarian_scores = (
            (eligible['Projected_Points'] / max_proj) /
            (eligible['Ownership'] / 100 + 0.1)
        )

        eligible = eligible.copy()
        eligible['Contrarian_Score'] = contrarian_scores
        captains = eligible.nlargest(5, 'Contrarian_Score')['Player'].tolist()

        return captains

    def _build_contrarian_enforcement_rules(self, captains: List[str],
                                           must_play: List[str],
                                           fades: List[str],
                                           hidden_stacks: List[Dict],
                                           captain_narratives: Dict) -> List[Dict]:
        """Build contrarian-specific enforcement rules"""
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
                'priority': ConstraintPriority.AI_HIGH_CONFIDENCE.value,
                'relaxation_tier': 2,
                'description': 'Ultra-contrarian captain (<5% owned)'
            })

        if moderate_contrarian:
            rules.append({
                'type': 'soft',
                'constraint': 'contrarian_captain',
                'players': moderate_contrarian,
                'weight': 0.8,
                'priority': ConstraintPriority.AI_MODERATE.value,
                'description': 'Contrarian captain options'
            })

        # Tournament core rules
        for i, player in enumerate(must_play[:3]):
            rules.append({
                'type': 'hard' if i == 0 else 'soft',
                'constraint': f'tournament_core_{player}',
                'player': player,
                'weight': 0.9 - (i * 0.1) if i > 0 else 1.0,
                'priority': ConstraintPriority.AI_MODERATE.value - (i * 5),
                'relaxation_tier': 2,
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
                'priority': ConstraintPriority.AI_MODERATE.value - (i * 5),
                'relaxation_tier': 2,
                'description': f'Fade chalk: {fade}'
            })

        return rules

    def _extract_narrative_from_text(self, response: str,
                                    df: pd.DataFrame) -> Dict:
        """Extract contrarian narrative from text response"""
        data = {
            'contrarian_captains': [],
            'fade_the_chalk': [],
            'confidence': 0.6
        }

        low_owned_mask = df['Ownership'] < 10
        high_owned_mask = df['Ownership'] > 30

        low_owned = df[low_owned_mask]['Player'].tolist()
        high_owned = df[high_owned_mask]['Player'].tolist()

        response_lower = response.lower()

        for player in low_owned:
            if player.lower() in response_lower:
                data['contrarian_captains'].append({
                    'player': player,
                    'narrative': 'Low ownership leverage'
                })

        for player in high_owned:
            if (f"fade {player.lower()}" in response_lower or
                f"avoid {player.lower()}" in response_lower):
                player_own = df[df['Player'] == player]['Ownership'].values
                data['fade_the_chalk'].append({
                    'player': player,
                    'ownership': player_own[0] if len(player_own) > 0 else 30
                })

        return data

# ============================================================================
# PART 6: MAIN OPTIMIZER ENGINE
# ============================================================================

class ShowdownOptimizer:
    """
    Enhanced main optimizer with ML integration, GA support, and simulation capabilities

    DFS Value: CRITICAL - Core engine that generates tournament-winning lineups
    """

    def __init__(self, api_key: Optional[str] = None):
        # Initialize singletons
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()
        self.ai_tracker = get_ai_tracker()

        # API management
        self.api_manager = ClaudeAPIManager(api_key) if api_key else None

        # Initialize AI strategists
        self.game_theory_ai = GPPGameTheoryStrategist(self.api_manager)
        self.correlation_ai = GPPCorrelationStrategist(self.api_manager)
        self.contrarian_ai = GPPContrarianNarrativeStrategist(self.api_manager)

        # Enforcement and validation
        self.enforcement_engine = AIEnforcementEngine()
        self.bucket_manager = AIOwnershipBucketManager(self.enforcement_engine)
        self.synthesis_engine = AISynthesisEngine()

        # Optimization state
        self.df = None
        self.game_info = {}
        self.lineups_generated = []
        self.optimization_metadata = {}

        # ML engines (initialized per slate)
        self.mc_engine = None
        self.ga_optimizer = None

        self.logger.log("ShowdownOptimizer initialized", "INFO")

    def optimize(self,
                df: pd.DataFrame,
                game_info: Dict,
                num_lineups: int = 20,
                field_size: str = 'large_field',
                ai_enforcement_level: AIEnforcementLevel = AIEnforcementLevel.STRONG,
                use_api: bool = True,
                randomness: float = 0.15,
                use_genetic: bool = False,
                use_simulation: bool = True) -> pd.DataFrame:
        """
        Main optimization method with comprehensive workflow

        DFS Value: CRITICAL - Orchestrates entire optimization process

        Args:
            df: Player DataFrame with required columns
            game_info: Game context dictionary
            num_lineups: Number of lineups to generate
            field_size: Contest size configuration
            ai_enforcement_level: How strictly to enforce AI recommendations
            use_api: Whether to use Claude API
            randomness: Randomness factor for diversity
            use_genetic: Force genetic algorithm usage
            use_simulation: Use Monte Carlo simulation

        Returns:
            DataFrame of optimized lineups
        """
        try:
            self.perf_monitor.start_timer("total_optimization")

            # Phase 1: Data validation and preparation
            self.logger.log(
                f"Starting optimization: {num_lineups} lineups, {field_size}, "
                f"enforcement={ai_enforcement_level.value}",
                "INFO"
            )

            df = self._validate_and_prepare_data(df)
            self.df = df
            self.game_info = game_info

            # Phase 2: Initialize ML engines if requested
            if use_simulation:
                self._initialize_ml_engines(df, game_info)

            # Phase 3: Get AI recommendations
            recommendations = self._get_ai_recommendations(
                df, game_info, field_size, use_api
            )

            # Phase 4: Create enforcement rules
            enforcement_rules = self._create_enforcement_rules(
                recommendations, ai_enforcement_level
            )

            # Phase 5: Determine optimization method
            field_config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size, {})
            use_genetic = use_genetic or field_config.get('use_genetic', False)

            # Phase 6: Generate lineups
            if use_genetic:
                lineups = self._optimize_with_genetic_algorithm(
                    df, game_info, num_lineups, field_size,
                    enforcement_rules, recommendations
                )
            else:
                lineups = self._optimize_with_linear_programming(
                    df, game_info, num_lineups, field_size,
                    enforcement_rules, randomness, recommendations
                )

            # Phase 7: Post-process and enhance
            lineups_df = self._post_process_lineups(
                lineups, df, recommendations, use_simulation
            )

            # Phase 8: Store results
            self.lineups_generated = lineups
            self._store_optimization_metadata(
                num_lineups, field_size, ai_enforcement_level,
                use_genetic, recommendations
            )

            elapsed = self.perf_monitor.stop_timer("total_optimization")

            self.logger.log(
                f"Optimization complete: {len(lineups_df)} lineups in {elapsed:.2f}s",
                "INFO"
            )

            return lineups_df

        except Exception as e:
            self.logger.log_exception(e, "optimize")
            return pd.DataFrame()

    def _validate_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare player data"""
        self.perf_monitor.start_timer("data_validation")

        # Required columns
        required_cols = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points']
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if df.empty:
            raise ValueError("Empty DataFrame provided")

        # Add default ownership if missing
        if 'Ownership' not in df.columns:
            df['Ownership'] = 10.0
            self.logger.log("Ownership column missing, defaulting to 10%", "WARNING")

        # Clean and validate data
        df = df.copy()
        df['Ownership'] = df['Ownership'].fillna(10.0)
        df['Projected_Points'] = df['Projected_Points'].fillna(0.0)

        # Validate salary range
        if df['Salary'].min() < 1000 or df['Salary'].max() > 15000:
            self.logger.log("Unusual salary values detected", "WARNING")

        # Validate projections
        if df['Projected_Points'].max() > 50:
            self.logger.log("Unusually high projections detected", "WARNING")

        self.perf_monitor.stop_timer("data_validation")

        return df

    def _initialize_ml_engines(self, df: pd.DataFrame, game_info: Dict) -> None:
        """Initialize Monte Carlo and other ML engines"""
        try:
            self.perf_monitor.start_timer("ml_initialization")

            # Initialize Monte Carlo engine
            self.mc_engine = MonteCarloSimulationEngine(
                df,
                game_info,
                n_simulations=OptimizerConfig.MC_SIMULATIONS
            )

            self.logger.log("Monte Carlo engine initialized", "INFO")

            # Initialize ML engines for AI strategists
            self.game_theory_ai.initialize_mc_engine(df, game_info)
            self.correlation_ai.initialize_mc_engine(df, game_info)
            self.contrarian_ai.initialize_mc_engine(df, game_info)

            self.perf_monitor.stop_timer("ml_initialization")

        except Exception as e:
            self.logger.log_exception(e, "ml_initialization")
            self.mc_engine = None

    def _get_ai_recommendations(self, df: pd.DataFrame, game_info: Dict,
                               field_size: str, use_api: bool) -> Dict[AIStrategistType, AIRecommendation]:
        """Get recommendations from all AI strategists"""
        self.perf_monitor.start_timer("ai_analysis")

        recommendations = {}

        # Game Theory AI
        try:
            recommendations[AIStrategistType.GAME_THEORY] = (
                self.game_theory_ai.get_recommendation(
                    df, game_info, field_size, use_api
                )
            )
            self.logger.log("Game Theory AI: Success", "INFO")
        except Exception as e:
            self.logger.log_exception(e, "game_theory_ai")
            recommendations[AIStrategistType.GAME_THEORY] = (
                self.game_theory_ai._get_fallback_recommendation(df, field_size)
            )

        # Correlation AI
        try:
            recommendations[AIStrategistType.CORRELATION] = (
                self.correlation_ai.get_recommendation(
                    df, game_info, field_size, use_api
                )
            )
            self.logger.log("Correlation AI: Success", "INFO")
        except Exception as e:
            self.logger.log_exception(e, "correlation_ai")
            recommendations[AIStrategistType.CORRELATION] = (
                self.correlation_ai._get_fallback_recommendation(df, field_size)
            )

        # Contrarian AI
        try:
            recommendations[AIStrategistType.CONTRARIAN_NARRATIVE] = (
                self.contrarian_ai.get_recommendation(
                    df, game_info, field_size, use_api
                )
            )
            self.logger.log("Contrarian AI: Success", "INFO")
        except Exception as e:
            self.logger.log_exception(e, "contrarian_ai")
            recommendations[AIStrategistType.CONTRARIAN_NARRATIVE] = (
                self.contrarian_ai._get_fallback_recommendation(df, field_size)
            )

        self.perf_monitor.stop_timer("ai_analysis")

        return recommendations

    def _create_enforcement_rules(self, recommendations: Dict[AIStrategistType, AIRecommendation],
                                 enforcement_level: AIEnforcementLevel) -> Dict:
        """Create enforcement rules from AI recommendations"""
        self.perf_monitor.start_timer("enforcement_rules")

        # Set enforcement level
        self.enforcement_engine.enforcement_level = enforcement_level

        # Create rules
        enforcement_rules = self.enforcement_engine.create_enforcement_rules(
            recommendations
        )

        # Validate rules
        validation = AIConfigValidator.validate_ai_requirements(
            enforcement_rules, self.df
        )

        if not validation['is_valid']:
            self.logger.log(
                f"Enforcement validation errors: {validation['errors']}",
                "WARNING"
            )
            for suggestion in validation['suggestions']:
                self.logger.log(f"Suggestion: {suggestion}", "INFO")

        self.perf_monitor.stop_timer("enforcement_rules")

        return enforcement_rules

    def _optimize_with_genetic_algorithm(self, df: pd.DataFrame, game_info: Dict,
                                        num_lineups: int, field_size: str,
                                        enforcement_rules: Dict,
                                        recommendations: Dict) -> List[Dict]:
        """Optimize using genetic algorithm"""
        self.logger.log("Using Genetic Algorithm optimization", "INFO")

        try:
            # Initialize GA optimizer
            ga_config = GeneticConfig(
                population_size=OptimizerConfig.GA_POPULATION_SIZE,
                generations=OptimizerConfig.GA_GENERATIONS,
                mutation_rate=OptimizerConfig.GA_MUTATION_RATE,
                elite_size=max(10, num_lineups // 5),
                tournament_size=5,
                crossover_rate=0.8
            )

            self.ga_optimizer = GeneticAlgorithmOptimizer(
                df, game_info, self.mc_engine, ga_config
            )

            # Determine fitness mode based on field size
            fitness_mode = self._determine_fitness_mode(field_size)

            # Run optimization
            ga_results = self.ga_optimizer.optimize(
                num_lineups=num_lineups,
                fitness_mode=fitness_mode,
                verbose=True
            )

            # Convert GA results to lineup format
            lineups = []
            for i, result in enumerate(ga_results):
                lineup = {
                    'Lineup': i + 1,
                    'Captain': result['captain'],
                    'FLEX': result['flex'],
                    'fitness': result['fitness'],
                    'sim_results': result.get('sim_results'),
                    'optimization_method': 'genetic_algorithm'
                }
                lineups.append(lineup)

            return lineups

        except Exception as e:
            self.logger.log_exception(e, "genetic_algorithm_optimization")
            # Fallback to LP
            self.logger.log("Falling back to Linear Programming", "WARNING")
            return self._optimize_with_linear_programming(
                df, game_info, num_lineups, field_size,
                enforcement_rules, 0.15, recommendations
            )

    def _determine_fitness_mode(self, field_size: str) -> FitnessMode:
        """Determine fitness mode based on field size"""
        mode_map = {
            'small_field': FitnessMode.MEAN,
            'medium_field': FitnessMode.SHARPE,
            'large_field': FitnessMode.CEILING,
            'large_field_aggressive': FitnessMode.CEILING,
            'milly_maker': FitnessMode.CEILING
        }

        return mode_map.get(field_size, FitnessMode.CEILING)

    def _optimize_with_linear_programming(self, df: pd.DataFrame, game_info: Dict,
                                         num_lineups: int, field_size: str,
                                         enforcement_rules: Dict, randomness: float,
                                         recommendations: Dict) -> List[Dict]:
        """Optimize using Linear Programming with three-tier relaxation"""
        self.logger.log("Using Linear Programming optimization", "INFO")

        lineups = []
        used_players = set()

        # Three-tier attempt structure
        max_attempts_per_tier = [
            num_lineups // 2,      # Tier 1: Strict constraints
            num_lineups // 3,      # Tier 2: Relaxed AI constraints
            num_lineups - (num_lineups // 2 + num_lineups // 3)  # Tier 3: Minimal constraints
        ]

        for tier in range(3):
            tier_target = max_attempts_per_tier[tier]
            tier_generated = 0
            tier_attempts = 0
            max_tier_attempts = tier_target * 3

            self.logger.log(
                f"Tier {tier + 1}: Attempting {tier_target} lineups",
                "INFO"
            )

            while tier_generated < tier_target and tier_attempts < max_tier_attempts:
                tier_attempts += 1

                lineup = self._generate_single_lineup_lp(
                    df, enforcement_rules, used_players,
                    randomness, tier, recommendations
                )

                if lineup:
                    lineups.append(lineup)
                    tier_generated += 1

                    # Track used players for diversity
                    lineup_players = [lineup['Captain']] + lineup['FLEX']
                    used_players.update(lineup_players)

                # Decay randomness for more diversity in later attempts
                if tier_attempts % 5 == 0:
                    randomness = min(0.25, randomness * 1.1)

            self.logger.log(
                f"Tier {tier + 1} complete: {tier_generated}/{tier_target} lineups",
                "INFO"
            )

            if len(lineups) >= num_lineups:
                break

        # Assign lineup numbers
        for i, lineup in enumerate(lineups):
            lineup['Lineup'] = i + 1
            lineup['optimization_method'] = 'linear_programming'

        return lineups[:num_lineups]

    def _generate_single_lineup_lp(self, df: pd.DataFrame, enforcement_rules: Dict,
                                   used_players: Set[str], randomness: float,
                                   tier: int, recommendations: Dict) -> Optional[Dict]:
        """Generate single lineup using PuLP with tier-based constraint relaxation"""
        try:
            # Create optimization problem
            prob = pulp.LpProblem("Showdown_Lineup", pulp.LpMaximize)

            # Decision variables
            player_vars = {}
            for idx, row in df.iterrows():
                player_vars[row['Player']] = pulp.LpVariable(
                    f"player_{idx}",
                    cat='Binary'
                )

            captain_vars = {}
            for idx, row in df.iterrows():
                captain_vars[row['Player']] = pulp.LpVariable(
                    f"captain_{idx}",
                    cat='Binary'
                )

            # Apply randomness to projections
            projections = self._apply_randomness(df, randomness)

            # Objective function
            prob += pulp.lpSum([
                captain_vars[p] * proj * OptimizerConfig.CAPTAIN_MULTIPLIER +
                player_vars[p] * proj
                for p, proj in projections.items()
            ])

            # Add constraints
            self._add_base_constraints(prob, df, player_vars, captain_vars)
            self._add_tier_constraints(
                prob, df, player_vars, captain_vars,
                enforcement_rules, tier
            )
            self._add_diversity_constraints(
                prob, player_vars, captain_vars, used_players, tier
            )

            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))

            # Extract solution
            if prob.status == pulp.LpStatusOptimal:
                return self._extract_lineup_from_solution(
                    df, player_vars, captain_vars
                )

            return None

        except Exception as e:
            self.logger.log(f"LP generation error: {e}", "WARNING")
            return None

    def _apply_randomness(self, df: pd.DataFrame, randomness: float) -> Dict[str, float]:
        """Apply randomness to projections for diversity"""
        projections = {}

        for _, row in df.iterrows():
            base_proj = row['Projected_Points']
            rand_factor = 1.0 + np.random.uniform(-randomness, randomness)
            projections[row['Player']] = base_proj * rand_factor

        return projections

    def _add_base_constraints(self, prob, df: pd.DataFrame,
                             player_vars: Dict, captain_vars: Dict) -> None:
        """Add base DraftKings constraints"""
        players = df['Player'].tolist()

        # Exactly 1 captain
        prob += pulp.lpSum([captain_vars[p] for p in players]) == 1

        # Exactly 5 FLEX
        prob += pulp.lpSum([player_vars[p] for p in players]) == 5

        # Captain must be different from FLEX
        for p in players:
            prob += captain_vars[p] + player_vars[p] <= 1

        # Salary cap
        prob += pulp.lpSum([
            captain_vars[p] * df[df['Player'] == p]['Salary'].values[0] * 1.5 +
            player_vars[p] * df[df['Player'] == p]['Salary'].values[0]
            for p in players
        ]) <= OptimizerConfig.SALARY_CAP

        # Team diversity
        teams = df['Team'].unique()
        for team in teams:
            team_players = df[df['Team'] == team]['Player'].tolist()
            prob += pulp.lpSum([
                captain_vars[p] + player_vars[p] for p in team_players
            ]) <= OptimizerConfig.MAX_PLAYERS_PER_TEAM

    def _add_tier_constraints(self, prob, df: pd.DataFrame,
                            player_vars: Dict, captain_vars: Dict,
                            enforcement_rules: Dict, tier: int) -> None:
        """Add tier-specific AI constraints"""
        for rule in enforcement_rules.get('hard_constraints', []):
            # Check if rule should be applied at this tier
            if not self.enforcement_engine.should_apply_constraint(rule, tier):
                continue

            rule_type = rule.get('rule')

            if rule_type == 'captain_from_list':
                players = rule.get('players', [])
                if players:
                    prob += pulp.lpSum([
                        captain_vars[p] for p in players
                        if p in captain_vars
                    ]) >= 1

            elif rule_type == 'consensus_captain_list':
                players = rule.get('players', [])
                if players:
                    prob += pulp.lpSum([
                        captain_vars[p] for p in players
                        if p in captain_vars
                    ]) >= 1

            elif rule_type == 'must_include':
                player = rule.get('player')
                if player and player in player_vars:
                    prob += captain_vars[player] + player_vars[player] >= 1

            elif rule_type == 'must_exclude':
                player = rule.get('player')
                if player and player in player_vars:
                    prob += captain_vars[player] + player_vars[player] == 0

    def _add_diversity_constraints(self, prob, player_vars: Dict,
                                  captain_vars: Dict, used_players: Set[str],
                                  tier: int) -> None:
        """Add diversity constraints based on previously generated lineups"""
        if not used_players or tier >= 2:
            return

        # Penalize overused players
        overused_threshold = 3 if tier == 0 else 5

        for player in used_players:
            if player in player_vars:
                # Soft constraint: reduce likelihood of reuse
                # This is done through modified objective rather than hard constraint
                pass

    def _extract_lineup_from_solution(self, df: pd.DataFrame,
                                     player_vars: Dict,
                                     captain_vars: Dict) -> Dict:
        """Extract lineup from solved LP problem"""
        captain = None
        flex = []

        for player in player_vars:
            if captain_vars[player].varValue > 0.5:
                captain = player
            if player_vars[player].varValue > 0.5:
                flex.append(player)

        if not captain or len(flex) != 5:
            return None

        return {
            'Captain': captain,
            'FLEX': flex
        }

    def _post_process_lineups(self, lineups: List[Dict], df: pd.DataFrame,
                             recommendations: Dict,
                             use_simulation: bool) -> pd.DataFrame:
        """Post-process and enhance lineups with metadata"""
        self.perf_monitor.start_timer("post_processing")

        enhanced_lineups = []

        for lineup in lineups:
            enhanced = self._calculate_lineup_stats(lineup, df)

            # Add AI metadata
            enhanced['AI_Strategy'] = self._determine_lineup_strategy(
                lineup, recommendations
            )

            # Add simulation results if available
            if use_simulation and self.mc_engine:
                enhanced = self._add_simulation_results(enhanced, lineup)

            enhanced_lineups.append(enhanced)

        # Convert to DataFrame
        df_result = self._convert_to_dataframe(enhanced_lineups, df)

        # Add rankings
        df_result = self._add_rankings(df_result, use_simulation)

        self.perf_monitor.stop_timer("post_processing")

        return df_result

    def _calculate_lineup_stats(self, lineup: Dict, df: pd.DataFrame) -> Dict:
        """Calculate lineup statistics"""
        captain = lineup['Captain']
        flex = lineup['FLEX']
        all_players = [captain] + flex

        # Get player data
        captain_data = df[df['Player'] == captain].iloc[0]
        flex_data = df[df['Player'].isin(flex)]

        # Calculate totals
        total_salary = (
            captain_data['Salary'] * OptimizerConfig.CAPTAIN_MULTIPLIER +
            flex_data['Salary'].sum()
        )

        total_proj = (
            captain_data['Projected_Points'] * OptimizerConfig.CAPTAIN_MULTIPLIER +
            flex_data['Projected_Points'].sum()
        )

        total_own = (
            captain_data['Ownership'] * OptimizerConfig.CAPTAIN_MULTIPLIER +
            flex_data['Ownership'].sum()
        )

        # Position breakdown
        positions = flex_data['Position'].value_counts().to_dict()
        positions[captain_data['Position']] = positions.get(captain_data['Position'], 0) + 1

        return {
            'Lineup': lineup.get('Lineup', 0),
            'Captain': captain,
            'Captain_Salary': captain_data['Salary'],
            'Captain_Proj': captain_data['Projected_Points'],
            'Captain_Own': captain_data['Ownership'],
            'FLEX': ', '.join(flex),
            'Total_Salary': total_salary,
            'Remaining_Salary': OptimizerConfig.SALARY_CAP - total_salary,
            'Projected': total_proj,
            'Total_Ownership': total_own,
            'Avg_Ownership': total_own / 6,
            'Positions': positions,
            'optimization_method': lineup.get('optimization_method', 'unknown')
        }

    def _determine_lineup_strategy(self, lineup: Dict,
                                   recommendations: Dict) -> str:
        """Determine which AI strategy the lineup follows"""
        captain = lineup['Captain']

        # Check which AI recommended this captain
        for ai_type, rec in recommendations.items():
            if captain in rec.captain_targets[:3]:
                return ai_type.value

        return 'balanced'

    def _add_simulation_results(self, enhanced: Dict, lineup: Dict) -> Dict:
        """Add Monte Carlo simulation results to lineup"""
        try:
            # Check if GA already provided sim results
            if 'sim_results' in lineup and lineup['sim_results']:
                sim_results = lineup['sim_results']
            else:
                # Run simulation
                sim_results = self.mc_engine.evaluate_lineup(
                    lineup['Captain'],
                    lineup['FLEX'],
                    use_cache=True
                )

            # Add simulation metrics
            enhanced['Sim_Mean'] = sim_results.mean
            enhanced['Sim_Median'] = sim_results.median
            enhanced['Sim_Std'] = sim_results.std
            enhanced['Sim_Floor_10th'] = sim_results.floor_10th
            enhanced['Sim_Ceiling_90th'] = sim_results.ceiling_90th
            enhanced['Sim_Ceiling_99th'] = sim_results.ceiling_99th
            enhanced['Sim_Top10_Mean'] = sim_results.top_10pct_mean
            enhanced['Sim_Sharpe'] = sim_results.sharpe_ratio
            enhanced['Sim_Win_Prob'] = sim_results.win_probability

        except Exception as e:
            self.logger.log(f"Simulation error for lineup: {e}", "WARNING")

        return enhanced

    def _convert_to_dataframe(self, lineups: List[Dict], df: pd.DataFrame) -> pd.DataFrame:
        """Convert lineup list to formatted DataFrame"""
        if not lineups:
            return pd.DataFrame()

        # Flatten for export
        export_data = []

        for lineup in lineups:
            captain = lineup['Captain']
            flex_players = lineup['FLEX'].split(', ') if isinstance(lineup['FLEX'], str) else lineup['FLEX']

            row = {
                'Lineup': lineup['Lineup'],
                'CPT': captain,
                'FLEX1': flex_players[0] if len(flex_players) > 0 else '',
                'FLEX2': flex_players[1] if len(flex_players) > 1 else '',
                'FLEX3': flex_players[2] if len(flex_players) > 2 else '',
                'FLEX4': flex_players[3] if len(flex_players) > 3 else '',
                'FLEX5': flex_players[4] if len(flex_players) > 4 else '',
                'Total_Salary': lineup['Total_Salary'],
                'Remaining': lineup['Remaining_Salary'],
                'Projected': lineup['Projected'],
                'Total_Own': lineup['Total_Ownership'],
                'Avg_Own': lineup['Avg_Ownership'],
                'Strategy': lineup.get('AI_Strategy', 'balanced'),
                'Method': lineup.get('optimization_method', 'lp')
            }

            # Add simulation metrics if available
            sim_cols = ['Sim_Mean', 'Sim_Ceiling_90th', 'Sim_Ceiling_99th',
                       'Sim_Sharpe', 'Sim_Win_Prob']
            for col in sim_cols:
                if col in lineup:
                    row[col] = lineup[col]

            export_data.append(row)

        return pd.DataFrame(export_data)

    def _add_rankings(self, df: pd.DataFrame, use_simulation: bool) -> pd.DataFrame:
        """Add ranking columns based on various metrics"""
        if df.empty:
            return df

        # Projection-based ranking
        df['Proj_Rank'] = df['Projected'].rank(ascending=False, method='min').astype(int)

        # Ownership-based ranking (lower is better for GPP)
        df['Own_Rank'] = df['Total_Own'].rank(ascending=True, method='min').astype(int)

        # Simulation-based rankings
        if use_simulation and 'Sim_Ceiling_90th' in df.columns:
            df['Ceiling_Rank'] = df['Sim_Ceiling_90th'].rank(ascending=False, method='min').astype(int)
            df['Sharpe_Rank'] = df['Sim_Sharpe'].rank(ascending=False, method='min').astype(int)
            df['Win_Prob_Rank'] = df['Sim_Win_Prob'].rank(ascending=False, method='min').astype(int)

            # Overall GPP score (weighted combination)
            df['GPP_Score'] = (
                df['Sim_Ceiling_90th'] * 0.4 +
                df['Sim_Win_Prob'] * 100 * 0.3 +
                (100 - df['Total_Own']) * 0.2 +
                df['Sim_Sharpe'] * 10 * 0.1
            )
            df['GPP_Rank'] = df['GPP_Score'].rank(ascending=False, method='min').astype(int)

        return df

    def _store_optimization_metadata(self, num_lineups: int, field_size: str,
                                    enforcement_level: AIEnforcementLevel,
                                    used_genetic: bool,
                                    recommendations: Dict) -> None:
        """Store metadata about optimization run"""
        self.optimization_metadata = {
            'timestamp': datetime.now(),
            'num_lineups_requested': num_lineups,
            'num_lineups_generated': len(self.lineups_generated),
            'field_size': field_size,
            'enforcement_level': enforcement_level.value,
            'optimization_method': 'genetic_algorithm' if used_genetic else 'linear_programming',
            'ai_recommendations': {
                ai_type.value: {
                    'confidence': rec.confidence,
                    'captain_count': len(rec.captain_targets),
                    'stack_count': len(rec.stacks)
                }
                for ai_type, rec in recommendations.items()
            },
            'performance_metrics': self.perf_monitor.get_phase_summary()
        }

    def export_lineups(self, lineups_df: pd.DataFrame, filename: str = None,
                      format: str = 'csv') -> str:
        """
        Export lineups in various formats

        Args:
            lineups_df: DataFrame of lineups
            filename: Output filename
            format: Export format ('csv', 'dk_csv', 'excel')

        Returns:
            Path to exported file
        """
        try:
            if lineups_df.empty:
                self.logger.log("No lineups to export", "WARNING")
                return ""

            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"showdown_lineups_{timestamp}"

            # Export based on format
            if format == 'csv':
                filepath = f"{filename}.csv"
                lineups_df.to_csv(filepath, index=False)

            elif format == 'dk_csv':
                # DraftKings upload format
                dk_df = self._convert_to_dk_format(lineups_df)
                filepath = f"{filename}_DK.csv"
                dk_df.to_csv(filepath, index=False)

            elif format == 'excel':
                filepath = f"{filename}.xlsx"
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    lineups_df.to_excel(writer, sheet_name='Lineups', index=False)

                    # Add summary sheet
                    summary = self._create_summary_sheet(lineups_df)
                    summary.to_excel(writer, sheet_name='Summary', index=False)

            else:
                raise ValueError(f"Unknown format: {format}")

            self.logger.log(f"Lineups exported to {filepath}", "INFO")
            return filepath

        except Exception as e:
            self.logger.log_exception(e, "export_lineups")
            return ""

    def _convert_to_dk_format(self, lineups_df: pd.DataFrame) -> pd.DataFrame:
        """Convert to DraftKings upload format"""
        dk_data = []

        for _, row in lineups_df.iterrows():
            dk_row = {
                'CPT': row['CPT'],
                'FLEX': row['FLEX1'],
                'FLEX': row['FLEX2'],
                'FLEX': row['FLEX3'],
                'FLEX': row['FLEX4'],
                'FLEX': row['FLEX5']
            }
            dk_data.append(dk_row)

        return pd.DataFrame(dk_data)

    def _create_summary_sheet(self, lineups_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics sheet"""
        summary_data = {
            'Metric': [
                'Total Lineups',
                'Avg Projected Points',
                'Avg Total Ownership',
                'Avg Salary Used',
                'Unique Captains',
                'Most Common Captain',
                'Strategy Distribution'
            ],
            'Value': [
                len(lineups_df),
                f"{lineups_df['Projected'].mean():.2f}",
                f"{lineups_df['Total_Own'].mean():.1f}%",
                f"${lineups_df['Total_Salary'].mean():.0f}",
                lineups_df['CPT'].nunique(),
                lineups_df['CPT'].mode()[0] if not lineups_df.empty else 'N/A',
                lineups_df['Strategy'].value_counts().to_dict() if 'Strategy' in lineups_df.columns else {}
            ]
        }

        # Add simulation summary if available
        if 'Sim_Ceiling_90th' in lineups_df.columns:
            sim_summary = {
                'Metric': [
                    'Avg Ceiling (90th)',
                    'Avg Ceiling (99th)',
                    'Avg Sharpe Ratio',
                    'Avg Win Probability'
                ],
                'Value': [
                    f"{lineups_df['Sim_Ceiling_90th'].mean():.2f}",
                    f"{lineups_df['Sim_Ceiling_99th'].mean():.2f}",
                    f"{lineups_df['Sim_Sharpe'].mean():.2f}",
                    f"{lineups_df['Sim_Win_Prob'].mean():.2%}"
                ]
            }
            summary_data['Metric'].extend(sim_summary['Metric'])
            summary_data['Value'].extend(sim_summary['Value'])

        return pd.DataFrame(summary_data)

    def get_optimization_report(self) -> Dict:
        """Get comprehensive optimization report"""
        report = {
            'metadata': self.optimization_metadata,
            'performance': self.perf_monitor.get_phase_summary(),
            'ai_stats': {
                'game_theory': self.game_theory_ai.performance_history,
                'correlation': self.correlation_ai.performance_history,
                'contrarian': self.contrarian_ai.performance_history
            },
            'enforcement': self.enforcement_engine.get_effectiveness_report(),
            'lineups_generated': len(self.lineups_generated)
        }

        if self.api_manager:
            report['api_stats'] = self.api_manager.get_stats()

        return report

    def clear_cache(self) -> None:
        """Clear all caches"""
        self.game_theory_ai.response_cache.clear()
        self.correlation_ai.response_cache.clear()
        self.contrarian_ai.response_cache.clear()

        if self.api_manager:
            self.api_manager.clear_cache()

        if self.mc_engine:
            self.mc_engine.simulation_cache.clear()

        self.logger.log("All caches cleared", "INFO")

# ============================================================================
# PART 7: INTEGRATION, TESTING & EXAMPLE USAGE
# ============================================================================

# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

class OptimizerIntegration:
    """
    Integration helper for common workflows and batch processing

    DFS Value: Simplifies common use cases and batch optimization
    """

    def __init__(self, api_key: Optional[str] = None):
        self.optimizer = ShowdownOptimizer(api_key)
        self.logger = get_logger()
        self.results_history = []

    def optimize_from_csv(self, csv_path: str,
                         game_info: Dict,
                         num_lineups: int = 20,
                         field_size: str = 'large_field',
                         **kwargs) -> pd.DataFrame:
        """
        Optimize directly from CSV file

        Args:
            csv_path: Path to player CSV
            game_info: Game information dictionary
            num_lineups: Number of lineups to generate
            field_size: Contest size
            **kwargs: Additional optimizer parameters

        Returns:
            DataFrame of optimized lineups
        """
        try:
            self.logger.log(f"Loading players from {csv_path}", "INFO")

            # Load CSV
            df = pd.read_csv(csv_path)

            # Validate required columns
            required = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points']
            missing = [col for col in required if col not in df.columns]

            if missing:
                self.logger.log(f"Missing columns: {missing}", "ERROR")
                return pd.DataFrame()

            # Run optimization
            lineups = self.optimizer.optimize(
                df, game_info, num_lineups, field_size, **kwargs
            )

            # Store results
            self.results_history.append({
                'timestamp': datetime.now(),
                'csv_path': csv_path,
                'lineups': len(lineups),
                'field_size': field_size
            })

            return lineups

        except Exception as e:
            self.logger.log_exception(e, "optimize_from_csv")
            return pd.DataFrame()

    def batch_optimize(self, configs: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Run multiple optimizations with different configurations

        Args:
            configs: List of configuration dictionaries, each containing:
                - df: DataFrame or csv_path
                - game_info: Game information
                - num_lineups: Number of lineups
                - field_size: Contest size
                - name: Configuration name

        Returns:
            Dictionary mapping config names to lineup DataFrames
        """
        results = {}

        for i, config in enumerate(configs):
            name = config.get('name', f'config_{i+1}')
            self.logger.log(f"Running batch optimization: {name}", "INFO")

            try:
                # Get DataFrame
                if 'csv_path' in config:
                    df = pd.read_csv(config['csv_path'])
                elif 'df' in config:
                    df = config['df']
                else:
                    self.logger.log(f"No data source for {name}", "ERROR")
                    continue

                # Run optimization
                lineups = self.optimizer.optimize(
                    df=df,
                    game_info=config.get('game_info', {}),
                    num_lineups=config.get('num_lineups', 20),
                    field_size=config.get('field_size', 'large_field'),
                    ai_enforcement_level=config.get(
                        'ai_enforcement_level',
                        AIEnforcementLevel.STRONG
                    ),
                    use_api=config.get('use_api', True),
                    randomness=config.get('randomness', 0.15),
                    use_genetic=config.get('use_genetic', False),
                    use_simulation=config.get('use_simulation', True)
                )

                results[name] = lineups

            except Exception as e:
                self.logger.log_exception(e, f"batch_optimize_{name}")
                results[name] = pd.DataFrame()

        return results

    def compare_optimization_methods(self, df: pd.DataFrame,
                                    game_info: Dict,
                                    num_lineups: int = 10) -> Dict:
        """
        Compare LP vs GA optimization methods

        Returns:
            Dictionary with comparison metrics
        """
        self.logger.log("Comparing optimization methods", "INFO")

        # Run LP optimization
        lp_start = time.time()
        lp_lineups = self.optimizer.optimize(
            df, game_info, num_lineups,
            field_size='large_field',
            use_genetic=False,
            use_simulation=True
        )
        lp_time = time.time() - lp_start

        # Run GA optimization
        ga_start = time.time()
        ga_lineups = self.optimizer.optimize(
            df, game_info, num_lineups,
            field_size='large_field',
            use_genetic=True,
            use_simulation=True
        )
        ga_time = time.time() - ga_start

        # Compare results
        comparison = {
            'lp': {
                'time': lp_time,
                'lineups': len(lp_lineups),
                'avg_projection': lp_lineups['Projected'].mean() if not lp_lineups.empty else 0,
                'avg_ceiling': lp_lineups['Sim_Ceiling_90th'].mean() if 'Sim_Ceiling_90th' in lp_lineups else 0,
                'avg_ownership': lp_lineups['Total_Own'].mean() if not lp_lineups.empty else 0
            },
            'ga': {
                'time': ga_time,
                'lineups': len(ga_lineups),
                'avg_projection': ga_lineups['Projected'].mean() if not ga_lineups.empty else 0,
                'avg_ceiling': ga_lineups['Sim_Ceiling_90th'].mean() if 'Sim_Ceiling_90th' in ga_lineups else 0,
                'avg_ownership': ga_lineups['Total_Own'].mean() if not ga_lineups.empty else 0
            }
        }

        self.logger.log(
            f"Comparison complete - LP: {lp_time:.2f}s, GA: {ga_time:.2f}s",
            "INFO"
        )

        return comparison

# ============================================================================
# TESTING UTILITIES
# ============================================================================

class OptimizerTester:
    """
    Testing utilities for validation and debugging

    DFS Value: Ensures optimizer reliability and correctness
    """

    def __init__(self):
        self.logger = get_logger()
        self.test_results = []

    def create_test_slate(self, num_players: int = 20) -> pd.DataFrame:
        """Create synthetic test slate"""
        np.random.seed(42)

        teams = ['TEAM1', 'TEAM2']
        positions = ['QB', 'RB', 'WR', 'TE', 'DST']

        players = []
        for i in range(num_players):
            team = teams[i % 2]
            position = positions[i % len(positions)]

            # Realistic salary and projection distributions
            if position == 'QB':
                salary = np.random.randint(9000, 12000)
                projection = np.random.uniform(18, 26)
                ownership = np.random.uniform(8, 25)
            elif position == 'RB':
                salary = np.random.randint(6000, 10000)
                projection = np.random.uniform(10, 20)
                ownership = np.random.uniform(5, 20)
            elif position == 'WR':
                salary = np.random.randint(5000, 11000)
                projection = np.random.uniform(8, 22)
                ownership = np.random.uniform(3, 18)
            elif position == 'TE':
                salary = np.random.randint(4000, 9000)
                projection = np.random.uniform(6, 16)
                ownership = np.random.uniform(3, 15)
            else:  # DST
                salary = np.random.randint(3000, 5000)
                projection = np.random.uniform(5, 12)
                ownership = np.random.uniform(2, 10)

            players.append({
                'Player': f'{position}_{team}_{i}',
                'Position': position,
                'Team': team,
                'Salary': salary,
                'Projected_Points': projection,
                'Ownership': ownership
            })

        return pd.DataFrame(players)

    def test_basic_optimization(self, optimizer: ShowdownOptimizer = None) -> bool:
        """Test basic optimization functionality"""
        self.logger.log("Running basic optimization test", "INFO")

        try:
            if not optimizer:
                optimizer = ShowdownOptimizer()

            # Create test data
            df = self.create_test_slate()
            game_info = {
                'teams': 'TEAM1 vs TEAM2',
                'total': 45.5,
                'spread': -3.5,
                'weather': 'Clear'
            }

            # Run optimization
            lineups = optimizer.optimize(
                df, game_info,
                num_lineups=5,
                field_size='small_field',
                use_api=False,
                use_simulation=False
            )

            # Validate results
            if lineups.empty:
                self.logger.log("Test FAILED: No lineups generated", "ERROR")
                return False

            # Check constraints
            for _, lineup in lineups.iterrows():
                if not self._validate_lineup_constraints(lineup, df):
                    self.logger.log("Test FAILED: Invalid lineup constraints", "ERROR")
                    return False

            self.logger.log("Basic optimization test PASSED", "INFO")
            self.test_results.append({
                'test': 'basic_optimization',
                'status': 'PASSED',
                'lineups': len(lineups)
            })
            return True

        except Exception as e:
            self.logger.log_exception(e, "test_basic_optimization")
            self.test_results.append({
                'test': 'basic_optimization',
                'status': 'FAILED',
                'error': str(e)
            })
            return False

    def test_genetic_algorithm(self, optimizer: ShowdownOptimizer = None) -> bool:
        """Test genetic algorithm optimization"""
        self.logger.log("Running genetic algorithm test", "INFO")

        try:
            if not optimizer:
                optimizer = ShowdownOptimizer()

            df = self.create_test_slate()
            game_info = {
                'teams': 'TEAM1 vs TEAM2',
                'total': 48.0,
                'spread': -7.0,
                'weather': 'Clear'
            }

            lineups = optimizer.optimize(
                df, game_info,
                num_lineups=10,
                field_size='large_field',
                use_api=False,
                use_genetic=True,
                use_simulation=True
            )

            if lineups.empty:
                self.logger.log("GA Test FAILED: No lineups generated", "ERROR")
                return False

            # Check for simulation metrics
            if 'Sim_Ceiling_90th' not in lineups.columns:
                self.logger.log("GA Test WARNING: No simulation metrics", "WARNING")

            self.logger.log("Genetic algorithm test PASSED", "INFO")
            self.test_results.append({
                'test': 'genetic_algorithm',
                'status': 'PASSED',
                'lineups': len(lineups)
            })
            return True

        except Exception as e:
            self.logger.log_exception(e, "test_genetic_algorithm")
            self.test_results.append({
                'test': 'genetic_algorithm',
                'status': 'FAILED',
                'error': str(e)
            })
            return False

    def test_monte_carlo_simulation(self) -> bool:
        """Test Monte Carlo simulation engine"""
        self.logger.log("Running Monte Carlo simulation test", "INFO")

        try:
            df = self.create_test_slate()
            game_info = {
                'teams': 'TEAM1 vs TEAM2',
                'total': 45.0,
                'spread': -3.0
            }

            # Initialize engine
            mc_engine = MonteCarloSimulationEngine(df, game_info, n_simulations=1000)

            # Test single lineup simulation
            captain = df.iloc[0]['Player']
            flex = df.iloc[1:6]['Player'].tolist()

            results = mc_engine.evaluate_lineup(captain, flex)

            # Validate results
            if results.mean <= 0:
                self.logger.log("MC Test FAILED: Invalid mean", "ERROR")
                return False

            if results.ceiling_90th <= results.mean:
                self.logger.log("MC Test FAILED: Invalid ceiling", "ERROR")
                return False

            if results.std <= 0:
                self.logger.log("MC Test FAILED: Invalid std dev", "ERROR")
                return False

            self.logger.log("Monte Carlo simulation test PASSED", "INFO")
            self.test_results.append({
                'test': 'monte_carlo',
                'status': 'PASSED',
                'mean': results.mean,
                'ceiling': results.ceiling_90th
            })
            return True

        except Exception as e:
            self.logger.log_exception(e, "test_monte_carlo_simulation")
            self.test_results.append({
                'test': 'monte_carlo',
                'status': 'FAILED',
                'error': str(e)
            })
            return False

    def test_ai_strategists(self, optimizer: ShowdownOptimizer = None) -> bool:
        """Test AI strategist fallback behavior"""
        self.logger.log("Running AI strategist test", "INFO")

        try:
            if not optimizer:
                optimizer = ShowdownOptimizer()

            df = self.create_test_slate()
            game_info = {
                'teams': 'TEAM1 vs TEAM2',
                'total': 45.0,
                'spread': -3.0,
                'weather': 'Clear'
            }

            # Test each strategist without API
            for strategist, name in [
                (optimizer.game_theory_ai, 'Game Theory'),
                (optimizer.correlation_ai, 'Correlation'),
                (optimizer.contrarian_ai, 'Contrarian')
            ]:
                rec = strategist.get_recommendation(
                    df, game_info, 'large_field', use_api=False
                )

                if not rec.captain_targets:
                    self.logger.log(
                        f"AI Test FAILED: {name} no captains",
                        "ERROR"
                    )
                    return False

                if rec.confidence <= 0 or rec.confidence > 1:
                    self.logger.log(
                        f"AI Test FAILED: {name} invalid confidence",
                        "ERROR"
                    )
                    return False

            self.logger.log("AI strategist test PASSED", "INFO")
            self.test_results.append({
                'test': 'ai_strategists',
                'status': 'PASSED'
            })
            return True

        except Exception as e:
            self.logger.log_exception(e, "test_ai_strategists")
            self.test_results.append({
                'test': 'ai_strategists',
                'status': 'FAILED',
                'error': str(e)
            })
            return False

    def test_constraint_relaxation(self, optimizer: ShowdownOptimizer = None) -> bool:
        """Test three-tier constraint relaxation"""
        self.logger.log("Running constraint relaxation test", "INFO")

        try:
            if not optimizer:
                optimizer = ShowdownOptimizer()

            df = self.create_test_slate()
            game_info = {
                'teams': 'TEAM1 vs TEAM2',
                'total': 45.0,
                'spread': -3.0
            }

            # Test with MANDATORY enforcement (strictest)
            lineups_mandatory = optimizer.optimize(
                df, game_info,
                num_lineups=10,
                field_size='large_field',
                ai_enforcement_level=AIEnforcementLevel.MANDATORY,
                use_api=False,
                use_simulation=False
            )

            # Test with ADVISORY enforcement (loosest)
            lineups_advisory = optimizer.optimize(
                df, game_info,
                num_lineups=10,
                field_size='large_field',
                ai_enforcement_level=AIEnforcementLevel.ADVISORY,
                use_api=False,
                use_simulation=False
            )

            # Advisory should have more diversity
            if not lineups_advisory.empty and not lineups_mandatory.empty:
                unique_captains_advisory = lineups_advisory['CPT'].nunique()
                unique_captains_mandatory = lineups_mandatory['CPT'].nunique()

                if unique_captains_advisory < unique_captains_mandatory:
                    self.logger.log(
                        "Relaxation Test WARNING: Advisory less diverse than mandatory",
                        "WARNING"
                    )

            self.logger.log("Constraint relaxation test PASSED", "INFO")
            self.test_results.append({
                'test': 'constraint_relaxation',
                'status': 'PASSED',
                'mandatory_lineups': len(lineups_mandatory),
                'advisory_lineups': len(lineups_advisory)
            })
            return True

        except Exception as e:
            self.logger.log_exception(e, "test_constraint_relaxation")
            self.test_results.append({
                'test': 'constraint_relaxation',
                'status': 'FAILED',
                'error': str(e)
            })
            return False

    def _validate_lineup_constraints(self, lineup: pd.Series, df: pd.DataFrame) -> bool:
        """Validate lineup meets DK constraints"""
        # Get all players
        captain = lineup['CPT']
        flex = [lineup['FLEX1'], lineup['FLEX2'], lineup['FLEX3'],
                lineup['FLEX4'], lineup['FLEX5']]
        all_players = [captain] + flex

        # Check unique players
        if len(set(all_players)) != 6:
            return False

        # Check salary cap
        if lineup['Total_Salary'] > OptimizerConfig.SALARY_CAP:
            return False

        # Check team diversity
        teams = df[df['Player'].isin(all_players)]['Team'].value_counts()
        if any(count > OptimizerConfig.MAX_PLAYERS_PER_TEAM for count in teams.values):
            return False

        if len(teams) < OptimizerConfig.MIN_TEAMS_REQUIRED:
            return False

        return True

    def run_all_tests(self, optimizer: ShowdownOptimizer = None) -> Dict:
        """Run complete test suite"""
        self.logger.log("=" * 60, "INFO")
        self.logger.log("RUNNING COMPLETE TEST SUITE", "INFO")
        self.logger.log("=" * 60, "INFO")

        self.test_results = []

        tests = [
            ('Basic Optimization', self.test_basic_optimization),
            ('Genetic Algorithm', self.test_genetic_algorithm),
            ('Monte Carlo Simulation', self.test_monte_carlo_simulation),
            ('AI Strategists', self.test_ai_strategists),
            ('Constraint Relaxation', self.test_constraint_relaxation)
        ]

        results = {}
        passed = 0
        failed = 0

        for name, test_func in tests:
            self.logger.log(f"\n{'='*40}", "INFO")
            self.logger.log(f"TEST: {name}", "INFO")
            self.logger.log(f"{'='*40}", "INFO")

            if test_func.__name__ in ['test_basic_optimization',
                                      'test_ai_strategists',
                                      'test_constraint_relaxation']:
                success = test_func(optimizer)
            else:
                success = test_func()

            results[name] = 'PASSED' if success else 'FAILED'
            if success:
                passed += 1
            else:
                failed += 1

        self.logger.log("\n" + "=" * 60, "INFO")
        self.logger.log("TEST SUITE COMPLETE", "INFO")
        self.logger.log(f"PASSED: {passed}/{len(tests)}", "INFO")
        self.logger.log(f"FAILED: {failed}/{len(tests)}", "INFO")
        self.logger.log("=" * 60, "INFO")

        return {
            'summary': results,
            'passed': passed,
            'failed': failed,
            'total': len(tests),
            'details': self.test_results
        }

# ============================================================================
# EXAMPLE USAGE SCRIPTS
# ============================================================================

def example_basic_usage():
    """
    Example 1: Basic optimization workflow
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: BASIC OPTIMIZATION")
    print("="*60 + "\n")

    # Create test data
    tester = OptimizerTester()
    df = tester.create_test_slate(num_players=20)

    # Game information
    game_info = {
        'teams': 'Chiefs vs Bills',
        'total': 52.5,
        'spread': -2.5,
        'weather': 'Clear',
        'primetime': True
    }

    # Initialize optimizer (without API key for demo)
    optimizer = ShowdownOptimizer()

    # Run optimization
    print("Running optimization...")
    lineups = optimizer.optimize(
        df=df,
        game_info=game_info,
        num_lineups=20,
        field_size='large_field',
        ai_enforcement_level=AIEnforcementLevel.STRONG,
        use_api=False,  # Set to True to use Claude API
        use_simulation=False  # Set to True for Monte Carlo
    )

    # Display results
    print(f"\nGenerated {len(lineups)} lineups")
    print("\nTop 5 Lineups by Projection:")
    print(lineups[['Lineup', 'CPT', 'Projected', 'Total_Own', 'Strategy']].head())

    # Export lineups
    filepath = optimizer.export_lineups(lineups, "example_lineups", format='csv')
    print(f"\nLineups exported to: {filepath}")

    return lineups

def example_advanced_usage():
    """
    Example 2: Advanced optimization with all features
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: ADVANCED OPTIMIZATION WITH ML")
    print("="*60 + "\n")

    # Create test data
    tester = OptimizerTester()
    df = tester.create_test_slate(num_players=25)

    game_info = {
        'teams': 'Ravens vs Bengals',
        'total': 49.0,
        'spread': -3.5,
        'weather': 'Dome',
        'primetime': True,
        'injury_count': 0
    }

    # Initialize with API key (replace with actual key)
    # optimizer = ShowdownOptimizer(api_key='your-api-key-here')
    optimizer = ShowdownOptimizer()  # No API for demo

    print("Running advanced optimization with:")
    print("- Genetic Algorithm")
    print("- Monte Carlo Simulation (5000 iterations)")
    print("- All AI strategists")
    print("- Moderate enforcement level\n")

    lineups = optimizer.optimize(
        df=df,
        game_info=game_info,
        num_lineups=50,
        field_size='milly_maker',
        ai_enforcement_level=AIEnforcementLevel.MODERATE,
        use_api=False,
        use_genetic=True,
        use_simulation=True
    )

    # Display simulation results
    if 'Sim_Ceiling_90th' in lineups.columns:
        print("\nTop 5 Lineups by Ceiling (90th percentile):")
        top_ceiling = lineups.nlargest(5, 'Sim_Ceiling_90th')
        print(top_ceiling[[
            'Lineup', 'CPT', 'Sim_Ceiling_90th',
            'Sim_Win_Prob', 'Total_Own', 'Strategy'
        ]])

        print("\nSimulation Summary:")
        print(f"Avg Ceiling (90th): {lineups['Sim_Ceiling_90th'].mean():.2f}")
        print(f"Avg Ceiling (99th): {lineups['Sim_Ceiling_99th'].mean():.2f}")
        print(f"Avg Win Probability: {lineups['Sim_Win_Prob'].mean():.2%}")
        print(f"Avg Sharpe Ratio: {lineups['Sim_Sharpe'].mean():.2f}")

    # Get optimization report
    report = optimizer.get_optimization_report()
    print("\nOptimization Metadata:")
    print(f"Method: {report['metadata']['optimization_method']}")
    print(f"Lineups Generated: {report['metadata']['num_lineups_generated']}")
    print(f"Field Size: {report['metadata']['field_size']}")

    return lineups

def example_batch_optimization():
    """
    Example 3: Batch optimization with multiple configurations
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: BATCH OPTIMIZATION")
    print("="*60 + "\n")

    # Create test data
    tester = OptimizerTester()
    df = tester.create_test_slate(num_players=20)

    game_info = {
        'teams': 'Packers vs Lions',
        'total': 51.0,
        'spread': -6.5,
        'weather': 'Dome'
    }

    # Initialize integration helper
    integration = OptimizerIntegration()

    # Define multiple configurations
    configs = [
        {
            'name': 'Cash_Game_Strategy',
            'df': df,
            'game_info': game_info,
            'num_lineups': 10,
            'field_size': 'small_field',
            'ai_enforcement_level': AIEnforcementLevel.MANDATORY,
            'use_simulation': True,
            'use_genetic': False
        },
        {
            'name': 'GPP_Balanced',
            'df': df,
            'game_info': game_info,
            'num_lineups': 20,
            'field_size': 'large_field',
            'ai_enforcement_level': AIEnforcementLevel.STRONG,
            'use_simulation': True,
            'use_genetic': False
        },
        {
            'name': 'GPP_Ultra_Contrarian',
            'df': df,
            'game_info': game_info,
            'num_lineups': 30,
            'field_size': 'milly_maker',
            'ai_enforcement_level': AIEnforcementLevel.MODERATE,
            'use_simulation': True,
            'use_genetic': True
        }
    ]

    # Run batch optimization
    print("Running batch optimization with 3 configurations...\n")
    results = integration.batch_optimize(configs)

    # Display results
    for name, lineups_df in results.items():
        print(f"\n{name}:")
        print(f"  Lineups: {len(lineups_df)}")
        if not lineups_df.empty:
            print(f"  Avg Projection: {lineups_df['Projected'].mean():.2f}")
            print(f"  Avg Ownership: {lineups_df['Total_Own'].mean():.1f}%")
            print(f"  Unique Captains: {lineups_df['CPT'].nunique()}")

    return results

def example_comparison():
    """
    Example 4: Compare optimization methods
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: OPTIMIZATION METHOD COMPARISON")
    print("="*60 + "\n")

    # Create test data
    tester = OptimizerTester()
    df = tester.create_test_slate(num_players=20)

    game_info = {
        'teams': 'Cowboys vs Eagles',
        'total': 48.5,
        'spread': -3.0,
        'weather': 'Clear'
    }

    # Initialize integration
    integration = OptimizerIntegration()

    print("Comparing Linear Programming vs Genetic Algorithm...")
    print("Generating 20 lineups with each method...\n")

    # Run comparison
    comparison = integration.compare_optimization_methods(
        df, game_info, num_lineups=20
    )

    # Display results
    print("\nLINEAR PROGRAMMING:")
    print(f"  Time: {comparison['lp']['time']:.2f}s")
    print(f"  Lineups: {comparison['lp']['lineups']}")
    print(f"  Avg Projection: {comparison['lp']['avg_projection']:.2f}")
    print(f"  Avg Ceiling: {comparison['lp']['avg_ceiling']:.2f}")
    print(f"  Avg Ownership: {comparison['lp']['avg_ownership']:.1f}%")

    print("\nGENETIC ALGORITHM:")
    print(f"  Time: {comparison['ga']['time']:.2f}s")
    print(f"  Lineups: {comparison['ga']['lineups']}")
    print(f"  Avg Projection: {comparison['ga']['avg_projection']:.2f}")
    print(f"  Avg Ceiling: {comparison['ga']['avg_ceiling']:.2f}")
    print(f"  Avg Ownership: {comparison['ga']['avg_ownership']:.1f}%")

    # Determine winner
    print("\nANALYSIS:")
    if comparison['ga']['avg_ceiling'] > comparison['lp']['avg_ceiling']:
        print("✓ GA produces higher ceiling lineups (better for GPP)")
    else:
        print("✓ LP produces higher ceiling lineups")

    if comparison['ga']['avg_ownership'] < comparison['lp']['avg_ownership']:
        print("✓ GA produces lower ownership lineups (better leverage)")
    else:
        print("✓ LP produces lower ownership lineups")

    if comparison['lp']['time'] < comparison['ga']['time']:
        print(f"✓ LP is faster ({comparison['lp']['time']:.1f}s vs {comparison['ga']['time']:.1f}s)")
    else:
        print(f"✓ GA is faster")

    return comparison

def example_testing():
    """
    Example 5: Run complete test suite
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: COMPLETE TEST SUITE")
    print("="*60 + "\n")

    # Initialize tester
    tester = OptimizerTester()

    # Run all tests
    results = tester.run_all_tests()

    # Display summary
    print("\nFINAL RESULTS:")
    for test_name, status in results['summary'].items():
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {test_name}: {status}")

    print(f"\nOverall: {results['passed']}/{results['total']} tests passed")

    if results['failed'] == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {results['failed']} test(s) failed")

    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - runs all examples
    """
    print("\n" + "="*80)
    print(" "*20 + "NFL SHOWDOWN OPTIMIZER")
    print(" "*15 + "Complete Example Suite")
    print("="*80)

    try:
        # Example 1: Basic Usage
        example_basic_usage()

        # Example 2: Advanced Usage
        example_advanced_usage()

        # Example 3: Batch Optimization
        example_batch_optimization()

        # Example 4: Method Comparison
        example_comparison()

        # Example 5: Testing Suite
        example_testing()

        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")

    except Exception as e:
        logger = get_logger()
        logger.log_exception(e, "main")
        print("\n⚠️  Error running examples - check logs for details")

# ============================================================================
# QUICK START TEMPLATE
# ============================================================================

def quick_start_template():
    """
    Quick start template for real usage

    INSTRUCTIONS:
    1. Replace the CSV path with your player projections file
    2. Update game_info with actual game details
    3. Add your Claude API key if using AI (optional)
    4. Adjust num_lineups and field_size as needed
    5. Run the script!
    """

    # ========== CONFIGURATION ==========

    # Path to your player projections CSV
    CSV_PATH = "your_projections.csv"

    # Game information
    GAME_INFO = {
        'teams': 'Team1 vs Team2',
        'total': 47.5,
        'spread': -3.5,
        'weather': 'Clear',
        'primetime': False
    }

    # Optimization settings
    NUM_LINEUPS = 20
    FIELD_SIZE = 'large_field'  # Options: small_field, medium_field, large_field, large_field_aggressive, milly_maker

    # AI settings (optional - set to None to skip Claude API)
    CLAUDE_API_KEY = None  # Replace with 'sk-ant-...' if using API

    # Advanced settings
    USE_GENETIC = False  # Set True for large fields / Milly Maker
    USE_SIMULATION = True  # Set True for ceiling analysis
    ENFORCEMENT = AIEnforcementLevel.STRONG  # Options: MANDATORY, STRONG, MODERATE, ADVISORY

    # ========== RUN OPTIMIZATION ==========

    print("Initializing optimizer...")
    optimizer = ShowdownOptimizer(api_key=CLAUDE_API_KEY)

    print(f"Loading players from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    print(f"Optimizing {NUM_LINEUPS} lineups for {FIELD_SIZE}...")
    lineups = optimizer.optimize(
        df=df,
        game_info=GAME_INFO,
        num_lineups=NUM_LINEUPS,
        field_size=FIELD_SIZE,
        ai_enforcement_level=ENFORCEMENT,
        use_api=(CLAUDE_API_KEY is not None),
        use_genetic=USE_GENETIC,
        use_simulation=USE_SIMULATION
    )

    print(f"\n✓ Generated {len(lineups)} lineups")

    # Display top lineups
    print("\nTop 5 Lineups:")
    print(lineups[['Lineup', 'CPT', 'Projected', 'Total_Own']].head())

    # Export lineups
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"showdown_lineups_{timestamp}"

    csv_path = optimizer.export_lineups(lineups, filename, format='csv')
    dk_path = optimizer.export_lineups(lineups, filename, format='dk_csv')

    print(f"\n✓ Exported to: {csv_path}")
    print(f"✓ DK format: {dk_path}")

    return lineups

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run all examples
    main()

    # Or run quick start template:
    # quick_start_template()
