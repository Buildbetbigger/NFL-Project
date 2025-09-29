
# ============================================================================
# PART 1: CONFIGURATION, IMPORTS, AND BASE CLASSES
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pulp
import json
import hashlib
import threading
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION CLASSES - ENHANCED WITH ALL IMPROVEMENTS
# ============================================================================

class OptimizerConfig:
    """Enhanced configuration with improved ownership projection and optimization modes"""

    # Core constraints
    SALARY_CAP = 50000
    MIN_SALARY = 3000
    MAX_SALARY = 12000
    CAPTAIN_MULTIPLIER = 1.5
    ROSTER_SIZE = 6
    FLEX_SPOTS = 5

    # DraftKings Showdown specific
    MIN_TEAMS_REQUIRED = 2  # Must have players from both teams
    MAX_PLAYERS_PER_TEAM = 5  # Max 5 from one team (leaving 1 for opponent)

    # Performance settings - Enhanced
    MAX_ITERATIONS = 1000
    OPTIMIZATION_TIMEOUT = 30  # seconds per lineup
    MAX_PARALLEL_THREADS = 8
    MAX_HISTORY_ENTRIES = 50  # Reduced for memory management
    CACHE_SIZE = 100  # Maximum cache entries

    # Default ownership values - Enhanced with position/salary correlation
    DEFAULT_OWNERSHIP = 5.0
    OWNERSHIP_BY_POSITION = {
        'QB': {'base': 15, 'salary_factor': 0.002},
        'RB': {'base': 12, 'salary_factor': 0.0015},
        'WR': {'base': 10, 'salary_factor': 0.0018},
        'TE': {'base': 8, 'salary_factor': 0.001},
        'DST': {'base': 5, 'salary_factor': 0.0005},
        'K': {'base': 3, 'salary_factor': 0.0003},
        'FLEX': {'base': 5, 'salary_factor': 0.001}
    }

    @classmethod
    def get_default_ownership(cls, position: str, salary: float) -> float:
        """Enhanced ownership projection based on position and salary"""
        pos_config = cls.OWNERSHIP_BY_POSITION.get(
            position,
            cls.OWNERSHIP_BY_POSITION['FLEX']
        )

        base = pos_config['base']
        salary_factor = pos_config['salary_factor']

        # Calculate ownership with salary correlation
        # Higher salary generally means higher ownership
        salary_adjustment = (salary - 5000) * salary_factor

        # Add some randomness for variation
        random_factor = np.random.normal(1.0, 0.1)

        ownership = (base + salary_adjustment) * random_factor

        # Bound between reasonable limits
        return max(1.0, min(40.0, ownership))

    # Contest field sizes - Enhanced
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

    # New: Optimization modes for ceiling/floor optimization
    OPTIMIZATION_MODES = {
        'balanced': {'ceiling_weight': 0.5, 'floor_weight': 0.5},
        'ceiling': {'ceiling_weight': 0.8, 'floor_weight': 0.2},
        'floor': {'ceiling_weight': 0.2, 'floor_weight': 0.8},
        'boom_or_bust': {'ceiling_weight': 1.0, 'floor_weight': 0.0}
    }

    # GPP Ownership targets by field size - Enhanced
    GPP_OWNERSHIP_TARGETS = {
        'small_field': (70, 110),          # Higher ownership acceptable
        'medium_field': (60, 90),          # Balanced
        'large_field': (50, 80),           # Lower ownership
        'large_field_aggressive': (40, 70), # Very low ownership
        'milly_maker': (30, 60),           # Ultra contrarian
        'super_contrarian': (20, 50)       # Extreme leverage
    }

    # Field-specific AI configurations - Enhanced
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
            'max_total_ownership': 110
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
            'max_total_ownership': 90
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
            'max_total_ownership': 80
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
            'max_total_ownership': 70
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
            'max_total_ownership': 60
        },
        'super_contrarian': {
            'max_exposure': 0.1,
            'min_unique_captains': 40,
            'max_chalk_players': 0,
            'min_leverage_players': 5,
            'ownership_leverage_weight': 0.6,
            'correlation_weight': 0.15,
            'narrative_weight': 0.25,
            'ai_enforcement': 'Mandatory',
            'min_total_ownership': 20,
            'max_total_ownership': 50
        }
    }

    # Enhanced AI system weights for multi-sport support structure
    AI_WEIGHTS = {
        'game_theory': 0.35,
        'correlation': 0.35,
        'contrarian': 0.30
    }

    # Sport-specific configurations (foundation for multi-sport)
    SPORT_CONFIGS = {
        'NFL': {
            'roster_size': 6,
            'salary_cap': 50000,
            'positions': ['QB', 'RB', 'WR', 'TE', 'DST'],
            'scoring': 'DK_NFL'
        },
        # Ready for expansion
        'NBA': {
            'roster_size': 6,
            'salary_cap': 50000,
            'positions': ['PG', 'SG', 'SF', 'PF', 'C'],
            'scoring': 'DK_NBA'
        }
    }

# ============================================================================
# ENUM CLASSES - ENHANCED
# ============================================================================

class AIStrategistType(Enum):
    """Types of AI strategists"""
    GAME_THEORY = "Game Theory"
    CORRELATION = "Correlation"
    CONTRARIAN_NARRATIVE = "Contrarian Narrative"

class AIEnforcementLevel(Enum):
    """AI enforcement levels - Enhanced"""
    ADVISORY = "Advisory"      # AI suggestions only
    MODERATE = "Moderate"      # Some constraints enforced
    STRONG = "Strong"          # Most constraints enforced
    MANDATORY = "Mandatory"    # All AI decisions enforced

class OptimizationMode(Enum):
    """Optimization modes for different strategies"""
    BALANCED = "balanced"
    CEILING = "ceiling"
    FLOOR = "floor"
    BOOM_OR_BUST = "boom_or_bust"

class StackType(Enum):
    """Enhanced stack types"""
    QB_RECEIVER = "qb_receiver"
    ONSLAUGHT = "onslaught"
    LEVERAGE = "leverage"
    BRING_BACK = "bring_back"
    DEFENSIVE = "defensive"
    HIDDEN = "hidden"

# ============================================================================
# BASE DATA CLASSES - ENHANCED
# ============================================================================

@dataclass
class AIRecommendation:
    """Enhanced AI recommendation with validation and metadata"""
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
        """Enhanced validation with detailed error reporting"""
        errors = []

        # Check for empty recommendations
        if not self.captain_targets and not self.must_play:
            errors.append("No captain targets or must-play players specified")

        # Validate confidence
        if not 0 <= self.confidence <= 1:
            errors.append(f"Invalid confidence score: {self.confidence}")
            self.confidence = max(0, min(1, self.confidence))

        # Check for conflicts
        conflicts = set(self.must_play) & set(self.never_play)
        if conflicts:
            errors.append(f"Conflicting players in must/never play: {conflicts}")

        # Validate stacks
        for stack in self.stacks:
            if not isinstance(stack, dict):
                errors.append("Invalid stack format")
            elif 'players' in stack and len(stack['players']) < 2:
                errors.append("Stack must have at least 2 players")

        # Validate enforcement rules
        for rule in self.enforcement_rules:
            if 'type' not in rule or 'constraint' not in rule:
                errors.append("Invalid enforcement rule format")

        is_valid = len(errors) == 0
        return is_valid, errors

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
        """Validate a lineup against constraints"""
        errors = []

        # Salary validation
        total_salary = lineup.get('Salary', 0)
        if total_salary < self.min_salary:
            errors.append(f"Salary too low: {total_salary} < {self.min_salary}")
        if total_salary > self.max_salary:
            errors.append(f"Salary too high: {total_salary} > {self.max_salary}")

        # Ownership validation
        total_ownership = lineup.get('Total_Ownership', 0)
        if total_ownership > self.max_ownership:
            errors.append(f"Ownership too high: {total_ownership}")
        if total_ownership < self.min_ownership:
            errors.append(f"Ownership too low: {total_ownership}")

        # Player validation
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
    """Track optimizer performance metrics"""
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

    def calculate_efficiency(self) -> float:
        """Calculate generation efficiency"""
        if self.total_iterations == 0:
            return 0
        return self.successful_lineups / self.total_iterations

    def get_summary(self) -> Dict:
        """Get performance summary"""
        return {
            'efficiency': self.calculate_efficiency(),
            'avg_time_per_lineup': self.lineup_generation_time / max(self.successful_lineups, 1),
            'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            'success_rate': self.successful_lineups / max(self.successful_lineups + self.failed_lineups, 1),
            'ai_metrics': {
                'api_calls': self.ai_api_calls,
                'cache_hits': self.ai_cache_hits,
                'avg_confidence': self.average_confidence
            }
        }

# ============================================================================
# SINGLETON INSTANCES - WITH MEMORY MANAGEMENT
# ============================================================================

_global_logger = None
_performance_monitor = None
_ai_tracker = None

def get_logger():
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = GlobalLogger()
    return _global_logger

def get_performance_monitor():
    """Get or create performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def get_ai_tracker():
    """Get or create AI decision tracker instance"""
    global _ai_tracker
    if _ai_tracker is None:
        _ai_tracker = AIDecisionTracker()
    return _ai_tracker

# ============================================================================
# END OF PART 1
# ============================================================================

# ============================================================================
# PART 2: LOGGING, MONITORING, AND TRACKING CLASSES
# ============================================================================

class GlobalLogger:
    """Enhanced global logger with memory management and better error messages"""

    def __init__(self):
        self.logs = deque(maxlen=OptimizerConfig.MAX_HISTORY_ENTRIES)
        self.error_logs = deque(maxlen=20)
        self.ai_decisions = deque(maxlen=50)
        self.optimization_events = deque(maxlen=30)
        self.performance_metrics = defaultdict(list)
        self._lock = threading.RLock()

        # Enhanced error tracking
        self.error_patterns = defaultdict(int)
        self.last_cleanup = datetime.now()

    def log(self, message: str, level: str = "INFO", context: Dict = None):
        """Enhanced logging with context and pattern detection"""
        with self._lock:
            entry = {
                'timestamp': datetime.now(),
                'level': level,
                'message': message,
                'context': context or {}
            }

            self.logs.append(entry)

            if level == "ERROR":
                self.error_logs.append(entry)
                # Track error patterns
                error_key = self._extract_error_pattern(message)
                self.error_patterns[error_key] += 1

            # Periodic cleanup
            if (datetime.now() - self.last_cleanup).seconds > 300:  # 5 minutes
                self._cleanup()

            # Print to console if streamlit is available
            if level in ["ERROR", "CRITICAL"]:
                print(f"[{level}] {message}")

    def _extract_error_pattern(self, message: str) -> str:
        """Extract error pattern for tracking common issues"""
        # Remove specific values to find patterns
        import re
        pattern = re.sub(r'\d+', 'N', message)  # Replace numbers
        pattern = re.sub(r'"[^"]*"', '"X"', pattern)  # Replace quoted strings
        return pattern[:100]  # Limit length

    def _cleanup(self):
        """Memory cleanup"""
        # Clear old performance metrics
        cutoff = datetime.now() - timedelta(hours=1)
        for key in list(self.performance_metrics.keys()):
            self.performance_metrics[key] = [
                m for m in self.performance_metrics[key]
                if m.get('timestamp', datetime.now()) > cutoff
            ]

        self.last_cleanup = datetime.now()

    def log_exception(self, exception: Exception, context: str = "", critical: bool = False):
        """Enhanced exception logging with helpful context"""
        with self._lock:
            error_msg = f"{context}: {str(exception)}" if context else str(exception)

            # Add helpful suggestions based on error type
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

            # Display helpful error message to user
            if suggestions:
                self.log(f"💡 Suggestion: {suggestions[0]}", "INFO")

    def _get_error_suggestions(self, exception: Exception, context: str) -> List[str]:
        """Provide helpful suggestions based on error type"""
        suggestions = []

        if isinstance(exception, KeyError):
            suggestions.append("Check that all required columns are present in the CSV")
            suggestions.append("Verify player names match exactly")
        elif isinstance(exception, ValueError):
            if "salary" in str(exception).lower():
                suggestions.append("Check salary cap constraints - may be too restrictive")
            elif "ownership" in str(exception).lower():
                suggestions.append("Verify ownership projections are between 0-100")
        elif isinstance(exception, pulp.PulpSolverError):
            suggestions.append("Optimization constraints may be infeasible")
            suggestions.append("Try relaxing AI enforcement level")
        elif "timeout" in str(exception).lower():
            suggestions.append("Reduce number of lineups or increase timeout")

        return suggestions

    def log_ai_decision(self, decision_type: str, ai_source: str,
                       success: bool, details: Dict = None, confidence: float = 0):
        """Log AI decision for tracking"""
        with self._lock:
            self.ai_decisions.append({
                'timestamp': datetime.now(),
                'type': decision_type,
                'source': ai_source,
                'success': success,
                'confidence': confidence,
                'details': details or {}
            })

    def log_optimization_start(self, num_lineups: int, field_size: str, settings: Dict):
        """Log optimization start with settings"""
        with self._lock:
            self.optimization_events.append({
                'timestamp': datetime.now(),
                'event': 'start',
                'num_lineups': num_lineups,
                'field_size': field_size,
                'settings': settings
            })

    def log_optimization_end(self, lineups_generated: int, time_taken: float, success_rate: float):
        """Log optimization completion"""
        with self._lock:
            self.optimization_events.append({
                'timestamp': datetime.now(),
                'event': 'complete',
                'lineups_generated': lineups_generated,
                'time_taken': time_taken,
                'success_rate': success_rate
            })

    def log_lineup_generation(self, strategy: str, lineup_num: int,
                            status: str, constraints_applied: int):
        """Log individual lineup generation"""
        with self._lock:
            self.logs.append({
                'timestamp': datetime.now(),
                'level': 'DEBUG',
                'message': f"Lineup {lineup_num} ({strategy}): {status}",
                'constraints': constraints_applied
            })

    def display_log_summary(self):
        """Display log summary in streamlit with better formatting"""
        import streamlit as st

        with self._lock:
            st.markdown("### 📊 System Log Summary")

            # Error summary with patterns
            if self.error_logs:
                st.markdown("#### ❌ Recent Errors")

                # Group errors by pattern
                if self.error_patterns:
                    st.write("**Common Error Patterns:**")
                    for pattern, count in sorted(
                        self.error_patterns.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]:
                        st.write(f"• {pattern[:50]}... ({count} occurrences)")

                # Show recent errors
                for error in list(self.error_logs)[-5:]:
                    with st.expander(f"Error: {error['timestamp'].strftime('%H:%M:%S')}"):
                        st.write(error['message'])
                        if error.get('suggestions'):
                            st.info("💡 " + error['suggestions'][0])

            # Optimization summary
            if self.optimization_events:
                st.markdown("#### 🎯 Recent Optimizations")
                recent = [e for e in self.optimization_events if e['event'] == 'complete'][-5:]

                if recent:
                    df_data = []
                    for event in recent:
                        df_data.append({
                            'Time': event['timestamp'].strftime('%H:%M:%S'),
                            'Lineups': event.get('lineups_generated', 0),
                            'Duration': f"{event.get('time_taken', 0):.1f}s",
                            'Success Rate': f"{event.get('success_rate', 0):.0%}"
                        })

                    st.dataframe(pd.DataFrame(df_data))

            # AI decisions summary
            if self.ai_decisions:
                st.markdown("#### 🤖 AI Decision Summary")

                # Calculate success rates by AI type
                ai_stats = defaultdict(lambda: {'success': 0, 'total': 0})
                for decision in self.ai_decisions:
                    source = decision['source']
                    ai_stats[source]['total'] += 1
                    if decision['success']:
                        ai_stats[source]['success'] += 1

                col1, col2, col3 = st.columns(3)
                for i, (ai_type, stats) in enumerate(ai_stats.items()):
                    success_rate = stats['success'] / max(stats['total'], 1)
                    cols = [col1, col2, col3]
                    with cols[i % 3]:
                        st.metric(
                            ai_type,
                            f"{success_rate:.0%}",
                            f"{stats['total']} calls"
                        )

    def display_ai_enforcement(self):
        """Display AI enforcement summary"""
        import streamlit as st

        with self._lock:
            recent_decisions = list(self.ai_decisions)[-20:]
            if recent_decisions:
                enforcement_count = sum(1 for d in recent_decisions if d['success'])
                st.info(f"🤖 AI Enforcement: {enforcement_count}/{len(recent_decisions)} decisions applied")

class PerformanceMonitor:
    """Enhanced performance monitoring with better metrics"""

    def __init__(self):
        self.timers = {}
        self.metrics = defaultdict(list)
        self._lock = threading.RLock()
        self.start_times = {}

        # Enhanced tracking
        self.operation_counts = defaultdict(int)
        self.operation_times = defaultdict(list)
        self.memory_snapshots = deque(maxlen=10)

    def start_timer(self, operation: str):
        """Start timing an operation"""
        with self._lock:
            self.start_times[operation] = time.time()
            self.operation_counts[operation] += 1

    def stop_timer(self, operation: str) -> float:
        """Stop timing and return elapsed time"""
        with self._lock:
            if operation in self.start_times:
                elapsed = time.time() - self.start_times[operation]
                del self.start_times[operation]

                # Store timing data
                self.operation_times[operation].append(elapsed)

                # Keep only recent timings for memory management
                if len(self.operation_times[operation]) > 100:
                    self.operation_times[operation] = self.operation_times[operation][-50:]

                return elapsed
            return 0

    def record_metric(self, metric_name: str, value: float, tags: Dict = None):
        """Record a metric with optional tags"""
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
                'total_time': sum(times)
            }

    def display_metrics(self):
        """Display performance metrics in streamlit"""
        import streamlit as st

        with self._lock:
            st.markdown("### ⚡ Performance Metrics")

            # Operation statistics
            if self.operation_counts:
                st.markdown("#### Operation Performance")

                ops_data = []
                for op in self.operation_counts:
                    stats = self.get_operation_stats(op)
                    if stats:
                        ops_data.append({
                            'Operation': op,
                            'Count': stats['count'],
                            'Avg Time': f"{stats['avg_time']:.2f}s",
                            'Total Time': f"{stats['total_time']:.1f}s"
                        })

                if ops_data:
                    st.dataframe(pd.DataFrame(ops_data))

            # Recent metrics
            if self.metrics:
                st.markdown("#### Recent Metrics")

                for metric_name, values in list(self.metrics.items())[:5]:
                    if values:
                        recent = values[-10:]
                        avg_value = np.mean([v['value'] for v in recent])
                        st.metric(metric_name, f"{avg_value:.2f}", f"Last {len(recent)} samples")

class AIDecisionTracker:
    """Track AI decisions and learn from performance"""

    def __init__(self):
        self.decisions = deque(maxlen=OptimizerConfig.MAX_HISTORY_ENTRIES)
        self.performance_feedback = deque(maxlen=100)
        self.decision_patterns = defaultdict(list)
        self._lock = threading.RLock()

        # Learning components
        self.successful_patterns = defaultdict(float)
        self.failed_patterns = defaultdict(float)
        self.confidence_calibration = defaultdict(list)

    def track_decision(self, ai_type: AIStrategistType, decision: AIRecommendation,
                      context: Dict = None):
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

            # Track patterns
            pattern_key = self._extract_pattern(decision)
            self.decision_patterns[pattern_key].append(entry)

    def _extract_pattern(self, decision: AIRecommendation) -> str:
        """Extract pattern from decision for learning"""
        pattern_elements = [
            f"conf_{int(decision.confidence*10)}",
            f"capt_{len(decision.captain_targets)}",
            f"must_{len(decision.must_play)}",
            f"stack_{len(decision.stacks)}"
        ]
        return "_".join(pattern_elements)

    def record_performance(self, lineup: Dict, actual_score: Optional[float] = None):
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
                    'success': actual_score > projected * 1.1  # 10% better than projected
                }

                self.performance_feedback.append(entry)

                # Update pattern success rates
                pattern_key = f"{lineup.get('Strategy', 'unknown')}_{lineup.get('Ownership_Tier', 'unknown')}"

                if entry['success']:
                    self.successful_patterns[pattern_key] += 1
                else:
                    self.failed_patterns[pattern_key] += 1

                # Update confidence calibration
                confidence = lineup.get('Confidence', 0.5)
                self.confidence_calibration[int(confidence * 10)].append(accuracy)

    def get_learning_insights(self) -> Dict:
        """Get insights from tracked performance"""
        with self._lock:
            insights = {
                'total_decisions': len(self.decisions),
                'avg_confidence': np.mean([d['confidence'] for d in self.decisions]) if self.decisions else 0
            }

            # Calculate pattern success rates
            pattern_stats = {}
            for pattern in set(list(self.successful_patterns.keys()) + list(self.failed_patterns.keys())):
                successes = self.successful_patterns.get(pattern, 0)
                failures = self.failed_patterns.get(pattern, 0)
                total = successes + failures

                if total > 0:
                    pattern_stats[pattern] = {
                        'success_rate': successes / total,
                        'total': total
                    }

            insights['pattern_performance'] = pattern_stats

            # Confidence calibration
            calibration = {}
            for conf_level, accuracies in self.confidence_calibration.items():
                if accuracies:
                    calibration[conf_level / 10] = np.mean(accuracies)

            insights['confidence_calibration'] = calibration

            return insights

    def get_recommended_adjustments(self) -> Dict:
        """Get recommended adjustments based on learning"""
        insights = self.get_learning_insights()
        adjustments = {}

        # Recommend confidence adjustments
        calibration = insights.get('confidence_calibration', {})
        if calibration:
            for conf_level, actual_accuracy in calibration.items():
                if abs(conf_level - actual_accuracy) > 0.2:
                    adjustments[f'confidence_{conf_level}'] = actual_accuracy

        # Recommend pattern adjustments
        pattern_perf = insights.get('pattern_performance', {})
        for pattern, stats in pattern_perf.items():
            if stats['total'] >= 10:  # Enough data
                if stats['success_rate'] > 0.7:
                    adjustments[f'boost_{pattern}'] = 1.2
                elif stats['success_rate'] < 0.3:
                    adjustments[f'reduce_{pattern}'] = 0.8

        return adjustments

# ============================================================================
# END OF PART 2
# ============================================================================

# ============================================================================
# PART 3: CORE ENFORCEMENT ENGINE, VALIDATORS, AND SYNTHESIS COMPONENTS
# ============================================================================

class AIEnforcementEngine:
    """Enhanced enforcement engine with advanced stacking rules"""

    def __init__(self, enforcement_level: AIEnforcementLevel = AIEnforcementLevel.STRONG):
        self.enforcement_level = enforcement_level
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()

        # Enhanced rule tracking
        self.applied_rules = deque(maxlen=100)
        self.rule_success_rate = defaultdict(float)
        self.violation_patterns = defaultdict(int)

    def create_enforcement_rules(self, recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """Create comprehensive enforcement rules from AI recommendations"""

        self.logger.log(f"Creating enforcement rules at {self.enforcement_level.value} level", "INFO")

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

        # Sort by priority
        for rule_type in rules:
            if isinstance(rules[rule_type], list):
                rules[rule_type].sort(key=lambda x: x.get('priority', 0), reverse=True)

        self.logger.log(f"Created {sum(len(v) for v in rules.values() if isinstance(v, list))} rules", "INFO")

        return rules

    def _create_mandatory_rules(self, recommendations: Dict) -> Dict:
        """Create mandatory enforcement rules (all AI decisions enforced)"""
        rules = {
            'hard_constraints': [],
            'soft_constraints': [],
            'stacking_rules': [],
            'ownership_rules': [],
            'correlation_rules': []
        }

        for ai_type, rec in recommendations.items():
            weight = OptimizerConfig.AI_WEIGHTS.get(ai_type.value.lower().replace(' ', '_'), 0.33)

            # Captain constraints (highest priority)
            if rec.captain_targets:
                rules['hard_constraints'].append({
                    'rule': 'captain_from_list',
                    'players': rec.captain_targets[:7],
                    'source': ai_type.value,
                    'priority': int(100 * weight * rec.confidence),
                    'type': 'hard'
                })

            # Must play constraints
            for player in rec.must_play[:3]:
                rules['hard_constraints'].append({
                    'rule': 'must_include',
                    'player': player,
                    'source': ai_type.value,
                    'priority': int(90 * weight * rec.confidence),
                    'type': 'hard'
                })

            # Never play constraints
            for player in rec.never_play[:3]:
                rules['hard_constraints'].append({
                    'rule': 'must_exclude',
                    'player': player,
                    'source': ai_type.value,
                    'priority': int(85 * weight * rec.confidence),
                    'type': 'hard'
                })

            # Stack constraints
            for stack in rec.stacks[:3]:
                rules['stacking_rules'].append({
                    'rule': 'must_stack',
                    'stack': stack,
                    'source': ai_type.value,
                    'priority': int(80 * weight * rec.confidence),
                    'type': 'hard'
                })

        return rules

    def _create_strong_rules(self, recommendations: Dict) -> Dict:
        """Create strong enforcement rules (most AI decisions enforced)"""
        rules = self._create_moderate_rules(recommendations)

        # Upgrade some soft constraints to hard
        for ai_type, rec in recommendations.items():
            if rec.confidence > 0.7:
                # High confidence recommendations become hard constraints
                weight = OptimizerConfig.AI_WEIGHTS.get(ai_type.value.lower().replace(' ', '_'), 0.33)

                if rec.captain_targets:
                    rules['hard_constraints'].append({
                        'rule': 'captain_selection',
                        'players': rec.captain_targets[:5],
                        'source': ai_type.value,
                        'priority': int(95 * weight * rec.confidence),
                        'type': 'hard'
                    })

        return rules

    def _create_moderate_rules(self, recommendations: Dict) -> Dict:
        """Create moderate enforcement rules (balanced approach)"""
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

        # Enforce consensus as hard constraints
        for captain, count in captain_counts.items():
            if count >= 2:  # At least 2 AIs agree
                rules['hard_constraints'].append({
                    'rule': 'consensus_captain',
                    'player': captain,
                    'agreement': count,
                    'priority': 90 + count * 5,
                    'type': 'hard'
                })

        # Add soft constraints for single AI recommendations
        for ai_type, rec in recommendations.items():
            weight = OptimizerConfig.AI_WEIGHTS.get(ai_type.value.lower().replace(' ', '_'), 0.33)

            for player in rec.must_play[:3]:
                if must_play_counts[player] == 1:  # Only one AI suggested
                    rules['soft_constraints'].append({
                        'rule': 'prefer_player',
                        'player': player,
                        'source': ai_type.value,
                        'weight': weight * rec.confidence,
                        'priority': int(70 * weight * rec.confidence),
                        'type': 'soft'
                    })

        return rules

    def _create_advisory_rules(self, recommendations: Dict) -> Dict:
        """Create advisory rules (suggestions only)"""
        rules = {
            'hard_constraints': [],
            'soft_constraints': [],
            'stacking_rules': [],
            'ownership_rules': [],
            'correlation_rules': []
        }

        # All recommendations become soft constraints
        for ai_type, rec in recommendations.items():
            weight = OptimizerConfig.AI_WEIGHTS.get(ai_type.value.lower().replace(' ', '_'), 0.33)

            # Captain preferences
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
        """Create advanced stacking rules including onslaught, leverage, and defensive stacks"""
        stacking_rules = []

        for ai_type, rec in recommendations.items():
            for stack in rec.stacks:
                stack_type = stack.get('type', 'standard')

                if stack_type == 'onslaught':
                    # 3-4+ players from same team
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

                elif stack_type == 'bring_back':
                    # Primary stack with opposing player
                    stacking_rules.append({
                        'rule': 'bring_back_stack',
                        'primary_players': stack.get('primary_stack', []),
                        'bring_back_player': stack.get('bring_back'),
                        'game_total': stack.get('game_total', 45),
                        'priority': 80,
                        'source': ai_type.value,
                        'correlation_strength': 0.5
                    })

                elif stack_type == 'leverage':
                    # Low ownership correlated plays
                    stacking_rules.append({
                        'rule': 'leverage_stack',
                        'players': [stack.get('player1'), stack.get('player2')],
                        'combined_ownership_max': stack.get('combined_ownership', 20),
                        'leverage_score_min': 3.0,
                        'priority': 75,
                        'source': ai_type.value,
                        'correlation_strength': 0.4
                    })

                elif stack_type == 'defensive':
                    # DST with negative game script
                    stacking_rules.append({
                        'rule': 'defensive_stack',
                        'dst_team': stack.get('dst_team'),
                        'opposing_players_max': 1,
                        'scenario': 'defensive_game',
                        'priority': 70,
                        'source': ai_type.value
                    })

                elif stack_type == 'hidden':
                    # Non-obvious correlations
                    stacking_rules.append({
                        'rule': 'hidden_correlation',
                        'players': [stack.get('player1'), stack.get('player2')],
                        'narrative': stack.get('narrative', 'hidden connection'),
                        'priority': 65,
                        'source': ai_type.value,
                        'correlation_strength': 0.35
                    })

                else:
                    # Standard QB stack
                    if 'player1' in stack and 'player2' in stack:
                        stacking_rules.append({
                            'rule': 'standard_stack',
                            'players': [stack['player1'], stack['player2']],
                            'correlation': stack.get('correlation', 0.5),
                            'priority': 70,
                            'source': ai_type.value
                        })

        # Remove duplicate stacks
        unique_stacks = []
        seen = set()

        for stack in stacking_rules:
            # Create unique identifier
            players = stack.get('players', [])
            if players:
                stack_id = "_".join(sorted(players[:2]))
                if stack_id not in seen:
                    seen.add(stack_id)
                    unique_stacks.append(stack)
            else:
                unique_stacks.append(stack)  # Non-standard stacks

        return unique_stacks

    def validate_lineup_against_ai(self, lineup: Dict, enforcement_rules: Dict) -> Tuple[bool, List[str]]:
        """Validate lineup against AI enforcement rules"""
        violations = []
        captain = lineup.get('Captain')
        flex = lineup.get('FLEX', [])
        all_players = [captain] + flex

        # Check hard constraints
        for rule in enforcement_rules.get('hard_constraints', []):
            if rule['rule'] == 'captain_from_list':
                if captain not in rule['players']:
                    violations.append(f"Captain not in AI list: {captain}")

            elif rule['rule'] == 'must_include':
                if rule['player'] not in all_players:
                    violations.append(f"Missing required player: {rule['player']}")

            elif rule['rule'] == 'must_exclude':
                if rule['player'] in all_players:
                    violations.append(f"Included banned player: {rule['player']}")

        # Check stacking rules
        for stack_rule in enforcement_rules.get('stacking_rules', []):
            if stack_rule.get('type') == 'hard':
                if not self._validate_stack_rule(all_players, stack_rule):
                    violations.append(f"Stack rule violation: {stack_rule.get('rule')}")

        # Track violations for learning
        for violation in violations:
            self.violation_patterns[violation[:50]] += 1

        is_valid = len(violations) == 0

        # Record rule application
        self.applied_rules.append({
            'timestamp': datetime.now(),
            'lineup_num': lineup.get('Lineup', 0),
            'valid': is_valid,
            'violations': len(violations)
        })

        return is_valid, violations

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
            return all(p in players for p in required)

        return True

class AIOwnershipBucketManager:
    """Enhanced ownership bucket management with better calculations"""

    def __init__(self, enforcement_engine: AIEnforcementEngine = None):
        self.enforcement_engine = enforcement_engine
        self.logger = get_logger()

        # Enhanced bucket definitions
        self.bucket_thresholds = {
            'mega_chalk': 35,
            'chalk': 20,
            'moderate': 15,
            'pivot': 10,
            'leverage': 5,
            'super_leverage': 2
        }

    def categorize_players(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize players into ownership buckets with enhanced logic"""
        buckets = {
            'mega_chalk': [],
            'chalk': [],
            'moderate': [],
            'pivot': [],
            'leverage': [],
            'super_leverage': []
        }

        for _, row in df.iterrows():
            player = row['Player']
            ownership = row.get('Ownership', 10)

            if ownership >= self.bucket_thresholds['mega_chalk']:
                buckets['mega_chalk'].append(player)
            elif ownership >= self.bucket_thresholds['chalk']:
                buckets['chalk'].append(player)
            elif ownership >= self.bucket_thresholds['moderate']:
                buckets['moderate'].append(player)
            elif ownership >= self.bucket_thresholds['pivot']:
                buckets['pivot'].append(player)
            elif ownership >= self.bucket_thresholds['leverage']:
                buckets['leverage'].append(player)
            else:
                buckets['super_leverage'].append(player)

        return buckets

    def calculate_gpp_leverage(self, players: List[str], df: pd.DataFrame) -> float:
        """Enhanced GPP leverage score calculation"""
        if not players:
            return 0

        total_projection = 0
        total_ownership = 0
        leverage_bonus = 0

        for player in players:
            player_data = df[df['Player'] == player]
            if not player_data.empty:
                row = player_data.iloc[0]
                projection = row.get('Projected_Points', 0)
                ownership = row.get('Ownership', 10)

                # Captain gets 1.5x weight
                if player == players[0]:  # Assuming first player is captain
                    projection *= 1.5
                    ownership *= 1.5

                total_projection += projection
                total_ownership += ownership

                # Bonus for leverage plays
                if ownership < self.bucket_thresholds['leverage']:
                    leverage_bonus += 10
                elif ownership < self.bucket_thresholds['pivot']:
                    leverage_bonus += 5

        # Calculate leverage score
        if total_ownership > 0:
            # Higher projection + lower ownership = higher leverage
            base_leverage = (total_projection / len(players)) / (total_ownership / len(players) + 1)
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

class AIConfigValidator:
    """Enhanced validator with dynamic strategy selection"""

    @staticmethod
    def validate_ai_requirements(enforcement_rules: Dict, df: pd.DataFrame) -> Dict:
        """Validate that AI requirements are feasible"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }

        available_players = set(df['Player'].values)

        # Check captain requirements
        captain_rules = [r for r in enforcement_rules.get('hard_constraints', [])
                        if r.get('rule') in ['captain_from_list', 'captain_selection']]

        for rule in captain_rules:
            valid_captains = [p for p in rule.get('players', []) if p in available_players]

            if not valid_captains:
                validation_result['errors'].append("No valid captains from AI recommendations")
                validation_result['is_valid'] = False
                validation_result['suggestions'].append("Relax captain constraints or check player pool")
            elif len(valid_captains) < 3:
                validation_result['warnings'].append(f"Only {len(valid_captains)} valid captains available")
                validation_result['suggestions'].append("Consider expanding captain pool")

        # Check must include players
        must_include = [r for r in enforcement_rules.get('hard_constraints', [])
                       if r.get('rule') == 'must_include']

        for rule in must_include:
            if rule.get('player') not in available_players:
                validation_result['errors'].append(f"Required player not available: {rule.get('player')}")
                validation_result['is_valid'] = False

        # Check stack feasibility
        stacking_rules = enforcement_rules.get('stacking_rules', [])

        for stack in stacking_rules:
            if stack.get('rule') == 'onslaught_stack':
                players = stack.get('players', [])
                valid = [p for p in players if p in available_players]

                if len(valid) < stack.get('min_players', 3):
                    validation_result['warnings'].append("Onslaught stack may not be feasible")

        # Check salary feasibility
        hard_constraints = enforcement_rules.get('hard_constraints', [])
        required_players = [r.get('player') for r in hard_constraints
                          if r.get('rule') == 'must_include' and r.get('player')]

        if required_players:
            min_required_salary = df[df['Player'].isin(required_players)]['Salary'].sum()

            if min_required_salary > OptimizerConfig.SALARY_CAP * 0.6:
                validation_result['warnings'].append("Required players use >60% of salary cap")
                validation_result['suggestions'].append("May have limited flexibility for other positions")

        return validation_result

    @staticmethod
    def get_ai_strategy_distribution(field_size: str, num_lineups: int,
                                    consensus_level: str = 'mixed') -> Dict:
        """Enhanced dynamic strategy distribution based on consensus"""

        # Base distribution by field size
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
        )

        # Adjust based on consensus level
        if consensus_level == 'high':
            # More balanced when AIs agree
            distribution['balanced'] = min(distribution.get('balanced', 0.3) * 1.3, 0.5)
            distribution['contrarian'] = distribution.get('contrarian', 0.2) * 0.7
        elif consensus_level == 'low':
            # More variety when AIs disagree
            distribution['contrarian'] = min(distribution.get('contrarian', 0.2) * 1.3, 0.4)
            distribution['balanced'] = distribution.get('balanced', 0.3) * 0.7

        # Normalize
        total = sum(distribution.values())
        distribution = {k: v/total for k, v in distribution.items()}

        # Convert to lineup counts
        lineup_distribution = {}
        allocated = 0

        for strategy, pct in distribution.items():
            count = int(num_lineups * pct)
            lineup_distribution[strategy] = count
            allocated += count

        # Allocate remaining lineups
        if allocated < num_lineups:
            lineup_distribution['balanced'] = lineup_distribution.get('balanced', 0) + (num_lineups - allocated)

        return lineup_distribution

class AISynthesisEngine:
    """Enhanced synthesis engine with pattern analysis"""

    def __init__(self):
        self.logger = get_logger()
        self.synthesis_history = deque(maxlen=20)

    def synthesize_recommendations(self, game_theory: AIRecommendation,
                                  correlation: AIRecommendation,
                                  contrarian: AIRecommendation) -> Dict:
        """Enhanced synthesis with pattern recognition"""

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

        # Captain synthesis with consensus tracking
        captain_votes = defaultdict(list)

        for ai_type, rec in [
            (AIStrategistType.GAME_THEORY, game_theory),
            (AIStrategistType.CORRELATION, correlation),
            (AIStrategistType.CONTRARIAN_NARRATIVE, contrarian)
        ]:
            for captain in rec.captain_targets[:5]:
                captain_votes[captain].append(ai_type.value)

        # Classify captains by consensus
        for captain, votes in captain_votes.items():
            if len(votes) == 3:
                synthesis['captain_strategy'][captain] = 'consensus'
            elif len(votes) == 2:
                synthesis['captain_strategy'][captain] = 'majority'
            else:
                synthesis['captain_strategy'][captain] = votes[0]

        # Player rankings synthesis
        player_scores = defaultdict(float)

        # Weight recommendations
        weights = {
            AIStrategistType.GAME_THEORY: OptimizerConfig.AI_WEIGHTS.get('game_theory', 0.33),
            AIStrategistType.CORRELATION: OptimizerConfig.AI_WEIGHTS.get('correlation', 0.33),
            AIStrategistType.CONTRARIAN_NARRATIVE: OptimizerConfig.AI_WEIGHTS.get('contrarian', 0

AIStrategistType.CONTRARIAN_NARRATIVE: OptimizerConfig.AI_WEIGHTS.get('contrarian', 0.34)
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
                synthesis['player_rankings'] = {
                    player: score / max_score
                    for player, score in player_scores.items()
                }

        # Stack synthesis - combine all unique stacks
        all_stacks = []
        stack_patterns = defaultdict(int)

        for rec in [game_theory, correlation, contrarian]:
            for stack in rec.stacks:
                all_stacks.append(stack)

                # Track patterns
                stack_type = stack.get('type', 'standard')
                stack_patterns[stack_type] += 1

        # Prioritize stacks that appear multiple times or have high confidence
        synthesis['stacking_rules'] = self._prioritize_stacks(all_stacks)

        # Pattern analysis
        synthesis['patterns'] = self._analyze_patterns(
            game_theory, correlation, contrarian, stack_patterns
        )

        # Calculate overall confidence
        confidences = [
            game_theory.confidence * weights[AIStrategistType.GAME_THEORY],
            correlation.confidence * weights[AIStrategistType.CORRELATION],
            contrarian.confidence * weights[AIStrategistType.CONTRARIAN_NARRATIVE]
        ]
        synthesis['confidence'] = sum(confidences)

        # Create narrative
        narratives = []
        if game_theory.narrative:
            narratives.append(f"Game Theory: {game_theory.narrative[:100]}")
        if correlation.narrative:
            narratives.append(f"Correlation: {correlation.narrative[:100]}")
        if contrarian.narrative:
            narratives.append(f"Contrarian: {contrarian.narrative[:100]}")

        synthesis['narrative'] = " | ".join(narratives)

        # Enforcement rules synthesis
        synthesis['enforcement_rules'] = self._synthesize_enforcement_rules(
            game_theory, correlation, contrarian
        )

        # Store in history
        self.synthesis_history.append({
            'timestamp': datetime.now(),
            'confidence': synthesis['confidence'],
            'captain_count': len(synthesis['captain_strategy']),
            'patterns': synthesis['patterns']
        })

        return synthesis

    def _prioritize_stacks(self, all_stacks: List[Dict]) -> List[Dict]:
        """Prioritize and deduplicate stacks"""

        # Group similar stacks
        stack_groups = defaultdict(list)

        for stack in all_stacks:
            # Create a key for grouping
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
                # Sort by correlation strength or confidence
                best = max(group, key=lambda s: s.get('correlation', 0.5))

                # Boost priority if multiple AIs suggested
                if len(group) > 1:
                    best['consensus'] = True
                    best['priority'] = best.get('priority', 50) + 10 * len(group)

                prioritized.append(best)

        # Sort by priority
        prioritized.sort(key=lambda s: s.get('priority', 50), reverse=True)

        return prioritized[:10]  # Top 10 stacks

    def _analyze_patterns(self, game_theory: AIRecommendation,
                         correlation: AIRecommendation,
                         contrarian: AIRecommendation,
                         stack_patterns: Dict) -> List[str]:
        """Analyze patterns in AI recommendations"""
        patterns = []

        # Check for unanimous agreement
        captain_overlap = set(game_theory.captain_targets) & \
                         set(correlation.captain_targets) & \
                         set(contrarian.captain_targets)

        if captain_overlap:
            patterns.append(f"Strong consensus on {len(captain_overlap)} captains")

        # Check for contrarian dominance
        if contrarian.confidence > max(game_theory.confidence, correlation.confidence):
            patterns.append("Contrarian approach favored")

        # Stack pattern analysis
        if stack_patterns.get('onslaught', 0) > 1:
            patterns.append("Multiple onslaught stacks recommended")

        if stack_patterns.get('bring_back', 0) > 0:
            patterns.append("Bring-back correlation identified")

        # Ownership pattern
        avg_ownership_targets = []
        for rec in [game_theory, correlation, contrarian]:
            if hasattr(rec, 'ownership_leverage'):
                target = rec.ownership_leverage.get('ownership_ceiling', 100)
                avg_ownership_targets.append(target)

        if avg_ownership_targets:
            avg_target = np.mean(avg_ownership_targets)
            if avg_target < 15:
                patterns.append("Ultra-contrarian ownership profile")
            elif avg_target < 25:
                patterns.append("Leverage-focused ownership")

        return patterns

    def _synthesize_enforcement_rules(self, game_theory: AIRecommendation,
                                     correlation: AIRecommendation,
                                     contrarian: AIRecommendation) -> List[Dict]:
        """Synthesize enforcement rules from all AIs"""
        rules = []

        # Combine all enforcement rules
        all_rules = (
            game_theory.enforcement_rules +
            correlation.enforcement_rules +
            contrarian.enforcement_rules
        )

        # Group similar rules
        rule_groups = defaultdict(list)

        for rule in all_rules:
            key = f"{rule.get('type')}_{rule.get('constraint')}"
            rule_groups[key].append(rule)

        # Consolidate groups
        for group in rule_groups.values():
            if len(group) > 1:
                # Multiple AIs suggest similar rule - increase priority
                consolidated = group[0].copy()
                consolidated['priority'] = max(r.get('priority', 50) for r in group) + 5 * len(group)
                consolidated['consensus_count'] = len(group)
                rules.append(consolidated)
            else:
                rules.append(group[0])

        # Sort by priority
        rules.sort(key=lambda r: r.get('priority', 50), reverse=True)

        return rules[:20]  # Top 20 rules

    def get_synthesis_quality_score(self) -> float:
        """Calculate quality score of recent syntheses"""
        if not self.synthesis_history:
            return 0.5

        recent = list(self.synthesis_history)[-5:]

        # Average confidence
        avg_confidence = np.mean([s['confidence'] for s in recent])

        # Captain diversity
        captain_counts = [s['captain_count'] for s in recent]
        captain_diversity = np.std(captain_counts) / (np.mean(captain_counts) + 1)

        # Pattern consistency
        all_patterns = []
        for s in recent:
            all_patterns.extend(s.get('patterns', []))

        unique_patterns = len(set(all_patterns))
        pattern_score = unique_patterns / max(len(all_patterns), 1)

        # Combine scores
        quality = (
            avg_confidence * 0.5 +
            min(captain_diversity, 0.3) +
            pattern_score * 0.2
        )

        return min(quality, 1.0)

# ============================================================================
# END OF PART 3
# ============================================================================

# ============================================================================
# BASE AI STRATEGIST CLASS - ENHANCED
# ============================================================================

class BaseAIStrategist:
    """Enhanced base class for all AI strategists with robust error handling"""

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

        # Fallback confidence levels
        self.fallback_confidence = {
            AIStrategistType.GAME_THEORY: 0.5,
            AIStrategistType.CORRELATION: 0.55,
            AIStrategistType.CONTRARIAN_NARRATIVE: 0.45
        }

        # Adaptive confidence based on performance
        self.adaptive_confidence_modifier = 1.0

    def get_recommendation(self, df: pd.DataFrame, game_info: Dict,
                          field_size: str, use_api: bool = True) -> AIRecommendation:
        """Get AI recommendation with comprehensive error handling and learning"""

        try:
            # Validate inputs
            if df.empty:
                self.logger.log(f"{self.strategist_type.value}: Empty DataFrame provided", "ERROR")
                return self._get_fallback_recommendation(df, field_size)

            # Analyze slate characteristics for dynamic adjustment
            slate_profile = self._analyze_slate_profile(df, game_info)

            # Generate cache key
            cache_key = self._generate_cache_key(df, game_info, field_size)

            # Check cache
            with self._cache_lock:
                if cache_key in self.response_cache:
                    self.logger.log(f"{self.strategist_type.value}: Using cached recommendation", "DEBUG")
                    cached_rec = self.response_cache[cache_key]
                    # Apply adaptive confidence
                    cached_rec.confidence *= self.adaptive_confidence_modifier
                    return cached_rec

            # Generate prompt with slate context
            prompt = self.generate_prompt(df, game_info, field_size, slate_profile)

            # Get response (API or manual)
            if use_api and self.api_manager and self.api_manager.client:
                response = self._get_api_response(prompt)
            else:
                response = self._get_fallback_response(df, game_info, field_size, slate_profile)

            # Parse response into recommendation
            recommendation = self.parse_response(response, df, field_size)

            # Apply adaptive learning adjustments
            recommendation = self._apply_learned_adjustments(recommendation, slate_profile)

            # Validate recommendation
            is_valid, errors = recommendation.validate()
            if not is_valid:
                self.logger.log(f"{self.strategist_type.value} validation errors: {errors}", "WARNING")
                recommendation = self._correct_recommendation(recommendation, df)

            # Add enforcement rules
            recommendation.enforcement_rules = self.create_enforcement_rules(
                recommendation, df, field_size, slate_profile
            )

            # Cache the result
            with self._cache_lock:
                self.response_cache[cache_key] = recommendation
                if len(self.response_cache) > self.max_cache_size:
                    for key in list(self.response_cache.keys())[:5]:
                        del self.response_cache[key]

            return recommendation

        except Exception as e:
            self.logger.log_exception(e, f"{self.strategist_type.value}.get_recommendation")
            return self._get_fallback_recommendation(df, field_size)

    def _analyze_slate_profile(self, df: pd.DataFrame, game_info: Dict) -> Dict:
        """Analyze slate characteristics for dynamic strategy adjustment"""
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
            'value_distribution': df['Projected_Points'].std() / df['Projected_Points'].mean(),
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

        # Adjust confidence based on slate type and historical performance
        slate_type = slate_profile.get('slate_type', 'standard')
        if slate_type in self.successful_patterns:
            confidence_boost = self.successful_patterns[slate_type] * 0.1
            recommendation.confidence = min(0.95, recommendation.confidence + confidence_boost)

        # Apply adaptive confidence modifier
        recommendation.confidence *= self.adaptive_confidence_modifier

        # Adjust captain targets based on slate profile
        if slate_type == 'shootout':
            # Prioritize pass catchers in shootouts
            qbs_and_receivers = set()
            for player in recommendation.captain_targets:
                player_data = self.df[self.df['Player'] == player]
                if not player_data.empty:
                    if player_data.iloc[0]['Position'] in ['QB', 'WR', 'TE']:
                        qbs_and_receivers.add(player)

            if qbs_and_receivers:
                # Move QB/pass catchers to front
                recommendation.captain_targets = list(qbs_and_receivers) + [
                    p for p in recommendation.captain_targets if p not in qbs_and_receivers
                ]

        return recommendation

    def track_performance(self, lineup: Dict, actual_points: float = None):
        """Track lineup performance for learning"""
        if actual_points is not None:
            performance_data = {
                'strategy': self.strategist_type.value,
                'projected': lineup.get('Projected', 0),
                'actual': actual_points,
                'accuracy': 1 - abs(actual_points - lineup.get('Projected', 0)) / actual_points,
                'timestamp': datetime.now()
            }

            self.performance_history.append(performance_data)

            # Update adaptive confidence based on recent performance
            if len(self.performance_history) >= 10:
                recent_accuracy = np.mean([p['accuracy'] for p in list(self.performance_history)[-10:]])
                self.adaptive_confidence_modifier = 0.5 + recent_accuracy  # Range 0.5 to 1.5

    def create_enforcement_rules(self, recommendation: AIRecommendation,
                                df: pd.DataFrame, field_size: str,
                                slate_profile: Dict) -> List[Dict]:
        """Create specific enforcement rules with slate context"""
        rules = []

        # Validate players exist in DataFrame
        available_players = set(df['Player'].values)

        # Dynamic priority based on slate type
        slate_type = slate_profile.get('slate_type', 'standard')
        priority_modifier = {
            'shootout': 1.2,
            'low_scoring': 0.8,
            'blowout_risk': 1.1,
            'flat_pricing': 0.9,
            'standard': 1.0
        }.get(slate_type, 1.0)

        # Captain enforcement with dynamic priority
        valid_captains = [c for c in recommendation.captain_targets if c in available_players]

        if valid_captains:
            base_priority = int(recommendation.confidence * 100)
            adjusted_priority = int(base_priority * priority_modifier)

            if recommendation.confidence > 0.8:
                rules.append({
                    'type': 'hard',
                    'constraint': f'captain_in_{self.strategist_type.value}',
                    'players': valid_captains[:5],
                    'priority': adjusted_priority,
                    'description': f'{self.strategist_type.value}: High-confidence captains',
                    'slate_context': slate_type
                })
            else:
                rules.append({
                    'type': 'soft',
                    'constraint': f'prefer_captain_{self.strategist_type.value}',
                    'players': valid_captains[:5],
                    'weight': recommendation.confidence,
                    'priority': int(adjusted_priority * 0.7),
                    'description': f'{self.strategist_type.value}: Preferred captains',
                    'slate_context': slate_type
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
                'priority': int((recommendation.confidence - i * 0.1) * 50 * priority_modifier),
                'description': f'{self.strategist_type.value}: Include {player}'
            })

        return rules

    # Abstract methods to be implemented by child classes
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str,
                       slate_profile: Dict) -> str:
        """Generate prompt - to be overridden by subclasses"""
        raise NotImplementedError

    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Parse response - to be overridden by subclasses"""
        raise NotImplementedError

# ============================================================================
# GPP GAME THEORY STRATEGIST - COMPLETE IMPLEMENTATION
# ============================================================================

class GPPGameTheoryStrategist(BaseAIStrategist):
    """AI Strategist 1: Game Theory and Ownership Leverage - Complete"""

    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.GAME_THEORY)
        self.df = None  # Store for reference in adjustments

    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str,
                       slate_profile: Dict) -> str:
        """Generate game theory focused prompt with slate context"""

        self.logger.log(f"Generating Game Theory prompt for {field_size}", "DEBUG")
        self.df = df  # Store for later use

        if df.empty:
            return "Error: Empty player pool"

        # Prepare ownership analysis
        bucket_manager = AIOwnershipBucketManager()
        buckets = bucket_manager.categorize_players(df)

        # Get low-owned high-upside plays
        low_owned_high_upside = df[df.get('Ownership', 10) < 10].nlargest(10, 'Projected_Points')

        # Field-specific strategies based on slate profile
        field_strategies = {
            'small_field': "Focus on slight differentiation while maintaining optimal plays",
            'medium_field': "Balance chalk with 2-3 strong leverage plays",
            'large_field': "Aggressive leverage with <15% owned captains",
            'large_field_aggressive': "Ultra-leverage approach with <10% captains",
            'milly_maker': "Maximum contrarian approach with <10% captains only",
            'super_contrarian': "Extreme leverage targeting <5% ownership"
        }

        # Adjust strategy based on slate type
        slate_adjustments = {
            'shootout': "Prioritize ceiling over floor, embrace variance",
            'low_scoring': "Target TD-dependent players for leverage",
            'blowout_risk': "Fade favorites heavily, target garbage time",
            'flat_pricing': "Ownership becomes primary differentiator",
            'standard': "Balanced approach with calculated risks"
        }

        prompt = f"""
        You are an expert DFS game theory strategist. Create an ENFORCEABLE lineup strategy for {field_size} GPP tournaments.

        GAME CONTEXT:
        Teams: {game_info.get('teams', 'Unknown')}
        Total: {game_info.get('total', 45)}
        Spread: {game_info.get('spread', 0)}
        Weather: {game_info.get('weather', 'Clear')}
        Slate Type: {slate_profile.get('slate_type', 'standard')}

        OWNERSHIP LANDSCAPE:
        Mega Chalk (>35%): {len(buckets['mega_chalk'])} players
        Chalk (20-35%): {len(buckets['chalk'])} players
        Pivot (10-20%): {len(buckets['pivot'])} players
        Leverage (5-10%): {len(buckets['leverage'])} players
        Super Leverage (<5%): {len(buckets['super_leverage'])} players

        HIGH LEVERAGE PLAYS (<10% ownership):
        {low_owned_high_upside[['Player', 'Position', 'Salary', 'Projected_Points', 'Ownership']].to_string()}

        FIELD STRATEGY:
        {field_strategies.get(field_size, 'Standard GPP strategy')}
        {slate_adjustments.get(slate_profile.get('slate_type', 'standard'), '')}

        PROVIDE SPECIFIC, ENFORCEABLE RULES IN JSON:
        {{
            "captain_rules": {{
                "must_be_one_of": ["exact_player_names"],
                "ownership_ceiling": 15,
                "min_projection": 15,
                "leverage_score_min": 3,
                "reasoning": "Why these specific captains win tournaments"
            }},
            "lineup_rules": {{
                "must_include": ["player_names"],
                "never_include": ["player_names"],
                "ownership_sum_range": [60, 90],
                "min_leverage_players": 2,
                "max_chalk_players": 2
            }},
            "correlation_rules": {{
                "required_stacks": [{{"player1": "name", "player2": "name", "type": "leverage"}}],
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

        Focus on ownership leverage, field tendencies, and exploitable patterns.
        """

        return prompt

    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Complete implementation of game theory response parsing"""

        try:
            # Try to parse JSON response
            if response and response != '{}':
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    self.logger.log("Failed to parse JSON, attempting text extraction", "WARNING")
                    data = self._extract_from_text_response(response)
            else:
                data = {}

            available_players = set(df['Player'].values)

            # Extract captain rules with validation
            captain_rules = data.get('captain_rules', {})
            captain_targets = captain_rules.get('must_be_one_of', [])
            valid_captains = [c for c in captain_targets if c in available_players]

            # If not enough valid captains, use game theory selection
            if len(valid_captains) < 3:
                ownership_ceiling = captain_rules.get('ownership_ceiling', 15)
                min_proj = captain_rules.get('min_projection', 15)

                # Filter by ownership and projection
                eligible = df[
                    (df.get('Ownership', 10) <= ownership_ceiling) &
                    (df['Projected_Points'] >= min_proj)
                ]

                if len(eligible) < 5:
                    eligible = df[df.get('Ownership', 10) <= ownership_ceiling * 1.5]

                # Calculate leverage score for each
                eligible = eligible.copy()
                eligible['Leverage_Score'] = (
                    eligible['Projected_Points'] / eligible['Projected_Points'].max() * 100 /
                    (eligible.get('Ownership', 10) + 5)
                )

                # Get top leverage plays
                leverage_captains = eligible.nlargest(5, 'Leverage_Score')['Player'].tolist()
                for captain in leverage_captains:
                    if captain not in valid_captains:
                        valid_captains.append(captain)
                    if len(valid_captains) >= 5:
                        break

            # Extract lineup rules
            lineup_rules = data.get('lineup_rules', {})
            must_include = [p for p in lineup_rules.get('must_include', []) if p in available_players]
            never_include = [p for p in lineup_rules.get('never_include', []) if p in available_players]
            ownership_range = lineup_rules.get('ownership_sum_range', [60, 90])
            min_leverage = lineup_rules.get('min_leverage_players', 2)

            # Extract correlation rules and create stacks
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

            # Extract game theory insights
            game_theory = data.get('game_theory_insights', {})
            key_insights = [
                data.get('key_insight', 'Ownership arbitrage opportunity'),
                game_theory.get('exploit_angle', ''),
                game_theory.get('unique_construction', ''),
                f"Target {ownership_range[0]}-{ownership_range[1]}% total ownership"
            ]
            key_insights = [i for i in key_insights if i]

            # Build comprehensive enforcement rules
            enforcement_rules = self._build_game_theory_enforcement_rules(
                valid_captains, must_include, never_include,
                ownership_range, min_leverage, stacks
            )

            confidence = data.get('confidence', 0.75)
            confidence = max(0.0, min(1.0, confidence))

            return AIRecommendation(
                captain_targets=valid_captains[:7],
                must_play=must_include[:5],
                never_play=never_include[:5],
                stacks=stacks[:5],
                key_insights=key_insights[:3],
                confidence=confidence,
                enforcement_rules=enforcement_rules,
                narrative=game_theory.get('exploit_angle', 'Game theory optimization'),
                source_ai=AIStrategistType.GAME_THEORY,
                ownership_leverage={
                    'ownership_range': ownership_range,
                    'ownership_ceiling': captain_rules.get('ownership_ceiling', 15),
                    'min_leverage': min_leverage,
                    'max_chalk': lineup_rules.get('max_chalk_players', 2)
                }
            )

        except Exception as e:
            self.logger.log_exception(e, "parse_game_theory_response")
            return self._get_fallback_recommendation(df, field_size)

    def _extract_from_text_response(self, response: str) -> Dict:
        """Extract structured data from non-JSON response"""
        data = {
            'captain_rules': {},
            'lineup_rules': {},
            'confidence': 0.6
        }

        # Simple text parsing for key patterns
        lines = response.lower().split('\n')

        for line in lines:
            if 'captain' in line and any(char.isdigit() for char in line):
                # Extract ownership numbers
                numbers = [int(s) for s in line.split() if s.isdigit()]
                if numbers:
                    data['captain_rules']['ownership_ceiling'] = min(numbers)

            if 'leverage' in line and 'players' in line:
                numbers = [int(s) for s in line.split() if s.isdigit()]
                if numbers:
                    data['lineup_rules']['min_leverage_players'] = max(1, min(numbers))

        return data

    def _build_game_theory_enforcement_rules(self, captains, must_include, never_include,
                                            ownership_range, min_leverage, stacks):
        """Build comprehensive game theory enforcement rules"""
        rules = []

        # Captain constraint (highest priority)
        if captains:
            rules.append({
                'type': 'hard',
                'constraint': 'game_theory_captain',
                'players': captains[:5],
                'priority': 100,
                'description': 'Game theory optimal captains'
            })

        # Ownership sum constraint
        rules.append({
            'type': 'hard',
            'constraint': 'ownership_sum',
            'min': ownership_range[0],
            'max': ownership_range[1],
            'priority': 90,
            'description': f'Total ownership {ownership_range[0]}-{ownership_range[1]}%'
        })

        # Minimum leverage players
        if min_leverage > 0:
            rules.append({
                'type': 'soft',
                'constraint': 'min_leverage',
                'count': min_leverage,
                'weight': 0.8,
                'priority': 70,
                'description': f'Include {min_leverage}+ leverage plays'
            })

        # Must include players
        for player in must_include[:3]:
            rules.append({
                'type': 'hard',
                'constraint': 'must_include',
                'player': player,
                'priority': 85,
                'description': f'Must include {player}'
            })

        # Never include players
        for player in never_include[:3]:
            rules.append({
                'type': 'hard',
                'constraint': 'must_exclude',
                'player': player,
                'priority': 80,
                'description': f'Fade chalk: {player}'
            })

        # Stacking rules
        for stack in stacks[:2]:
            rules.append({
                'type': 'soft',
                'constraint': 'leverage_stack',
                'players': [stack['player1'], stack['player2']],
                'weight': 0.7,
                'priority': 60,
                'description': f"Leverage stack: {stack['player1']} + {stack['player2']}"
            })

        return rules

# ============================================================================
# GPP CORRELATION STRATEGIST - COMPLETE IMPLEMENTATION
# ============================================================================

class GPPCorrelationStrategist(BaseAIStrategist):
    """AI Strategist 2: Correlation and Stacking Patterns - Complete"""

    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.CORRELATION)
        self.correlation_matrix = {}

    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str,
                       slate_profile: Dict) -> str:
        """Generate correlation focused prompt with advanced stacking"""

        self.logger.log(f"Generating Correlation prompt for {field_size}", "DEBUG")

        if df.empty:
            return "Error: Empty player pool"

        # Team analysis
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

        # Calculate correlation opportunities
        qbs = df[df['Position'] == 'QB']['Player'].tolist()
        pass_catchers = df[df['Position'].isin(['WR', 'TE'])]['Player'].tolist()
        rbs = df[df['Position'] == 'RB']['Player'].tolist()

        # Determine game flow expectations
        total = game_info.get('total', 45)
        spread = game_info.get('spread', 0)
        favorite = team1 if spread < 0 else team2
        underdog = team2 if favorite == team1 else team1

        prompt = f"""
        You are an expert DFS correlation strategist. Create SPECIFIC stacking rules for {field_size} GPP.

        GAME ENVIRONMENT:
        Total: {total} | Spread: {spread}
        Favorite: {favorite} by {abs(spread)} points
        Underdog: {underdog}
        Slate Type: {slate_profile.get('slate_type', 'standard')}

        TEAM 1 - {team1} ({'Favorite' if team1 == favorite else 'Underdog'}):
        {team1_df[['Player', 'Position', 'Salary', 'Projected_Points']].head(8).to_string() if not team1_df.empty else 'No data'}

        TEAM 2 - {team2} ({'Favorite' if team2 == favorite else 'Underdog'}):
        {team2_df[['Player', 'Position', 'Salary', 'Projected_Points']].head(8).to_string() if not team2_df.empty else 'No data'}

        CORRELATION TARGETS:
        QBs: {qbs}
        Pass Catchers: {pass_catchers[:8]}
        RBs: {rbs}

        CREATE ADVANCED CORRELATION RULES IN JSON:
        {{
            "primary_stacks": [
                {{"type": "QB_WR1", "player1": "exact_qb", "player2": "exact_wr", "correlation": 0.7, "narrative": "why"}}
            ],
            "onslaught_stacks": [
                {{"team": "winning_team", "players": ["qb", "wr1", "wr2", "rb"], "scenario": "blowout correlation"}}
            ],
            "leverage_stacks": [
                {{"type": "contrarian", "player1": "low_own_qb", "player2": "opposing_wr1", "combined_ownership": 15}}
            ],
            "bring_back_stacks": [
                {{"primary": ["qb", "wr"], "bring_back": "opposing_wr1", "game_total": 50}}
            ],
            "negative_correlation": [
                {{"avoid_together": ["rb1", "rb2"], "reason": "same backfield"}},
                {{"avoid_together": ["wr1", "wr2", "wr3"], "reason": "target competition"}}
            ],
            "game_script_stacks": {{
                "shootout": ["qb1", "wr1", "opp_wr1"],
                "blowout": ["fav_rb", "fav_dst", "fav_wr2"],
                "upset": ["dog_qb", "dog_wr1", "dog_wr2"]
            }},
            "captain_correlation": {{
                "best_captains_for_stacking": ["player_names"],
                "stack_multipliers": {{"QB": 1.5, "WR1": 1.3, "TE": 1.2}}
            }},
            "confidence": 0.8,
            "stack_narrative": "Primary correlation thesis"
        }}

        Provide exact player names and focus on correlations that maximize ceiling.
        """

        return prompt

    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Complete implementation of correlation response parsing"""

        try:
            # Parse JSON response
            if response and response != '{}':
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    self.logger.log("Failed to parse JSON, using text extraction", "WARNING")
                    data = self._extract_correlation_from_text(response, df)
            else:
                data = {}

            available_players = set(df['Player'].values)
            all_stacks = []

            # Process primary stacks with validation
            for stack in data.get('primary_stacks', []):
                if self._validate_stack(stack, available_players):
                    stack['priority'] = 'high'
                    stack['enforced'] = True
                    all_stacks.append(stack)

            # Process onslaught stacks (3+ players from winning team)
            for onslaught in data.get('onslaught_stacks', []):
                players = onslaught.get('players', [])
                valid_players = [p for p in players if p in available_players]

                if len(valid_players) >= 3:
                    # Create an onslaught stack structure
                    all_stacks.append({
                        'type': 'onslaught',
                        'players': valid_players,
                        'team': onslaught.get('team', ''),
                        'scenario': onslaught.get('scenario', 'Blowout correlation'),
                        'priority': 'high',
                        'correlation': 0.6
                    })

            # Process leverage stacks
            for stack in data.get('leverage_stacks', []):
                if self._validate_stack(stack, available_players):
                    stack['priority'] = 'medium'
                    stack['leverage'] = True
                    all_stacks.append(stack)

            # Process bring-back stacks
            for bring_back in data.get('bring_back_stacks', []):
                primary = bring_back.get('primary', [])
                opponent = bring_back.get('bring_back', '')

                valid_primary = [p for p in primary if p in available_players]

                if valid_primary and opponent in available_players:
                    all_stacks.append({
                        'type': 'bring_back',
                        'primary_stack': valid_primary,
                        'bring_back': opponent,
                        'game_total': bring_back.get('game_total', 45),
                        'priority': 'high',
                        'correlation': 0.5
                    })

            # If no valid stacks from AI, create statistical ones
            if len(all_stacks) < 2:
                all_stacks.extend(self._create_statistical_stacks(df))

            # Extract captain correlation
            captain_rules = data.get('captain_correlation', {})
            captain_targets = captain_rules.get('best_captains_for_stacking', [])
            valid_captains = [c for c in captain_targets if c in available_players]

            # If no valid captains, use correlation-based selection
            if len(valid_captains) < 3:
                valid_captains.extend(self._get_correlation_captains(df, all_stacks))
                valid_captains = list(set(valid_captains))[:7]

            # Process negative correlations
            avoid_pairs = []
            for neg_corr in data.get('negative_correlation', []):
                players = neg_corr.get('avoid_together', [])
                if len(players) >= 2 and all(p in available_players for p in players[:2]):
                    avoid_pairs.append({
                        'players': players,
                        'reason': neg_corr.get('reason', 'negative correlation')
                    })

            # Build comprehensive enforcement rules
            enforcement_rules = self._build_correlation_enforcement_rules(
                all_stacks, avoid_pairs, valid_captains
            )

            # Create correlation matrix for reference
            self.correlation_matrix = self._build_correlation_matrix(all_stacks, avoid_pairs)

            confidence = data.get('confidence', 0.75)
            confidence = max(0.0, min(1.0, confidence))

            # Extract key insights
            key_insights = [
                data.get('stack_narrative', 'Correlation-based construction'),
                f"Primary focus: {all_stacks[0]['type'] if all_stacks else 'standard'} stacks",
                f"{len(all_stacks)} correlation plays identified"
            ]

            return AIRecommendation(
                captain_targets=valid_captains,
                must_play=[],  # Will be populated by stacks
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

    def _validate_stack(self, stack: Dict, available_players: Set[str]) -> bool:
        """Validate stack players exist"""
        player1 = stack.get('player1')
        player2 = stack.get('player2')

        return (player1 and player2 and
                player1 in available_players and
                player2 in available_players)

    def _create_statistical_stacks(self, df: pd.DataFrame) -> List[Dict]:
        """Create correlation stacks using statistical analysis"""
        stacks = []

        try:
            # QB stacks
            qbs = df[df['Position'] == 'QB']

            for _, qb in qbs.iterrows():
                team = qb['Team']

                # Find pass catchers from same team
                teammates = df[(df['Team'] == team) & (df['Position'].isin(['WR', 'TE']))]

                if not teammates.empty:
                    # Sort by projection for best stacks
                    top_teammates = teammates.nlargest(3, 'Projected_Points')

                    for i, (_, teammate) in enumerate(top_teammates.iterrows()):
                        correlation_strength = 0.7 - (i * 0.1)  # Decreasing correlation
                        stacks.append({
                            'player1': qb['Player'],
                            'player2': teammate['Player'],
                            'type': f"QB_{teammate['Position']}",
                            'correlation': correlation_strength,
                            'priority': 'high' if i == 0 else 'medium'
                        })

                # Create bring-back stacks
                opponents = df[(df['Team'] != team) & (df['Position'].isin(['WR', 'TE']))]
                if not opponents.empty:
                    top_opponent = opponents.nlargest(1, 'Projected_Points').iloc[0]
                    stacks.append({
                        'type': 'bring_back',
                        'primary_stack': [qb['Player'], top_teammates.iloc[0]['Player']] if not teammates.empty else [qb['Player']],
                        'bring_back': top_opponent['Player'],
                        'correlation': 0.5,
                        'priority': 'medium'
                    })

            # Create onslaught stacks for favorites
            if len(df['Team'].unique()) >= 2:
                teams = df['Team'].unique()
                for team in teams:
                    team_df = df[df['Team'] == team]
                    if len(team_df) >= 4:
                        # Top 4 players from same team
                        top_players = team_df.nlargest(4, 'Projected_Points')['Player'].tolist()
                        stacks.append({
                            'type': 'onslaught',
                            'players': top_players,
                            'team': team,
                            'correlation': 0.5,
                            'priority': 'low'
                        })

        except Exception as e:
            self.logger.log(f"Error creating statistical stacks: {e}", "WARNING")

        return stacks[:8]  # Return top 8 stacks

    def _get_correlation_captains(self, df: pd.DataFrame, stacks: List[Dict]) -> List[str]:
        """Get captain targets based on correlation analysis"""
        captains = []

        # Prioritize QBs involved in stacks
        for stack in stacks:
            if stack.get('type') in ['QB_WR', 'QB_TE', 'primary']:
                player1 = stack.get('player1')
                if player1:
                    player_data = df[df['Player'] == player1]
                    if not player_data.empty and player_data.iloc[0]['Position'] == 'QB':
                        captains.append(player1)

        # Add primary pass catchers
        for stack in stacks:
            if 'player2' in stack:
                player2 = stack['player2']
                if player2 not in captains:
                    captains.append(player2)

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
        """Build correlation-specific enforcement rules"""
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

        # Process stacks by priority
        high_priority_stacks = [s for s in stacks if s.get('priority') == 'high'][:3]
        medium_priority_stacks = [s for s in stacks if s.get('priority') == 'medium'][:2]

        # Enforce high priority stacks
        for i, stack in enumerate(high_priority_stacks):
            if stack.get('type') == 'onslaught':
                # Onslaught requires 3+ players
                rules.append({
                    'type': 'hard' if i == 0 else 'soft',
                    'constraint': 'onslaught_stack',
                    'players': stack['players'][:4],
                    'min_players': 3,
                    'weight': 0.9 if i > 0 else 1.0,
                    'priority': 90 - (i * 5),
                    'description': f"Onslaught: {stack.get('team', 'team')} correlation"
                })
            elif stack.get('type') == 'bring_back':
                # Bring-back stack
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
                # Standard two-player stack
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

        # Add medium priority stacks as soft constraints
        for stack in medium_priority_stacks:
            if 'player1' in stack and 'player2' in stack:
                rules.append({
                    'type': 'soft',
                    'constraint': 'prefer_stack',
                    'players': [stack['player1'], stack['player2']],
                    'weight': 0.6,
                    'priority': 60,
                    'description': f"Prefer: {stack.get('type', 'stack')}"
                })

        # Enforce negative correlations
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
        """Build correlation matrix for reference"""
        matrix = {}

        # Positive correlations from stacks
        for stack in stacks:
            if 'player1' in stack and 'player2' in stack:
                key = f"{stack['player1']}_{stack['player2']}"
                matrix[key] = stack.get('correlation', 0.5)
            elif stack.get('type') == 'onslaught' and 'players' in stack:
                # Create pairwise correlations for onslaught
                players = stack['players']
                for i in range(len(players)):
                    for j in range(i+1, len(players)):
                        key = f"{players[i]}_{players[j]}"
                        matrix[key] = 0.4  # Moderate correlation within team

        # Negative correlations
        for avoid in avoid_pairs:
            players = avoid['players']
            if len(players) >= 2:
                key = f"{players[0]}_{players[1]}"
                matrix[key] = -0.5  # Negative correlation

        return matrix

    def _extract_correlation_from_text(self, response: str, df: pd.DataFrame) -> Dict:
        """Extract correlation data from text response"""
        data = {
            'primary_stacks': [],
            'confidence': 0.6
        }

        # Look for QB-receiver pairs mentioned
        qbs = df[df['Position'] == 'QB']['Player'].tolist()
        receivers = df[df['Position'].isin(['WR', 'TE'])]['Player'].tolist()

        # Simple pattern matching
        for qb in qbs:
            for receiver in receivers:
                if qb in response and receiver in response:
                    # Check if they're from same team
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

# ============================================================================
# GPP CONTRARIAN NARRATIVE STRATEGIST - COMPLETE IMPLEMENTATION
# ============================================================================

class GPPContrarianNarrativeStrategist(BaseAIStrategist):
    """AI Strategist 3: Contrarian Narratives and Hidden Angles - Complete"""

    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.CONTRARIAN_NARRATIVE)

    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str,
                       slate_profile: Dict) -> str:
        """Generate contrarian narrative focused prompt"""

        self.logger.log(f"Generating Contrarian Narrative prompt for {field_size}", "DEBUG")

        if df.empty:
            return "Error: Empty player pool"

        # Calculate contrarian opportunities
        df_copy = df.copy()

        # Value calculations
        df_copy['Value'] = df_copy['Projected_Points'] / (df_copy['Salary'] / 1000)
        df_copy['Contrarian_Score'] = (
            df_copy['Projected_Points'] / df_copy['Projected_Points'].max()
        ) / (df_copy.get('Ownership', 10) / 100 + 0.1)

        # Find contrarian plays
        low_owned_high_ceiling = df_copy[df_copy.get('Ownership', 10) < 10].nlargest(10, 'Projected_Points')
        hidden_value = df_copy[df_copy.get('Ownership', 10) < 15].nlargest(10, 'Value')
        contrarian_captains = df_copy.nlargest(10, 'Contrarian_Score')

        # Identify chalk to fade
        chalk_plays = df_copy[df_copy.get('Ownership', 10) > 30].nlargest(5, 'Ownership')

        # Teams and game info
        teams = df['Team'].unique()[:2]
        total = game_info.get('total', 45)
        spread = game_info.get('spread', 0)

        prompt = f"""
        You are a contrarian DFS strategist who finds the NON-OBVIOUS narratives that win GPP tournaments.

        GAME SETUP:
        {teams[0] if len(teams) > 0 else 'Team1'} vs {teams[1] if len(teams) > 1 else 'Team2'}
        Total: {total} | Spread: {spread}
        Slate Type: {slate_profile.get('slate_type', 'standard')}

        CONTRARIAN OPPORTUNITIES:

        LOW-OWNED HIGH CEILING (<10% owned):
        {low_owned_high_ceiling[['Player', 'Position', 'Team', 'Projected_Points', 'Ownership']].to_string()}

        HIDDEN VALUE PLAYS:
        {hidden_value[['Player', 'Position', 'Salary', 'Value', 'Ownership']].to_string()}

        CONTRARIAN CAPTAIN SCORES:
        {contrarian_captains[['Player', 'Contrarian_Score', 'Ownership']].to_string()}

        CHALK TO FADE (>30% owned):
        {chalk_plays[['Player', 'Position', 'Ownership', 'Salary']].to_string()}

        CREATE CONTRARIAN TOURNAMENT-WINNING NARRATIVES IN JSON:
        {{
            "primary_narrative": "The ONE scenario that creates a unique winning lineup",
            "contrarian_captains": [
                {{"player": "exact_name", "narrative": "Why this 5% captain wins", "ceiling_path": "How they hit 30+ points"}}
            ],
            "hidden_correlations": [
                {{"player1": "name1", "player2": "name2", "narrative": "Non-obvious connection"}}
            ],
            "fade_the_chalk": [
                {{"player": "chalk_name", "ownership": 35, "fade_reason": "Specific bust risk", "pivot_to": "alternative"}}
            ],
            "leverage_scenarios": [
                {{"scenario": "Game script", "beneficiaries": ["player1", "player2"], "probability": "10% but wins if hits"}}
            ],
            "contrarian_game_theory": {{
                "what_field_expects": "Common narrative",
                "fatal_flaw": "Why the field is wrong",
                "exploit_angle": "How to capitalize",
                "unique_construction": "Roster construction edge"
            }},
            "boom_paths": [
                {{"player": "name", "path_to_ceiling": "Specific scenario for 3x value", "indicators": "What to watch for"}}
            ],
            "tournament_winner": {{
                "captain": "exact_contrarian_captain",
                "core": ["player1", "player2", "player3"],
                "differentiatorsifferentiators": ["unique1", "unique2"],
                "total_ownership": 65,
                "win_condition": "What needs to happen for this to take down the field"
            }},
            "confidence": 0.7
        }}

        Find the narrative that makes sub-5% plays optimal. Think about game flow scenarios the field ignores.
        """

        return prompt

    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Complete implementation of contrarian narrative parsing"""

        try:
            # Parse JSON response
            if response and response != '{}':
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    self.logger.log("Failed to parse JSON, using narrative extraction", "WARNING")
                    data = self._extract_narrative_from_text(response, df)
            else:
                data = {}

            available_players = set(df['Player'].values)

            # Extract contrarian captains with narratives
            contrarian_captains = []
            captain_narratives = {}

            for captain_data in data.get('contrarian_captains', []):
                player = captain_data.get('player')
                if player and player in available_players:
                    contrarian_captains.append(player)
                    captain_narratives[player] = {
                        'narrative': captain_data.get('narrative', ''),
                        'ceiling_path': captain_data.get('ceiling_path', ''),
                        'ownership': df[df['Player'] == player]['Ownership'].values[0] if not df[df['Player'] == player].empty else 10
                    }

            # If no valid contrarian captains from AI, find them statistically
            if len(contrarian_captains) < 3:
                contrarian_captains.extend(self._find_statistical_contrarian_captains(df, contrarian_captains))
                contrarian_captains = contrarian_captains[:7]

            # Extract tournament winner lineup
            tournament_winner = data.get('tournament_winner', {})
            tw_captain = tournament_winner.get('captain')
            tw_core = tournament_winner.get('core', [])
            tw_differentiators = tournament_winner.get('differentiators', [])

            must_play = []

            # Validate and add tournament winner captain
            if tw_captain and tw_captain in available_players:
                if tw_captain not in contrarian_captains:
                    contrarian_captains.insert(0, tw_captain)
                    captain_narratives[tw_captain] = {
                        'narrative': 'Tournament winner captain',
                        'ceiling_path': tournament_winner.get('win_condition', ''),
                        'ownership': df[df['Player'] == tw_captain]['Ownership'].values[0] if not df[df['Player'] == tw_captain].empty else 5
                    }

            # Validate tournament winner core
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

            # Extract leverage scenarios
            leverage_plays = []

            for scenario in data.get('leverage_scenarios', []):
                beneficiaries = scenario.get('beneficiaries', [])
                valid_beneficiaries = [p for p in beneficiaries if p in available_players]

                if valid_beneficiaries:
                    leverage_plays.append({
                        'scenario': scenario.get('scenario', ''),
                        'players': valid_beneficiaries,
                        'probability': scenario.get('probability', 'low'),
                        'leverage': True
                    })

            # Extract boom paths for ceiling optimization
            boom_players = []

            for boom in data.get('boom_paths', []):
                player = boom.get('player')
                if player and player in available_players:
                    boom_players.append({
                        'player': player,
                        'ceiling_path': boom.get('path_to_ceiling', ''),
                        'indicators': boom.get('indicators', ''),
                        'boom_potential': True
                    })

            # Build contrarian angles from game theory insights
            game_theory = data.get('contrarian_game_theory', {})
            contrarian_angles = [
                game_theory.get('fatal_flaw', ''),
                game_theory.get('exploit_angle', ''),
                game_theory.get('unique_construction', '')
            ]
            contrarian_angles = [a for a in contrarian_angles if a]

            # Build comprehensive enforcement rules
            enforcement_rules = self._build_contrarian_enforcement_rules(
                contrarian_captains, must_play, fades, hidden_stacks,
                leverage_plays, captain_narratives
            )

            # Extract key insights
            key_insights = [
                data.get('primary_narrative', 'Contrarian approach'),
                f"Fade {len(fades)} chalk plays",
                f"{len(contrarian_captains)} contrarian captains identified"
            ]

            confidence = data.get('confidence', 0.7)
            confidence = max(0.0, min(1.0, confidence))

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
                contrarian_angles=contrarian_angles[:3],
                boosts=[b['player'] for b in boom_players[:3]],
                fades=fades[:3]
            )

        except Exception as e:
            self.logger.log_exception(e, "parse_contrarian_response")
            return self._get_fallback_recommendation(df, field_size)

    def _find_statistical_contrarian_captains(self, df: pd.DataFrame,
                                             existing: List[str]) -> List[str]:
        """Find contrarian captains using statistical analysis"""
        captains = []

        # Calculate contrarian score
        df_copy = df.copy()
        df_copy['Contrarian_Score'] = (
            df_copy['Projected_Points'] / df_copy['Projected_Points'].max()
        ) / (df_copy.get('Ownership', 10) / 100 + 0.1)

        # Filter out existing captains and high ownership
        eligible = df_copy[
            (~df_copy['Player'].isin(existing)) &
            (df_copy.get('Ownership', 10) < 15)
        ]

        # Get top contrarian plays
        contrarian_plays = eligible.nlargest(5, 'Contrarian_Score')

        for _, row in contrarian_plays.iterrows():
            captains.append(row['Player'])

        return captains

    def _build_contrarian_enforcement_rules(self, captains: List[str], must_play: List[str],
                                           fades: List[str], hidden_stacks: List[Dict],
                                           leverage_plays: List[Dict],
                                           captain_narratives: Dict) -> List[Dict]:
        """Build contrarian-specific enforcement rules"""
        rules = []

        # Contrarian captain rule (highest priority)
        if captains:
            # Separate ultra-contrarian (<5% owned) from moderate contrarian
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

        # Tournament core rules
        for i, player in enumerate(must_play[:3]):
            rules.append({
                'type': 'hard' if i == 0 else 'soft',
                'constraint': f'tournament_core_{player}',
                'player': player,
                'weight': 0.9 - (i * 0.1) if i > 0 else 1.0,
                'priority': 85 - (i * 5),
                'description': f'Tournament core: {player}'
            })

        # Fade rules (avoid chalk)
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

        # Hidden correlation rules
        for i, stack in enumerate(hidden_stacks[:2]):
            rules.append({
                'type': 'soft',
                'constraint': 'hidden_correlation',
                'players': [stack['player1'], stack['player2']],
                'weight': 0.7 - (i * 0.1),
                'priority': 60 - (i * 5),
                'description': stack.get('narrative', 'Hidden correlation')
            })

        # Leverage scenario rules
        for scenario in leverage_plays[:1]:
            if scenario.get('players'):
                rules.append({
                    'type': 'soft',
                    'constraint': 'leverage_scenario',
                    'players': scenario['players'][:2],
                    'weight': 0.6,
                    'priority': 55,
                    'description': f"Leverage: {scenario.get('scenario', 'scenario')}"
                })

        # Ownership ceiling rule for contrarian approach
        rules.append({
            'type': 'soft',
            'constraint': 'ownership_ceiling',
            'max_ownership': 75,  # Lower than normal
            'weight': 0.7,
            'priority': 50,
            'description': 'Maintain contrarian ownership profile'
        })

        return rules

    def _extract_narrative_from_text(self, response: str, df: pd.DataFrame) -> Dict:
        """Extract

def _extract_narrative_from_text(self, response: str, df: pd.DataFrame) -> Dict:
        """Extract contrarian narrative from text response"""
        data = {
            'contrarian_captains': [],
            'fade_the_chalk': [],
            'confidence': 0.6
        }

        # Look for low ownership players mentioned
        low_owned = df[df.get('Ownership', 10) < 10]['Player'].tolist()
        high_owned = df[df.get('Ownership', 10) > 30]['Player'].tolist()

        # Find mentioned players
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

# ============================================================================
# ENHANCED CLAUDE API MANAGER WITH FALLBACK SUPPORT
# ============================================================================

class ClaudeAPIManager:
    """Enhanced Claude API manager with robust error handling and caching"""

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
                AIStrategistType.GAME_THEORY: {'requests': 0, 'errors': 0, 'tokens': 0, 'avg_time': 0},
                AIStrategistType.CORRELATION: {'requests': 0, 'errors': 0, 'tokens': 0, 'avg_time': 0},
                AIStrategistType.CONTRARIAN_NARRATIVE: {'requests': 0, 'errors': 0, 'tokens': 0, 'avg_time': 0}
            }
        }

        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()

        # Response time tracking
        self.response_times = deque(maxlen=100)

        self.initialize_client()

    def initialize_client(self):
        """Initialize Claude client with validation"""
        try:
            # Validate API key format
            if not self.api_key or not self.api_key.startswith('sk-'):
                raise ValueError("Invalid API key format")

            # Only import when needed and with error handling
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)

                # Test the connection with a minimal request
                self.validate_connection()

            except ImportError:
                self.logger.log("Anthropic library not installed. Install with: pip install anthropic", "ERROR")
                self.client = None
                return

            self.logger.log("Claude API client initialized successfully", "INFO")

        except Exception as e:
            self.logger.log(f"Failed to initialize Claude API: {e}", "ERROR")
            self.client = None

    def get_ai_response(self, prompt: str, ai_type: Optional[AIStrategistType] = None) -> str:
        """Get response from Claude API with caching and comprehensive error handling"""

        # Generate cache key
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        # Check cache
        with self._cache_lock:
            if prompt_hash in self.cache:
                self.stats['cache_hits'] += 1
                self.logger.log(f"Cache hit for {ai_type.value if ai_type else 'unknown'}", "DEBUG")
                return self.cache[prompt_hash]

        # Update request statistics
        self.stats['requests'] += 1
        if ai_type:
            self.stats['by_ai'][ai_type]['requests'] += 1

        try:
            if not self.client:
                raise Exception("API client not initialized")

            self.perf_monitor.start_timer("claude_api_call")
            start_time = time.time()

            # Make API call with appropriate model and system prompt
            message = self.client.messages.create(
                model="claude-3-sonnet-20241022",  # Latest model
                max_tokens=2000,
                temperature=0.7,
                system="""You are an expert Daily Fantasy Sports (DFS) strategist specializing in NFL tournament optimization.
                         You provide specific, actionable recommendations using exact player names and clear reasoning.
                         Always respond with valid JSON containing specific player recommendations.
                         Focus on game theory, correlations, and contrarian angles that win GPP tournaments.
                         Your recommendations must be enforceable as optimization constraints.""",
                messages=[{"role": "user", "content": prompt}]
            )

            elapsed = self.perf_monitor.stop_timer("claude_api_call")

            # Extract response
            response = message.content[0].text if message.content else "{}"

            # Update statistics
            response_time = time.time() - start_time
            self.response_times.append(response_time)

            if ai_type:
                self.stats['by_ai'][ai_type]['tokens'] += len(response) // 4  # Approximate
                # Update average time
                current_avg = self.stats['by_ai'][ai_type]['avg_time']
                total_requests = self.stats['by_ai'][ai_type]['requests']
                self.stats['by_ai'][ai_type]['avg_time'] = (
                    (current_avg * (total_requests - 1) + response_time) / total_requests
                )

            self.stats['total_tokens'] += len(response) // 4
            self.stats['avg_response_time'] = np.mean(list(self.response_times))

            # Cache response
            with self._cache_lock:
                self.cache[prompt_hash] = response
                self.stats['cache_size'] = len(self.cache)

                # Manage cache size
                if len(self.cache) > self.max_cache_size:
                    # Remove oldest entries (FIFO)
                    for key in list(self.cache.keys())[:10]:
                        del self.cache[key]

            self.logger.log(
                f"AI response received for {ai_type.value if ai_type else 'unknown'} "
                f"({len(response)} chars, {elapsed:.2f}s)",
                "DEBUG"
            )

            return response

        except Exception as e:
            self.stats['errors'] += 1
            if ai_type:
                self.stats['by_ai'][ai_type]['errors'] += 1

            self.logger.log(f"API error for {ai_type.value if ai_type else 'unknown'}: {e}", "ERROR")

            # Return empty JSON for parser to handle
            return "{}"

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
                'avg_response_time': self.stats['avg_response_time'],
                'by_ai': dict(self.stats['by_ai'])
            }

    def clear_cache(self):
        """Clear response cache"""
        with self._cache_lock:
            self.cache.clear()
            self.stats['cache_size'] = 0

        self.logger.log("API cache cleared", "INFO")

    def validate_connection(self) -> bool:
        """Validate API connection is working"""
        try:
            if not self.client:
                return False

            # Try a minimal test request
            test_prompt = "Respond with only the word: OK"

            message = self.client.messages.create(
                model="claude-3-sonnet-20241022",
                max_tokens=10,
                temperature=0,
                messages=[{"role": "user", "content": test_prompt}]
            )

            return bool(message.content)

        except Exception as e:
            self.logger.log(f"API validation failed: {e}", "ERROR")
            return False

# ============================================================================
# AI-DRIVEN GPP OPTIMIZER WITH COMPLETE FUNCTIONALITY
# ============================================================================

class AIChefGPPOptimizer:
    """Main optimizer where AI is the chef and optimization is just execution"""

    def __init__(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field',
                 api_manager=None):

        # Validate inputs
        self._validate_inputs(df, game_info, field_size)

        self.df = df.copy()  # Work with copy to avoid modifying original
        self.game_info = game_info
        self.field_size = field_size
        self.api_manager = api_manager

        # Initialize the three AI strategists
        self.game_theory_ai = GPPGameTheoryStrategist(api_manager)
        self.correlation_ai = GPPCorrelationStrategist(api_manager)
        self.contrarian_ai = GPPContrarianNarrativeStrategist(api_manager)

        # Initialize core components
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()

        # Get field configuration
        field_config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(
            field_size,
            OptimizerConfig.FIELD_SIZE_CONFIGS['large_field']
        )

        # AI enforcement and synthesis
        self.enforcement_engine = AIEnforcementEngine(
            field_config.get('ai_enforcement', AIEnforcementLevel.MANDATORY)
        )
        self.synthesis_engine = AISynthesisEngine()

        # Supporting components
        self.bucket_manager = AIOwnershipBucketManager(self.enforcement_engine)

        # Tracking
        self.ai_decisions_log = []
        self.optimization_log = []
        self.generated_lineups = []
        self.lineup_generation_stats = {
            'attempts': 0,
            'successes': 0,
            'failures_by_reason': defaultdict(int)
        }

        # Threading for parallel generation
        self.max_workers = min(OptimizerConfig.MAX_PARALLEL_THREADS, 4)
        self.generation_timeout = OptimizerConfig.OPTIMIZATION_TIMEOUT

        # Prepare data
        self._prepare_data()

    def _validate_inputs(self, df: pd.DataFrame, game_info: Dict, field_size: str):
        """Validate all inputs before initialization"""
        if df is None or df.empty:
            raise ValueError("Player pool DataFrame cannot be empty")

        required_columns = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if field_size not in OptimizerConfig.FIELD_SIZE_CONFIGS:
            self.logger.log(f"Unknown field size {field_size}, using large_field", "WARNING")

    def _prepare_data(self):
        """Prepare data with additional calculations"""
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

        # Add team counts for validation
        self.team_counts = self.df['Team'].value_counts().to_dict()

    def get_triple_ai_strategies(self, use_api: bool = True) -> Dict[AIStrategistType, AIRecommendation]:
        """Get strategies from all three AIs with parallel execution and manual input support"""

        self.logger.log("Getting strategies from three AI strategists", "INFO")
        self.perf_monitor.start_timer("get_ai_strategies")

        recommendations = {}

        if use_api and self.api_manager and self.api_manager.client:
            # API mode - get recommendations in parallel
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
                        self.logger.log(f"{ai_type.value} recommendation received", "INFO")
                    except Exception as e:
                        self.logger.log(f"{ai_type.value} failed: {e}", "ERROR")
                        recommendations[ai_type] = self._get_fallback_recommendation(ai_type)
        else:
            # Manual mode or no API - get strategies through UI input
            recommendations = self._get_manual_ai_strategies()

        # Validate we have all recommendations
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

    def _get_manual_ai_strategies(self) -> Dict[AIStrategistType, AIRecommendation]:
        """Get AI strategies through manual input with complete UI"""
        recommendations = {}

        import streamlit as st

        st.subheader("Manual AI Strategy Input")
        st.info("Copy prompts below and paste responses from your AI assistant")

        tab1, tab2, tab3 = st.tabs(["Game Theory", "Correlation", "Contrarian"])

        with tab1:
            response = self._get_manual_ai_input("Game Theory", self.game_theory_ai)
            try:
                recommendations[AIStrategistType.GAME_THEORY] = self.game_theory_ai.parse_response(
                    response, self.df, self.field_size
                )
                st.success("Game Theory strategy parsed successfully")
            except Exception as e:
                st.error(f"Error parsing Game Theory: {e}")
                recommendations[AIStrategistType.GAME_THEORY] = self._get_fallback_recommendation(
                    AIStrategistType.GAME_THEORY
                )

        with tab2:
            response = self._get_manual_ai_input("Correlation", self.correlation_ai)
            try:
                recommendations[AIStrategistType.CORRELATION] = self.correlation_ai.parse_response(
                    response, self.df, self.field_size
                )
                st.success("Correlation strategy parsed successfully")
            except Exception as e:
                st.error(f"Error parsing Correlation: {e}")
                recommendations[AIStrategistType.CORRELATION] = self._get_fallback_recommendation(
                    AIStrategistType.CORRELATION
                )

        with tab3:
            response = self._get_manual_ai_input("Contrarian", self.contrarian_ai)
            try:
                recommendations[AIStrategistType.CONTRARIAN_NARRATIVE] = self.contrarian_ai.parse_response(
                    response, self.df, self.field_size
                )
                st.success("Contrarian strategy parsed successfully")
            except Exception as e:
                st.error(f"Error parsing Contrarian: {e}")
                recommendations[AIStrategistType.CONTRARIAN_NARRATIVE] = self._get_fallback_recommendation(
                    AIStrategistType.CONTRARIAN_NARRATIVE
                )

        return recommendations

    def _get_manual_ai_input(self, ai_name: str, strategist) -> str:
        """Complete implementation of manual AI input UI"""
        import streamlit as st

        # Generate the prompt
        slate_profile = strategist._analyze_slate_profile(self.df, self.game_info)
        prompt = strategist.generate_prompt(self.df, self.game_info, self.field_size, slate_profile)

        # Display prompt for copying
        with st.expander(f"📋 {ai_name} Prompt (Click to expand)", expanded=False):
            st.text_area(
                "Copy this prompt to your AI assistant:",
                value=prompt,
                height=300,
                key=f"{ai_name}_prompt_display"
            )

            # Add copy button functionality hint
            st.caption("💡 Tip: Click in the text area and press Ctrl+A (Cmd+A on Mac) then Ctrl+C to copy")

        # Input area for response
        st.markdown(f"**Paste {ai_name} Response:**")

        # Add example response format
        with st.expander("See example response format"):
            example = {
                "captain_rules": {"must_be_one_of": ["Player Name 1", "Player Name 2"]},
                "lineup_rules": {"must_include": ["Player Name"], "ownership_sum_range": [60, 90]},
                "confidence": 0.8
            }
            st.json(example)

        response = st.text_area(
            f"Paste the JSON response here:",
            height=250,
            key=f"{ai_name}_response",
            placeholder='{"captain_rules": {...}, "lineup_rules": {...}, ...}'
        )

        # Validate JSON
        if response and response.strip() != '':
            try:
                json.loads(response)
                st.success(f"✅ Valid JSON for {ai_name}")
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON: {str(e)}")
                if st.checkbox(f"Use fallback for {ai_name}?", key=f"{ai_name}_use_fallback"):
                    response = '{}'
        else:
            response = '{}'

        return response

    def _get_fallback_recommendation(self, ai_type: AIStrategistType) -> AIRecommendation:
        """Get fallback recommendation when AI fails"""
        if ai_type == AIStrategistType.GAME_THEORY:
            strategist = self.game_theory_ai
        elif ai_type == AIStrategistType.CORRELATION:
            strategist = self.correlation_ai
        else:
            strategist = self.contrarian_ai

        return strategist._get_fallback_recommendation(self.df, self.field_size)

    def synthesize_ai_strategies(self, recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """Synthesize three AI perspectives into unified strategy"""

        self.logger.log("Synthesizing triple AI strategies", "INFO")

        try:
            # Use synthesis engine
            synthesis = self.synthesis_engine.synthesize_recommendations(
                recommendations.get(AIStrategistType.GAME_THEORY),
                recommendations.get(AIStrategistType.CORRELATION),
                recommendations.get(AIStrategistType.CONTRARIAN_NARRATIVE)
            )

            # Create enforcement rules
            enforcement_rules = self.enforcement_engine.create_enforcement_rules(recommendations)

            # Validate rules are feasible
            validation = AIConfigValidator.validate_ai_requirements(enforcement_rules, self.df)

            if not validation['is_valid']:
                self.logger.log(f"AI requirements validation failed: {validation['errors']}", "WARNING")

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
        """Create fallback synthesis when main synthesis fails"""
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
        """Generate lineups with enhanced performance and error handling"""

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
        import streamlit as st
        st.info(f"Generating {num_lineups} lineups with {self.field_size} settings...")

        # Quick feasibility check
        min_salary_lineup = self.df.nsmallest(6, 'Salary')['Salary'].sum()
        if min_salary_lineup > OptimizerConfig.SALARY_CAP:
            st.error("Cannot create valid lineup - minimum salary exceeds cap!")
            return pd.DataFrame()

        # Prepare data structures
        players = self.df['Player'].tolist()
        positions = self.df.set_index('Player')['Position'].to_dict()
        teams = self.df.set_index('Player')['Team'].to_dict()
        salaries = self.df.set_index('Player')['Salary'].to_dict()
        points = self.df.set_index('Player')['Projected_Points'].to_dict()
        ownership = self.df.set_index('Player')['Ownership'].to_dict()

        # Apply AI modifications to projections
        ai_adjusted_points = self._apply_ai_adjustments(points, synthesis)

        # Get strategy distribution
        consensus_level = self._determine_consensus_level(synthesis)
        strategy_distribution = AIConfigValidator.get_ai_strategy_distribution(
            self.field_size, num_lineups, consensus_level
        )

        self.logger.log(f"AI Strategy distribution: {strategy_distribution}", "INFO")

        all_lineups = []
        used_captains = set()

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Generate lineups by strategy
        lineup_tasks = []
        for strategy, count in strategy_distribution.items():
            strategy_name = strategy if isinstance(strategy, str) else strategy.value
            for i in range(count):
                lineup_tasks.append((len(lineup_tasks) + 1, strategy_name))

        # Use parallel generation for better performance
        if self.max_workers > 1 and len(lineup_tasks) > 5:
            all_lineups = self._generate_lineups_parallel_optimized(
                lineup_tasks, players, salaries, ai_adjusted_points,
                ownership, positions, teams, enforcement_rules,
                synthesis, used_captains, progress_bar, status_text
            )
        else:
            # Sequential generation for small counts
            for i, (lineup_num, strategy_name) in enumerate(lineup_tasks):
                progress = (i + 1) / len(lineup_tasks)
                progress_bar.progress(progress)
                status_text.text(f"Generating lineup {lineup_num} ({strategy_name})...")

                lineup = self._build_ai_enforced_lineup(
                    lineup_num=lineup_num,
                    strategy=strategy_name,
                    players=players,
                    salaries=salaries,
                    points=ai_adjusted_points,
                    ownership=ownership,
                    positions=positions,
                    teams=teams,
                    enforcement_rules=enforcement_rules,
                    synthesis=synthesis,
                    used_captains=used_captains
                )

                if lineup:
                    is_valid, violations = self.enforcement_engine.validate_lineup_against_ai(
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

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Calculate metrics
        total_time = time.time() - start_time
        success_rate = len(all_lineups) / max(num_lineups, 1)

        self.logger.log_optimization_end(len(all_lineups), total_time, success_rate)

        # Display results summary
        if len(all_lineups) == 0:
            st.error("No valid lineups generated")
            self._display_generation_diagnostics()
        elif len(all_lineups) < num_lineups:
            st.warning(f"Generated {len(all_lineups)}/{num_lineups} lineups")
            self._display_partial_generation_reasons(num_lineups - len(all_lineups))
        else:
            st.success(f"Generated {len(all_lineups)} lineups in {total_time:.1f}s!")

        # Store generated lineups
        self.generated_lineups = all_lineups

        return pd.DataFrame(all_lineups)

    def _generate_lineups_parallel_optimized(self, lineup_tasks, players, salaries, points,
                                            ownership, positions, teams, enforcement_rules,
                                            synthesis, used_captains, progress_bar, status_text):
        """Optimized parallel lineup generation with better thread management"""
        import streamlit as st
        all_lineups = []
        completed = 0
        captain_lock = threading.Lock()

        def generate_single_lineup(task_data):
            lineup_num, strategy_name = task_data

            # Thread-safe captain selection
            local_used_captains = set()
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
                is_valid, violations = self.enforcement_engine.validate_lineup_against_ai(
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
                completed += 1
                progress_bar.progress(completed / len(lineup_tasks))
                status_text.text(f"Processing lineup {completed}/{len(lineup_tasks)}...")

                try:
                    lineup = future.result(timeout=10)
                    if lineup:
                        all_lineups.append(lineup)
                except Exception as e:
                    self.logger.log(f"Parallel generation error: {e}", "DEBUG")

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

                # AI-modified objective function
                player_weights = synthesis.get('player_rankings', {})

                objective = pulp.lpSum([
                    points[p] * player_weights.get(p, 1.0) * flex[p] +
                    1.5 * points[p] * player_weights.get(p, 1.0) * captain[p]
                    for p in players
                ])

                model += objective

                # Basic DraftKings constraints
                model += pulp.lpSum(captain.values()) == 1
                model += pulp.lpSum(flex.values()) == 5

                for p in players:
                    model += flex[p] + captain[p] <= 1

                # Salary constraint with slight relaxation on retries
                salary_cap = OptimizerConfig.SALARY_CAP
                if attempt > 0:
                    salary_cap += 500 * attempt

                model += pulp.lpSum([
                    salaries[p] * flex[p] + 1.5 * salaries[p] * captain[p]
                    for p in players
                ]) <= salary_cap

                # DraftKings team diversity requirements
                unique_teams = list(set(teams.values()))

                # Must have at least 1 player from each team
                for team in unique_teams:
                    team_players = [p for p in players if teams.get(p) == team]
                    if team_players:
                        model += pulp.lpSum([flex[p] + captain[p] for p in team_players]) >= 1

                # Max 5 players from one team (DK rule)
                max_from_team = 5 if attempt > 1 else OptimizerConfig.MAX_PLAYERS_PER_TEAM
                for team in unique_teams:
                    team_players = [p for p in players if teams.get(p) == team]
                    if team_players:
                        model += pulp.lpSum([flex[p] + captain[p] for p in team_players]) <= max_from_team

                # Apply AI constraints based on attempt
                relaxation_factor = constraint_relaxation[attempt]

                if attempt == 0:
                    # Strict constraints
                    self._apply_strict_ai_constraints(
                        model, flex, captain, enforcement_rules, players,
                        used_captains, synthesis, strategy, teams
                    )
                elif attempt == 1:
                    # Relaxed constraints
                    self._apply_relaxed_ai_constraints(
                        model, flex, captain, enforcement_rules, players,
                        used_captains, relaxation_factor
                    )
                else:
                    # Minimal constraints
                    self._apply_minimal_constraints(
                        model, captain, players, used_captains, ownership
                    )

                # Solve with timeout
                timeout = 5 + (attempt * 5)
                model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout))

                if pulp.LpStatus[model.status] == 'Optimal':
                    lineup = self._extract_lineup_from_solution(
                        flex, captain, players, salaries, points, ownership,
                        lineup_num, strategy, synthesis
                    )

                    if lineup and self._verify_dk_requirements(lineup, teams):
                        self.lineup_generation_stats['successes'] += 1
                        if attempt > 0:
                            self.logger.log(f"Lineup {lineup_num} succeeded on attempt {attempt + 1}", "DEBUG")
                        return lineup
                    elif lineup:
                        self.lineup_generation_stats['failures_by_reason']['dk_requirements'] += 1
                else:
                    self.lineup_generation_stats['failures_by_reason']['no_solution'] += 1

            except Exception as e:
                self.lineup_generation_stats['failures_by_reason']['exception'] += 1
                self.logger.log(f"Lineup {lineup_num} attempt {attempt + 1} error: {str(e)}", "DEBUG")

        return None

    def _apply_strict_ai_constraints(self, model, flex, captain, enforcement_rules,
                                     players, used_captains, synthesis, strategy, teams):
        """Apply strict AI constraints for first attempt"""

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

        # Apply other hard constraints
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

            elif rule in ['must_stack', 'correlation_stack', 'onslaught_stack']:
                if 'players' in constraint:
                    stack_players = [p for p in constraint['players'] if p in players]
                    if len(stack_players) >= 2:
                        # Ensure stack players are in lineup
                        min_stack = constraint.get('min_players', 2)
                        model += pulp.lpSum([flex[p] + captain[p] for p in stack_players]) >= min(min_stack, len(stack_players))

    def _apply_relaxed_ai_constraints(self, model, flex, captain, enforcement_rules,
                                      players, used_captains, relaxation_factor):
        """Apply relaxed constraints for second attempt"""

        # Expand captain pool
        top_players = self.df.nlargest(15, 'Projected_Points')['Player'].tolist()
        valid_captains = [p for p in top_players if p in players and p not in used_captains]

        if valid_captains:
            model += pulp.lpSum([captain[c] for c in valid_captains]) == 1

        # Apply soft constraints with reduced weight
        for constraint in enforcement_rules.get('soft_constraints', []):
            # Soft constraints are suggestions only in relaxed mode
            pass

    def _apply_minimal_constraints(self, model, captain, players, used_captains, ownership):
        """Apply minimal constraints for final attempt"""

        # Just avoid used captains and extreme chalk
        available_captains = []
        for p in players:
            if p not in used_captains:
                player_own = ownership.get(p, 10)
                if player_own < 60:  # Very lenient
                    available_captains.append(p)

        if available_captains:
            model += pulp.lpSum([captain[c] for c in available_captains[:20]]) == 1

    def _verify_dk_requirements(self, lineup: Dict, teams: Dict) -> bool:
        """Verify lineup meets DraftKings Showdown requirements"""

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
        """Extract lineup from solved model with complete metadata"""

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

        for player, score in rankings.items():
            if player in adjusted:
                # Normalize score to multiplier (0.7 to 1.4 range)
                if score > 0:
                    multiplier = 1.0 + min(score * 0.2, 0.4)
                else:
                    multiplier = max(0.7, 1.0 + score * 0.3)

                adjusted[player] *= multiplier

        # Apply avoidance rules
        for player in synthesis.get('avoidance_rules', []):
            if player in adjusted:
                adjusted[player] *= 0.6  # Heavy penalty for fades

        return adjusted

    def _determine_consensus_level(self, synthesis: Dict) -> str:
        """Determine the level of AI consensus"""
        captain_strategy = synthesis.get('captain_strategy', {})

        if not captain_strategy:
            return 'low'

        consensus_count = len([c for c, level in captain_strategy.items() if level == 'consensus'])
        total = len(captain_strategy)

        if total == 0:
            return 'low'

        consensus_ratio = consensus_count / total

        if consensus_ratio > 0.3:
            return 'high'
        elif consensus_ratio > 0.1:
            return 'mixed'
        else:
            return 'low'

    def _display_generation_diagnostics(self):
        """Display detailed diagnostics when generation fails"""
        import streamlit as st

        with st.expander("Generation Diagnostics", expanded=True):
            st.write("**Generation Statistics:**")
            st.write(f"Total attempts: {self.lineup_generation_stats['attempts']}")
            st.write(f"Successes: {self.lineup_generation_stats['successes']}")

            st.write("\n**Failure Reasons:**")
            for reason, count in self.lineup_generation_stats['failures_by_reason'].items():
                if count > 0:
                    st.write(f"- {reason}: {count}")

            st.write("\n**Possible Issues:**")
            st.write("• AI constraints may be too strict")
            st.write("• Salary cap constraints with required players")
            st.write("• Insufficient captain diversity")
            st.write("• Team representation requirements")

    def _display_partial_generation_reasons(self, missing_count: int):
        """Display reasons for partial generation"""
        import streamlit as st

        with st.expander("Why some lineups couldn't be generated", expanded=False):
            st.write(f"**{missing_count} lineups could not be generated**")

            # Analyze reasons
            if self.lineup_generation_stats['failures_by_reason']['dk_requirements'] > 0:
                st.write("• DraftKings team diversity requirements")

            if self.lineup_generation_stats['failures_by_reason']['no_solution'] > 0:
                st.write("• No valid solution within constraints")

            if len(self.generated_lineups) > 0:
                unique_captains = len(set([l['Captain'] for l in self.generated_lineups]))
                st.write(f"• Only {unique_captains} unique captains available")

            st.write("\n**Suggestions:**")
            st.write("• Try reducing the number of lineups")
            st.write("• Adjust AI enforcement level in sidebar")
            st.write("• Check player pool size and diversity")

# ============================================================================
# HELPER FUNCTIONS AND UTILITIES - ENHANCED
# ============================================================================

def init_ai_session_state():
    """Initialize session state for AI-driven optimization with memory management"""
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

def validate_and_process_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Enhanced validation and processing of uploaded DataFrame"""
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'fixes_applied': [],
        'stats': {}
    }

    try:
        # Column mapping for different CSV formats
        column_mappings = {
            # DraftKings format
            'first_name': 'First_Name',
            'last_name': 'Last_Name',
            'position': 'Position',
            'team': 'Team',
            'salary': 'Salary',
            'ppg_projection': 'Projected_Points',
            'ownership_projection': 'Ownership',
            # Alternative formats
            'name': 'Player',
            'proj': 'Projected_Points',
            'own': 'Ownership',
            'sal': 'Salary',
            'pos': 'Position'
        }

        # Rename columns to standard format
        df = df.rename(columns={k.lower(): v for k, v in column_mappings.items()})

        # Create Player column if needed
        if 'Player' not in df.columns:
            if 'First_Name' in df.columns and 'Last_Name' in df.columns:
                df['Player'] = df['First_Name'].fillna('') + ' ' + df['Last_Name'].fillna('')
                df['Player'] = df['Player'].str.strip()
                validation['fixes_applied'].append("Created Player names from first/last name")
            elif 'Name' in df.columns:
                df['Player'] = df['Name']
            else:
                validation['errors'].append("Cannot determine player names")
                validation['is_valid'] = False
                return df, validation

        # Ensure numeric columns are numeric
        numeric_columns = ['Salary', 'Projected_Points', 'Ownership']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    na_count = df[col].isna().sum()
                    validation['warnings'].append(f"{na_count} {col} values couldn't be converted")
                    # Fill with sensible defaults
                    if col == 'Salary':
                        df[col] = df[col].fillna(OptimizerConfig.MIN_SALARY)
                    elif col == 'Projected_Points':
                        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 10)
                    elif col == 'Ownership':
                        df[col] = df[col].fillna(OptimizerConfig.DEFAULT_OWNERSHIP)

        # Add ownership if missing using enhanced projection
        if 'Ownership' not in df.columns:
            df['Ownership'] = df.apply(
                lambda row: OptimizerConfig.get_default_ownership(
                    row.get('Position', 'FLEX'),
                    row.get('Salary', 5000)
                ), axis=1
            )
            validation['fixes_applied'].append("Added default ownership projections based on position/salary")

        # Validate ownership values
        if 'Ownership' in df.columns:
            invalid_ownership = df[(df['Ownership'] < 0) | (df['Ownership'] > 100)]
            if not invalid_ownership.empty:
                df.loc[df['Ownership'] < 0, 'Ownership'] = 0
                df.loc[df['Ownership'] > 100, 'Ownership'] = 100
                validation['warnings'].append(f"Corrected {len(invalid_ownership)} invalid ownership values")

        # Remove duplicates
        if df.duplicated(subset=['Player']).any():
            dup_count = df.duplicated(subset=['Player']).sum()
            df = df.drop_duplicates(subset=['Player'], keep='first')
            validation['warnings'].append(f"Removed {dup_count} duplicate players")

        # Check for minimum requirements
        validation['stats']['total_players'] = len(df)
        if len(df) < 6:
            validation['errors'].append(f"Only {len(df)} players (minimum 6 required)")
            validation['is_valid'] = False
        elif len(df) < 12:
            validation['warnings'].append(f"Only {len(df)} players (12+ recommended)")

        # Validate team count
        teams = df['Team'].unique()
        validation['stats']['teams'] = len(teams)
        if len(teams) != 2:
            validation['warnings'].append(f"Expected 2 teams, found {len(teams)}")

        # Position distribution
        positions = df['Position'].value_counts()
        validation['stats']['positions'] = positions.to_dict()

        # Check for QBs
        if 'QB' not in positions or positions.get('QB', 0) == 0:
            validation['warnings'].append("No QB in player pool - unusual for Showdown")

        # Validate salary cap feasibility
        min_lineup_salary = df.nsmallest(6, 'Salary')['Salary'].sum()
        max_lineup_salary = df.nlargest(6, 'Salary')['Salary'].sum()

        validation['stats']['min_possible_salary'] = min_lineup_salary
        validation['stats']['max_possible_salary'] = max_lineup_salary

        if min_lineup_salary > OptimizerConfig.SALARY_CAP:
            validation['errors'].append(f"Minimum salary ({min_lineup_salary}) exceeds cap")
            validation['is_valid'] = False

        if max_lineup_salary < 35000:
            validation['warnings'].append("Low salary players - limited lineup diversity")

    except Exception as e:
        validation['errors'].append(f"Processing error: {str(e)}")
        validation['is_valid'] = False

    return df, validation

def display_ai_recommendations(recommendations: Dict[AIStrategistType, AIRecommendation]):
    """Enhanced display of AI recommendations with better visualization"""
    st.markdown("### 🤖 Triple AI Strategic Analysis")

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

    # Detailed recommendations in tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 Game Theory",
        "🔗 Correlation",
        "🎭 Contrarian"
    ])

    with tab1:
        display_single_ai_recommendation(
            recommendations.get(AIStrategistType.GAME_THEORY),
            "Game Theory",
            "🎯"
        )

    with tab2:
        display_single_ai_recommendation(
            recommendations.get(AIStrategistType.CORRELATION),
            "Correlation",
            "🔗"
        )

    with tab3:
        display_single_ai_recommendation(
            recommendations.get(AIStrategistType.CONTRARIAN_NARRATIVE),
            "Contrarian",
            "🎭"
        )

def display_single_ai_recommendation(rec: AIRecommendation, name: str, icon: str):
    """Enhanced display of single AI recommendation"""
    if not rec:
        st.warning(f"No {name} recommendation available")
        return

    try:
        # Confidence indicator with color coding
        confidence_color = (
            "🟢" if rec.confidence > 0.7 else
            "🟡" if rec.confidence > 0.5 else
            "🔴"
        )

        st.markdown(f"#### {icon} {name} Strategy {confidence_color}")

        # Create three columns for better layout
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.metric("Confidence", f"{rec.confidence:.0%}")

            if rec.narrative:
                with st.expander("📖 Narrative", expanded=True):
                    st.write(rec.narrative[:300])

            if rec.captain_targets:
                st.markdown("**Captain Targets:**")
                for i, captain in enumerate(rec.captain_targets[:5], 1):
                    confidence_icon = "⭐" if i <= 3 else "☆"
                    st.write(f"{confidence_icon} {captain}")

        with col2:
            if rec.must_play:
                st.markdown("**Must Play:**")
                cols = st.columns(2)
                for i, player in enumerate(rec.must_play[:6]):
                    cols[i % 2].write(f"✓ {player}")

            if rec.never_play:
                st.markdown("**Fade:**")
                cols = st.columns(2)
                for i, player in enumerate(rec.never_play[:4]):
                    cols[i % 2].write(f"✗ {player}")

        with col3:
            if rec.stacks:
                st.markdown("**Stacks:**")
                for stack in rec.stacks[:3]:
                    if isinstance(stack, dict):
                        stack_type = stack.get('type', 'standard')
                        icon = {"onslaught": "💥", "bring_back": "↔️", "hidden": "🔍"}.get(stack_type, "📊")

                        if 'players' in stack and len(stack['players']) > 2:
                            # Onslaught stack
                            st.write(f"{icon} {len(stack['players'])}-man")
                        elif 'player1' in stack and 'player2' in stack:
                            st.write(f"{icon} {stack['player1'][:10]}+")

            # Enforcement summary
            if rec.enforcement_rules:
                hard_rules = len([r for r in rec.enforcement_rules if r.get('type') == 'hard'])
                soft_rules = len([r for r in rec.enforcement_rules if r.get('type') == 'soft'])
                st.markdown("**Rules:**")
                st.write(f"Hard: {hard_rules}")
                st.write(f"Soft: {soft_rules}")

        # Special insights for each AI type
        if hasattr(rec, 'ownership_leverage') and rec.ownership_leverage:
            with st.expander("🎮 Game Theory Details"):
                st.write(f"Ownership Range: {rec.ownership_leverage.get('ownership_range', [0,100])}")
                st.write(f"Min Leverage Players: {rec.ownership_leverage.get('min_leverage', 0)}")

        elif hasattr(rec, 'correlation_matrix') and rec.correlation_matrix:
            with st.expander("📊 Correlation Matrix"):
                # Show top correlations
                sorted_corr = sorted(rec.correlation_matrix.items(),
                                   key=lambda x: abs(x[1]), reverse=True)[:5]
                for pair, corr in sorted_corr:
                    corr_str = f"+{corr:.2f}" if corr > 0 else f"{corr:.2f}"
                    st.write(f"{pair}: {corr_str}")

        elif hasattr(rec, 'contrarian_angles') and rec.contrarian_angles:
            with st.expander("🎭 Contrarian Angles"):
                for angle in rec.contrarian_angles[:3]:
                    if angle:
                        st.write(f"• {angle}")

    except Exception as e:
        st.error(f"Error displaying {name}: {str(e)}")

def display_ai_synthesis(synthesis: Dict):
    """Enhanced display of AI synthesis with visual improvements"""
    try:
        st.markdown("### 🧬 AI Synthesis & Consensus")

        # Visual consensus meter
        consensus_score = synthesis.get('confidence', 0) * 100

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Consensus Level")
            if consensus_score > 70:
                st.success(f"HIGH ({consensus_score:.0f}%)")
            elif consensus_score > 50:
                st.warning(f"MODERATE ({consensus_score:.0f}%)")
            else:
                st.info(f"LOW ({consensus_score:.0f}%)")

        with col2:
            captain_strategy = synthesis.get('captain_strategy', {})
            consensus_captains = len([c for c, l in captain_strategy.items() if l == 'consensus'])
            majority_captains = len([c for c, l in captain_strategy.items() if l == 'majority'])
            st.markdown("#### Captain Agreement")
            st.write(f"Consensus: {consensus_captains}")
            st.write(f"Majority: {majority_captains}")

        with col3:
            st.markdown("#### Enforcement")
            st.write(f"Rules: {len(synthesis.get('enforcement_rules', []))}")
            st.write(f"Stacks: {len(synthesis.get('stacking_rules', []))}")

        # Captain consensus details in expandable section
        with st.expander("📋 Captain Strategy Details", expanded=False):
            captain_strategy = synthesis.get('captain_strategy', {})
            if captain_strategy:
                # Group by consensus level
                consensus = []
                majority = []
                single = []

                for captain, level in captain_strategy.items():
                    if level == 'consensus':
                        consensus.append(captain)
                    elif level == 'majority':
                        majority.append(captain)
                    else:
                        single.append((captain, level))

                if consensus:
                    st.markdown("**🏆 Consensus Captains (All 3 AIs agree):**")
                    for captain in consensus[:5]:
                        st.write(f"  • {captain}")

                if majority:
                    st.markdown("**🤝 Majority Captains (2 AIs agree):**")
                    for captain in majority[:5]:
                        st.write(f"  • {captain}")

                if single:
                    st.markdown("**💭 Single AI Captains:**")
                    for captain, ai in single[:5]:
                        st.write(f"  • {captain} ({ai})")
            else:
                st.write("No captain consensus data available")

    except Exception as e:
        st.error(f"Error displaying synthesis: {str(e)}")

def display_ai_lineup_analysis(lineups_df: pd.DataFrame, df: pd.DataFrame,
                              synthesis: Dict, field_size: str):
    """Enhanced lineup analysis with advanced visualizations"""
    if lineups_df.empty:
        st.warning("No lineups to analyze")
        return

    try:
        st.markdown("### 📊 AI-Driven Lineup Analysis")

        # Summary cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Lineups", len(lineups_df))
            unique_captains = lineups_df['Captain'].nunique() if 'Captain' in lineups_df.columns else 0
            st.metric("Unique Captains", unique_captains)

        with col2:
            if 'Projected' in lineups_df.columns:
                st.metric("Avg Projection", f"{lineups_df['Projected'].mean():.1f}")
                st.metric("Max Projection", f"{lineups_df['Projected'].max():.1f}")

        with col3:
            if 'Total_Ownership' in lineups_df.columns:
                st.metric("Avg Ownership", f"{lineups_df['Total_Ownership'].mean():.1f}%")
                st.metric("Ownership Range",
                         f"{lineups_df['Total_Ownership'].min():.0f}-{lineups_df['Total_Ownership'].max():.0f}%")

        with col4:
            if 'Leverage_Score' in lineups_df.columns:
                st.metric("Avg Leverage", f"{lineups_df['Leverage_Score'].mean():.1f}")
            if 'Salary' in lineups_df.columns:
                st.metric("Avg Salary Used", f"${lineups_df['Salary'].mean():.0f}")

        # Detailed visualizations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Use a professional color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

        # 1. Strategy Distribution
        ax1 = axes[0, 0]
        if 'AI_Strategy' in lineups_df.columns:
            strategy_counts = lineups_df['AI_Strategy'].value_counts()
            wedges, texts, autotexts = ax1.pie(
                strategy_counts.values,
                labels=strategy_counts.index,
                autopct='%1.0f%%',
                colors=colors[:len(strategy_counts)],
                startangle=90
            )
            ax1.set_title('Strategy Distribution', fontweight='bold')

            # Make percentage text more visible
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

        # 2. Captain Usage Heatmap
        ax2 = axes[0, 1]
        if 'Captain' in lineups_df.columns:
            captain_usage = lineups_df['Captain'].value_counts().head(10)

            # Create color gradient based on usage
            norm = plt.Normalize(vmin=captain_usage.min(), vmax=captain_usage.max())
            colors_captain = plt.cm.YlOrRd(norm(captain_usage.values))

            bars = ax2.barh(range(len(captain_usage)), captain_usage.values, color=colors_captain)
            ax2.set_yticks(range(len(captain_usage)))
            ax2.set_yticklabels(captain_usage.index, fontsize=9)
            ax2.set_xlabel('Times Used')
            ax2.set_title('Top 10 Captain Usage', fontweight='bold')
            ax2.invert_yaxis()

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, captain_usage.values)):
                ax2.text(val + 0.1, i, str(val), va='center')

        # 3. Ownership Distribution with target zones
        ax3 = axes[0, 2]
        if 'Total_Ownership' in lineups_df.columns:
            n, bins, patches = ax3.hist(lineups_df['Total_Ownership'], bins=20,
                                        alpha=0.7, color=colors[0], edgecolor='black')

            # Add target ownership zones
            target_range = OptimizerConfig.GPP_OWNERSHIP_TARGETS.get(field_size, (60, 90))
            ax3.axvspan(target_range[0], target_range[1], alpha=0.2, color='green',
                       label=f'Target: {target_range[0]}-{target_range[1]}%')

            # Add mean line
            mean_own = lineups_df['Total_Ownership'].mean()
            ax3.axvline(mean_own, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_own:.1f}%')

            ax3.set_xlabel('Total Ownership %')
            ax3.set_ylabel('Number of Lineups')
            ax3.set_title('Ownership Distribution', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Salary Efficiency
        ax4 = axes[1, 0]
        if 'Salary' in lineups_df.columns and 'Projected' in lineups_df.columns:
            # Scatter plot of salary vs projection
            scatter = ax4.scatter(lineups_df['Salary'], lineups_df['Projected'],
                                 c=lineups_df['Total_Ownership'] if 'Total_Ownership' in lineups_df.columns else 'blue',
                                 cmap='RdYlGn_r', alpha=0.6, edgecolors='black', linewidth=0.5)

            # Add trend line
            z = np.polyfit(lineups_df['Salary'], lineups_df['Projected'], 1)
            p = np.poly1d(z)
            ax4.plot(lineups_df['Salary'].sort_values(),
                    p(lineups_df['Salary'].sort_values()),
                    "r--", alpha=0.5, label='Trend')

            ax4.set_xlabel('Salary Used')
            ax4.set_ylabel('Projected Points')
            ax4.set_title('Salary Efficiency', fontweight='bold')
            ax4.grid(True, alpha=0.3)

            if 'Total_Ownership' in lineups_df.columns:
                plt.colorbar(scatter, ax=ax4, label='Ownership %')

        # 5. Projection Distribution with percentiles
        ax5 = axes[1, 1]
        if 'Projected' in lineups_df.columns:
            n, bins, patches = ax5.hist(lineups_df['Projected'], bins=15,
                                        alpha=0.7, color=colors[1], edgecolor='black')

            # Add percentile lines
            percentiles = [25, 50, 75, 90]
            for p in percentiles:
                val = np.percentile(lineups_df['Projected'], p)
                ax5.axvline(val, linestyle='--', alpha=0.5,
                           label=f'P{p}: {val:.1f}')

            ax5.set_xlabel('Projected Points')
            ax5.set_ylabel('Number of Lineups')
            ax5.set_title('Projection Distribution', fontweight='bold')
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)

        # 6. Player Exposure
        ax6 = axes[1, 2]

        # Calculate player exposure across all lineups
        player_exposure = {}
        for _, row in lineups_df.iterrows():
            captain = row.get('Captain')
            if captain:
                player_exposure[captain] = player_exposure.get(captain, 0) + 1.5  # Captain weight

            flex_players = row.get('FLEX', [])
            for player in flex_players:
                player_exposure[player] = player_exposure.get(player, 0) + 1

        # Convert to percentage
        total_spots = len(lineups_df) * 6  # 6 players per lineup
        player_exposure_pct = {k: (v/total_spots)*100 for k, v in player_exposure.items()}

        # Get top exposed players
        top_exposure = sorted(player_exposure_pct.items(), key=lambda x: x[1], reverse=True)[:10]

        if top_exposure:
            players, exposures = zip(*top_exposure)

            # Create gradient colors
            norm = plt.Normalize(vmin=min(exposures), vmax=max(exposures))
            colors_exp = plt.cm.RdYlGn_r(norm(exposures))

            bars = ax6.barh(range(len(players)), exposures, color=colors_exp)
            ax6.set_yticks(range(len(players)))
            ax6.set_yticklabels(players, fontsize=9)
            ax6.set_xlabel('Exposure %')
            ax6.set_title('Top 10 Player Exposure', fontweight='bold')
            ax6.invert_yaxis()

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, exposures)):
                ax6.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=8)

        plt.suptitle(f'Lineup Analysis - {field_size.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        st.pyplot(fig)

        # Additional Analysis Sections
        st.markdown("---")

        # Stack Analysis
        if 'FLEX' in lineups_df.columns:
            st.markdown("#### 📈 Stacking Patterns")

            stack_patterns = analyze_stacking_patterns(lineups_df, df)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Lineups with QB Stack", f"{stack_patterns['qb_stacks']}%")

            with col2:
                st.metric("Bring-back Stacks", f"{stack_patterns['bring_backs']}%")

            with col3:
                st.metric("Onslaught Stacks", f"{stack_patterns['onslaughts']}%")

            if stack_patterns['common_stacks']:
                with st.expander("Most Common Stack Combinations"):
                    for stack, count in stack_patterns['common_stacks'][:5]:
                        st.write(f"• {stack}: {count} lineups")

    except Exception as e:
        st.error(f"Error in lineup analysis: {str(e)}")
        get_logger().log_exception(e, "display_ai_lineup_analysis")

def analyze_stacking_patterns(lineups_df: pd.DataFrame, player_df: pd.DataFrame) -> Dict:
    """Analyze stacking patterns across lineups"""
    patterns = {
        'qb_stacks': 0,
        'bring_backs': 0,
        'onslaughts': 0,
        'common_stacks': []
    }

    try:
        # Create player position/team lookup
        player_info = player_df.set_index('Player')[['Position', 'Team']].to_dict('index')

        stack_combos = []

        for _, lineup in lineups_df.iterrows():
            captain = lineup.get('Captain')
            flex_players = lineup.get('FLEX', [])

            if not captain or not flex_players:
                continue

            all_players = [captain] + flex_players

            # Check for QB stacks
            captain_info = player_info.get(captain, {})
            if captain_info.get('Position') == 'QB':
                captain_team = captain_info.get('Team')
                teammates = [p for p in flex_players
                           if player_info.get(p, {}).get('Team') == captain_team]

                if teammates:
                    patterns['qb_stacks'] += 1

                    # Check for bring-back
                    opp_players = [p for p in flex_players
                                 if player_info.get(p, {}).get('Team') != captain_team]
                    if opp_players:
                        patterns['bring_backs'] += 1

            # Check for onslaught (4+ from same team)
            team_counts = {}
            for player in all_players:
                team = player_info.get(player, {}).get('Team')
                if team:
                    team_counts[team] = team_counts.get(team, 0) + 1

            for team, count in team_counts.items():
                if count >= 4:
                    patterns['onslaughts'] += 1
                    break

            # Track common stacks
            if len(all_players) >= 2:
                # Create sorted tuple of first 3 players for consistency
                stack_key = tuple(sorted(all_players[:3]))
                stack_combos.append(stack_key)

        # Convert to percentages
        total_lineups = len(lineups_df)
        if total_lineups > 0:
            patterns['qb_stacks'] = round(100 * patterns['qb_stacks'] / total_lineups, 1)
            patterns['bring_backs'] = round(100 * patterns['bring_backs'] / total_lineups, 1)
            patterns['onslaughts'] = round(100 * patterns['onslaughts'] / total_lineups, 1)

        # Find most common stacks
        if stack_combos:
            stack_counter = Counter(stack_combos)
            patterns['common_stacks'] = [
                (" + ".join(stack[:2]), count)
                for stack, count in stack_counter.most_common(5)
            ]

    except Exception as e:
        get_logger().log(f"Error analyzing stacks: {e}", "WARNING")

    return patterns

def export_lineups_draftkings(lineups_df: pd.DataFrame) -> str:
    """Export lineups in DraftKings Showdown format"""
    try:
        dk_lineups = []

        for idx, row in lineups_df.iterrows():
            flex_players = row['FLEX'] if isinstance(row['FLEX'], list) else []

            # Ensure exactly 5 FLEX
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

        dk_df = pd.DataFrame(dk_lineups)
        return dk_df.to_csv(index=False)

    except Exception as e:
        get_logger().log(f"Export error: {e}", "ERROR")
        return ""

def export_detailed_lineups(lineups_df: pd.DataFrame) -> str:
    """Export detailed lineup information with all metadata"""
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
                'Salary_Remaining': row.get('Salary_Remaining', 0),
                'Total_Ownership': row.get('Total_Ownership', 0),
                'Ownership_Tier': row.get('Ownership_Tier', ''),
                'Leverage_Score': row.get('Leverage_Score', 0),
                'AI_Enforced': row.get('AI_Enforced', False),
                'Confidence': row.get('Confidence', 0)
            }

            # Add FLEX players
            flex_players = row.get('FLEX', [])
            for i, player in enumerate(flex_players):
                lineup_detail[f'FLEX_{i+1}'] = player

            # Ensure all 5 FLEX columns exist
            for i in range(len(flex_players), 5):
                lineup_detail[f'FLEX_{i+1}'] = ''

            detailed.append(lineup_detail)

        detailed_df = pd.DataFrame(detailed)
        return detailed_df.to_csv(index=False)

    except Exception as e:
        get_logger().log(f"Detailed export error: {e}", "ERROR")
        return ""

def display_lineup_editor(lineups_df: pd.DataFrame, player_df: pd.DataFrame) -> pd.DataFrame:
    """Allow manual editing of generated lineups"""
    st.markdown("### ✏️ Lineup Editor")

    if lineups_df.empty:
        st.warning("No lineups to edit")
        return lineups_df

    edited_df = lineups_df.copy()

    # Select lineup to edit
    lineup_num = st.selectbox(
        "Select lineup to edit",
        options=lineups_df['Lineup'].tolist(),
        format_func=lambda x: f"Lineup {x} - {lineups_df[lineups_df['Lineup']==x]['Strategy'].values[0]}"
    )

    # Get current lineup
    current_lineup = lineups_df[lineups_df['Lineup'] == lineup_num].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Current Lineup")
        st.write(f"**Captain:** {current_lineup['Captain']}")
        st.write("**FLEX:**")
        for i, player in enumerate(current_lineup['FLEX']):
            st.write(f"{i+1}. {player}")

        st.metric("Projected", f"{current_lineup['Projected']:.1f}")
        st.metric("Salary", f"${current_lineup['Salary']:,}")
        st.metric("Ownership", f"{current_lineup['Total_Ownership']:.1f}%")

    with col2:
        st.markdown("#### Edit Lineup")

        # Captain selection
        available_players = player_df['Player'].tolist()
        new_captain = st.selectbox(
            "Captain",
            options=available_players,
            index=available_players.index(current_lineup['Captain']) if current_lineup['Captain'] in available_players else 0
        )

        # FLEX selections
        new_flex = []
        for i in range(5):
            current_player = current_lineup['FLEX'][i] if i < len(current_lineup['FLEX']) else None

            # Filter out captain and already selected FLEX
            available_for_flex = [p for p in available_players
                                 if p != new_captain and p not in new_flex]

            flex_player = st.selectbox(
                f"FLEX {i+1}",
                options=available_for_flex,
                index=available_for_flex.index(current_player) if current_player in available_for_flex else 0,
                key=f"flex_{lineup_num}_{i}"
            )
            new_flex.append(flex_player)

        if st.button("Apply Changes", type="primary"):
            # Validate lineup
            validation_errors = validate_lineup_edit(
                new_captain, new_flex, player_df
            )

            if validation_errors:
                for error in validation_errors:
                    st.error(error)
            else:
                # Update lineup
                idx = edited_df[edited_df['Lineup'] == lineup_num].index[0]

                # Recalculate metrics
                captain_salary = player_df[player_df['Player'] == new_captain]['Salary'].values[0]
                flex_salaries = [player_df[player_df['Player'] == p]['Salary'].values[0] for p in new_flex]
                total_salary = captain_salary * 1.5 + sum(flex_salaries)

                captain_proj = player_df[player_df['Player'] == new_captain]['Projected_Points'].values[0]
                flex_proj = [player_df[player_df['Player'] == p]['Projected_Points'].values[0] for p in new_flex]
                total_proj = captain_proj * 1.5 + sum(flex_proj)

                captain_own = player_df[player_df['Player'] == new_captain]['Ownership'].values[0]
                flex_own = [player_df[player_df['Player'] == p]['Ownership'].values[0] for p in new_flex]
                total_own = captain_own * 1.5 + sum(flex_own)

                edited_df.at[idx, 'Captain'] = new_captain
                edited_df.at[idx, 'FLEX'] = new_flex
                edited_df.at[idx, 'Salary'] = int(total_salary)
                edited_df.at[idx, 'Projected'] = round(total_proj, 2)
                edited_df.at[idx, 'Total_Ownership'] = round(total_own, 1)
                edited_df.at[idx, 'Salary_Remaining'] = int(50000 - total_salary)

                st.success("Lineup updated successfully!")
                st.rerun()

    return edited_df

def validate_lineup_edit(captain: str, flex: List[str], player_df: pd.DataFrame) -> List[str]:
    """Validate edited lineup"""
    errors = []

    # Check for duplicates
    all_players = [captain] + flex
    if len(all_players) != len(set(all_players)):
        errors.append("Duplicate players detected")

    # Check salary
    captain_salary = player_df[player_df['Player'] == captain]['Salary'].values[0]
    flex_salaries = [player_df[player_df['Player'] == p]['Salary'].values[0] for p in flex]
    total_salary = captain_salary * 1.5 + sum(flex_salaries)

    if total_salary > 50000:
        errors.append(f"Salary exceeds cap: ${total_salary:,}")

    # Check team diversity
    teams = {}
    for player in all_players:
        team = player_df[player_df['Player'] == player]['Team'].values[0]
        teams[team] = teams.get(team, 0) + 1

    if len(teams) < 2:
        errors.append("Must have players from both teams")

    for team, count in teams.items():
        if count > 5:
            errors.append(f"Too many players from {team}: {count}")

    return errors

# ============================================================================
# MAIN STREAMLIT APPLICATION - COMPLETE
# ============================================================================

def main():
    """Main application with complete functionality"""

    # Page configuration
    st.set_page_config(
        page_title="NFL GPP AI-Chef Optimizer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better UI
    st.markdown("""
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            padding-left: 20px;
            padding-right: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🏈 NFL GPP Tournament Optimizer - AI-as-Chef Edition")
    st.markdown("*Version 6.4 - Triple AI System with Enhanced Features*")

    # Initialize session state
    init_ai_session_state()

    # Track optimization count
    if st.session_state['optimization_count'] > 10:
        # Memory cleanup
        st.session_state['optimization_history'] = deque(maxlen=5)
        st.session_state['optimization_count'] = 0

    # Sidebar configuration
    with st.sidebar:
        st.header("🤖 AI-Chef Configuration")

        # Quick Settings
        with st.expander("⚡ Quick Settings", expanded=True):
            quick_mode = st.radio(
                "Optimization Mode",
                ["Balanced", "Aggressive", "Conservative"],
                help="Quick preset configurations"
            )

            if quick_mode == "Aggressive":
                enforcement_level = 'Mandatory'
                field_preset = 'large_field_aggressive'
            elif quick_mode == "Conservative":
                enforcement_level = 'Moderate'
                field_preset = 'small_field'
            else:
                enforcement_level = 'Strong'
                field_preset = 'large_field'

        st.markdown("---")

        # AI Mode Selection
        st.markdown("### AI Enforcement Level")
        enforcement_level = st.select_slider(
            "How strictly to enforce AI decisions",
            options=['Advisory', 'Moderate', 'Strong', 'Mandatory'],
            value=enforcement_level,
            help="Mandatory = AI decisions are hard constraints"
        )

        if enforcement_level != 'Mandatory':
            st.info("💡 AI-as-Chef mode works best with Mandatory enforcement")

        # Contest Type
        st.markdown("### Contest Type")
        contest_type = st.selectbox(
            "Select GPP Type",
            list(OptimizerConfig.FIELD_SIZES.keys()),
            index=list(OptimizerConfig.FIELD_SIZES.values()).index(field_preset),
            help="Different contests require different AI strategies"
        )
        field_size = OptimizerConfig.FIELD_SIZES[contest_type]

        # Display field configuration
        ai_config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size, {})
        with st.expander("📋 Field Configuration"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Min Captains:** {ai_config.get('min_unique_captains', 10)}")
                st.write(f"**Max Chalk:** {ai_config.get('max_chalk_players', 2)}")
            with col2:
                st.write(f"**Min Leverage:** {ai_config.get('min_leverage_players', 2)}")
                st.write(f"**Ownership Target:** {ai_config.get('min_total_ownership', 60)}-{ai_config.get('max_total_ownership', 90)}%")

        st.markdown("---")

        # API Configuration
        st.markdown("### 🔌 AI Connection")
        api_mode = st.radio(
            "AI Input Mode",
            ["Manual (Copy/Paste)", "API (Automated)"],
            help="API mode for automatic AI analysis"
        )

        api_manager = None
        use_api = False

        if api_mode == "API (Automated)":
            api_key = st.text_input(
                "Claude API Key",
                type="password",
                placeholder="sk-ant-api03-...",
                help="Get your key from console.anthropic.com"
            )

            if api_key:
                if st.button("🔗 Connect to Claude"):
                    try:
                        with st.spinner("Connecting..."):
                            api_manager = ClaudeAPIManager(api_key)
                            if api_manager.validate_connection():
                                st.success("✅ Connected to Claude AI")
                                st.session_state['api_manager'] = api_manager
                                use_api = True
                            else:
                                st.error("❌ Connection validation failed")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

                # Check existing connection
                if 'api_manager' in st.session_state and st.session_state['api_manager']:
                    api_manager = st.session_state['api_manager']
                    use_api = True
                    st.success("✅ Using existing connection")

                    # Show API stats
                    with st.expander("📊 API Statistics"):
                        stats = api_manager.get_stats()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Requests", stats['requests'])
                            st.metric("Cache Hits", f"{stats['cache_hit_rate']*100:.0f}%")
                        with col2:
                            st.metric("Errors", stats['errors'])
                            st.metric("Avg Response", f"{stats['avg_response_time']:.1f}s")

        st.markdown("---")

        # Advanced Settings
        with st.expander("⚙️ Advanced Settings"):
            st.markdown("### AI Weights")

            col1, col2, col3 = st.columns(3)
            with col1:
                gt_weight = st.slider("Game Theory", 0.0, 1.0, 0.35, 0.05)
            with col2:
                corr_weight = st.slider("Correlation", 0.0, 1.0, 0.35, 0.05)
            with col3:
                contra_weight = st.slider("Contrarian", 0.0, 1.0, 0.30, 0.05)

            # Normalize weights
            total_weight = gt_weight + corr_weight + contra_weight
            if total_weight > 0:
                OptimizerConfig.AI_WEIGHTS = {
                    AIStrategistType.GAME_THEORY: gt_weight / total_weight,
                    AIStrategistType.CORRELATION: corr_weight / total_weight,
                    AIStrategistType.CONTRARIAN_NARRATIVE: contra_weight / total_weight
                }

            st.markdown("### Performance")
            parallel_threads = st.slider("Parallel Threads", 1, 8, 4)
            OptimizerConfig.MAX_PARALLEL_THREADS = parallel_threads

        # Debug Panel
        with st.expander("🐛 Debug & Monitoring"):
            if st.button("📊 Show Performance"):
                perf = get_performance_monitor()
                perf.display_metrics()

            if st.button("📝 Show Logs"):
                logger = get_logger()
                logger.display_log_summary()

            if st.button("🗑️ Clear Cache"):
                st.cache_data.clear()
                if api_manager:
                    api_manager.clear_cache()
                st.success("Cache cleared")

    # Main Content Area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Data Upload",
        "🤖 AI Analysis",
        "🎯 Generate Lineups",
        "📊 Results & Analysis",
        "📤 Export"
    ])

    with tab1:
        st.markdown("## Data & Game Configuration")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload DraftKings Showdown CSV",
                type=['csv'],
                help="Export player pool from DraftKings Showdown contest"
            )

            # Sample data option
            if st.checkbox("Use sample data for testing"):
                sample_data = create_sample_data()
                st.session_state['df'] = sample_data
                st.success("Sample data loaded")

        with col2:
            st.markdown("### File Requirements")
            st.write("Required columns:")
            st.write("• Player names")
            st.write("• Position")
            st.write("• Team")
            st.write("• Salary")
            st.write("• Projected Points")
            st.write("• Ownership % (optional)")

        if uploaded_file is not None or 'df' in st.session_state and not st.session_state['df'].empty:
            try:
                # Load and validate data
                if uploaded_file is not None:
                    raw_df = pd.read_csv(uploaded_file)
                    df, validation = validate_and_process_dataframe(raw_df)
                else:
                    df = st.session_state['df']
                    validation = {'is_valid': True, 'errors': [], 'warnings': [], 'fixes_applied': []}

                # Display validation results
                if validation['errors']:
                    st.error("❌ Validation Errors:")
                    for error in validation['errors']:
                        st.write(f"  • {error}")

                if validation['warnings']:
                    st.warning("⚠️ Warnings:")
                    for warning in validation['warnings']:
                        st.write(f"  • {warning}")

                if validation['fixes_applied']:
                    st.info("✅ Fixes Applied:")
                    for fix in validation['fixes_applied']:
                        st.write(f"  • {fix}")

                if not validation['is_valid']:
                    st.error("Cannot proceed with optimization due to validation errors")
                    st.stop()

                # Store in session state
                st.session_state['df'] = df

                st.success(f"✅ Loaded {len(df)} players successfully!")

                # Game configuration
                st.markdown("### Game Setup")

                col1, col2, col3, col4 = st.columns(4)

                teams = df['Team'].unique()
                with col1:
                    team_display = f"{teams[0]} vs {teams[1]}" if len(teams) >= 2 else "Teams"
                    game_teams = st.text_input("Teams", team_display)

                with col2:
                    total = st.number_input("O/U Total", 30.0, 70.0, 48.5, 0.5)

                with col3:
                    spread = st.number_input("Spread", -20.0, 20.0, -3.0, 0.5)

                with col4:
                    weather = st.selectbox("Weather", ["Clear", "Wind", "Rain", "Snow"])

                game_info = {
                    'teams': game_teams,
                    'total': total,
                    'spread': spread,
                    'weather': weather
                }

                st.session_state['game_info'] = game_info

                # Display player pool summary
                st.markdown("### Player Pool Summary")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Players", len(df))
                    st.metric("Teams", len(df['Team'].unique()))

                with col2:
                    st.metric("Avg Salary", f"${df['Salary'].mean():.0f}")
                    st.metric("Salary Range", f"${df['Salary'].min()}-${df['Salary'].max()}")

                with col3:
                    st.metric("Avg Projection", f"{df['Projected_Points'].mean():.1f}")
                    st.metric("Proj Range", f"{df['Projected_Points'].min():.1f}-{df['Projected_Points'].max():.1f}")

                with col4:
                    st.metric("Avg Ownership", f"{df['Ownership'].mean():.1f}%")
                    positions = df['Position'].value_counts()
                    st.metric("Positions", f"{len(positions)} types")

                # Show player data
                with st.expander("👀 View Player Pool", expanded=False):
                    # Sorting options
                    sort_col = st.selectbox(
                        "Sort by",
                        ['Projected_Points', 'Salary', 'Ownership', 'Value'],
                        key="sort_players"
                    )

                    # Add value column for display
                    display_df = df.copy()
                    display_df['Value'] = display_df['Projected_Points'] / (display_df['Salary'] / 1000)

                    # Sort and display
                    display_df = display_df.sort_values(sort_col, ascending=False)

                    # Format columns
                    st.dataframe(
                        display_df[['Player', 'Position', 'Team', 'Salary',
                                  'Projected_Points', 'Ownership', 'Value']].style.format({
                            'Salary': '${:,.0f}',
                            'Projected_Points': '{:.1f}',
                            'Ownership': '{:.1f}%',
                            'Value': '{:.2f}'
                        }),
                        use_container_width=True,
                        height=400
                    )

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                get_logger().log_exception(e, "file_processing")

    with tab2:
        st.markdown("## 🤖 AI Strategic Analysis")

        if 'df' not in st.session_state or st.session_state['df'].empty:
            st.warning("Please upload data first")
            st.stop()

        df = st.session_state['df']
        game_info = st.session_state.get('game_info', {})

        # AI Strategy Generation
        if st.button("🧠 Generate AI Strategies", type="primary", use_container_width=True):
            try:
                # Initialize optimizer
                optimizer = AIChefGPPOptimizer(df, game_info, field_size, api_manager)

                # Get AI strategies
                with st.spinner("🤖 Consulting Triple AI System..."):
                    ai_recommendations = optimizer.get_triple_ai_strategies(use_api=use_api)

                if not ai_recommendations:
                    st.error("Failed to get AI recommendations")
                else:
                    # Store recommendations
                    st.session_state['ai_recommendations'] = ai_recommendations
                    st.session_state['optimizer'] = optimizer

                    # Display recommendations
                    display_ai_recommendations(ai_recommendations)

                    # Synthesize strategies
                    with st.spinner("🧬 Synthesizing AI strategies..."):
                        ai_strategy = optimizer.synthesize_ai_strategies(ai_recommendations)

                    # Store synthesis
                    st.session_state['ai_synthesis'] = ai_strategy['synthesis']
                    st.session_state['ai_strategy'] = ai_strategy

                    # Display synthesis
                    display_ai_synthesis(ai_strategy['synthesis'])

                    # Validation display
                    if ai_strategy['validation']['errors']:
                        st.error("⚠️ Validation Issues:")
                        for error in ai_strategy['validation']['errors'][:3]:
                            st.write(f"• {error}")

                    st.success("✅ AI Analysis Complete!")

            except Exception as e:
                st.error(f"Error during AI analysis: {str(e)}")
                get_logger().log_exception(e, "ai_analysis")

        # Display existing recommendations if available
        elif 'ai_recommendations' in st.session_state and st.session_state['ai_recommendations']:
            display_ai_recommendations(st.session_state['ai_recommendations'])
            if 'ai_synthesis' in st.session_state:
                display_ai_synthesis(st.session_state['ai_synthesis'])

    with tab3:
        st.markdown("## 🎯 Generate AI-Driven Lineups")

        if 'ai_strategy' not in st.session_state:
            st.warning("Please generate AI strategies first")
            st.stop()

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            num_lineups = st.slider(
                "Number of Lineups",
                min_value=1,
                max_value=150,
                value=20,
                step=5,
                help="AI will determine optimal strategy distribution"
            )

        with col2:
            st.markdown("### Settings")
            st.metric("Field Size", field_size.replace('_', ' ').title())
            st.metric("AI Mode", enforcement_level)

        with col3:
            st.markdown("### Quick Stats")
            synthesis = st.session_state.get('ai_synthesis', {})
            st.metric("AI Confidence", f"{synthesis.get('confidence', 0):.0%}")
            st.metric("Consensus Captains", len([
                c for c, l in synthesis.get('captain_strategy', {}).items()
                if l == 'consensus'
            ]))

        if st.button("🚀 Generate Lineups", type="primary", use_container_width=True):
            try:
                optimizer = st.session_state.get('optimizer')
                ai_strategy = st.session_state.get('ai_strategy')

                if not optimizer or not ai_strategy:
                    st.error("Missing optimizer or strategy")
                    st.stop()

                # Track optimization
                st.session_state['optimization_count'] += 1

                # Generate lineups
                with st.spinner(f"🔨 Building {num_lineups} AI-enforced lineups..."):
                    lineups_df = optimizer.generate_ai_driven_lineups(num_lineups, ai_strategy)

                if not lineups_df.empty:
                    # Store in session
                    st.session_state['lineups_df'] = lineups_df
                    st.session_state['last_optimization_time'] = datetime.now()

                    # Display success metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Lineups Generated", len(lineups_df))

                    with col2:
                        unique_captains = lineups_df['Captain'].nunique()
                        st.metric("Unique Captains", unique_captains)

                    with col3:
                        if 'Total_Ownership' in lineups_df.columns:
                            avg_own = lineups_df['Total_Ownership'].mean()
                            st.metric("Avg Ownership", f"{avg_own:.1f}%")

                    # Quick lineup preview
                    st.markdown("### 👀 Quick Preview")
                    for idx, row in lineups_df.head(3).iterrows():
                        with st.expander(f"Lineup {row['Lineup']} - {row['Strategy']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Captain:** {row['Captain']} ({row['Captain_Own%']:.1f}%)")
                                st.write("**FLEX:**")
                                for player in row['FLEX']:
                                    st.write(f"• {player}")
                            with col2:
                                st.metric("Projected", f"{row['Projected']:.1f}")
                                st.metric("Salary", f"${row['Salary']:,}")
                                st.metric("Ownership", f"{row['Total_Ownership']:.1f}%")

                    st.success(f"✅ Successfully generated {len(lineups_df)} lineups!")

                    # Display enforcement
                    logger = get_logger()
                    logger.display_ai_enforcement()
                else:
                    st.error("No lineups generated. Check constraints and try again.")

            except Exception as e:
                st.error(f"Generation error: {str(e)}")
                get_logger().log_exception(e, "lineup_generation", critical=True)

    with tab4:
        st.markdown("## 📊 Results & Analysis")

        if 'lineups_df' not in st.session_state or st.session_state['lineups_df'].empty:
            st.warning("No lineups to analyze. Generate lineups first.")
            st.stop()

        lineups_df = st.session_state['lineups_df']
        df = st.session_state.get('df', pd.DataFrame())
        synthesis = st.session_state.get('ai_synthesis', {})

        # Analysis tabs
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
            "📊 Statistical Analysis",
            "📝 Lineup Details",
            "✏️ Edit Lineups"
        ])

        with analysis_tab1:
            display_ai_lineup_analysis(lineups_df, df, synthesis, field_size)

        with analysis_tab2:
            st.markdown("### 📝 Detailed Lineup View")

            # Filters
            col1, col2, col3 = st.columns(3)

            with col1:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Lineup", "Projected", "Total_Ownership", "Salary", "Leverage_Score"],
                    key="sort_lineups"
                )

            with col2:
                filter_strategy = st.selectbox(
                    "Filter by Strategy",
                    ["All"] + lineups_df['Strategy'].unique().tolist(),
                    key="filter_strategy"
                )

            with col3:
                show_count = st.slider("Show lineups", 5, len(lineups_df

show_count = st.slider("Show lineups", 5, len(lineups_df), min(10, len(lineups_df)))

            # Apply filters
            display_df = lineups_df.copy()
            if filter_strategy != "All":
                display_df = display_df[display_df['Strategy'] == filter_strategy]

            # Sort
            display_df = display_df.sort_values(sort_by, ascending=(sort_by == "Lineup"))

            # Display lineups
            for idx, row in display_df.head(show_count).iterrows():
                with st.expander(
                    f"Lineup {row['Lineup']} - {row.get('AI_Strategy', 'Unknown')} | "
                    f"Proj: {row.get('Projected', 0):.1f} | Own: {row.get('Total_Ownership', 0):.1f}%"
                ):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown("**Roster:**")
                        st.write(f"**CPT:** {row['Captain']}")
                        for i, player in enumerate(row['FLEX']):
                            st.write(f"FLEX {i+1}: {player}")

                    with col2:
                        st.markdown("**Financial:**")
                        st.write(f"Salary: ${row.get('Salary', 0):,}")
                        st.write(f"Remaining: ${row.get('Salary_Remaining', 0):,}")
                        st.write(f"")
                        st.markdown("**Performance:**")
                        st.write(f"Projected: {row.get('Projected', 0):.1f}")
                        st.write(f"Leverage: {row.get('Leverage_Score', 0):.1f}")

                    with col3:
                        st.markdown("**Ownership:**")
                        st.write(f"Captain: {row.get('Captain_Own%', 0):.1f}%")
                        st.write(f"Total: {row.get('Total_Ownership', 0):.1f}%")
                        st.write(f"Tier: {row.get('Ownership_Tier', 'Unknown')}")

                    with col4:
                        st.markdown("**AI Info:**")
                        st.write(f"Strategy: {row.get('AI_Strategy', 'N/A')}")
                        st.write(f"Confidence: {row.get('Confidence', 0):.0%}")
                        st.write(f"Enforced: {'✅' if row.get('AI_Enforced') else '❌'}")

        with analysis_tab3:
            # Lineup editor
            edited_df = display_lineup_editor(lineups_df, df)
            if not edited_df.equals(lineups_df):
                st.session_state['lineups_df'] = edited_df
                st.success("Lineups updated!")

    with tab5:
        st.markdown("## 📤 Export Options")

        if 'lineups_df' not in st.session_state or st.session_state['lineups_df'].empty:
            st.warning("No lineups to export. Generate lineups first.")
            st.stop()

        lineups_df = st.session_state['lineups_df']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🏈 DraftKings Format")
            st.write("Standard DK upload format with CPT and FLEX positions")

            dk_csv = export_lineups_draftkings(lineups_df)
            if dk_csv:
                st.download_button(
                    label="📥 Download DK CSV",
                    data=dk_csv,
                    file_name=f"dk_lineups_{field_size}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col2:
            st.markdown("### 📊 Detailed Format")
            st.write("Full lineup details with all metrics and metadata")

            detailed_csv = export_detailed_lineups(lineups_df)
            if detailed_csv:
                st.download_button(
                    label="📥 Download Detailed CSV",
                    data=detailed_csv,
                    file_name=f"detailed_lineups_{field_size}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col3:
            st.markdown("### 🧬 AI Strategy Export")
            st.write("AI recommendations and synthesis for future reference")

            synthesis = st.session_state.get('ai_synthesis', {})
            if synthesis:
                strategy_export = {
                    'timestamp': datetime.now().isoformat(),
                    'field_size': field_size,
                    'contest_type': contest_type,
                    'synthesis': synthesis,
                    'lineup_count': len(lineups_df),
                    'ai_weights': OptimizerConfig.AI_WEIGHTS,
                    'enforcement_level': enforcement_level
                }

                strategy_json = json.dumps(strategy_export, default=str, indent=2)
                st.download_button(
                    label="📥 Download AI Strategy",
                    data=strategy_json,
                    file_name=f"ai_strategy_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )

        # Session export
        st.markdown("---")
        st.markdown("### 💾 Session Backup")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📦 Export Full Session", use_container_width=True):
                try:
                    session_data = {
                        'timestamp': datetime.now().isoformat(),
                        'lineups': lineups_df.to_dict('records'),
                        'ai_synthesis': st.session_state.get('ai_synthesis'),
                        'game_info': st.session_state.get('game_info'),
                        'field_size': field_size,
                        'settings': {
                            'enforcement_level': enforcement_level,
                            'ai_weights': OptimizerConfig.AI_WEIGHTS
                        }
                    }

                    # Compress with pickle
                    import pickle
                    import base64

                    session_bytes = pickle.dumps(session_data)
                    session_b64 = base64.b64encode(session_bytes).decode()

                    st.download_button(
                        label="💾 Download Session Backup",
                        data=session_b64,
                        file_name=f"session_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl",
                        mime="application/octet-stream"
                    )

                except Exception as e:
                    st.error(f"Export error: {str(e)}")

        with col2:
            uploaded_session = st.file_uploader(
                "Restore Session Backup",
                type=['pkl'],
                help="Upload a previously saved session"
            )

            if uploaded_session:
                try:
                    import pickle
                    import base64

                    session_data = pickle.loads(uploaded_session.read())

                    # Restore session
                    st.session_state['lineups_df'] = pd.DataFrame(session_data['lineups'])
                    st.session_state['ai_synthesis'] = session_data['ai_synthesis']
                    st.session_state['game_info'] = session_data['game_info']

                    st.success("Session restored successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Restore error: {str(e)}")

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

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        # Set up logging
        logger = get_logger()
        logger.log("Application started", "INFO")

        # Run main application
        main()

    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        get_logger().log_exception(e, "main_entry", critical=True)

        # Show debug information
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
