# NFL GPP DUAL-AI OPTIMIZER - CONSOLIDATED VERSION
# Part 1: Configuration, Monitoring, and Base Classes
# Version 6.4 - Single File Structure

# ============================================================================
# ALL IMPORTS CONSOLIDATED AT TOP
# ============================================================================

# Standard library imports
import pandas as pd
import numpy as np
import json
import threading
import hashlib
import os
import traceback
import time
import pickle
import base64
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from enum import Enum
from collections import deque, defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

# Third-party imports
import streamlit as st
import pulp
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class StrategyType(Enum):
    """GPP Strategy Types - AI-Driven and Legacy"""
    AI_CONSENSUS = 'ai_consensus'
    AI_MAJORITY = 'ai_majority'
    AI_CONTRARIAN = 'ai_contrarian'
    AI_CORRELATION = 'ai_correlation'
    AI_GAME_THEORY = 'ai_game_theory'
    AI_MIXED = 'ai_mixed'
    LEVERAGE = 'leverage'
    CONTRARIAN = 'contrarian'
    GAME_STACK = 'game_stack'
    STARS_SCRUBS = 'stars_scrubs'
    CORRELATION = 'correlation'
    SUPER_CONTRARIAN = 'super_contrarian'

class AIEnforcementLevel(Enum):
    """How strictly to enforce AI decisions"""
    MANDATORY = 'mandatory'
    STRONG = 'strong'
    MODERATE = 'moderate'
    SUGGESTION = 'suggestion'

class AIStrategistType(Enum):
    """Three AI Strategist Types"""
    GAME_THEORY = 'game_theory'
    CORRELATION = 'correlation'
    CONTRARIAN_NARRATIVE = 'contrarian_narrative'

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================

class OptimizerConfig:
    """Core configuration for GPP optimizer - AI-as-Chef version"""

    # Salary and roster constraints
    SALARY_CAP = 50000
    MIN_SALARY = 3000
    MAX_PLAYERS_PER_TEAM = 5
    ROSTER_SIZE = 6

    # Enhanced default ownership with position and salary correlation
    @staticmethod
    def get_default_ownership(position: str, salary: int = None) -> float:
        """Get position and salary-based default ownership"""
        base_ownership = {
            'QB': 15.0,
            'RB': 12.0,
            'WR': 10.0,
            'TE': 8.0,
            'K': 5.0,
            'DST': 5.0,
            'FLEX': 7.0
        }
        
        ownership = base_ownership.get(position, 5.0)
        
        # Adjust for salary if provided
        if salary:
            if salary >= 9000:  # High salary
                ownership *= 1.5
            elif salary <= 4000:  # Min salary
                ownership *= 0.7
                
        return min(ownership, 50.0)  # Cap at 50%

    # Default static ownership for backward compatibility
    DEFAULT_OWNERSHIP = 5.0

    # AI Configuration
    AI_ENFORCEMENT_MODE = AIEnforcementLevel.MANDATORY
    REQUIRE_AI_FOR_GENERATION = False
    MIN_AI_CONFIDENCE = 0.5
    MAX_AI_CONFIDENCE_OVERRIDE = 0.9

    # Triple AI Weights
    AI_WEIGHTS = {
        AIStrategistType.GAME_THEORY: 0.35,
        AIStrategistType.CORRELATION: 0.35,
        AIStrategistType.CONTRARIAN_NARRATIVE: 0.30
    }

    # AI Consensus Requirements
    AI_CONSENSUS_THRESHOLDS = {
        'captain': 2,
        'must_play': 3,
        'never_play': 2,
        'stack_required': 2
    }

    # GPP Ownership targets
    GPP_OWNERSHIP_TARGETS = {
        'small_field': (80, 120),
        'medium_field': (70, 100),
        'large_field': (60, 90),
        'large_field_aggressive': (45, 75),
        'milly_maker': (40, 70),
        'super_contrarian': (30, 60)
    }

    # Field size configurations
    FIELD_SIZE_CONFIGS = {
        'small_field': {
            'min_unique_captains': 3,
            'max_chalk_players': 4,
            'min_leverage_players': 0,
            'max_ownership_per_player': 40,
            'min_total_ownership': 80,
            'max_total_ownership': 120,
            'ai_enforcement': AIEnforcementLevel.MODERATE,
            'ai_strategy_distribution': {
                'ai_consensus': 0.4,
                'ai_majority': 0.3,
                'ai_mixed': 0.3
            },
            'stack_requirements': {
                'min_stacks': 1,
                'max_stacks': 2,
                'bring_back_required': False
            }
        },
        'medium_field': {
            'min_unique_captains': 6,
            'max_chalk_players': 3,
            'min_leverage_players': 1,
            'max_ownership_per_player': 25,
            'min_total_ownership': 70,
            'max_total_ownership': 100,
            'ai_enforcement': AIEnforcementLevel.STRONG,
            'ai_strategy_distribution': {
                'ai_consensus': 0.2,
                'ai_majority': 0.3,
                'ai_contrarian': 0.2,
                'ai_correlation': 0.2,
                'ai_game_theory': 0.1
            },
            'stack_requirements': {
                'min_stacks': 1,
                'max_stacks': 3,
                'bring_back_required': True
            }
        },
        'large_field': {
            'min_unique_captains': 10,
            'max_chalk_players': 2,
            'min_leverage_players': 2,
            'max_ownership_per_player': 20,
            'min_total_ownership': 60,
            'max_total_ownership': 90,
            'ai_enforcement': AIEnforcementLevel.MANDATORY,
            'ai_strategy_distribution': {
                'ai_consensus': 0.1,
                'ai_majority': 0.2,
                'ai_contrarian': 0.3,
                'ai_correlation': 0.2,
                'ai_game_theory': 0.2
            },
            'stack_requirements': {
                'min_stacks': 2,
                'max_stacks': 3,
                'bring_back_required': True
            }
        },
        'large_field_aggressive': {
            'min_unique_captains': 15,
            'max_chalk_players': 1,
            'min_leverage_players': 3,
            'max_ownership_per_player': 15,
            'min_total_ownership': 45,
            'max_total_ownership': 75,
            'ai_enforcement': AIEnforcementLevel.MANDATORY,
            'ai_strategy_distribution': {
                'ai_consensus': 0.05,
                'ai_majority': 0.15,
                'ai_contrarian': 0.35,
                'ai_correlation': 0.25,
                'ai_game_theory': 0.2
            },
            'stack_requirements': {
                'min_stacks': 2,
                'max_stacks': 4,
                'bring_back_required': True
            }
        },
        'milly_maker': {
            'min_unique_captains': 20,
            'max_chalk_players': 0,
            'min_leverage_players': 3,
            'max_ownership_per_player': 10,
            'min_total_ownership': 40,
            'max_total_ownership': 70,
            'ai_enforcement': AIEnforcementLevel.MANDATORY,
            'ai_strategy_distribution': {
                'ai_contrarian': 0.4,
                'ai_game_theory': 0.3,
                'ai_correlation': 0.2,
                'ai_mixed': 0.1
            },
            'stack_requirements': {
                'min_stacks': 2,
                'max_stacks': 4,
                'bring_back_required': True
            }
        },
        'super_contrarian': {
            'min_unique_captains': 25,
            'max_chalk_players': 0,
            'min_leverage_players': 4,
            'max_ownership_per_player': 7,
            'min_total_ownership': 30,
            'max_total_ownership': 60,
            'ai_enforcement': AIEnforcementLevel.MANDATORY,
            'ai_strategy_distribution': {
                'ai_contrarian': 0.5,
                'ai_game_theory': 0.35,
                'ai_correlation': 0.15
            },
            'stack_requirements': {
                'min_stacks': 1,
                'max_stacks': 5,
                'bring_back_required': False
            }
        }
    }

    # AI Strategy Requirements
    AI_STRATEGY_REQUIREMENTS = {
        'ai_consensus': {
            'must_use_consensus_captain': True,
            'must_include_agreed_players': True,
            'must_avoid_fade_players': True,
            'min_confidence_required': 0.7
        },
        'ai_majority': {
            'must_use_majority_captain': True,
            'should_include_agreed_players': True,
            'min_confidence_required': 0.6
        },
        'ai_contrarian': {
            'must_use_contrarian_captain': True,
            'must_include_narrative_plays': True,
            'must_fade_chalk_narrative': True,
            'min_confidence_required': 0.5
        },
        'ai_correlation': {
            'must_use_correlation_captain': True,
            'must_include_primary_stack': True,
            'should_include_secondary_correlation': True,
            'min_confidence_required': 0.6
        },
        'ai_game_theory': {
            'must_use_leverage_captain': True,
            'must_meet_ownership_targets': True,
            'must_avoid_ownership_traps': True,
            'min_confidence_required': 0.5
        }
    }

    # Ownership buckets
    OWNERSHIP_BUCKETS = {
        'mega_chalk': 35,
        'chalk': 20,
        'pivot': 10,
        'leverage': 5,
        'super_leverage': 0
    }

    # Contest types
    FIELD_SIZES = {
        'Single Entry': 'small_field',
        '3-Max Entry': 'small_field',
        '20-Max Entry': 'medium_field',
        '150-Max Entry': 'large_field',
        '150-Max Aggressive': 'large_field_aggressive',
        'Milly Maker': 'milly_maker',
        'Super Contrarian': 'super_contrarian'
    }

    # Correlation thresholds
    CORRELATION_THRESHOLDS = {
        'strong_positive': 0.6,
        'moderate_positive': 0.3,
        'weak': 0.0,
        'moderate_negative': -0.3,
        'strong_negative': -0.6,
        'game_stack_min': 0.4,
        'bring_back_min': 0.2
    }

    # Memory management settings
    MAX_HISTORY_ENTRIES = 500
    MAX_LOG_ENTRIES = 2000
    MAX_SYNTHESIS_HISTORY = 100
    MAX_DECISION_CACHE = 50
    
    # Performance settings
    PARALLEL_LINEUP_GENERATION = True
    MAX_PARALLEL_THREADS = 4
    OPTIMIZATION_TIMEOUT = 300

# ============================================================================
# AI DECISION TRACKER
# ============================================================================

class AIDecisionTracker:
    """Enhanced tracker with analytics and caching"""

    def __init__(self, max_decisions: int = 2000):
        self.max_decisions = max_decisions
        self.ai_decisions = deque(maxlen=max_decisions)
        self.decision_cache = {}
        self.enforcement_stats = {
            'total_rules': 0,
            'enforced_rules': 0,
            'violated_rules': 0,
            'consensus_decisions': 0,
            'majority_decisions': 0,
            'single_ai_decisions': 0,
            'override_decisions': 0
        }
        self.ai_performance = {
            AIStrategistType.GAME_THEORY: {
                'suggestions': 0, 'used': 0, 'success_rate': 0.0
            },
            AIStrategistType.CORRELATION: {
                'suggestions': 0, 'used': 0, 'success_rate': 0.0
            },
            AIStrategistType.CONTRARIAN_NARRATIVE: {
                'suggestions': 0, 'used': 0, 'success_rate': 0.0
            }
        }
        self._lock = threading.RLock()
        self.analytics = {'decision_patterns': {}, 'consensus_patterns': {}}

    def track_decision(self, decision_type: str, ai_source: str,
                      enforced: bool, details: Dict, confidence: float = 0.5):
        """Enhanced decision tracking with confidence and caching"""
        with self._lock:
            decision = {
                'timestamp': datetime.now(),
                'type': decision_type,
                'source': ai_source,
                'enforced': enforced,
                'confidence': confidence,
                'details': details
            }
            
            self.ai_decisions.append(decision)
            
            # Update cache
            cache_key = f"{decision_type}_{ai_source}"
            self.decision_cache[cache_key] = decision

            # Update statistics
            self.enforcement_stats['total_rules'] += 1
            if enforced:
                self.enforcement_stats['enforced_rules'] += 1
            else:
                self.enforcement_stats['violated_rules'] += 1
                
            if confidence > OptimizerConfig.MAX_AI_CONFIDENCE_OVERRIDE:
                self.enforcement_stats['override_decisions'] += 1
                
            self._update_analytics(decision_type, ai_source, enforced)

    def track_consensus(self, consensus_type: str, ais_agreeing: List[str], 
                       confidence_levels: List[float] = None):
        """Enhanced consensus tracking with confidence levels"""
        with self._lock:
            num_agreeing = len(ais_agreeing)
            
            if num_agreeing == 3:
                self.enforcement_stats['consensus_decisions'] += 1
            elif num_agreeing == 2:
                self.enforcement_stats['majority_decisions'] += 1
            else:
                self.enforcement_stats['single_ai_decisions'] += 1
                
            pattern_key = tuple(sorted(ais_agreeing))
            if pattern_key not in self.analytics['consensus_patterns']:
                self.analytics['consensus_patterns'][pattern_key] = 0
            self.analytics['consensus_patterns'][pattern_key] += 1

    def get_enforcement_rate(self) -> float:
        """Get the rate of AI decision enforcement"""
        with self._lock:
            if self.enforcement_stats['total_rules'] == 0:
                return 0.0
            return self.enforcement_stats['enforced_rules'] / self.enforcement_stats['total_rules']

    def get_summary(self) -> Dict:
        """Get comprehensive summary with analytics"""
        with self._lock:
            return {
                'enforcement_rate': self.get_enforcement_rate(),
                'stats': dict(self.enforcement_stats),
                'ai_performance': {k: dict(v) for k, v in self.ai_performance.items()},
                'recent_decisions': list(self.ai_decisions)[-10:] if self.ai_decisions else [],
                'analytics': dict(self.analytics),
                'cache_size': len(self.decision_cache),
                'total_decisions': len(self.ai_decisions)
            }

    def _update_analytics(self, decision_type: str, ai_source: str, enforced: bool):
        """Update internal analytics"""
        pattern_key = f"{decision_type}_{ai_source}_{enforced}"
        if pattern_key not in self.analytics['decision_patterns']:
            self.analytics['decision_patterns'][pattern_key] = 0
        self.analytics['decision_patterns'][pattern_key] += 1

# ============================================================================
# COMPLETE GLOBAL LOGGER WITH ALL METHODS
# ============================================================================

class GlobalLogger:
    """Complete production-ready logger with all required methods"""
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.entries = deque(maxlen=OptimizerConfig.MAX_LOG_ENTRIES)
            self.ai_tracker = AIDecisionTracker()
            self.verbose = False
            self.log_to_file = False
            self.log_file_path = 'optimizer_log.txt'
            self.initialized = True
            self._entry_lock = threading.RLock()
            self.log_levels = {
                'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3,
                'AI_DECISION': 1, 'AI_CONSENSUS': 1
            }
            self.min_log_level = 'INFO'
            self.stats = {'total_logs': 0, 'errors': 0, 'warnings': 0}

    def set_log_level(self, level: str):
        """Set minimum log level"""
        self.min_log_level = level

    def log(self, message: str, level: str = "INFO", **kwargs):
        """Enhanced logging with metadata"""
        with self._entry_lock:
            if self.log_levels.get(level, 1) < self.log_levels.get(self.min_log_level, 1):
                return
                
            entry = {
                'timestamp': datetime.now(),
                'level': level,
                'message': str(message),
                'metadata': kwargs
            }
            
            self.entries.append(entry)
            self.stats['total_logs'] += 1
            
            if level == 'ERROR':
                self.stats['errors'] += 1
            elif level == 'WARNING':
                self.stats['warnings'] += 1

            if self.verbose or level in ["ERROR", "WARNING"]:
                timestamp = entry['timestamp'].strftime('%H:%M:%S')
                print(f"[{timestamp}] {level}: {message}")

            if self.log_to_file:
                self._write_to_file(entry)

    def log_optimization_start(self, num_lineups: int, field_size: str, settings: Dict):
        """Log optimization start with full context"""
        self.log(f"Starting optimization: {num_lineups} lineups for {field_size}", "INFO")
        self.log(f"Settings: {json.dumps(settings, default=str)}", "DEBUG", settings=settings)
        self.log(f"AI Enforcement: {settings.get('enforcement', 'UNKNOWN')}", "INFO")

    def log_optimization_end(self, lineups_generated: int, total_time: float, success_rate: float = None):
        """Log optimization completion with metrics"""
        self.log(f"Optimization complete: {lineups_generated} lineups in {total_time:.2f}s", "INFO")
        
        if lineups_generated > 0:
            avg_time = total_time / lineups_generated
            self.log(f"Average time per lineup: {avg_time:.3f}s", "DEBUG")
            
            if success_rate is not None:
                self.log(f"Success rate: {success_rate:.1%}", "INFO")

    def log_lineup_generation(self, strategy: str, lineup_num: int, status: str, 
                             ai_rules_enforced: int = 0, violations: List[str] = None):
        """Log individual lineup generation"""
        message = f"Lineup {lineup_num} ({strategy}): {status}"
        if ai_rules_enforced > 0:
            message += f" - {ai_rules_enforced} AI rules enforced"
        if violations:
            message += f" - Violations: {', '.join(violations[:3])}"
        
        level = "DEBUG" if status == "SUCCESS" else "WARNING"
        self.log(message, level, lineup_num=lineup_num, strategy=strategy)

    def log_ai_decision(self, decision_type: str, ai_source: str, enforced: bool, 
                       details: Dict, confidence: float = 0.5):
        """Log AI decision with tracking"""
        self.ai_tracker.track_decision(decision_type, ai_source, enforced, details, confidence)
        message = f"AI Decision [{ai_source}]: {decision_type} - {'ENFORCED' if enforced else 'VIOLATED'} (conf: {confidence:.2f})"
        self.log(message, "AI_DECISION", confidence=confidence)

    def log_ai_consensus(self, consensus_type: str, ais_agreeing: List[str],
                        decision: str, confidence_levels: List[float] = None):
        """Log AI consensus with confidence tracking"""
        self.ai_tracker.track_consensus(consensus_type, ais_agreeing, confidence_levels)
        avg_conf = np.mean(confidence_levels) if confidence_levels else 0.5
        message = f"AI Consensus ({len(ais_agreeing)}/3): {decision} (avg conf: {avg_conf:.2f})"
        self.log(message, "AI_CONSENSUS", avg_confidence=avg_conf)

    def log_exception(self, exception: Exception, context: str = "", critical: bool = False):
        """Log exception with optional stack trace"""
        message = f"Exception in {context}: {str(exception)}"
        if critical:
            message += f"\nStack trace: {traceback.format_exc()}"
        
        self.log(message, "ERROR", context=context, exception_type=type(exception).__name__)

    def get_ai_summary(self) -> Dict:
        """Get comprehensive AI tracking summary"""
        return self.ai_tracker.get_summary()

    def display_ai_enforcement(self):
        """Display AI enforcement statistics in Streamlit"""
        try:
            summary = self.get_ai_summary()
            
            st.markdown("### AI Decision Enforcement")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rate = summary['enforcement_rate'] * 100
                st.metric("Enforcement Rate", f"{rate:.1f}%",
                         delta=f"{rate-80:.1f}%" if rate < 80 else None)
            
            with col2:
                stats = summary['stats']
                st.metric("Rules Enforced",
                         f"{stats['enforced_rules']}/{stats['total_rules']}")
            
            with col3:
                st.metric("Consensus Decisions", stats.get('consensus_decisions', 0))
            
            with col4:
                st.metric("AI Overrides", stats.get('override_decisions', 0),
                         help="Times AI confidence exceeded override threshold")
            
            # AI Performance breakdown
            if summary.get('ai_performance'):
                st.markdown("#### AI Performance by Type")
                perf_cols = st.columns(3)
                
                for i, (ai_type, perf) in enumerate(summary['ai_performance'].items()):
                    with perf_cols[i % 3]:
                        if isinstance(ai_type, AIStrategistType):
                            ai_name = ai_type.value.replace('_', ' ').title()
                        else:
                            ai_name = str(ai_type)
                        
                        success_rate = perf.get('success_rate', 0) * 100
                        st.metric(ai_name, f"{success_rate:.1f}%",
                                 delta=f"{perf.get('used', 0)}/{perf.get('suggestions', 0)}")
                        
        except Exception as e:
            st.error(f"Error displaying AI enforcement: {e}")

    def display_log_summary(self):
        """Display log summary in Streamlit"""
        try:
            st.markdown("### Optimization Log Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Logs", self.stats['total_logs'])
            with col2:
                st.metric("Errors", self.stats['errors'])
            with col3:
                st.metric("Warnings", self.stats['warnings'])
            with col4:
                success_rate = 100 * (1 - self.stats['errors'] / max(self.stats['total_logs'], 1))
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            # Level distribution
            with self._entry_lock:
                level_counts = {}
                for entry in self.entries:
                    level = entry.get('level', 'UNKNOWN')
                    level_counts[level] = level_counts.get(level, 0) + 1
            
            if level_counts:
                st.markdown("#### Log Level Distribution")
                cols = st.columns(len(level_counts))
                for i, (level, count) in enumerate(level_counts.items()):
                    cols[i].metric(level, count)
            
            # Recent critical logs
            st.markdown("#### Recent Critical Logs")
            with self._entry_lock:
                critical_logs = [e for e in self.entries if e.get('level') in ['ERROR', 'WARNING']][-5:]
                
            for entry in critical_logs:
                timestamp = entry['timestamp'].strftime('%H:%M:%S')
                level_icon = '🔴' if entry.get('level') == 'ERROR' else '🟡'
                st.text(f"{level_icon} [{timestamp}] {entry.get('message', '')}")
                
        except Exception as e:
            st.error(f"Error displaying log summary: {e}")

    def export_logs(self, format: str = 'text') -> Union[str, Dict]:
        """Export logs in multiple formats"""
        try:
            with self._entry_lock:
                if format == 'json':
                    return {
                        'logs': [
                            {
                                'timestamp': e['timestamp'].isoformat(),
                                'level': e.get('level'),
                                'message': e.get('message'),
                                'metadata': e.get('metadata', {})
                            } for e in list(self.entries)[-100:]
                        ],
                        'summary': self.get_ai_summary(),
                        'stats': dict(self.stats)
                    }
                else:
                    output = "=== OPTIMIZATION LOG ===\n\n"
                    for entry in list(self.entries)[-100:]:
                        timestamp = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                        output += f"[{timestamp}] {entry.get('level', 'UNKNOWN')}: {entry.get('message', '')}\n"
                    
                    output += "\n=== AI DECISION SUMMARY ===\n"
                    summary = self.get_ai_summary()
                    output += f"Enforcement Rate: {summary['enforcement_rate']*100:.1f}%\n"
                    output += f"Total AI Rules: {summary['stats']['total_rules']}\n"
                    output += f"Enforced: {summary['stats']['enforced_rules']}\n"
                    output += f"Violated: {summary['stats']['violated_rules']}\n"
                    
                    return output
                    
        except Exception as e:
            return f"Error exporting logs: {e}"

    def _write_to_file(self, entry: Dict):
        """Thread-safe file writing with rotation"""
        try:
            if os.path.exists(self.log_file_path):
                if os.path.getsize(self.log_file_path) > 10 * 1024 * 1024:  # 10MB
                    os.rename(self.log_file_path, f"{self.log_file_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                timestamp = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                metadata = json.dumps(entry.get('metadata', {})) if entry.get('metadata') else ''
                f.write(f"[{timestamp}] {entry.get('level', 'UNKNOWN')}: {entry.get('message', '')} {metadata}\n")
                
        except Exception:
            pass

# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """Production performance monitoring with detailed metrics"""
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.timers = {}
            self.counters = {}
            self.gauges = {}
            self.histograms = {}
            self.max_timers = 200
            self.initialized = True
            self._timer_lock = threading.RLock()

    def start_timer(self, name: str, metadata: Dict = None):
        """Start a named timer with optional metadata"""
        with self._timer_lock:
            self.timers[name] = {
                'start': datetime.now(),
                'metadata': metadata or {}
            }
            
            if len(self.timers) > self.max_timers:
                oldest = sorted(self.timers.items(), 
                              key=lambda x: x[1]['start'])[:20]
                for timer_name, _ in oldest:
                    del self.timers[timer_name]

    def stop_timer(self, name: str) -> float:
        """Stop timer and record in histogram"""
        with self._timer_lock:
            if name in self.timers:
                elapsed = (datetime.now() - self.timers[name]['start']).total_seconds()
                
                if name not in self.histograms:
                    self.histograms[name] = []
                self.histograms[name].append(elapsed)
                
                if len(self.histograms[name]) > 100:
                    self.histograms[name] = self.histograms[name][-100:]
                
                del self.timers[name]
                return elapsed
            return 0.0

    def increment_counter(self, name: str, value: int = 1):
        """Increment a named counter"""
        with self._timer_lock:
            if name not in self.counters:
                self.counters[name] = 0
            self.counters[name] += value

    def set_gauge(self, name: str, value: float):
        """Set a gauge value"""
        with self._timer_lock:
            self.gauges[name] = value

    def get_metrics(self) -> Dict:
        """Get comprehensive metrics"""
        with self._timer_lock:
            histogram_stats = {}
            for name, values in self.histograms.items():
                if values:
                    histogram_stats[name] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'p95': np.percentile(values, 95),
                        'p99': np.percentile(values, 99),
                        'count': len(values)
                    }
            
            return {
                'active_timers': list(self.timers.keys()),
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histogram_stats': histogram_stats
            }

    def display_metrics(self):
        """Display comprehensive metrics in Streamlit"""
        try:
            metrics = self.get_metrics()
            st.markdown("### Performance Metrics")
            
            if metrics['active_timers']:
                st.markdown("#### Active Operations")
                for timer in metrics['active_timers'][:5]:
                    st.write(f"• {timer}")
            
            if metrics['counters']:
                st.markdown("#### Operation Counts")
                counter_cols = st.columns(4)
                for i, (name, value) in enumerate(list(metrics['counters'].items())[:8]):
                    counter_cols[i % 4].metric(name.replace('_', ' ').title(), value)
            
            if metrics['histogram_stats']:
                st.markdown("#### Performance Statistics (seconds)")
                for operation, stats in list(metrics['histogram_stats'].items())[:5]:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric(operation, f"{stats['mean']:.3f}s", help="Mean time")
                    col2.metric("Median", f"{stats['median']:.3f}s")
                    col3.metric("P95", f"{stats['p95']:.3f}s")
                    col4.metric("Count", stats['count'])
                    
        except Exception as e:
            st.error(f"Error displaying metrics: {e}")

    def reset(self):
        """Reset all metrics"""
        with self._timer_lock:
            self.timers.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()

# ============================================================================
# SINGLETON GETTERS
# ============================================================================

def get_logger() -> GlobalLogger:
    """Get the singleton logger instance"""
    return GlobalLogger()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the singleton performance monitor instance"""
    return PerformanceMonitor()

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AIRecommendation:
    """Enhanced AI recommendation with validation"""
    captain_targets: List[str]
    must_play: List[str]
    never_play: List[str]
    stacks: List[Dict]
    key_insights: List[str]
    confidence: float
    enforcement_rules: List[Dict]
    narrative: str
    source_ai: AIStrategistType
    
    # Optional specialized fields
    ownership_leverage: Optional[Dict] = None
    correlation_matrix: Optional[Dict] = None
    contrarian_angles: Optional[List[str]] = None
    boosts: Optional[List[str]] = None
    fades: Optional[List[str]] = None
    
    # Enhanced tracking
    generation_time: datetime = field(default_factory=datetime.now)
    validation_status: str = 'pending'
    enforcement_priority: int = 1
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate recommendation structure"""
        errors = []
        
        if not self.captain_targets:
            errors.append("No captain targets provided")
        if self.confidence < 0 or self.confidence > 1:
            errors.append(f"Invalid confidence: {self.confidence}")
        if not self.source_ai:
            errors.append("No source AI specified")
            
        self.validation_status = 'valid' if not errors else 'invalid'
        return len(errors) == 0, errors
        # NFL GPP DUAL-AI OPTIMIZER - PART 2: CORE COMPONENTS (AI-AS-CHEF VERSION)
# Version 6.3 - Enhanced Core Components with Robust Validation
# NOTE: This continues from Part 1 - all imports already consolidated at top

# ============================================================================
# AI ENFORCEMENT ENGINE WITH VALIDATION
# ============================================================================

class AIEnforcementEngine:
    """Core engine for enforcing AI decisions throughout optimization"""

    def __init__(self, enforcement_level: AIEnforcementLevel = AIEnforcementLevel.MANDATORY):
        self.enforcement_level = enforcement_level
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()
        self.enforcement_history = []
        self.rule_cache = {}  # Cache processed rules
        self._lock = threading.RLock()
        
        # Rule priorities for conflict resolution
        self.rule_priorities = {
            'captain_selection': 100,
            'must_include': 90,
            'must_exclude': 85,
            'must_stack': 80,
            'ownership_sum': 70,
            'avoid_together': 60,
            'soft_include': 50,
            'soft_exclude': 40
        }

    def create_enforcement_rules(self, ai_recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """Convert AI recommendations into enforceable optimization rules with validation"""
        
        with self._lock:
            # Check cache first
            cache_key = self._generate_cache_key(ai_recommendations)
            if cache_key in self.rule_cache:
                self.logger.log("Using cached enforcement rules", "DEBUG")
                return self.rule_cache[cache_key]
            
            self.perf_monitor.start_timer("create_enforcement_rules")
            
            rules = {
                'hard_constraints': [],
                'soft_constraints': [],
                'objective_modifiers': {},
                'variable_locks': {},
                'metadata': {
                    'enforcement_level': self.enforcement_level.value,
                    'ai_count': len(ai_recommendations),
                    'timestamp': pd.Timestamp.now()
                }
            }

            # Find consensus among AIs with confidence weighting
            consensus = self._find_weighted_consensus(ai_recommendations)
            
            # Create rules based on consensus and enforcement level
            if self.enforcement_level == AIEnforcementLevel.MANDATORY:
                # All high-confidence consensus decisions become hard constraints
                rules['hard_constraints'].extend(
                    self._create_hard_constraints(consensus['must_play'], 'include', priority=90)
                )
                rules['hard_constraints'].extend(
                    self._create_hard_constraints(consensus['never_play'], 'exclude', priority=85)
                )
                
                # Captain requirements are highest priority
                if consensus['consensus_captains']:
                    rules['variable_locks']['captain'] = consensus['consensus_captains']
                    rules['hard_constraints'].append({
                        'type': 'hard',
                        'constraint': 'captain_selection',
                        'players': consensus['consensus_captains'],
                        'priority': 100,
                        'description': f"Must use consensus captain from: {consensus['consensus_captains'][:3]}"
                    })
                    
            elif self.enforcement_level == AIEnforcementLevel.STRONG:
                # Consensus becomes hard, majority becomes soft with high weight
                rules['hard_constraints'].extend(
                    self._create_hard_constraints(consensus['must_play'], 'include', priority=85)
                )
                rules['soft_constraints'].extend(
                    self._create_soft_constraints(consensus['should_play'], 'include', weight=0.8)
                )
                
                # Majority captains as soft constraint
                if consensus['majority_captains']:
                    rules['soft_constraints'].append({
                        'type': 'soft',
                        'constraint': 'prefer_captains',
                        'players': consensus['majority_captains'],
                        'weight': 0.7,
                        'description': f"Prefer majority captains"
                    })
                    
            elif self.enforcement_level == AIEnforcementLevel.MODERATE:
                # More soft constraints, fewer hard constraints
                rules['soft_constraints'].extend(
                    self._create_soft_constraints(consensus['must_play'], 'include', weight=0.7)
                )
                rules['soft_constraints'].extend(
                    self._create_soft_constraints(consensus['should_play'], 'include', weight=0.5)
                )
                
            # Add AI-specific modifiers with confidence scaling
            for ai_type, rec in ai_recommendations.items():
                if rec.confidence > OptimizerConfig.MIN_AI_CONFIDENCE:
                    modifiers = self._create_objective_modifiers(rec)
                    rules['objective_modifiers'].update(modifiers)
                    
                    # Track which AI contributed what
                    if 'ai_contributions' not in rules['metadata']:
                        rules['metadata']['ai_contributions'] = {}
                    rules['metadata']['ai_contributions'][ai_type.value] = {
                        'confidence': rec.confidence,
                        'rules_contributed': len(modifiers)
                    }

            # Add stacking rules from consensus
            stack_rules = self._create_stacking_rules(consensus, ai_recommendations)
            rules['hard_constraints'].extend(stack_rules['hard'])
            rules['soft_constraints'].extend(stack_rules['soft'])

            # Validate and resolve conflicts
            rules = self._validate_and_resolve_conflicts(rules)
            
            # Cache the result
            self.rule_cache[cache_key] = rules
            
            # Clean old cache entries
            if len(self.rule_cache) > 50:
                self.rule_cache = dict(list(self.rule_cache.items())[-25:])
            
            elapsed = self.perf_monitor.stop_timer("create_enforcement_rules")
            
            self.logger.log_ai_decision(
                "enforcement_rules_created",
                "AIEnforcementEngine",
                True,
                {
                    'num_hard': len(rules['hard_constraints']),
                    'num_soft': len(rules['soft_constraints']),
                    'num_modifiers': len(rules['objective_modifiers']),
                    'generation_time': elapsed
                },
                confidence=max([r.confidence for r in ai_recommendations.values()] + [0])
            )

            return rules

    def _find_weighted_consensus(self, ai_recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """Find consensus with confidence weighting"""
        
        consensus = {
            'consensus_captains': [],  # All 3 AIs agree with high confidence
            'majority_captains': [],   # 2 of 3 agree
            'must_play': [],           # High confidence agreement
            'should_play': [],         # Medium confidence agreement
            'never_play': [],          # Agreement on fades
            'stacks': []               # Agreed stacking patterns
        }

        # Weight votes by confidence
        captain_votes = defaultdict(lambda: {'voters': [], 'total_confidence': 0})
        must_play_votes = defaultdict(lambda: {'voters': [], 'total_confidence': 0})
        never_play_votes = defaultdict(lambda: {'voters': [], 'total_confidence': 0})
        stack_votes = defaultdict(lambda: {'voters': [], 'total_confidence': 0})

        for ai_type, rec in ai_recommendations.items():
            # Captain votes with confidence weighting
            for captain in rec.captain_targets:
                captain_votes[captain]['voters'].append(ai_type)
                captain_votes[captain]['total_confidence'] += rec.confidence

            # Must play votes
            for player in rec.must_play:
                must_play_votes[player]['voters'].append(ai_type)
                must_play_votes[player]['total_confidence'] += rec.confidence

            # Never play votes
            for player in rec.never_play:
                never_play_votes[player]['voters'].append(ai_type)
                never_play_votes[player]['total_confidence'] += rec.confidence
                
            # Stack votes
            for stack in rec.stacks:
                if isinstance(stack, dict):
                    stack_key = f"{stack.get('player1', '')}_{stack.get('player2', '')}"
                    stack_votes[stack_key]['voters'].append(ai_type)
                    stack_votes[stack_key]['total_confidence'] += rec.confidence

        # Classify captains by agreement level and confidence
        for captain, data in captain_votes.items():
            num_voters = len(data['voters'])
            avg_confidence = data['total_confidence'] / max(num_voters, 1)
            
            if num_voters == 3 and avg_confidence >= 0.7:
                consensus['consensus_captains'].append(captain)
                self.logger.log_ai_consensus("captain", data['voters'], 
                                            f"Consensus captain: {captain}", 
                                            [rec.confidence for rec in ai_recommendations.values()])
            elif num_voters >= 2 and avg_confidence >= 0.6:
                consensus['majority_captains'].append(captain)

        # Classify must_play by weighted agreement
        for player, data in must_play_votes.items():
            num_voters = len(data['voters'])
            avg_confidence = data['total_confidence'] / max(num_voters, 1)
            
            if num_voters == 3 or (num_voters == 2 and avg_confidence >= 0.8):
                consensus['must_play'].append(player)
            elif num_voters >= 2 or avg_confidence >= 0.7:
                consensus['should_play'].append(player)

        # Classify never_play (lower threshold for fades)
        for player, data in never_play_votes.items():
            if len(data['voters']) >= 2 or data['total_confidence'] >= 1.2:
                consensus['never_play'].append(player)
                
        # Classify stacks
        for stack_key, data in stack_votes.items():
            if len(data['voters']) >= 2:
                players = stack_key.split('_')
                if len(players) == 2:
                    consensus['stacks'].append({
                        'player1': players[0],
                        'player2': players[1],
                        'confidence': data['total_confidence'] / len(data['voters'])
                    })

        return consensus

    def _create_hard_constraints(self, players: List[str], constraint_type: str, 
                                priority: int = 50) -> List[Dict]:
        """Create hard constraints with priority"""
        constraints = []
        
        for player in players:
            if constraint_type == 'include':
                constraints.append({
                    'type': 'hard',
                    'player': player,
                    'rule': 'must_include',
                    'priority': priority,
                    'description': f"AI consensus: must include {player}"
                })
            elif constraint_type == 'exclude':
                constraints.append({
                    'type': 'hard',
                    'player': player,
                    'rule': 'must_exclude',
                    'priority': priority,
                    'description': f"AI consensus: must exclude {player}"
                })
                
        return constraints

    def _create_soft_constraints(self, players: List[str], constraint_type: str, 
                                weight: float) -> List[Dict]:
        """Create weighted soft constraints"""
        constraints = []
        
        for player in players:
            constraints.append({
                'type': 'soft',
                'player': player,
                'rule': f"should_{constraint_type}",
                'weight': weight,
                'priority': int(weight * 50),  # Convert weight to priority
                'description': f"AI majority: should {constraint_type} {player}"
            })
            
        return constraints

    def _create_stacking_rules(self, consensus: Dict, 
                             ai_recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """Create stacking rules from consensus"""
        stack_rules = {'hard': [], 'soft': []}
        
        # Process consensus stacks
        for stack in consensus.get('stacks', []):
            if stack.get('confidence', 0) >= 0.7:
                stack_rules['hard'].append({
                    'type': 'hard',
                    'constraint': 'must_stack',
                    'players': [stack['player1'], stack['player2']],
                    'priority': 80,
                    'description': f"Required stack: {stack['player1']} + {stack['player2']}"
                })
            else:
                stack_rules['soft'].append({
                    'type': 'soft',
                    'constraint': 'prefer_stack',
                    'players': [stack['player1'], stack['player2']],
                    'weight': stack.get('confidence', 0.5),
                    'priority': int(stack.get('confidence', 0.5) * 50),
                    'description': f"Preferred stack: {stack['player1']} + {stack['player2']}"
                })
                
        return stack_rules

    def _create_objective_modifiers(self, recommendation: AIRecommendation) -> Dict:
        """Create objective function modifiers based on AI recommendation"""
        modifiers = {}
        
        # Scale modifiers by confidence
        confidence_multiplier = recommendation.confidence
        
        # Boost recommended captains significantly
        for captain in recommendation.captain_targets:
            modifiers[captain] = modifiers.get(captain, 1.0) * (1.0 + confidence_multiplier * 0.4)
        
        # Boost must-play players
        for player in recommendation.must_play:
            modifiers[player] = modifiers.get(player, 1.0) * (1.0 + confidence_multiplier * 0.25)
        
        # Penalize fade targets
        for player in recommendation.never_play:
            modifiers[player] = modifiers.get(player, 1.0) * (1.0 - confidence_multiplier * 0.4)
        
        # Apply boosts if specified
        if recommendation.boosts:
            for player in recommendation.boosts:
                modifiers[player] = modifiers.get(player, 1.0) * 1.15
                
        return modifiers

    def _validate_and_resolve_conflicts(self, rules: Dict) -> Dict:
        """Validate rules and resolve any conflicts"""
        
        # Check for player conflicts (can't include and exclude same player)
        included_players = set()
        excluded_players = set()
        
        for constraint in rules['hard_constraints']:
            if constraint['rule'] == 'must_include':
                included_players.add(constraint['player'])
            elif constraint['rule'] == 'must_exclude':
                excluded_players.add(constraint['player'])
        
        conflicts = included_players.intersection(excluded_players)
        
        if conflicts:
            self.logger.log(f"Resolving conflicts for players: {conflicts}", "WARNING")
            
            # Resolve by priority - keep higher priority rule
            for player in conflicts:
                include_priority = max([c['priority'] for c in rules['hard_constraints'] 
                                      if c.get('player') == player and c['rule'] == 'must_include'] + [0])
                exclude_priority = max([c['priority'] for c in rules['hard_constraints'] 
                                      if c.get('player') == player and c['rule'] == 'must_exclude'] + [0])
                
                if include_priority >= exclude_priority:
                    # Remove exclude constraint
                    rules['hard_constraints'] = [
                        c for c in rules['hard_constraints'] 
                        if not (c.get('player') == player and c['rule'] == 'must_exclude')
                    ]
                else:
                    # Remove include constraint
                    rules['hard_constraints'] = [
                        c for c in rules['hard_constraints'] 
                        if not (c.get('player') == player and c['rule'] == 'must_include')
                    ]
        
        # Sort constraints by priority for consistent application
        rules['hard_constraints'].sort(key=lambda x: x.get('priority', 0), reverse=True)
        rules['soft_constraints'].sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        return rules

    def validate_lineup_against_ai(self, lineup: Dict, ai_rules: Dict) -> Tuple[bool, List[str]]:
        """Validate that a lineup satisfies AI requirements"""
        violations = []
        
        lineup_players = [lineup.get('Captain')] + lineup.get('FLEX', [])
        
        # Check hard constraints
        for constraint in ai_rules.get('hard_constraints', []):
            player = constraint.get('player')
            rule = constraint['rule']
            
            if rule == 'must_include' and player and player not in lineup_players:
                violations.append(f"Missing required player: {player}")
            elif rule == 'must_exclude' and player and player in lineup_players:
                violations.append(f"Included banned player: {player}")
            elif rule == 'must_stack':
                stack_players = constraint.get('players', [])
                if len(stack_players) == 2:
                    if not all(p in lineup_players for p in stack_players):
                        violations.append(f"Missing required stack: {' + '.join(stack_players)}")
        
        # Check captain requirements
        required_captains = ai_rules.get('variable_locks', {}).get('captain', [])
        if required_captains and lineup.get('Captain') not in required_captains:
            violations.append(f"Captain {lineup.get('Captain')} not in AI requirements")
        
        is_valid = len(violations) == 0
        
        # Track enforcement
        self.enforcement_history.append({
            'lineup': lineup.get('Lineup', 0),
            'valid': is_valid,
            'violations': violations,
            'timestamp': pd.Timestamp.now()
        })
        
        # Keep history manageable
        if len(self.enforcement_history) > 1000:
            self.enforcement_history = self.enforcement_history[-500:]
        
        return is_valid, violations

    def _generate_cache_key(self, ai_recommendations: Dict) -> str:
        """Generate cache key for recommendations"""
        key_parts = []
        for ai_type in sorted(ai_recommendations.keys(), key=lambda x: x.value):
            rec = ai_recommendations[ai_type]
            key_parts.append(f"{ai_type.value}_{rec.confidence:.2f}_{len(rec.captain_targets)}")
        return "_".join(key_parts)

    def get_enforcement_summary(self) -> Dict:
        """Get summary of enforcement history"""
        if not self.enforcement_history:
            return {'total': 0, 'valid': 0, 'invalid': 0, 'rate': 0.0}
            
        total = len(self.enforcement_history)
        valid = sum(1 for h in self.enforcement_history if h['valid'])
        
        return {
            'total': total,
            'valid': valid,
            'invalid': total - valid,
            'rate': valid / total if total > 0 else 0.0,
            'common_violations': Counter(
                v for h in self.enforcement_history 
                for v in h.get('violations', [])
            ).most_common(5)
        }

# ============================================================================
# AI-DRIVEN OWNERSHIP BUCKET MANAGER
# ============================================================================

class AIOwnershipBucketManager:
    """Enhanced bucket manager that respects AI decisions"""

    def __init__(self, ai_enforcement_engine: Optional[AIEnforcementEngine] = None):
        self.thresholds = OptimizerConfig.OWNERSHIP_BUCKETS
        self.ai_engine = ai_enforcement_engine
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()
        self.bucket_cache = {}
        
    def categorize_players_with_ai(self, df: pd.DataFrame,
                                   ai_recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """Categorize players with AI override capabilities"""
        
        self.perf_monitor.start_timer("categorize_players_ai")
        
        # Standard categorization first
        buckets = self.categorize_players(df)
        
        if self.ai_engine and ai_recommendations:
            # Apply AI overrides
            consensus = self.ai_engine._find_weighted_consensus(ai_recommendations)
            
            # Create special AI categories
            buckets['ai_consensus_captains'] = consensus['consensus_captains']
            buckets['ai_must_play'] = consensus['must_play']
            buckets['ai_never_play'] = consensus['never_play']
            
            # Remove AI players from standard buckets to avoid duplication
            ai_players = set(consensus['consensus_captains'] + 
                           consensus['must_play'] + 
                           consensus['never_play'])
            
            for bucket_name, players in buckets.items():
                if not bucket_name.startswith('ai_'):
                    buckets[bucket_name] = [p for p in players if p not in ai_players]
            
            # Create AI leverage category (contrarian picks)
            buckets['ai_leverage'] = []
            for ai_type, rec in ai_recommendations.items():
                if ai_type == AIStrategistType.CONTRARIAN_NARRATIVE:
                    unique_contrarian = [
                        p for p in rec.captain_targets
                        if p not in consensus['consensus_captains'] and
                        df[df['Player'] == p]['Ownership'].values[0] < 10
                    ]
                    buckets['ai_leverage'].extend(unique_contrarian)
            
            # Remove duplicates from ai_leverage
            buckets['ai_leverage'] = list(set(buckets['ai_leverage']))
            
            self.logger.log(
                f"AI-enhanced buckets: {len(buckets['ai_consensus_captains'])} consensus, "
                f"{len(buckets['ai_leverage'])} leverage plays", 
                "DEBUG"
            )
        
        self.perf_monitor.stop_timer("categorize_players_ai")
        return buckets

    def categorize_players(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Standard categorization by ownership with validation"""
        buckets = {
            'mega_chalk': [],
            'chalk': [],
            'pivot': [],
            'leverage': [],
            'super_leverage': []
        }
        
        # Validate dataframe has required columns
        if 'Player' not in df.columns or 'Ownership' not in df.columns:
            self.logger.log("Missing required columns for categorization", "WARNING")
            return buckets
        
        for _, row in df.iterrows():
            player = row['Player']
            ownership = row.get('Ownership', OptimizerConfig.DEFAULT_OWNERSHIP)
            
            # Handle NaN or invalid ownership
            if pd.isna(ownership) or ownership < 0:
                ownership = OptimizerConfig.DEFAULT_OWNERSHIP
                
            if ownership >= self.thresholds['mega_chalk']:
                buckets['mega_chalk'].append(player)
            elif ownership >= self.thresholds['chalk']:
                buckets['chalk'].append(player)
            elif ownership >= self.thresholds['pivot']:
                buckets['pivot'].append(player)
            elif ownership >= self.thresholds['leverage']:
                buckets['leverage'].append(player)
            else:
                buckets['super_leverage'].append(player)
        
        return buckets

    def calculate_ai_leverage_score(self, lineup_players: List[str], df: pd.DataFrame,
                                   ai_recommendations: Dict[AIStrategistType, AIRecommendation]) -> float:
        """Calculate leverage score with AI weighting and validation"""
        
        # Base GPP leverage score
        base_score = self.calculate_gpp_leverage(lineup_players, df)
        
        if not ai_recommendations:
            return base_score
        
        # AI bonus calculation with confidence weighting
        ai_bonus = 0
        
        for player in lineup_players:
            for ai_type, rec in ai_recommendations.items():
                confidence_multiplier = rec.confidence
                
                if ai_type == AIStrategistType.CONTRARIAN_NARRATIVE:
                    if player in rec.captain_targets:
                        ai_bonus += 4 * confidence_multiplier
                    elif player in rec.must_play:
                        ai_bonus += 2.5 * confidence_multiplier
                elif ai_type == AIStrategistType.GAME_THEORY:
                    if player in rec.captain_targets:
                        # Check if it's a low-owned captain
                        player_own = df[df['Player'] == player]['Ownership'].values
                        if len(player_own) > 0 and player_own[0] < 10:
                            ai_bonus += 3 * confidence_multiplier
                elif player in rec.captain_targets:
                    ai_bonus += 1.5 * confidence_multiplier
        
        return base_score + ai_bonus

    def calculate_gpp_leverage(self, lineup_players: List[str], df: pd.DataFrame) -> float:
        """Calculate standard GPP leverage score with enhanced logic"""
        score = 0
        
        # Validate inputs
        if not lineup_players or df.empty:
            return 0
        
        # Calculate ownership distribution bonus/penalty
        ownership_values = []
        
        for player in lineup_players:
            player_data = df[df['Player'] == player]
            if not player_data.empty:
                ownership = player_data['Ownership'].values[0]
                ownership_values.append(ownership)
                
                # Tiered scoring system
                if ownership < 3:      # Ultra leverage
                    score += 4
                elif ownership < 5:    # Super leverage
                    score += 3
                elif ownership < 10:   # Leverage
                    score += 2
                elif ownership < 15:   # Pivot
                    score += 1
                elif ownership < 20:   # Balanced
                    score += 0
                elif ownership < 30:   # Slightly chalky
                    score -= 1
                elif ownership < 40:   # Chalky
                    score -= 2
                else:                  # Mega chalk
                    score -= 3
        
        # Bonus for good ownership distribution (not all chalk or all leverage)
        if ownership_values:
            std_dev = np.std(ownership_values)
            if 10 < std_dev < 20:  # Good mix
                score += 2
            elif std_dev > 20:     # Very diverse
                score += 3
        
        return score

    def get_gpp_summary(self, lineup_players: List[str], df: pd.DataFrame,
                       field_size: str, ai_enforced: bool = False) -> str:
        """Get detailed GPP summary with AI indicator"""
        
        ownership_counts = {
            '<3%': 0, '3-5%': 0, '5-10%': 0, '10-20%': 0, 
            '20-30%': 0, '30-40%': 0, '40%+': 0
        }
        
        total_ownership = 0
        
        for player in lineup_players:
            player_data = df[df['Player'] == player]
            if not player_data.empty:
                ownership = player_data['Ownership'].values[0]
                total_ownership += ownership
                
                if ownership < 3:
                    ownership_counts['<3%'] += 1
                elif ownership < 5:
                    ownership_counts['3-5%'] += 1
                elif ownership < 10:
                    ownership_counts['5-10%'] += 1
                elif ownership < 20:
                    ownership_counts['10-20%'] += 1
                elif ownership < 30:
                    ownership_counts['20-30%'] += 1
                elif ownership < 40:
                    ownership_counts['30-40%'] += 1
                else:
                    ownership_counts['40%+'] += 1
        
        # Create summary string
        summary_parts = [f"{k}:{v}" for k, v in ownership_counts.items() if v > 0]
        summary = ' | '.join(summary_parts)
        
        # Add total ownership
        summary += f" | Total: {total_ownership:.1f}%"
        
        # Add field-specific indicator
        target_range = OptimizerConfig.GPP_OWNERSHIP_TARGETS.get(field_size, (60, 90))
        if total_ownership < target_range[0]:
            summary += " [LOW]"
        elif total_ownership > target_range[1]:
            summary += " [HIGH]"
        else:
            summary += " [OPTIMAL]"
        
        if ai_enforced:
            summary = "AI-ENFORCED | " + summary
        
        return summary

# ============================================================================
# ENHANCED CONFIG VALIDATORS
# ============================================================================

class AIConfigValidator:
    """Enhanced validator that ensures AI requirements are feasible"""

    @staticmethod
    def validate_ai_requirements(ai_rules: Dict, player_pool: pd.DataFrame) -> Dict:
        """Comprehensive validation of AI requirements"""
        
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'adjustments': [],
            'stats': {
                'available_players': len(player_pool),
                'required_inclusions': 0,
                'required_exclusions': 0,
                'captain_options': 0
            }
        }
        
        # Validate player pool has minimum required columns
        required_columns = ['Player', 'Salary', 'Position', 'Team']
        missing_columns = [col for col in required_columns if col not in player_pool.columns]
        
        if missing_columns:
            validation['errors'].append(f"Missing columns: {missing_columns}")
            validation['is_valid'] = False
            return validation
        
        available_players = set(player_pool['Player'].tolist())
        validation['stats']['available_players'] = len(available_players)
        
        # Check hard constraints feasibility
        must_include = set()
        must_exclude = set()
        
        for constraint in ai_rules.get('hard_constraints', []):
            player = constraint.get('player')
            rule = constraint.get('rule')
            
            if not player:
                continue
                
            if player not in available_players:
                validation['warnings'].append(f"AI required player '{player}' not in pool")
                continue
            
            if rule == 'must_include':
                must_include.add(player)
            elif rule == 'must_exclude':
                must_exclude.add(player)
        
        validation['stats']['required_inclusions'] = len(must_include)
        validation['stats']['required_exclusions'] = len(must_exclude)
        
        # Check for conflicts
        conflicts = must_include.intersection(must_exclude)
        if conflicts:
            validation['errors'].append(f"Conflicting requirements for: {conflicts}")
            validation['is_valid'] = False
        
        # Check if we have enough players after exclusions
        available_after_exclusions = len(available_players - must_exclude)
        required_spots = 6  # 1 CPT + 5 FLEX
        
        if len(must_include) > required_spots:
            validation['errors'].append(
                f"AI requires {len(must_include)} players but lineup has {required_spots} spots"
            )
            validation['is_valid'] = False
        
        if available_after_exclusions < required_spots:
            validation['errors'].append(
                f"Only {available_after_exclusions} players available after exclusions (need {required_spots})"
            )
            validation['is_valid'] = False
        
        # Check captain requirements
        required_captains = ai_rules.get('variable_locks', {}).get('captain', [])
        if required_captains:
            valid_captains = [c for c in required_captains if c in available_players]
            validation['stats']['captain_options'] = len(valid_captains)
            
            if not valid_captains:
                validation['errors'].append("No valid AI captains in player pool")
                validation['is_valid'] = False
            elif len(valid_captains) < len(required_captains):
                validation['warnings'].append(
                    f"Only {len(valid_captains)} of {len(required_captains)} AI captains available"
                )
                ai_rules['variable_locks']['captain'] = valid_captains
                validation['adjustments'].append("Reduced captain pool to available players")
        
        # Check salary feasibility with requirements
        if must_include:
            must_include_df = player_pool[player_pool['Player'].isin(must_include)]
            min_required_salary = must_include_df['Salary'].sum()
            
            # Account for captain multiplier
            if validation['stats']['captain_options'] > 0:
                # Assume cheapest required player could be captain
                cheapest_required = must_include_df['Salary'].min()
                min_required_salary += cheapest_required * 0.5  # 1.5x multiplier
            
            if min_required_salary > OptimizerConfig.SALARY_CAP:
                validation['errors'].append(
                    f"AI required players cost minimum ${min_required_salary:.0f} (exceeds ${OptimizerConfig.SALARY_CAP} cap)"
                )
                validation['is_valid'] = False
            elif min_required_salary > OptimizerConfig.SALARY_CAP * 0.9:
                validation['warnings'].append(
                    f"AI requirements use {min_required_salary/OptimizerConfig.SALARY_CAP:.0%} of salary cap"
                )
        
        # Check stacking requirements
        for constraint in ai_rules.get('hard_constraints', []):
            if constraint.get('rule') == 'must_stack':
                stack_players = constraint.get('players', [])
                missing_stack_players = [p for p in stack_players if p not in available_players]
                if missing_stack_players:
                    validation['warnings'].append(
                        f"Stack players not available: {missing_stack_players}"
                    )
        
        return validation

    @staticmethod
    def validate_field_config_with_ai(field_size: str, num_lineups: int,
                                      ai_recommendations: Dict) -> Dict:
        """Validate and adjust field configuration based on AI consensus"""
        
        base_config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(
            field_size, 
            OptimizerConfig.FIELD_SIZE_CONFIGS['large_field']
        )
        
        # Deep copy to avoid modifying original
        config = json.loads(json.dumps(base_config))
        
        # Get AI strategy distribution
        ai_distribution = config.get('ai_strategy_distribution', {})
        
        # Adjust based on AI consensus level
        if ai_recommendations:
            # Calculate consensus strength
            high_confidence_count = sum(
                1 for rec in ai_recommendations.values() 
                if rec.confidence > 0.8
            )
            
            consensus_agreement = len(set(
                captain 
                for rec in ai_recommendations.values() 
                for captain in rec.captain_targets[:1]
            ))
            
            if high_confidence_count == 3 and consensus_agreement <= 2:
                # Strong consensus - increase consensus strategy allocation
                if 'ai_consensus' in ai_distribution:
                    ai_distribution['ai_consensus'] = min(
                        0.6, 
                        ai_distribution.get('ai_consensus', 0.2) * 1.5
                    )
                    # Normalize other strategies
                    total = sum(ai_distribution.values())
                    if total > 0:
                        ai_distribution = {
                            k: v/total for k, v in ai_distribution.items()
                        }
            
            config['ai_strategy_distribution'] = ai_distribution
        
        # Validate lineup count against unique captain requirement
        if num_lineups < config.get('min_unique_captains', 1):
            config['min_unique_captains'] = max(1, num_lineups // 2)
            
        return config

    @staticmethod
    def get_ai_strategy_distribution(field_size: str, num_lineups: int,
                                     ai_consensus_level: str = 'mixed') -> Dict[StrategyType, int]:
        """Get optimal lineup distribution across AI strategies"""
        
        config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(
            field_size,
            OptimizerConfig.FIELD_SIZE_CONFIGS['large_field']
        )
        distribution = config.get('ai_strategy_distribution', {})
        
        # Adjust distribution based on consensus level
        if ai_consensus_level == 'high':
            # Prioritize consensus strategies
            priority_order = [
                'ai_consensus', 'ai_majority', 'ai_mixed', 
                'ai_correlation', 'ai_contrarian', 'ai_game_theory'
            ]
        elif ai_consensus_level == 'low':
            # Prioritize diverse strategies
            priority_order = [
                'ai_contrarian', 'ai_game_theory', 'ai_correlation',
                'ai_mixed', 'ai_majority', 'ai_consensus'
            ]
        else:
            # Balanced approach
            priority_order = list(distribution.keys())
        
        strategy_counts = {}
        remaining = num_lineups
        
        # Allocate lineups based on priority and percentages
        for strategy in priority_order:
            if strategy in distribution and remaining > 0:
                # Calculate count with rounding
                count = round(distribution[strategy] * num_lineups)
                count = min(count, remaining)
                
                if count > 0:
                    # Try to convert to StrategyType enum
                    try:
                        strategy_enum = StrategyType[strategy.upper()]
                        strategy_counts[strategy_enum] = count
                    except (KeyError, AttributeError):
                        # Use string key if enum doesn't exist
                        strategy_counts[strategy] = count
                    
                    remaining -= count
        
        # Distribute remaining lineups to first strategy
        if remaining > 0 and strategy_counts:
            first_key = list(strategy_counts.keys())[0]
            strategy_counts[first_key] += remaining
        elif remaining == num_lineups:
            # No valid distribution, use default
            try:
                default_strategy = StrategyType.AI_MIXED
            except:
                default_strategy = 'ai_mixed'
            strategy_counts[default_strategy] = num_lineups
        
        return strategy_counts

class ConfigValidator:
    """Standard configuration validator with enhanced validation"""

    @staticmethod
    def validate_field_config(field_size: str, num_lineups: int) -> Dict:
        """Validate and return field configuration"""
        config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size)
        
        if not config:
            get_logger().log(f"Unknown field size '{field_size}', using large_field defaults", "WARNING")
            config = OptimizerConfig.FIELD_SIZE_CONFIGS['large_field'].copy()
        else:
            # Deep copy to avoid modifying original
            config = json.loads(json.dumps(config))
        
        # Adjust unique captains if needed
        if num_lineups < config['min_unique_captains']:
            original = config['min_unique_captains']
            config['min_unique_captains'] = max(1, min(num_lineups, num_lineups // 2))
            get_logger().log(
                f"Adjusted min_unique_captains from {original} to {config['min_unique_captains']}", 
                "DEBUG"
            )
        
        # Validate ownership targets
        min_own, max_own = config['min_total_ownership'], config['max_total_ownership']
        if min_own >= max_own:
            config['max_total_ownership'] = min_own + 20
            get_logger().log("Adjusted ownership range", "DEBUG")
            
        return config

    @staticmethod
    def validate_player_pool(df: pd.DataFrame, field_size: str) -> Dict:
        """Comprehensive player pool validation"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for required columns
        required_columns = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation['errors'].append(f"Missing required columns: {missing_columns}")
            validation['is_valid'] = False
            return validation
        
        # Check minimum players
        player_count = len(df)
        validation['stats']['total_players'] = player_count
        
        if player_count < 12:
            validation['warnings'].append(f"Only {player_count} players (12+ recommended)")
            if player_count < 6:
                validation['errors'].append(f"Insufficient players: {player_count} (minimum 6)")
                validation['is_valid'] = False
        
        # Check position distribution
        positions = df['Position'].value_counts()
        validation['stats']['positions'] = positions.to_dict()
        
        if 'QB' not in positions or positions.get('QB', 0) == 0:
            validation['warnings'].append("No QB in player pool")
        
        # Check for minimum pass catchers
        pass_catchers = positions.get('WR', 0) + positions.get('TE', 0)
        if pass_catchers < 3:
            validation['warnings'].append(f"Only {pass_catchers} pass catchers (WR+TE)")
        
        # Check team distribution
        teams = df['Team'].unique()
        validation['stats']['teams'] = len(teams)
        
        if len(teams) != 2:
            validation['errors'].append(f"Expected 2 teams, found {len(teams)}: {teams}")
            validation['is_valid'] = False
        
        # Check salary distribution
        avg_salary = df['Salary'].mean()
        min_salary = df['Salary'].min()
        max_salary = df['Salary'].max()
        
        validation['stats']['salary'] = {
            'avg': avg_salary,
            'min': min_salary,
            'max': max_salary
        }
        
        if avg_salary < 5000:
            validation['warnings'].append(f"Low average salary (${avg_salary:.0f})")
        
        if max_salary < 9000:
            validation['warnings'].append("No premium players (max salary < $9000)")
        
        # Check for duplicate players
        duplicates = df[df.duplicated(subset=['Player'], keep=False)]
        if not duplicates.empty:
            validation['errors'].append(f"Duplicate players found: {duplicates['Player'].unique()}")
            validation['is_valid'] = False
        
        # Check ownership values if present
        if 'Ownership' in df.columns:
            invalid_ownership = df[(df['Ownership'] < 0) | (df['Ownership'] > 100)]
            if not invalid_ownership.empty:
                validation['warnings'].append(
                    f"{len(invalid_ownership)} players with invalid ownership values"
                )
        
        return validation

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Quick validation for DataFrame structure"""
        required_columns = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 'Ownership']
        missing = [col for col in required_columns if col not in df.columns]
        return len(missing) == 0, missing

# ============================================================================
# AI SYNTHESIS ENGINE
# ============================================================================

class AISynthesisEngine:
    """Synthesizes recommendations from multiple AI strategists"""

    def __init__(self):
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()
        self.synthesis_history = []
        self.max_history = OptimizerConfig.MAX_SYNTHESIS_HISTORY
        
    def synthesize_recommendations(self,
                                  game_theory_rec: AIRecommendation,
                                  correlation_rec: AIRecommendation,
                                  contrarian_rec: AIRecommendation) -> Dict:
        """Combine three AI perspectives into unified strategy with validation"""
        
        self.perf_monitor.start_timer("synthesize_recommendations")
        
        # Validate inputs
        for rec, name in [(game_theory_rec, "Game Theory"), 
                         (correlation_rec, "Correlation"),
                         (contrarian_rec, "Contrarian")]:
            is_valid, errors = rec.validate()
            if not is_valid:
                self.logger.log(f"{name} recommendation validation errors: {errors}", "WARNING")
        
        synthesis = {
            'captain_strategy': {},
            'player_rankings': {},
            'stacking_rules': [],
            'avoidance_rules': [],
            'narrative': "",
            'confidence': 0.0,
            'enforcement_rules': [],
            'metadata': {
                'timestamp': pd.Timestamp.now(),
                'ai_confidences': {
                    'game_theory': game_theory_rec.confidence,
                    'correlation': correlation_rec.confidence,
                    'contrarian': contrarian_rec.confidence
                }
            }
        }
        
        # Combine captain recommendations with weighted voting
        captain_votes = defaultdict(lambda: {'score': 0, 'voters': [], 'narratives': []})
        
        for rec, weight in [(game_theory_rec, OptimizerConfig.AI_WEIGHTS[AIStrategistType.GAME_THEORY]),
                           (correlation_rec, OptimizerConfig.AI_WEIGHTS[AIStrategistType.CORRELATION]),
                           (contrarian_rec, OptimizerConfig.AI_WEIGHTS[AIStrategistType.CONTRARIAN_NARRATIVE])]:
            for captain in rec.captain_targets[:7]:  # Consider top 7 from each
                captain_votes[captain]['score'] += weight * rec.confidence
                captain_votes[captain]['voters'].append(rec.source_ai)
                if rec.narrative:
                    captain_votes[captain]['narratives'].append(rec.narrative)
        
        # Classify captains by consensus level
        for captain, data in captain_votes.items():
            num_voters = len(data['voters'])
            if num_voters == 3:
                synthesis['captain_strategy'][captain] = 'consensus'
            elif num_voters == 2:
                synthesis['captain_strategy'][captain] = 'majority'
            else:
                # Single AI recommendation - tag with specific AI
                synthesis['captain_strategy'][captain] = data['voters'][0].value
        
        # Sort captains by score
        sorted_captains = sorted(captain_votes.items(), 
                                key=lambda x: x[1]['score'], 
                                reverse=True)
        
        # Keep top captains based on field requirements
        synthesis['captain_strategy'] = {
            captain: synthesis['captain_strategy'][captain]
            for captain, _ in sorted_captains[:20]
        }
        
        # Rank all players with confidence weighting
        player_scores = defaultdict(float)
        
        for rec in [game_theory_rec, correlation_rec, contrarian_rec]:
            weight = OptimizerConfig.AI_WEIGHTS[rec.source_ai]
            
            # Captain bonus
            for i, captain in enumerate(rec.captain_targets):
                player_scores[captain] += weight * rec.confidence * (2.0 - i * 0.1)
            
            # Must play bonus
            for i, player in enumerate(rec.must_play):
                player_scores[player] += weight * rec.confidence * (1.0 - i * 0.05)
            
            # Never play penalty
            for player in rec.never_play:
                player_scores[player] -= weight * rec.confidence * 0.5
                
            # Apply boosts if specified
            if rec.boosts:
                for player in rec.boosts:
                    player_scores[player] += weight * 0.3
        
        # Create rankings (top 50 players)
        synthesis['player_rankings'] = dict(
            sorted(player_scores.items(), key=lambda x: x[1], reverse=True)[:50]
        )
        
        # Combine stacking rules with priority
        stack_map = defaultdict(lambda: {'stack': None, 'support': [], 'confidence': 0})
        
        for rec in [game_theory_rec, correlation_rec, contrarian_rec]:
            weight = OptimizerConfig.AI_WEIGHTS[rec.source_ai]
            
            for stack in rec.stacks:
                if isinstance(stack, dict):
                    p1 = stack.get('player1', '')
                    p2 = stack.get('player2', '')
                    if p1 and p2:
                        stack_key = tuple(sorted([p1, p2]))
                        
                        if stack_map[stack_key]['stack'] is None:
                            stack_map[stack_key]['stack'] = stack
                        
                        stack_map[stack_key]['support'].append(rec.source_ai)
                        stack_map[stack_key]['confidence'] += weight * rec.confidence
        
        # Process stacks by support level
        for stack_key, data in sorted(stack_map.items(), 
                                     key=lambda x: x[1]['confidence'], 
                                     reverse=True)[:15]:
            stack = data['stack'].copy()
            
            if len(data['support']) >= 2:
                stack['strength'] = 'strong'
            else:
                stack['strength'] = 'moderate'
            
            stack['support'] = [ai.value for ai in data['support']]
            stack['combined_confidence'] = data['confidence']
            
            synthesis['stacking_rules'].append(stack)
        
        # Combine avoidance rules (players to fade)
        avoid_votes = defaultdict(float)
        
        for rec in [game_theory_rec, correlation_rec, contrarian_rec]:
            weight = OptimizerConfig.AI_WEIGHTS[rec.source_ai]
            for player in rec.never_play:
                avoid_votes[player] += weight * rec.confidence
        
        # Keep players with strong fade consensus
        synthesis['avoidance_rules'] = [
            player for player, score in avoid_votes.items()
            if score >= 0.5  # At least 50% weighted agreement
        ]
        
        # Create comprehensive narrative
        narratives = []
        if game_theory_rec.narrative:
            narratives.append(f"GT: {game_theory_rec.narrative[:100]}")
        if correlation_rec.narrative:
            narratives.append(f"CORR: {correlation_rec.narrative[:100]}")
        if contrarian_rec.narrative:
            narratives.append(f"CONTRA: {contrarian_rec.narrative[:100]}")
        
        synthesis['narrative'] = " | ".join(narratives)
        
        # Calculate combined confidence (weighted average)
        synthesis['confidence'] = (
            game_theory_rec.confidence * OptimizerConfig.AI_WEIGHTS[AIStrategistType.GAME_THEORY] +
            correlation_rec.confidence * OptimizerConfig.AI_WEIGHTS[AIStrategistType.CORRELATION] +
            contrarian_rec.confidence * OptimizerConfig.AI_WEIGHTS[AIStrategistType.CONTRARIAN_NARRATIVE]
        )
        
        # Create enforcement rules from synthesis
        synthesis['enforcement_rules'] = self._create_enforcement_rules(synthesis)
        
        # Log synthesis
        elapsed = self.perf_monitor.stop_timer("synthesize_recommendations")
        
        self.logger.log(
            f"AI Synthesis complete: {len(synthesis['captain_strategy'])} captains, "
            f"{len(synthesis['stacking_rules'])} stacks, confidence: {synthesis['confidence']:.2f}, "
            f"time: {elapsed:.3f}s",
            "INFO"
        )
        
        # Store in history with memory management
        self.synthesis_history.append(synthesis)
        if len(self.synthesis_history) > self.max_history:
            self.synthesis_history = self.synthesis_history[-self.max_history:]
        
        return synthesis

    def _create_enforcement_rules(self, synthesis: Dict) -> List[Dict]:
        """Convert synthesis into prioritized enforcement rules"""
        rules = []
        
        # Captain rules (highest priority)
        consensus_captains = [
            c for c, level in synthesis['captain_strategy'].items() 
            if level == 'consensus'
        ][:5]  # Top 5 consensus captains
        
        if consensus_captains:
            rules.append({
                'type': 'hard',
                'rule': 'captain_from_list',
                'players': consensus_captains,
                'priority': 100,
                'description': 'Must use consensus captain'
            })
        
        # Strong stacking rules
        strong_stacks = [
            s for s in synthesis['stacking_rules'] 
            if s.get('strength') == 'strong' and s.get('combined_confidence', 0) > 0.6
        ][:3]  # Top 3 strong stacks
        
        for i, stack in enumerate(strong_stacks):
            rules.append({
                'type': 'soft' if i > 0 else 'hard',  # First stack is hard, others soft
                'rule': 'include_stack',
                'players': [stack.get('player1', ''), stack.get('player2', '')],
                'weight': 0.8 - (i * 0.1),
                'priority': 80 - (i * 5),
                'description': f"Stack: {stack.get('player1', '')} + {stack.get('player2', '')}"
            })
        
        # Avoidance rules
        for i, player in enumerate(synthesis['avoidance_rules'][:5]):
            rules.append({
                'type': 'soft',
                'rule': 'avoid_player',
                'player': player,
                'weight': 0.7 - (i * 0.1),
                'priority': 60 - i,
                'description': f"Avoid: {player}"
            })
        
        # Top ranked players as soft includes
        top_players = list(synthesis['player_rankings'].keys())[:3]
        for i, player in enumerate(top_players):
            if player not in consensus_captains:  # Avoid duplication
                rules.append({
                    'type': 'soft',
                    'rule': 'prefer_player',
                    'player': player,
                    'weight': 0.6 - (i * 0.1),
                    'priority': 50 - i,
                    'description': f"Prefer high-ranked: {player}"
                })
        
        return rules

    def get_synthesis_summary(self) -> Dict:
        """Get comprehensive summary of synthesis history"""
        if not self.synthesis_history:
            return {
                'total_syntheses': 0,
                'latest_confidence': 0,
                'consensus_captains': 0
            }
        
        latest = self.synthesis_history[-1]
        
        # Analyze consensus trends
        consensus_trend = []
        for synth in self.synthesis_history[-5:]:
            consensus_count = len([
                c for c, l in synth['captain_strategy'].items() 
                if l == 'consensus'
            ])
            consensus_trend.append(consensus_count)
        
        return {
            'total_syntheses': len(self.synthesis_history),
            'latest_confidence': latest['confidence'],
            'consensus_captains': len([
                c for c, l in latest['captain_strategy'].items() 
                if l == 'consensus'
            ]),
            'total_captains': len(latest['captain_strategy']),
            'strong_stacks': len([
                s for s in latest['stacking_rules'] 
                if s.get('strength') == 'strong'
            ]),
            'consensus_trend': consensus_trend,
            'avg_confidence': np.mean([
                s['confidence'] for s in self.synthesis_history[-10:]
            ]) if self.synthesis_history else 0
        }
        # NFL GPP DUAL-AI OPTIMIZER - PART 3: AI STRATEGISTS (AI-AS-CHEF VERSION)
# Version 6.3 - Triple AI System with Enhanced Robustness
# NOTE: This continues from Parts 1-2 - all imports already consolidated at top

# ============================================================================
# BASE AI STRATEGIST CLASS WITH ENHANCED ERROR HANDLING
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
        
        # Fallback confidence levels
        self.fallback_confidence = {
            AIStrategistType.GAME_THEORY: 0.5,
            AIStrategistType.CORRELATION: 0.55,
            AIStrategistType.CONTRARIAN_NARRATIVE: 0.45
        }

    def get_recommendation(self, df: pd.DataFrame, game_info: Dict,
                          field_size: str, use_api: bool = True) -> AIRecommendation:
        """Get AI recommendation with comprehensive error handling"""
        
        try:
            # Validate inputs
            if df.empty:
                self.logger.log(f"{self.strategist_type.value}: Empty DataFrame provided", "ERROR")
                return self._get_fallback_recommendation(df, field_size)
            
            # Generate cache key
            cache_key = self._generate_cache_key(df, game_info, field_size)
            
            # Check cache
            with self._cache_lock:
                if cache_key in self.response_cache:
                    self.logger.log(f"{self.strategist_type.value}: Using cached recommendation", "DEBUG")
                    return self.response_cache[cache_key]
            
            # Generate prompt
            prompt = self.generate_prompt(df, game_info, field_size)
            
            # Get response (API or manual)
            if use_api and self.api_manager and self.api_manager.client:
                response = self._get_api_response(prompt)
            else:
                response = self._get_fallback_response(df, game_info, field_size)
            
            # Parse response into recommendation
            recommendation = self.parse_response(response, df, field_size)
            
            # Validate recommendation
            is_valid, errors = recommendation.validate()
            if not is_valid:
                self.logger.log(f"{self.strategist_type.value} validation errors: {errors}", "WARNING")
                # Apply corrections
                recommendation = self._correct_recommendation(recommendation, df)
            
            # Add enforcement rules
            recommendation.enforcement_rules = self.create_enforcement_rules(recommendation, df, field_size)
            
            # Cache the result
            with self._cache_lock:
                self.response_cache[cache_key] = recommendation
                # Manage cache size
                if len(self.response_cache) > self.max_cache_size:
                    # Remove oldest entries
                    for key in list(self.response_cache.keys())[:5]:
                        del self.response_cache[key]
            
            return recommendation
            
        except Exception as e:
            self.logger.log_exception(e, f"{self.strategist_type.value}.get_recommendation")
            return self._get_fallback_recommendation(df, field_size)

    def _get_api_response(self, prompt: str) -> str:
        """Get API response with retry logic and error handling"""
        self.perf_monitor.start_timer(f"ai_{self.strategist_type.value}_api")
        
        max_retries = 2
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                if not self.api_manager or not self.api_manager.client:
                    raise Exception("API client not available")
                
                response = self.api_manager.get_ai_response(prompt, self.strategist_type)
                
                if response and response != '{}':
                    self.perf_monitor.increment_counter("ai_api_success")
                    return response
                    
            except Exception as e:
                self.logger.log(
                    f"API attempt {attempt + 1} failed for {self.strategist_type.value}: {e}", 
                    "WARNING"
                )
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        
        self.perf_monitor.increment_counter("ai_api_failures")
        return '{}'  # Return empty JSON on failure
        
    def _get_fallback_response(self, df: pd.DataFrame, game_info: Dict, field_size: str) -> str:
        """Generate statistical fallback response"""
        # This will be overridden by child classes
        return '{}'

    def create_enforcement_rules(self, recommendation: AIRecommendation,
                                df: pd.DataFrame, field_size: str) -> List[Dict]:
        """Create specific enforcement rules with validation"""
        rules = []
        
        # Validate players exist in DataFrame
        available_players = set(df['Player'].values)
        
        # Captain enforcement
        valid_captains = [c for c in recommendation.captain_targets if c in available_players]
        
        if valid_captains:
            if recommendation.confidence > 0.8:
                rules.append({
                    'type': 'hard',
                    'constraint': f'captain_in_{self.strategist_type.value}',
                    'players': valid_captains[:5],
                    'priority': int(recommendation.confidence * 100),
                    'description': f'{self.strategist_type.value}: High-confidence captains'
                })
            else:
                rules.append({
                    'type': 'soft',
                    'constraint': f'prefer_captain_{self.strategist_type.value}',
                    'players': valid_captains[:5],
                    'weight': recommendation.confidence,
                    'priority': int(recommendation.confidence * 50),
                    'description': f'{self.strategist_type.value}: Preferred captains'
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
        
        # Never play enforcement
        valid_never_play = [p for p in recommendation.never_play[:3] if p in available_players]
        
        for player in valid_never_play:
            rule_type = 'hard' if recommendation.confidence > 0.8 else 'soft'
            rules.append({
                'type': rule_type,
                'constraint': f'exclude_{player}',
                'player': player,
                'weight': recommendation.confidence if rule_type == 'soft' else 1.0,
                'priority': int(recommendation.confidence * 40),
                'description': f'{self.strategist_type.value}: Exclude {player}'
            })
        
        # Stack enforcement
        for stack in recommendation.stacks[:2]:
            if isinstance(stack, dict):
                players = [stack.get('player1'), stack.get('player2')]
                if all(p and p in available_players for p in players):
                    rules.append({
                        'type': 'soft',
                        'constraint': f'stack_{players[0]}_{players[1]}',
                        'players': players,
                        'weight': recommendation.confidence * 0.8,
                        'priority': int(recommendation.confidence * 60),
                        'description': f'{self.strategist_type.value}: Stack {players[0]} + {players[1]}'
                    })
        
        return rules

    def _correct_recommendation(self, recommendation: AIRecommendation, 
                               df: pd.DataFrame) -> AIRecommendation:
        """Correct invalid recommendations"""
        available_players = set(df['Player'].values)
        
        # Filter to valid players only
        recommendation.captain_targets = [
            p for p in recommendation.captain_targets if p in available_players
        ]
        recommendation.must_play = [
            p for p in recommendation.must_play if p in available_players
        ]
        recommendation.never_play = [
            p for p in recommendation.never_play if p in available_players
        ]
        
        # Ensure minimum captains
        if len(recommendation.captain_targets) < 3:
            # Add top projected players as captains
            top_players = df.nlargest(5, 'Projected_Points')['Player'].tolist()
            for player in top_players:
                if player not in recommendation.captain_targets:
                    recommendation.captain_targets.append(player)
                if len(recommendation.captain_targets) >= 5:
                    break
        
        # Adjust confidence if corrections were needed
        if len(recommendation.captain_targets) < 3:
            recommendation.confidence *= 0.8
        
        recommendation.validation_status = 'corrected'
        
        return recommendation

    def _generate_cache_key(self, df: pd.DataFrame, game_info: Dict, 
                           field_size: str) -> str:
        """Generate cache key for recommendation"""
        # Create hash from key inputs
        key_parts = [
            field_size,
            str(len(df)),
            str(game_info.get('total', 0)),
            str(game_info.get('spread', 0)),
            str(df['Salary'].sum()),
            self.strategist_type.value
        ]
        
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_fallback_recommendation(self, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Generate statistical fallback recommendation"""
        try:
            # Calculate value metric
            df = df.copy()
            df['Value'] = df['Projected_Points'] / (df['Salary'] / 1000)
            df['GPP_Score'] = df['Value'] * (30 / (df.get('Ownership', 10) + 10))
            
            # Get diverse captain pool
            top_proj = df.nlargest(3, 'Projected_Points')['Player'].tolist()
            top_value = df.nlargest(3, 'Value')['Player'].tolist()
            top_gpp = df.nlargest(3, 'GPP_Score')['Player'].tolist()
            
            captain_targets = []
            for player_list in [top_proj, top_value, top_gpp]:
                for player in player_list:
                    if player not in captain_targets:
                        captain_targets.append(player)
            
            # Get must play (high value players)
            must_play = df[df['Value'] > df['Value'].quantile(0.8)]['Player'].tolist()[:5]
            
            # Get never play (low value, high ownership)
            never_play = df[
                (df.get('Ownership', 10) > 30) & 
                (df['Value'] < df['Value'].median())
            ]['Player'].tolist()[:3]
            
            return AIRecommendation(
                captain_targets=captain_targets[:7],
                must_play=must_play,
                never_play=never_play,
                stacks=[],
                key_insights=[f"Statistical fallback for {self.strategist_type.value}"],
                confidence=self.fallback_confidence.get(self.strategist_type, 0.4),
                enforcement_rules=[],
                narrative=f"Using statistical analysis due to API unavailability",
                source_ai=self.strategist_type
            )
            
        except Exception as e:
            self.logger.log_exception(e, "fallback_recommendation")
            # Return minimal valid recommendation
            return AIRecommendation(
                captain_targets=df.nlargest(5, 'Projected_Points')['Player'].tolist(),
                must_play=[],
                never_play=[],
                stacks=[],
                key_insights=["Minimal fallback recommendation"],
                confidence=0.3,
                enforcement_rules=[],
                narrative="Error generating recommendation",
                source_ai=self.strategist_type
            )

    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str) -> str:
        """Generate prompt - to be overridden by subclasses"""
        raise NotImplementedError

    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Parse response - to be overridden by subclasses"""
        raise NotImplementedError

# ============================================================================
# GPP GAME THEORY STRATEGIST
# ============================================================================

class GPPGameTheoryStrategist(BaseAIStrategist):
    """AI Strategist 1: Game Theory and Ownership Leverage"""

    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.GAME_THEORY)
        
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field') -> str:
        """Generate game theory focused prompt with validation"""
        
        self.logger.log(f"Generating Game Theory prompt for {field_size}", "DEBUG")
        
        # Validate and prepare data
        if df.empty:
            return "Error: Empty player pool"
        
        # Prepare data summaries with error handling
        try:
            bucket_manager = AIOwnershipBucketManager()
            buckets = bucket_manager.categorize_players(df)
            
            # Create ownership distribution
            ownership_bins = [0, 5, 10, 20, 30, 100]
            ownership_labels = ['0-5%', '5-10%', '10-20%', '20-30%', '30%+']
            df['OwnershipBin'] = pd.cut(df.get('Ownership', 0), bins=ownership_bins, labels=ownership_labels, include_lowest=True)
            ownership_summary = df['OwnershipBin'].value_counts().to_dict()
            
        except Exception as e:
            self.logger.log(f"Error preparing data for prompt: {e}", "WARNING")
            ownership_summary = {"Unknown": len(df)}
        
        # Field-specific strategies
        field_strategies = {
            'small_field': "Focus on slight differentiation while maintaining optimal plays",
            'medium_field': "Balance chalk with 2-3 strong leverage plays",
            'large_field': "Aggressive leverage with <15% owned captains",
            'large_field_aggressive': "Ultra-leverage approach with <10% captains",
            'milly_maker': "Maximum contrarian approach with <10% captains only",
            'super_contrarian': "Extreme leverage targeting <5% ownership"
        }
        
        # Get low-owned high-upside plays
        low_owned_high_upside = df[df.get('Ownership', 10) < 10].nlargest(10, 'Projected_Points')
        
        # Build prompt
        prompt = f"""
        You are an expert DFS game theory strategist. Create an ENFORCEABLE lineup strategy for {field_size} GPP tournaments.

        GAME CONTEXT:
        Teams: {game_info.get('teams', 'Unknown')}
        Total: {game_info.get('total', 45)}
        Spread: {game_info.get('spread', 0)}
        Weather: {game_info.get('weather', 'Clear')}

        PLAYER POOL ANALYSIS:
        Total players: {len(df)}
        Ownership distribution: {ownership_summary}
        Average salary: ${df['Salary'].mean():.0f}

        HIGH LEVERAGE PLAYS (<10% ownership):
        {low_owned_high_upside[['Player', 'Position', 'Salary', 'Projected_Points', 'Ownership']].to_string() if not low_owned_high_upside.empty else 'No low-owned plays found'}

        FIELD STRATEGY:
        {field_strategies.get(field_size, 'Standard GPP strategy')}

        PROVIDE SPECIFIC, ENFORCEABLE RULES IN JSON:
        {{
            "captain_rules": {{
                "must_be_one_of": ["player1", "player2", "player3"],
                "ownership_ceiling": 15,
                "min_projection": 15,
                "reasoning": "Why these specific captains win tournaments"
            }},
            "lineup_rules": {{
                "must_include": ["player_x"],
                "never_include": ["player_y"],
                "ownership_sum_range": [60, 90],
                "min_leverage_players": 2
            }},
            "correlation_rules": {{
                "required_stacks": [{{"player1": "name1", "player2": "name2", "type": "QB_WR"}}],
                "avoid_negative_correlation": ["player_a", "player_b"]
            }},
            "game_theory_insights": {{
                "field_will_play": "What most will do",
                "exploit_this": "How to exploit the field",
                "unique_angle": "Your contrarian approach"
            }},
            "confidence": 0.85,
            "key_insight": "The ONE thing that makes this lineup win tournaments"
        }}

        Be SPECIFIC with exact player names. Focus on ownership leverage and game theory edge.
        """
        
        return prompt

    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Parse response with comprehensive validation"""
        
        try:
            # Try to parse JSON response
            if response and response != '{}':
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    self.logger.log("Failed to parse JSON response, using fallback", "WARNING")
                    data = {}
            else:
                data = {}
            
            # Extract and validate captain rules
            captain_rules = data.get('captain_rules', {})
            captain_targets = captain_rules.get('must_be_one_of', [])
            ownership_ceiling = captain_rules.get('ownership_ceiling', 15)
            
            # Validate captains exist in pool
            available_players = set(df['Player'].values)
            valid_captains = [c for c in captain_targets if c in available_players]
            
            # If not enough valid captains from AI, use statistical selection
            if len(valid_captains) < 3:
                # Get players under ownership ceiling
                eligible = df[df.get('Ownership', 10) <= ownership_ceiling]
                
                if len(eligible) < 5:
                    eligible = df.copy()
                
                # Sort by projection and add to valid captains
                additional = eligible.nlargest(5, 'Projected_Points')['Player'].tolist()
                for player in additional:
                    if player not in valid_captains:
                        valid_captains.append(player)
                    if len(valid_captains) >= 5:
                        break
            
            # Extract lineup rules
            lineup_rules = data.get('lineup_rules', {})
            must_include = [p for p in lineup_rules.get('must_include', []) if p in available_players]
            never_include = [p for p in lineup_rules.get('never_include', []) if p in available_players]
            ownership_range = lineup_rules.get('ownership_sum_range', [60, 90])
            
            # Validate ownership range
            if ownership_range[0] >= ownership_range[1]:
                ownership_range = [60, 90]
            
            # Extract stacks
            correlation_rules = data.get('correlation_rules', {})
            stacks = []
            
            for stack_data in correlation_rules.get('required_stacks', []):
                p1 = stack_data.get('player1')
                p2 = stack_data.get('player2')
                
                if p1 in available_players and p2 in available_players:
                    stacks.append({
                        'player1': p1,
                        'player2': p2,
                        'type': stack_data.get('type', 'generic'),
                        'correlation': 0.7  # Default correlation
                    })
            
            # Extract insights
            game_theory = data.get('game_theory_insights', {})
            key_insights = [
                data.get('key_insight', 'Leverage game theory for GPP edge'),
                game_theory.get('exploit_this', ''),
                game_theory.get('unique_angle', '')
            ]
            key_insights = [i for i in key_insights if i]  # Remove empty insights
            
            # Build enforcement rules
            enforcement_rules = []
            
            # Captain constraint
            if valid_captains:
                enforcement_rules.append({
                    'type': 'hard',
                    'constraint': 'captain_selection',
                    'players': valid_captains[:5],
                    'description': 'Game theory optimal captains'
                })
            
            # Ownership sum constraint
            enforcement_rules.append({
                'type': 'hard',
                'constraint': 'ownership_sum',
                'min': ownership_range[0],
                'max': ownership_range[1],
                'description': f'Total ownership {ownership_range[0]}-{ownership_range[1]}%'
            })
            
            # Minimum leverage players
            min_leverage = lineup_rules.get('min_leverage_players', 2)
            if min_leverage > 0:
                enforcement_rules.append({
                    'type': 'soft',
                    'constraint': 'min_leverage',
                    'count': min_leverage,
                    'weight': 0.7,
                    'description': f'Include {min_leverage}+ leverage plays'
                })
            
            confidence = data.get('confidence', 0.7)
            
            # Validate confidence
            confidence = max(0.0, min(1.0, confidence))
            
            return AIRecommendation(
                captain_targets=valid_captains[:7],
                must_play=must_include[:5],
                never_play=never_include[:5],
                stacks=stacks[:5],
                key_insights=key_insights[:3],
                confidence=confidence,
                enforcement_rules=enforcement_rules,
                narrative=data.get('key_insight', 'Game theory optimization'),
                source_ai=AIStrategistType.GAME_THEORY,
                ownership_leverage={
                    'ownership_range': ownership_range,
                    'ownership_ceiling': ownership_ceiling,
                    'min_leverage': min_leverage
                }
            )
            
        except Exception as e:
            self.logger.log_exception(e, "parse_game_theory_response")
            return self._get_fallback_recommendation(df, field_size)

    def _get_fallback_response(self, df: pd.DataFrame, game_info: Dict, 
                              field_size: str) -> str:
        """Generate game theory focused fallback"""
        
        # Statistical game theory analysis
        ownership_threshold = 15 if field_size in ['large_field', 'milly_maker'] else 20
        
        # Find leverage captains
        leverage_captains = df[df.get('Ownership', 10) < ownership_threshold].nlargest(
            5, 'Projected_Points'
        )['Player'].tolist()
        
        # Find chalk to fade
        chalk_to_fade = df[df.get('Ownership', 10) > 30]['Player'].tolist()[:3]
        
        # Build response
        response = {
            "captain_rules": {
                "must_be_one_of": leverage_captains[:3],
                "ownership_ceiling": ownership_threshold,
                "reasoning": "Statistical leverage plays"
            },
            "lineup_rules": {
                "must_include": [],
                "never_include": chalk_to_fade,
                "ownership_sum_range": [60, 90]
            },
            "confidence": 0.5
        }
        
        return json.dumps(response)

# ============================================================================
# GPP CORRELATION STRATEGIST
# ============================================================================

class GPPCorrelationStrategist(BaseAIStrategist):
    """AI Strategist 2: Correlation and Stacking Patterns"""

    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.CORRELATION)

    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field') -> str:
        """Generate correlation focused prompt"""
        
        self.logger.log(f"Generating Correlation prompt for {field_size}", "DEBUG")
        
        # Validate inputs
        if df.empty:
            return "Error: Empty player pool"
        
        # Team analysis with error handling
        try:
            teams = df['Team'].unique()[:2]
            
            if len(teams) >= 2:
                team1, team2 = teams[0], teams[1]
                team1_df = df[df['Team'] == team1]
                team2_df = df[df['Team'] == team2]
            else:
                # Handle single team or missing team data
                team1 = teams[0] if len(teams) > 0 else "Unknown"
                team2 = "Unknown"
                team1_df = df if len(teams) > 0 else pd.DataFrame()
                team2_df = pd.DataFrame()
                
        except Exception as e:
            self.logger.log(f"Error analyzing teams: {e}", "WARNING")
            team1, team2 = "Team1", "Team2"
            team1_df = df[:len(df)//2]
            team2_df = df[len(df)//2:]
        
        # Game environment
        total = game_info.get('total', 45)
        spread = game_info.get('spread', 0)
        
        # Calculate correlation opportunities
        qbs = df[df['Position'] == 'QB']['Player'].tolist()
        pass_catchers = df[df['Position'].isin(['WR', 'TE'])]['Player'].tolist()
        
        prompt = f"""
        You are an expert DFS correlation strategist. Create SPECIFIC stacking rules for {field_size} GPP.

        GAME ENVIRONMENT:
        Total: {total} | Spread: {spread}
        Favorite: {team1 if spread < 0 else team2} by {abs(spread)} points

        TEAM 1 - {team1}:
        Key Players:
        {team1_df[['Player', 'Position', 'Salary', 'Projected_Points']].head(8).to_string() if not team1_df.empty else 'No data'}

        TEAM 2 - {team2}:
        Key Players:
        {team2_df[['Player', 'Position', 'Salary', 'Projected_Points']].head(8).to_string() if not team2_df.empty else 'No data'}

        CORRELATION OPPORTUNITIES:
        QBs available: {qbs[:3] if qbs else ['None']}
        Top pass catchers: {pass_catchers[:5] if pass_catchers else ['None']}

        CREATE ENFORCEABLE CORRELATION RULES IN JSON:
        {{
            "primary_stacks": [
                {{"type": "QB_WR1", "player1": "exact_qb_name", "player2": "exact_wr_name", "correlation": 0.7}},
                {{"type": "QB_TE", "player1": "exact_qb_name", "player2": "exact_te_name", "correlation": 0.6}}
            ],
            "game_stacks": [
                {{"players": ["qb1", "wr1", "opp_wr1"], "narrative": "shootout correlation", "expected_score": 50}}
            ],
            "leverage_stacks": [
                {{"type": "contrarian", "player1": "low_own_qb", "player2": "low_own_wr", "combined_ownership": 15}}
            ],
            "negative_correlation": [
                {{"avoid_together": ["rb1", "rb2"], "reason": "same backfield"}},
                {{"avoid_together": ["wr1", "wr2"], "reason": "target competition"}}
            ],
            "bring_back_rules": [
                {{"if_stacking": "team1", "bring_back_from": "team2", "position": "WR"}}
            ],
            "captain_correlation": {{
                "best_captains_for_stacking": ["qb_name", "primary_wr"],
                "correlation_multiplier": 1.5
            }},
            "confidence": 0.8,
            "stack_narrative": "WHY this correlation wins GPPs"
        }}

        Use EXACT player names. Focus on correlations that maximize ceiling.
        """
        
        return prompt

    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Parse correlation response with validation"""
        
        try:
            # Parse JSON
            if response and response != '{}':
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}
            
            available_players = set(df['Player'].values)
            all_stacks = []
            
            # Process primary stacks
            for stack in data.get('primary_stacks', []):
                if self._validate_stack(stack, available_players):
                    stack['priority'] = 'high'
                    stack['enforced'] = True
                    all_stacks.append(stack)
            
            # Process game stacks
            for game_stack in data.get('game_stacks', []):
                players = game_stack.get('players', [])
                valid_players = [p for p in players if p in available_players]
                
                if len(valid_players) >= 2:
                    # Create pairwise stacks from game stack
                    for i in range(len(valid_players) - 1):
                        all_stacks.append({
                            'player1': valid_players[i],
                            'player2': valid_players[i + 1],
                            'type': 'game_stack',
                            'priority': 'medium',
                            'narrative': game_stack.get('narrative', ''),
                            'correlation': 0.5
                        })
            
            # Process leverage stacks
            for stack in data.get('leverage_stacks', []):
                if self._validate_stack(stack, available_players):
                    stack['priority'] = 'high'
                    stack['leverage'] = True
                    all_stacks.append(stack)
            
            # If no valid stacks from AI, create statistical ones
            if len(all_stacks) < 2:
                all_stacks.extend(self._create_statistical_stacks(df))
            
            # Extract captain correlation
            captain_rules = data.get('captain_correlation', {})
            captain_targets = captain_rules.get('best_captains_for_stacking', [])
            valid_captains = [c for c in captain_targets if c in available_players]
            
            # If no valid captains, use QBs and top receivers
            if len(valid_captains) < 3:
                qbs = df[df['Position'] == 'QB']['Player'].tolist()
                top_receivers = df[df['Position'].isin(['WR', 'TE'])].nlargest(
                    3, 'Projected_Points'
                )['Player'].tolist()
                
                for player in qbs + top_receivers:
                    if player not in valid_captains:
                        valid_captains.append(player)
                    if len(valid_captains) >= 5:
                        break
            
            # Process negative correlations
            avoid_pairs = []
            for neg_corr in data.get('negative_correlation', []):
                players = neg_corr.get('avoid_together', [])
                if len(players) == 2 and all(p in available_players for p in players):
                    avoid_pairs.append({
                        'players': players,
                        'reason': neg_corr.get('reason', 'negative correlation')
                    })
            
            # Build enforcement rules
            enforcement_rules = []
            
            # Enforce top stacks
            high_priority_stacks = [s for s in all_stacks if s.get('priority') == 'high'][:2]
            
            for i, stack in enumerate(high_priority_stacks):
                enforcement_rules.append({
                    'type': 'hard' if i == 0 else 'soft',
                    'constraint': 'must_stack',
                    'players': [stack['player1'], stack['player2']],
                    'weight': 0.8 if i > 0 else 1.0,
                    'description': f"Correlation: {stack.get('type', 'stack')}"
                })
            
            # Enforce negative correlations
            for avoid in avoid_pairs[:2]:
                enforcement_rules.append({
                    'type': 'soft',
                    'constraint': 'avoid_together',
                    'players': avoid['players'],
                    'weight': 0.7,
                    'description': avoid['reason']
                })
            
            # Bring-back rules
            bring_back = data.get('bring_back_rules', [])
            if bring_back and field_size in ['large_field', 'milly_maker']:
                enforcement_rules.append({
                    'type': 'soft',
                    'constraint': 'bring_back',
                    'weight': 0.6,
                    'description': 'Include opponent in game stack'
                })
            
            confidence = data.get('confidence', 0.75)
            confidence = max(0.0, min(1.0, confidence))
            
            return AIRecommendation(
                captain_targets=valid_captains[:7],
                must_play=[],  # Will be filled by stacks
                never_play=[],
                stacks=all_stacks[:10],
                key_insights=[
                    data.get('stack_narrative', 'Correlation-based construction'),
                    f"Focus on {len(high_priority_stacks)} primary stacks"
                ],
                confidence=confidence,
                enforcement_rules=enforcement_rules,
                narrative=data.get('stack_narrative', ''),
                source_ai=AIStrategistType.CORRELATION,
                correlation_matrix={
                    f"{s['player1']}_{s['player2']}": s.get('correlation', 0.5)
                    for s in all_stacks[:5]
                }
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
                    # Sort by projection
                    top_teammates = teammates.nlargest(2, 'Projected_Points')
                    
                    for _, teammate in top_teammates.iterrows():
                        stacks.append({
                            'player1': qb['Player'],
                            'player2': teammate['Player'],
                            'type': f"QB_{teammate['Position']}",
                            'correlation': 0.6,
                            'priority': 'medium'
                        })
            
            # Limit to top stacks
            stacks = stacks[:6]
            
        except Exception as e:
            self.logger.log(f"Error creating statistical stacks: {e}", "WARNING")
        
        return stacks

# ============================================================================
# GPP CONTRARIAN NARRATIVE STRATEGIST
# ============================================================================

class GPPContrarianNarrativeStrategist(BaseAIStrategist):
    """AI Strategist 3: Contrarian Narratives and Hidden Angles"""

    def __init__(self, api_manager=None):
        super().__init__(api_manager, AIStrategistType.CONTRARIAN_NARRATIVE)

    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field') -> str:
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
        low_owned_high_ceiling = df_copy[df_copy.get('Ownership', 10) < 10].nlargest(
            10, 'Projected_Points'
        )
        
        hidden_value = df_copy[df_copy.get('Ownership', 10) < 15].nlargest(10, 'Value')
        
        contrarian_captains = df_copy.nlargest(10, 'Contrarian_Score')
        
        # Teams and game info
        teams = df['Team'].unique()[:2]
        total = game_info.get('total', 45)
        spread = game_info.get('spread', 0)
        weather = game_info.get('weather', 'Clear')
        
        prompt = f"""
        You are a contrarian DFS strategist who finds the NON-OBVIOUS narratives that win GPP tournaments.
        Your job is to identify the scenarios that create massive scores while the field follows chalk.

        GAME SETUP:
        {teams[0] if len(teams) > 0 else 'Team1'} vs {teams[1] if len(teams) > 1 else 'Team2'}
        Total: {total} | Spread: {spread} | Weather: {weather}

        CONTRARIAN OPPORTUNITIES:
        
        LOW-OWNED HIGH CEILING (<10% owned):
        {low_owned_high_ceiling[['Player', 'Position', 'Team', 'Projected_Points', 'Ownership']].head(7).to_string() if not low_owned_high_ceiling.empty else 'None found'}

        HIDDEN VALUE PLAYS:
        {hidden_value[['Player', 'Position', 'Salary', 'Value', 'Ownership']].head(7).to_string() if not hidden_value.empty else 'None found'}

        CONTRARIAN CAPTAIN SCORES:
        {contrarian_captains[['Player', 'Contrarian_Score', 'Ownership']].head(7).to_string() if not contrarian_captains.empty else 'None found'}

        CREATE CONTRARIAN TOURNAMENT-WINNING NARRATIVES IN JSON:
        {{
            "primary_narrative": "The ONE scenario everyone is missing that creates a tournament-winning lineup",
            "contrarian_captains": [
                {{"player": "exact_name", "narrative": "Why this 5% captain wins", "ceiling_scenario": "specific game flow that makes them explode"}}
            ],
            "hidden_correlations": [
                {{"player1": "name1", "player2": "name2", "narrative": "Non-obvious connection that the field misses"}}
            ],
            "fade_the_field": [
                {{"player": "chalky_player_name", "ownership": 35, "fade_reason": "Specific reason field is wrong", "pivot_to": "alternative_player"}}
            ],
            "boom_scenarios": [
                {{"players": ["player1", "player2"], "scenario": "Game script that creates ceiling", "probability": "5% but wins tournament if hits"}}
            ],
            "contrarian_game_theory": {{
                "what_field_expects": "Common narrative everyone follows",
                "why_field_is_wrong": "Specific flaw in common thinking",
                "exploit_this": "Exact players/stacks to exploit this edge"
            }},
            "tournament_winner": {{
                "captain": "exact_low_owned_captain",
                "core": ["player1", "player2", "player3"],
                "total_ownership": 65,
                "narrative": "The complete story of how this 0.1% lineup takes down the million"
            }},
            "confidence": 0.7
        }}

        Be EXTREMELY SPECIFIC with player names. Find the narrative that makes a sub-5% owned player the optimal captain.
        Think about game script, injuries, weather, and situations the field ignores.
        """
        
        return prompt

    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Parse contrarian narrative response"""
        
        try:
            # Parse JSON
            if response and response != '{}':
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    data = {}
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
                        'ceiling': captain_data.get('ceiling_scenario', '')
                    }
            
            # If no valid contrarian captains from AI, find them statistically
            if len(contrarian_captains) < 3:
                # Calculate contrarian score
                df_copy = df.copy()
                df_copy['Contrarian_Score'] = (
                    df_copy['Projected_Points'] / df_copy['Projected_Points'].max()
                ) / (df_copy.get('Ownership', 10) / 100 + 0.1)
                
                # Get top contrarian plays
                contrarian_plays = df_copy[df_copy.get('Ownership', 10) < 10].nlargest(
                    5, 'Contrarian_Score'
                )
                
                for _, row in contrarian_plays.iterrows():
                    player = row['Player']
                    if player not in contrarian_captains:
                        contrarian_captains.append(player)
                        captain_narratives[player] = {
                            'narrative': f"Hidden upside at {row.get('Ownership', 10):.1f}% ownership",
                            'ceiling': f"Contrarian score: {row['Contrarian_Score']:.2f}"
                        }
            
            # Extract tournament winner lineup
            tournament_winner = data.get('tournament_winner', {})
            tw_captain = tournament_winner.get('captain')
            tw_core = tournament_winner.get('core', [])
            
            must_play = []
            
            # Validate tournament winner captain
            if tw_captain and tw_captain in available_players:
                if tw_captain not in contrarian_captains:
                    contrarian_captains.insert(0, tw_captain)
            
            # Validate tournament winner core
            for player in tw_core:
                if player in available_players:
                    must_play.append(player)
            
            # Extract fades and pivots
            fades = []
            pivots = {}
            
            for fade_data in data.get('fade_the_field', []):
                fade_player = fade_data.get('player')
                pivot_player = fade_data.get('pivot_to')
                
                if fade_player and fade_player in available_players:
                    # Only fade if ownership is actually high
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
            
            # Extract boom scenarios
            boom_scenarios = []
            
            for scenario in data.get('boom_scenarios', []):
                players = scenario.get('players', [])
                valid_players = [p for p in players if p in available_players]
                
                if valid_players:
                    boom_scenarios.append({
                        'players': valid_players,
                        'scenario': scenario.get('scenario', ''),
                        'probability': scenario.get('probability', 'low')
                    })
            
            # Build enforcement rules
            enforcement_rules = []
            
            # Enforce contrarian captain
            if contrarian_captains:
                enforcement_rules.append({
                    'type': 'hard',
                    'constraint': 'contrarian_captain',
                    'players': contrarian_captains[:5],
                    'description': 'Must use contrarian captain for tournament upside'
                })
            
            # Enforce tournament winner core
            for i, player in enumerate(must_play[:3]):
                enforcement_rules.append({
                    'type': 'hard' if i == 0 else 'soft',
                    'constraint': f'tournament_core_{player}',
                    'player': player,
                    'weight': 0.8 if i > 0 else 1.0,
                    'description': f'Tournament core: {player}'
                })
            
            # Enforce fades
            for fade in fades[:2]:
                enforcement_rules.append({
                    'type': 'soft',
                    'constraint': f'fade_{fade}',
                    'player': fade,
                    'exclude': True,
                    'weight': 0.7,
                    'description': f"Fade chalk: {fade}"
                })
            
            # Enforce hidden correlations
            for stack in hidden_stacks[:2]:
                enforcement_rules.append({
                    'type': 'soft',
                    'constraint': 'hidden_correlation',
                    'players': [stack['player1'], stack['player2']],
                    'weight': 0.6,
                    'description': stack['narrative']
                })
            
            # Extract insights
            insights = []
            
            primary_narrative = data.get('primary_narrative', '')
            if primary_narrative:
                insights.append(primary_narrative)
            
            game_theory = data.get('contrarian_game_theory', {})
            if game_theory.get('exploit_this'):
                insights.append(f"Exploit: {game_theory['exploit_this']}")
            
            # Add boom scenario insights
            for scenario in boom_scenarios[:1]:
                insights.append(f"Boom: {scenario['scenario']}")
            
            confidence = data.get('confidence', 0.7)
            confidence = max(0.0, min(1.0, confidence))
            
            return AIRecommendation(
                captain_targets=contrarian_captains[:7],
                must_play=must_play[:5],
                never_play=fades[:5],
                stacks=hidden_stacks[:5],
                key_insights=insights[:3],
                confidence=confidence,
                enforcement_rules=enforcement_rules,
                narrative=data.get('primary_narrative', 'Contrarian approach'),
                source_ai=AIStrategistType.CONTRARIAN_NARRATIVE,
                contrarian_angles=[
                    captain_narratives.get(c, {}).get('narrative', '')
                    for c in contrarian_captains[:3]
                ],
                boosts=list(pivots.values())[:3],
                fades=fades[:3]
            )
            
        except Exception as e:
            self.logger.log_exception(e, "parse_contrarian_response")
            return self._get_fallback_recommendation(df, field_size)

# ============================================================================
# ENHANCED CLAUDE API MANAGER
# ============================================================================

class ClaudeAPIManager:
    """Enhanced Claude API manager with robust error handling"""

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
            'by_ai': {
                AIStrategistType.GAME_THEORY: {'requests': 0, 'errors': 0, 'tokens': 0},
                AIStrategistType.CORRELATION: {'requests': 0, 'errors': 0, 'tokens': 0},
                AIStrategistType.CONTRARIAN_NARRATIVE: {'requests': 0, 'errors': 0, 'tokens': 0}
            }
        }
        
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()
        
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
            except ImportError:
                self.logger.log("Anthropic library not installed", "ERROR")
                self.client = None
                return
            
            # Test the connection with a minimal request
            self.logger.log("Claude API client initialized successfully", "INFO")
            
        except Exception as e:
            self.logger.log(f"Failed to initialize Claude API: {e}", "ERROR")
            self.client = None

    def get_ai_response(self, prompt: str, ai_type: Optional[AIStrategistType] = None) -> str:
        """Get response from Claude API with caching and error handling"""
        
        # Generate cache key
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        # Check cache
        with self._cache_lock:
            if prompt_hash in self.cache:
                self.stats['cache_hits'] += 1
                self.logger.log("Cache hit for AI response", "DEBUG")
                return self.cache[prompt_hash]
        
        # Update statistics
        self.stats['requests'] += 1
        if ai_type:
            self.stats['by_ai'][ai_type]['requests'] += 1
        
        try:
            if not self.client:
                raise Exception("API client not initialized")
            
            self.perf_monitor.start_timer("claude_api_call")
            
            # Make API call with appropriate model
            message = self.client.messages.create(
                model="claude-3-sonnet-20241022",  # Use most capable model
                max_tokens=2000,  # Increased for comprehensive responses
                temperature=0.7,
                system="""You are an expert DFS (Daily Fantasy Sports) optimizer specializing in NFL tournament strategy. 
                         You provide specific, actionable recommendations using exact player names and clear reasoning.
                         Always respond with valid JSON containing specific player recommendations.""",
                messages=[{"role": "user", "content": prompt}]
            )
            
            elapsed = self.perf_monitor.stop_timer("claude_api_call")
            
            # Extract response
            response = message.content[0].text if message.content else "{}"
            
            # Update statistics
            if ai_type:
                self.stats['by_ai'][ai_type]['tokens'] += len(response) // 4  # Approximate
            self.stats['total_tokens'] += len(response) // 4
            
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
                f"AI response received ({len(response)} chars, {elapsed:.2f}s)", 
                "DEBUG"
            )
            
            return response
            
        except Exception as e:
            self.stats['errors'] += 1
            if ai_type:
                self.stats['by_ai'][ai_type]['errors'] += 1
            
            self.logger.log(f"API error: {e}", "ERROR")
            
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
            test_prompt = "Respond with only: OK"
            response = self.get_ai_response(test_prompt)
            
            return response and len(response) > 0
            
        except Exception as e:
            self.logger.log(f"API validation failed: {e}", "ERROR")
            return False

# ============================================================================
# AI RESPONSE FALLBACK SYSTEM
# ============================================================================

class AIFallbackSystem:
    """Enhanced fallback system with strategy-specific logic"""

    @staticmethod
    def get_game_theory_fallback(df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Generate game theory rules without AI"""
        
        logger = get_logger()
        logger.log("Using game theory fallback", "INFO")
        
        # Calculate ownership tiers
        ownership_thresholds = {
            'small_field': 20,
            'medium_field': 15,
            'large_field': 12,
            'milly_maker': 10,
            'super_contrarian': 7
        }
        
        threshold = ownership_thresholds.get(field_size, 15)
        
        # Find leverage captains
        low_owned = df[df.get('Ownership', 10) < threshold]
        
        if len(low_owned) < 3:
            low_owned = df.copy()
        
        captains = low_owned.nlargest(5, 'Projected_Points')['Player'].tolist()
        
        # Find chalk to fade
        chalk = df[df.get('Ownership', 10) > 35]['Player'].tolist()[:3]
        
        enforcement_rules = [
            {
                'type': 'hard',
                'constraint': 'leverage_captain',
                'players': captains[:3],
                'description': 'Statistical leverage captains'
            },
            {
                'type': 'soft',
                'constraint': 'ownership_target',
                'min': 60,
                'max': 90,
                'weight': 0.7,
                'description': 'Target ownership range'
            }
        ]
        
        return AIRecommendation(
            captain_targets=captains,
            must_play=[],
            never_play=chalk,
            stacks=[],
            key_insights=['Statistical game theory optimization'],
            confidence=0.5,
            enforcement_rules=enforcement_rules,
            narrative='Fallback game theory strategy',
            source_ai=AIStrategistType.GAME_THEORY,
            ownership_leverage={'threshold': threshold}
        )

    @staticmethod
    def get_correlation_fallback(df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Generate correlation rules without AI"""
        
        logger = get_logger()
        logger.log("Using correlation fallback", "INFO")
        
        stacks = []
        teams = df['Team'].unique()
        
        # Create QB stacks
        for team in teams[:2]:
            team_df = df[df['Team'] == team]
            
            qbs = team_df[team_df['Position'] == 'QB']['Player'].tolist()
            pass_catchers = team_df[
                team_df['Position'].isin(['WR', 'TE'])
            ].nlargest(3, 'Projected_Points')['Player'].tolist()
            
            for qb in qbs:
                for pc in pass_catchers[:2]:
                    stacks.append({
                        'player1': qb,
                        'player2': pc,
                        'type': 'QB_stack',
                        'correlation': 0.6
                    })
        
        # Captain targets are QBs and top pass catchers
        captains = (
            df[df['Position'] == 'QB']['Player'].tolist()[:2] +
            df[df['Position'].isin(['WR', 'TE'])].nlargest(3, 'Projected_Points')['Player'].tolist()
        )
        
        enforcement_rules = []
        
        if stacks:
            enforcement_rules.append({
                'type': 'soft',
                'constraint': 'correlation_stack',
                'players': [stacks[0]['player1'], stacks[0]['player2']],
                'weight': 0.7,
                'description': 'Primary correlation stack'
            })
        
        return AIRecommendation(
            captain_targets=captains[:5],
            must_play=[],
            never_play=[],
            stacks=stacks[:5],
            key_insights=['Statistical correlation analysis'],
            confidence=0.5,
            enforcement_rules=enforcement_rules,
            narrative='Fallback correlation strategy',
            source_ai=AIStrategistType.CORRELATION,
            correlation_matrix={}
        )

    @staticmethod
    def get_contrarian_fallback(df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Generate contrarian rules without AI"""
        
        logger = get_logger()
        logger.log("Using contrarian fallback", "INFO")
        
        # Calculate contrarian score
        df_copy = df.copy()
        df_copy['Contrarian_Score'] = (
            df_copy['Projected_Points'] / df_copy['Projected_Points'].max()
        ) / (df_copy.get('Ownership', 10) / 100 + 0.1)
        
        # Get contrarian plays
        contrarian = df_copy.nlargest(7, 'Contrarian_Score')
        captains = contrarian['Player'].tolist()
        
        # Low-owned must plays
        must_play = df_copy[df_copy.get('Ownership', 10) < 5]['Player'].tolist()[:3]
        
        # High-owned fades
        fades = df_copy[df_copy.get('Ownership', 10) > 30]['Player'].tolist()[:3]
        
        enforcement_rules = [
            {
                'type': 'hard',
                'constraint': 'contrarian_captain',
                'players': captains[:3],
                'description': 'Contrarian captain requirement'
            }
        ]
        
        return AIRecommendation(
            captain_targets=captains,
            must_play=must_play,
            never_play=fades,
            stacks=[],
            key_insights=['Statistical contrarian analysis'],
            confidence=0.5,
            enforcement_rules=enforcement_rules,
            narrative='Fallback contrarian strategy',
            source_ai=AIStrategistType.CONTRARIAN_NARRATIVE,
            contrarian_angles=['Low ownership leverage']
        )
        # NFL GPP DUAL-AI OPTIMIZER - PART 4: MAIN OPTIMIZER & LINEUP GENERATION
# Version 6.3 - Enhanced AI-Chef Functionality with Robust Generation
# NOTE: This continues from Parts 1-3 - all imports already consolidated at top

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
        self.pivot_generator = GPPCaptainPivotGenerator()
        
        # Tracking
        self.ai_decisions_log = []
        self.optimization_log = []
        self.generated_lineups = []
        
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
        """Get strategies from all three AIs with parallel execution"""
        
        self.logger.log("Getting strategies from three AI strategists", "INFO")
        self.perf_monitor.start_timer("get_ai_strategies")
        
        recommendations = {}
        
        if use_api and self.api_manager and self.api_manager.client:
            # API mode - get recommendations in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(
                        self._get_ai_recommendation,
                        self.game_theory_ai,
                        "Game Theory AI",
                        use_api
                    ): AIStrategistType.GAME_THEORY,
                    executor.submit(
                        self._get_ai_recommendation,
                        self.correlation_ai,
                        "Correlation AI",
                        use_api
                    ): AIStrategistType.CORRELATION,
                    executor.submit(
                        self._get_ai_recommendation,
                        self.contrarian_ai,
                        "Contrarian AI",
                        use_api
                    ): AIStrategistType.CONTRARIAN_NARRATIVE
                }
                
                for future in as_completed(futures):
                    ai_type = futures[future]
                    try:
                        recommendation = future.result(timeout=30)
                        recommendations[ai_type] = recommendation
                    except Exception as e:
                        self.logger.log(f"{ai_type.value} failed: {e}", "ERROR")
                        recommendations[ai_type] = self._get_fallback_recommendation(ai_type)
        else:
            # Manual mode or no API
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

    def _get_ai_recommendation(self, strategist, display_name: str, use_api: bool) -> AIRecommendation:
        """Get recommendation from a single AI strategist"""
        with st.spinner(f"{display_name} analyzing..."):
            try:
                return strategist.get_recommendation(
                    self.df, self.game_info, self.field_size, use_api=use_api
                )
            except Exception as e:
                self.logger.log(f"{display_name} error: {e}", "ERROR")
                return strategist._get_fallback_recommendation(self.df, self.field_size)

    def _get_manual_ai_strategies(self) -> Dict[AIStrategistType, AIRecommendation]:
        """Get AI strategies through manual input"""
        recommendations = {}
        
        st.subheader("Triple AI Strategy Input")
        
        tab1, tab2, tab3 = st.tabs(["Game Theory", "Correlation", "Contrarian"])
        
        with tab1:
            gt_response = self._get_manual_ai_input("Game Theory", self.game_theory_ai)
            try:
                recommendations[AIStrategistType.GAME_THEORY] = self.game_theory_ai.parse_response(
                    gt_response, self.df, self.field_size
                )
            except:
                recommendations[AIStrategistType.GAME_THEORY] = self._get_fallback_recommendation(
                    AIStrategistType.GAME_THEORY
                )
        
        with tab2:
            corr_response = self._get_manual_ai_input("Correlation", self.correlation_ai)
            try:
                recommendations[AIStrategistType.CORRELATION] = self.correlation_ai.parse_response(
                    corr_response, self.df, self.field_size
                )
            except:
                recommendations[AIStrategistType.CORRELATION] = self._get_fallback_recommendation(
                    AIStrategistType.CORRELATION
                )
        
        with tab3:
            contra_response = self._get_manual_ai_input("Contrarian", self.contrarian_ai)
            try:
                recommendations[AIStrategistType.CONTRARIAN_NARRATIVE] = self.contrarian_ai.parse_response(
                    contra_response, self.df, self.field_size
                )
            except:
                recommendations[AIStrategistType.CONTRARIAN_NARRATIVE] = self._get_fallback_recommendation(
                    AIStrategistType.CONTRARIAN_NARRATIVE
                )
        
        return recommendations

    def _get_manual_ai_input(self, ai_name: str, strategist) -> str:
        """Get manual AI input with validation"""
        with st.expander(f"View {ai_name} Prompt"):
            prompt = strategist.generate_prompt(self.df, self.game_info, self.field_size)
            st.text_area(
                f"Copy this prompt:",
                value=prompt,
                height=250,
                key=f"{ai_name}_prompt_display"
            )
        
        response = st.text_area(
            f"Paste {ai_name} Response (JSON):",
            height=200,
            key=f"{ai_name}_response",
            value='{}'
        )
        
        if response and response != '{}':
            try:
                json.loads(response)
                st.success(f"Valid {ai_name} JSON")
            except:
                st.error(f"Invalid {ai_name} JSON - will use fallback")
                response = '{}'
        
        return response

    def _get_fallback_recommendation(self, ai_type: AIStrategistType) -> AIRecommendation:
        """Get fallback recommendation when AI fails"""
        if ai_type == AIStrategistType.GAME_THEORY:
            return AIFallbackSystem.get_game_theory_fallback(self.df, self.field_size)
        elif ai_type == AIStrategistType.CORRELATION:
            return AIFallbackSystem.get_correlation_fallback(self.df, self.field_size)
        else:
            return AIFallbackSystem.get_contrarian_fallback(self.df, self.field_size)

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
                st.warning("Some AI requirements cannot be satisfied:")
                for error in validation['errors'][:5]:
                    st.write(f"  - {error}")
                
                # Adjust rules if needed
                if validation.get('adjustments'):
                    st.info("Applying adjustments to make requirements feasible")
            
            return {
                'synthesis': synthesis,
                'enforcement_rules': enforcement_rules,
                'validation': validation,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.log_exception(e, "synthesize_ai_strategies")
            # Return minimal synthesis
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
        """Generate lineups following AI directives with parallel processing"""
        
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
        
        # Show AI consensus
        self._display_ai_consensus(synthesis)

        # Pre-generation validation
        st.info(f"Attempting to generate {num_lineups} lineups with {self.field_size} settings...")

        # Quick feasibility check
        min_salary_lineup = self.df.nsmallest(6, 'Salary')['Salary'].sum()
        max_salary_lineup = self.df.nlargest(6, 'Salary')['Salary'].sum()

        if min_salary_lineup > 50000:
            st.error("⚠️ Cannot create valid lineup - even cheapest 6 players exceed salary cap!")
            st.stop()
    
        if max_salary_lineup < 30000:
            st.warning("⚠️ Player pool has very low salaries - may limit lineup diversity")
        
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
        generation_errors = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate lineups by strategy
        lineup_tasks = []
        for strategy, count in strategy_distribution.items():
            strategy_name = strategy if isinstance(strategy, str) else strategy.value
            for i in range(count):
                lineup_tasks.append((len(lineup_tasks) + 1, strategy_name))
        
        # Parallel or sequential generation based on configuration
        if self.max_workers > 1 and len(lineup_tasks) > 10:
            all_lineups = self._generate_lineups_parallel(
                lineup_tasks, players, salaries, ai_adjusted_points,
                ownership, positions, teams, enforcement_rules,
                synthesis, used_captains, progress_bar, status_text
            )
        else:
            all_lineups = self._generate_lineups_sequential(
                lineup_tasks, players, salaries, ai_adjusted_points,
                ownership, positions, teams, enforcement_rules,
                synthesis, used_captains, progress_bar, status_text
            )
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Calculate metrics
        total_time = time.time() - start_time
        success_rate = len(all_lineups) / max(num_lineups, 1)
        
        self.logger.log_optimization_end(len(all_lineups), total_time, success_rate)
        
        # Handle results
        if len(all_lineups) == 0:
            st.error("No valid lineups generated with AI constraints")
            self._display_optimization_issues()
            return pd.DataFrame()
        
        if len(all_lineups) < num_lineups:
            st.warning(f"Generated {len(all_lineups)}/{num_lineups} AI-compliant lineups")
            
            # Add detailed diagnostics
            with st.expander("Why couldn't all lineups be generated?", expanded=True):
                st.write("**Possible issues:**")
        
                # Check salary cap issues
                avg_salary_used = sum([lineup['Salary'] for lineup in all_lineups]) / max(len(all_lineups), 1)
                if avg_salary_used > 48000:
                    st.write("• Salary cap is very tight - most lineups using $48k+")
                # Check captain diversity
                if len(all_lineups) > 0:
                    unique_captains = len(set([lineup['Captain'] for lineup in all_lineups]))
                    st.write(f"• Only {unique_captains} unique captains available")
                    # Check AI constraints
                if enforcement_rules.get('hard_constraints'):
                    st.write(f"• {len(enforcement_rules['hard_constraints'])} hard AI constraints active")
                    # Show which constraints are causing issues
                    if hasattr(self, 'enforcement_engine'):
                        summary = self.enforcement_engine.get_enforcement_summary()
                        if summary['common_violations']:
                            st.write("**Most common constraint violations:**")
                            for violation, count in summary['common_violations'][:3]:
                                st.write(f"  - {violation}: {count} times")
        else:
            st.success(f"Generated {len(all_lineups)} AI-driven lineups in {total_time:.1f}s!")
        
        # Display AI enforcement statistics
        self.logger.display_ai_enforcement()
        
        # Store generated lineups
        self.generated_lineups = all_lineups
        
        return pd.DataFrame(all_lineups)

    def _generate_lineups_sequential(self, lineup_tasks, players, salaries, points,
                                    ownership, positions, teams, enforcement_rules,
                                    synthesis, used_captains, progress_bar, status_text):
        """Generate lineups sequentially"""
        all_lineups = []
        
        for i, (lineup_num, strategy_name) in enumerate(lineup_tasks):
            # Update progress
            progress = (i + 1) / len(lineup_tasks)
            progress_bar.progress(progress)
            status_text.text(f"Generating lineup {lineup_num} ({strategy_name}) - Success: {len(all_lineups)}/{lineup_num-1}")
            
            # Build lineup
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
                # Validate against AI requirements
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
                    self.logger.log_lineup_generation(
                        strategy_name, lineup_num, "FAILED",
                        violations=violations[:3]
                    )
            else:
                self.logger.log(f"Failed to generate lineup {lineup_num}", "WARNING")
        
        return all_lineups

    def _generate_lineups_parallel(self, lineup_tasks, players, salaries, points,
                                  ownership, positions, teams, enforcement_rules,
                                  synthesis, used_captains, progress_bar, status_text):
        """Generate lineups in parallel"""
        all_lineups = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for lineup_num, strategy_name in lineup_tasks:
                future = executor.submit(
                    self._build_ai_enforced_lineup,
                    lineup_num, strategy_name, players, salaries,
                    points, ownership, positions, teams,
                    enforcement_rules, synthesis, set()  # Empty set for parallel
                )
                futures[future] = (lineup_num, strategy_name)
            
            for future in as_completed(futures):
                lineup_num, strategy_name = futures[future]
                completed += 1
                
                # Update progress
                progress = completed / len(lineup_tasks)
                progress_bar.progress(progress)
                status_text.text(f"Processing lineup {completed}/{len(lineup_tasks)}...")
                
                try:
                    lineup = future.result(timeout=10)
                    
                    if lineup:
                        # Check captain uniqueness
                        if lineup['Captain'] not in used_captains:
                            # Validate
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
                                self.logger.log_lineup_generation(
                                    strategy_name, lineup_num, "FAILED",
                                    violations=violations[:3]
                                )
                
                except Exception as e:
                    self.logger.log(f"Lineup {lineup_num} generation error: {e}", "ERROR")
        
        return all_lineups

    def _build_ai_enforced_lineup(self, lineup_num: int, strategy: str, players: List[str],
                             salaries: Dict, points: Dict, ownership: Dict,
                             positions: Dict, teams: Dict, enforcement_rules: Dict,
                             synthesis: Dict, used_captains: Set[str]) -> Optional[Dict]:
    """Build a single lineup enforcing AI rules with DK Showdown requirements"""
    
    max_attempts = 3
    constraint_relaxation = [1.0, 0.8, 0.6]
    
    for attempt in range(max_attempts):
        try:
            model = pulp.LpProblem(f"AI_Lineup_{lineup_num}_{strategy}_attempt{attempt}", pulp.LpMaximize)
            
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
            
            # Basic constraints
            model += pulp.lpSum(captain.values()) == 1
            model += pulp.lpSum(flex.values()) == 5
            
            # Player can't be both captain and flex
            for p in players:
                model += flex[p] + captain[p] <= 1
            
            # Salary constraint
            salary_cap = OptimizerConfig.SALARY_CAP
            if attempt > 0:
                salary_cap += 500 * attempt  # Allow slight overage on retries
                
            model += pulp.lpSum([
                salaries[p] * flex[p] + 1.5 * salaries[p] * captain[p]
                for p in players
            ]) <= salary_cap
            
            # CRITICAL DK SHOWDOWN CONSTRAINTS
            unique_teams = list(set(teams.values()))
            
            # 1. Must have at least 1 player from each team
            for team in unique_teams:
                team_players = [p for p in players if teams.get(p) == team]
                if team_players:
                    model += pulp.lpSum([
                        flex[p] + captain[p] for p in team_players
                    ]) >= 1
            
            # 2. Max players from one team (usually 5, but we'll use config)
            max_from_team = OptimizerConfig.MAX_PLAYERS_PER_TEAM
            if attempt > 1:
                max_from_team = 5  # DK allows up to 5 from one team
                
            for team in unique_teams:
                team_players = [p for p in players if teams.get(p) == team]
                if team_players:
                    model += pulp.lpSum([
                        flex[p] + captain[p] for p in team_players
                    ]) <= max_from_team
            
            # Apply AI constraints with relaxation
            relaxation_factor = constraint_relaxation[attempt]
            
            # Captain constraints based on attempt
            if attempt == 0:
                self._apply_strict_captain_constraints(model, captain, enforcement_rules, 
                                                      players, used_captains, synthesis, strategy)
            elif attempt == 1:
                self._apply_relaxed_captain_constraints(model, captain, enforcement_rules,
                                                       players, used_captains, strategy)
            else:
                self._apply_minimal_captain_constraints(model, captain, players, 
                                                       used_captains, ownership)
            
            # Apply other hard constraints with relaxation
            if relaxation_factor >= 0.8:
                self._apply_hard_constraints_with_validation(model, flex, captain, 
                                                            enforcement_rules, players, teams)
            
            # Apply soft constraints
            soft_penalty = self._calculate_soft_penalties(
                flex, captain, enforcement_rules, players, weight_multiplier=relaxation_factor
            )
            
            if soft_penalty:
                model += objective - soft_penalty
            
            # Solve
            timeout = 5 + (attempt * 5)
            model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout))
            
            if pulp.LpStatus[model.status] == 'Optimal':
                lineup = self._extract_lineup_from_solution(
                    flex, captain, players, salaries, points, ownership,
                    lineup_num, strategy, synthesis
                )
                
                if lineup:
                    # Verify DK requirements are met
                    if self._verify_dk_requirements(lineup, teams):
                        if attempt > 0:
                            self.logger.log(f"Lineup {lineup_num} generated on attempt {attempt + 1}", "DEBUG")
                        return lineup
                    else:
                        self.logger.log(f"Lineup {lineup_num} failed DK verification", "DEBUG")
            else:
                self.logger.log(f"No solution found for lineup {lineup_num} attempt {attempt + 1}", "DEBUG")
                
        except Exception as e:
            self.logger.log(f"Error in lineup {lineup_num} attempt {attempt + 1}: {str(e)}", "DEBUG")
            continue
    
    return None

def _verify_dk_requirements(self, lineup: Dict, teams: Dict) -> bool:
    """Verify lineup meets DraftKings Showdown requirements"""
    
    captain = lineup.get('Captain')
    flex_players = lineup.get('FLEX', [])
    
    if not captain or len(flex_players) != 5:
        return False
    
    all_players = [captain] + flex_players
    
    # Check team representation
    team_counts = {}
    for player in all_players:
        team = teams.get(player)
        if team:
            team_counts[team] = team_counts.get(team, 0) + 1
    
    # Must have at least 1 from each team
    unique_teams = set(teams.values())
    if len(team_counts) < len(unique_teams):
        self.logger.log(f"Missing team representation: {team_counts}", "DEBUG")
        return False
    
    # Check max from one team (5 for DK)
    for team, count in team_counts.items():
        if count > 5:
            self.logger.log(f"Too many from {team}: {count}", "DEBUG")
            return False
    
    return True

def _apply_hard_constraints_with_validation(self, model, flex, captain, 
                                           enforcement_rules, players, teams):
    """Apply hard constraints while ensuring DK requirements can be met"""
    
    # Get unique teams
    unique_teams = list(set(teams.values()))
    
    for constraint in enforcement_rules.get('hard_constraints', []):
        rule = constraint.get('rule')
        
        if rule == 'must_include':
            player = constraint.get('player')
            if player and player in players:
                # Check if forcing this player still allows team diversity
                player_team = teams.get(player)
                same_team_players = [p for p in players if teams.get(p) == player_team]
                
                # Only enforce if we have enough roster spots for other team
                if len(same_team_players) <= 5:
                    model += flex[player] + captain[player] >= 1
                    
        elif rule == 'must_exclude':
            player = constraint.get('player')
            if player and player in players:
                # Check if excluding this player still allows minimum team requirements
                player_team = teams.get(player)
                same_team_players = [p for p in players if teams.get(p) == player_team and p != player]
                
                # Only exclude if team still has enough players
                if len(same_team_players) >= 1:
                    model += flex[player] + captain[player] == 0
                    
        elif rule == 'must_stack':
            stack_players = constraint.get('players', [])
            valid_stack = [p for p in stack_players if p in players]
            
            # Check stack doesn't violate team limits
            stack_teams = [teams.get(p) for p in valid_stack]
            if len(set(stack_teams)) > 0:  # Has valid team data
                team_counts = {}
                for t in stack_teams:
                    team_counts[t] = team_counts.get(t, 0) + 1
                
                # Only enforce if doesn't exceed team limits
                if all(count <= 4 for count in team_counts.values()):
                    model += pulp.lpSum([
                        flex[p] + captain[p] for p in valid_stack
                    ]) >= len(valid_stack)

                    def _apply_strict_captain_constraints(self, model, captain, enforcement_rules, 
                                      players, used_captains, synthesis, strategy):
    """Apply strict captain constraints for first attempt"""
    
    # Get valid captain candidates based on strategy
    valid_captains = []
    
    # First check AI enforcement rules
    for constraint in enforcement_rules.get('hard_constraints', []):
        if constraint.get('rule') in ['captain_selection', 'captain_from_list']:
            rule_captains = [p for p in constraint.get('players', []) if p in players]
            valid_captains.extend(rule_captains)
    
    # Then check synthesis captain strategy
    if not valid_captains and synthesis:
        captain_strategy = synthesis.get('captain_strategy', {})
        if strategy in ['ai_consensus', 'AI_CONSENSUS']:
            valid_captains = [c for c, level in captain_strategy.items() 
                            if level == 'consensus' and c in players]
        elif strategy in ['ai_majority', 'AI_MAJORITY']:
            valid_captains = [c for c, level in captain_strategy.items() 
                            if level in ['consensus', 'majority'] and c in players]
    
    # Remove already used captains
    valid_captains = [c for c in valid_captains if c not in used_captains]
    
    # Apply constraint if we have valid captains
    if valid_captains:
        model += pulp.lpSum([captain[c] for c in valid_captains]) == 1

def _apply_relaxed_captain_constraints(self, model, captain, enforcement_rules,
                                      players, used_captains, strategy):
    """Apply relaxed captain constraints for second attempt"""
    
    # Expand captain pool to include top projected players
    top_players = self.df.nlargest(10, 'Projected_Points')['Player'].tolist()
    valid_captains = [p for p in top_players if p in players and p not in used_captains]
    
    if valid_captains:
        model += pulp.lpSum([captain[c] for c in valid_captains]) == 1

def _apply_minimal_captain_constraints(self, model, captain, players, 
                                      used_captains, ownership):
    """Apply minimal captain constraints for final attempt"""
    
    # Just avoid used captains and super high ownership
    available_captains = []
    for p in players:
        if p not in used_captains:
            player_own = ownership.get(p, 10)
            if player_own < 50:  # Avoid only extreme chalk
                available_captains.append(p)
    
    if available_captains:
        model += pulp.lpSum([captain[c] for c in available_captains]) == 1

def _verify_dk_requirements(self, lineup: Dict, teams: Dict) -> bool:
    """Verify lineup meets DraftKings Showdown requirements"""
    
    captain = lineup.get('Captain')
    flex_players = lineup.get('FLEX', [])
    
    if not captain or len(flex_players) != 5:
        return False
    
    all_players = [captain] + flex_players
    
    # Check team representation
    team_counts = {}
    for player in all_players:
        team = teams.get(player)
        if team:
            team_counts[team] = team_counts.get(team, 0) + 1
    
    # Must have at least 1 from each team
    unique_teams = set(teams.values())
    if len(team_counts) < len(unique_teams):
        self.logger.log(f"Missing team representation: {team_counts}", "DEBUG")
        return False
    
    # Check max from one team (5 for DK)
    for team, count in team_counts.items():
        if count > 5:
            self.logger.log(f"Too many from {team}: {count}", "DEBUG")
            return False
    
    return True

def _apply_hard_constraints_with_validation(self, model, flex, captain, 
                                           enforcement_rules, players, teams):
    """Apply hard constraints while ensuring DK requirements can be met"""
    
    # Get unique teams
    unique_teams = list(set(teams.values()))
    
    for constraint in enforcement_rules.get('hard_constraints', []):
        rule = constraint.get('rule')
        
        if rule == 'must_include':
            player = constraint.get('player')
            if player and player in players:
                # Check if forcing this player still allows team diversity
                player_team = teams.get(player)
                same_team_players = [p for p in players if teams.get(p) == player_team]
                
                # Only enforce if we have enough roster spots for other team
                if len(same_team_players) <= 5:
                    model += flex[player] + captain[player] >= 1
                    
        elif rule == 'must_exclude':
            player = constraint.get('player')
            if player and player in players:
                # Check if excluding this player still allows minimum team requirements
                player_team = teams.get(player)
                same_team_players = [p for p in players if teams.get(p) == player_team and p != player]
                
                # Only exclude if team still has enough players
                if len(same_team_players) >= 1:
                    model += flex[player] + captain[player] == 0
                    
        elif rule == 'must_stack':
            stack_players = constraint.get('players', [])
            valid_stack = [p for p in stack_players if p in players]
            
            # Check stack doesn't violate team limits
            stack_teams = [teams.get(p) for p in valid_stack]
            if len(set(stack_teams)) > 0:  # Has valid team data
                team_counts = {}
                for t in stack_teams:
                    team_counts[t] = team_counts.get(t, 0) + 1
                
                # Only enforce if doesn't exceed team limits
                if all(count <= 4 for count in team_counts.values()):
                    model += pulp.lpSum([
                        flex[p] + captain[p] for p in valid_stack
                    ]) >= len(valid_stack)

    def _apply_hard_constraints(self, model, flex, captain, enforcement_rules, players):
        """Apply hard AI constraints to the model"""
        
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
                    
            elif rule == 'captain_from_list' or rule == 'captain_selection':
                valid_captains = [
                    p for p in constraint.get('players', []) 
                    if p in players
                ]
                if valid_captains:
                    model += pulp.lpSum([captain[p] for p in valid_captains]) == 1
                    
            elif rule == 'must_stack':
                stack_players = constraint.get('players', [])
                valid_stack = [p for p in stack_players if p in players]
                if len(valid_stack) == 2:
                    # Both players must be in lineup (as captain or flex)
                    model += pulp.lpSum([
                        flex[p] + captain[p] for p in valid_stack
                    ]) >= 2

    def _calculate_soft_penalties(self, flex, captain, enforcement_rules, players, 
                             weight_multiplier=1.0):
    """Calculate penalties for soft constraints with adjustable weight"""
    
    penalties = []
    
    for constraint in enforcement_rules.get('soft_constraints', []):
        base_weight = constraint.get('weight', 0.5)
        adjusted_weight = base_weight * weight_multiplier
        rule = constraint.get('rule')
        
        if rule == 'should_include':
            player = constraint.get('player')
            if player and player in players:
                penalties.append(
                    adjusted_weight * 10 * (1 - flex[player] - captain[player])
                )
                
        elif rule == 'should_exclude':
            player = constraint.get('player')
            if player and player in players:
                penalties.append(
                    adjusted_weight * 10 * (flex[player] + captain[player])
                )
    
    return pulp.lpSum(penalties) if penalties else None

    def _apply_strategy_constraints(self, model, flex, captain, strategy, synthesis,
                                   players, ownership, used_captains):
        """Apply strategy-specific constraints"""
        
        # AI Consensus strategy
        if strategy in ['ai_consensus', 'AI_CONSENSUS']:
            consensus_captains = [
                c for c, level in synthesis.get('captain_strategy', {}).items()
                if level == 'consensus' and c in players
            ]
            
            available_consensus = [
                c for c in consensus_captains 
                if c not in used_captains
            ]
            
            if available_consensus:
                model += pulp.lpSum([captain[c] for c in available_consensus]) == 1
        
        # AI Contrarian strategy
        elif strategy in ['ai_contrarian', 'AI_CONTRARIAN']:
            # Prefer low ownership captains
            low_owned = [
                p for p in players 
                if ownership.get(p, 10) < 10 and p not in used_captains
            ]
            
            if low_owned:
                model += pulp.lpSum([captain[p] for p in low_owned]) >= 0.5
        
        # Ownership constraints based on strategy
        if strategy in ['ai_game_theory', 'AI_GAME_THEORY']:
            # Add ownership sum constraint
            total_own = pulp.lpSum([
                ownership.get(p, 10) * (flex[p] + 1.5 * captain[p])
                for p in players
            ])
            
            # Target ownership based on field size
            targets = OptimizerConfig.GPP_OWNERSHIP_TARGETS.get(self.field_size, (60, 90))
            
            # Soft constraint via objective (can't use hard constraint on non-linear)
            # This is approximated in the objective function

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
            
            # Determine AI sources for this captain
            ai_sources = []
            captain_strategy = synthesis.get('captain_strategy', {})
            if captain_pick in captain_strategy:
                ai_sources.append(captain_strategy[captain_pick])
            
            # Calculate leverage score
            leverage_score = self.bucket_manager.calculate_ai_leverage_score(
                [captain_pick] + flex_picks, self.df, {}
            )
            
            # Determine ownership tier
            if total_ownership < 60:
                ownership_tier = 'Elite'
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
                'AI_Sources': ai_sources,
                'AI_Enforced': True,
                'Confidence': synthesis.get('confidence', 0.5),
                'GPP_Summary': self.bucket_manager.get_gpp_summary(
                    [captain_pick] + flex_picks, self.df, self.field_size, True
                ),
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
                # Normalize score to multiplier (0.8 to 1.3 range)
                if score > 0:
                    multiplier = 1.0 + min(score * 0.15, 0.3)
                else:
                    multiplier = max(0.8, 1.0 + score * 0.2)
                
                adjusted[player] *= multiplier
        
        # Apply avoidance rules
        for player in synthesis.get('avoidance_rules', []):
            if player in adjusted:
                adjusted[player] *= 0.7  # Reduce by 30%
        
        return adjusted

    def _determine_consensus_level(self, synthesis: Dict) -> str:
        """Determine the level of AI consensus"""
        captain_strategy = synthesis.get('captain_strategy', {})
        
        if not captain_strategy:
            return 'low'
        
        consensus_count = len([c for c, level in captain_strategy.items() if level == 'consensus'])
        majority_count = len([c for c, level in captain_strategy.items() if level == 'majority'])
        
        total = len(captain_strategy)
        
        if total == 0:
            return 'low'
        
        consensus_ratio = consensus_count / total
        majority_ratio = (consensus_count + majority_count) / total
        
        if consensus_ratio > 0.3:
            return 'high'
        elif majority_ratio > 0.5:
            return 'mixed'
        else:
            return 'low'

    def _display_ai_consensus(self, synthesis: Dict):
        """Display AI consensus analysis"""
        try:
            st.markdown("### AI Consensus Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                consensus_captains = len([
                    c for c, l in synthesis.get('captain_strategy', {}).items()
                    if l == 'consensus'
                ])
                st.metric("Consensus Captains", consensus_captains)
            
            with col2:
                strong_stacks = len([
                    s for s in synthesis.get('stacking_rules', [])
                    if s.get('strength') == 'strong'
                ])
                st.metric("Strong Stacks", strong_stacks)
            
            with col3:
                st.metric("AI Confidence", f"{synthesis.get('confidence', 0):.0%}")
            
            with col4:
                enforcement = len(synthesis.get('enforcement_rules', []))
                st.metric("Enforcement Rules", enforcement)
            
            # Show narrative if present
            if synthesis.get('narrative'):
                with st.expander("AI Narrative"):
                    st.write(synthesis['narrative'])
                    
        except Exception as e:
            self.logger.log_exception(e, "_display_ai_consensus")

    def _display_optimization_issues(self):
        """Display optimization issues for debugging"""
        if self.optimization_log:
            with st.expander("Optimization Issues", expanded=True):
                for issue in self.optimization_log[-10:]:
                    st.write(f"- {issue}")
        
        # Show enforcement summary
        enforcement_summary = self.enforcement_engine.get_enforcement_summary()
        if enforcement_summary['common_violations']:
            st.markdown("#### Common Violations:")
            for violation, count in enforcement_summary['common_violations']:
                st.write(f"- {violation}: {count} times")

# ============================================================================
# CAPTAIN PIVOT GENERATOR (AI-Enhanced)
# ============================================================================

class GPPCaptainPivotGenerator:
    """Generate captain pivots with AI guidance"""

    def __init__(self):
        self.logger = get_logger()

    def generate_ai_guided_pivots(self, lineup: Dict, df: pd.DataFrame,
                                 synthesis: Dict, max_pivots: int = 5) -> List[Dict]:
        """Generate pivots following AI recommendations"""
        
        pivots = []
        
        try:
            current_captain = lineup.get('Captain')
            flex_players = lineup.get('FLEX', [])
            
            if not current_captain or not flex_players:
                return pivots
            
            # Get AI-recommended captains
            captain_strategy = synthesis.get('captain_strategy', {})
            ai_captains = list(captain_strategy.keys())
            
            # Prioritize AI captains not yet used
            for new_captain in ai_captains:
                if new_captain != current_captain and new_captain in df['Player'].values:
                    # Check if new captain is in current flex
                    if new_captain in flex_players:
                        # Swap captain with flex
                        new_flex = flex_players.copy()
                        new_flex.remove(new_captain)
                        new_flex.append(current_captain)
                        
                        # Calculate new metrics
                        salaries = df.set_index('Player')['Salary'].to_dict()
                        new_salary = (
                            sum(salaries.get(p, 0) for p in new_flex) +
                            1.5 * salaries.get(new_captain, 0)
                        )
                        
                        if new_salary <= OptimizerConfig.SALARY_CAP:
                            pivot = {
                                'Original_Captain': current_captain,
                                'Captain': new_captain,
                                'FLEX': new_flex,
                                'Salary': new_salary,
                                'Pivot_Type': f"AI-{captain_strategy.get(new_captain, 'suggested')}",
                                'AI_Recommended': True
                            }
                            
                            pivots.append(pivot)
                            
                            if len(pivots) >= max_pivots:
                                break
            
            # If not enough AI pivots, add statistical pivots
            if len(pivots) < max_pivots:
                pivots.extend(self._generate_statistical_pivots(
                    lineup, df, max_pivots - len(pivots)
                ))
                
        except Exception as e:
            self.logger.log_exception(e, "generate_ai_guided_pivots")
        
        return pivots[:max_pivots]

    def _generate_statistical_pivots(self, lineup: Dict, df: pd.DataFrame,
                                    max_pivots: int) -> List[Dict]:
        """Generate statistical pivots as fallback"""
        
        pivots = []
        current_captain = lineup.get('Captain')
        flex_players = lineup.get('FLEX', [])
        
        if not flex_players:
            return pivots
        
        # Calculate value scores for flex players
        flex_df = df[df['Player'].isin(flex_players)].copy()
        flex_df['PivotScore'] = (
            flex_df['Projected_Points'] * 1.5 / 
            (flex_df.get('Ownership', 10) + 5)
        )
        
        # Sort by pivot score
        flex_df = flex_df.sort_values('PivotScore', ascending=False)
        
        for _, row in flex_df.iterrows():
            if len(pivots) >= max_pivots:
                break
                
            new_captain = row['Player']
            new_flex = flex_players.copy()
            new_flex.remove(new_captain)
            new_flex.append(current_captain)
            
            pivots.append({
                'Original_Captain': current_captain,
                'Captain': new_captain,
                'FLEX': new_flex,
                'Pivot_Type': 'Statistical',
                'AI_Recommended': False
            })
        
        return pivots
        # NFL GPP DUAL-AI OPTIMIZER - PART 5: MAIN UI AND HELPER FUNCTIONS
# Version 6.3 - Enhanced UI with Robust Error Handling
# NOTE: This continues from Parts 1-4 - all imports already consolidated at top

# ============================================================================
# PERFORMANCE OPTIMIZATION - AI-FOCUSED CACHING
# ============================================================================

@st.cache_data(ttl=3600, max_entries=10)
def cache_ai_synthesis(gt_json: str, corr_json: str, contra_json: str,
                      field_size: str) -> Dict:
    """Cache AI synthesis results with memory limit"""
    try:
        # Parse JSON recommendations safely
        gt_rec = json.loads(gt_json) if gt_json else {}
        corr_rec = json.loads(corr_json) if corr_json else {}
        contra_rec = json.loads(contra_json) if contra_json else {}
        
        # Convert to AIRecommendation objects
        # This would need proper reconstruction logic
        
        # Synthesize
        synthesis_engine = AISynthesisEngine()
        # Simplified for caching - would need full reconstruction
        
        return {
            'captain_strategy': {},
            'player_rankings': {},
            'confidence': 0.5
        }
    except Exception as e:
        get_logger().log(f"Cache synthesis error: {e}", "ERROR")
        return {}

@st.cache_data(ttl=1800, max_entries=5)
def cache_ai_enforcement_rules(synthesis_json: str, df_json: str,
                              field_size: str) -> Dict:
    """Cache AI enforcement rule generation"""
    try:
        synthesis = json.loads(synthesis_json)
        df = pd.read_json(df_json)
        
        enforcement_engine = AIEnforcementEngine()
        # Simplified for caching
        
        validation = AIConfigValidator.validate_ai_requirements({}, df)
        
        return {
            'rules': {},
            'validation': validation
        }
    except Exception as e:
        get_logger().log(f"Cache enforcement error: {e}", "ERROR")
        return {'rules': {}, 'validation': {'is_valid': True}}

# ============================================================================
# SESSION STATE MANAGEMENT - AI TRACKING
# ============================================================================

def init_ai_session_state():
    """Initialize session state for AI-driven optimization"""
    if 'ai_recommendations' not in st.session_state:
        st.session_state['ai_recommendations'] = {}
    if 'ai_synthesis' not in st.session_state:
        st.session_state['ai_synthesis'] = None
    if 'ai_enforcement_history' not in st.session_state:
        st.session_state['ai_enforcement_history'] = []
    if 'optimization_history' not in st.session_state:
        st.session_state['optimization_history'] = []
    if 'ai_mode' not in st.session_state:
        st.session_state['ai_mode'] = 'enforced'
    if 'api_manager' not in st.session_state:
        st.session_state['api_manager'] = None
    if 'lineups_df' not in st.session_state:
        st.session_state['lineups_df'] = pd.DataFrame()
    if 'df' not in st.session_state:
        st.session_state['df'] = pd.DataFrame()

def save_ai_optimization_session(lineups_df: pd.DataFrame, ai_synthesis: Dict,
                                settings: Dict):
    """Save AI-driven optimization session with memory management"""
    try:
        session = {
            'timestamp': datetime.now().isoformat(),
            'lineups': lineups_df.to_dict('records') if not lineups_df.empty else [],
            'ai_synthesis': ai_synthesis,
            'settings': settings,
            'ai_consensus_level': determine_consensus_level(ai_synthesis),
            'enforcement_stats': get_logger().get_ai_summary()
        }
        
        if 'optimization_history' not in st.session_state:
            st.session_state['optimization_history'] = []
        
        st.session_state['optimization_history'].append(session)
        
        # Limit history to prevent memory issues
        max_history = 10
        if len(st.session_state['optimization_history']) > max_history:
            st.session_state['optimization_history'] = st.session_state['optimization_history'][-max_history:]
            
    except Exception as e:
        get_logger().log(f"Error saving session: {e}", "ERROR")

def determine_consensus_level(synthesis: Dict) -> str:
    """Determine AI consensus level"""
    if not synthesis:
        return 'none'
    
    captain_strategy = synthesis.get('captain_strategy', {})
    if not captain_strategy:
        return 'none'
    
    consensus = len([c for c, l in captain_strategy.items() if l == 'consensus'])
    total = len(captain_strategy)
    
    if total == 0:
        return 'none'
    
    ratio = consensus / total
    if ratio > 0.5:
        return 'high'
    elif ratio > 0.2:
        return 'medium'
    else:
        return 'low'

# ============================================================================
# DATA VALIDATION AND PROCESSING
# ============================================================================

def validate_and_process_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Validate and process uploaded DataFrame"""
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'fixes_applied': []
    }
    
    try:
        # Check for required columns
        required_mapping = {
            'first_name': 'First_Name',
            'last_name': 'Last_Name',
            'position': 'Position',
            'team': 'Team',
            'salary': 'Salary',
            'ppg_projection': 'Projected_Points'
        }
        
        # Rename columns to standard format
        df = df.rename(columns=required_mapping)
        
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
        
        # Ensure numeric columns are numeric
        numeric_columns = ['Salary', 'Projected_Points']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    validation['warnings'].append(f"Some {col} values could not be converted to numbers")
                    df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
        
        # Add ownership if missing
        if 'Ownership' not in df.columns:
            df['Ownership'] = df.apply(
                lambda row: OptimizerConfig.get_default_ownership(
                    row.get('Position', 'FLEX'),
                    row.get('Salary', 5000)
                ), axis=1
            )
            validation['fixes_applied'].append("Added default ownership projections")
        
        # Remove duplicates
        if df.duplicated(subset=['Player']).any():
            df = df.drop_duplicates(subset=['Player'], keep='first')
            validation['warnings'].append("Removed duplicate players")
        
        # Check for minimum requirements
        if len(df) < 6:
            validation['errors'].append(f"Only {len(df)} players available (minimum 6 required)")
            validation['is_valid'] = False
        
        # Validate team count
        teams = df['Team'].unique()
        if len(teams) != 2:
            validation['warnings'].append(f"Expected 2 teams, found {len(teams)}")
        
        # Validate salary cap feasibility
        min_lineup_salary = df.nsmallest(6, 'Salary')['Salary'].sum()
        if min_lineup_salary > OptimizerConfig.SALARY_CAP:
            validation['errors'].append("Cannot create valid lineup within salary cap")
            validation['is_valid'] = False
            
    except Exception as e:
        validation['errors'].append(f"Processing error: {str(e)}")
        validation['is_valid'] = False
    
    return df, validation

# ============================================================================
# AI VALIDATION AND DISPLAY
# ============================================================================

def display_ai_recommendations(recommendations: Dict[AIStrategistType, AIRecommendation]):
    """Display the three AI recommendations with enhanced visuals"""
    st.markdown("### AI Strategic Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Game Theory", "Correlation", "Contrarian"])
    
    with tab1:
        display_single_ai_recommendation(
            recommendations.get(AIStrategistType.GAME_THEORY),
            "Game Theory",
            "Target"
        )
    
    with tab2:
        display_single_ai_recommendation(
            recommendations.get(AIStrategistType.CORRELATION),
            "Correlation",
            "Link"
        )
    
    with tab3:
        display_single_ai_recommendation(
            recommendations.get(AIStrategistType.CONTRARIAN_NARRATIVE),
            "Contrarian",
            "Theater"
        )

def display_single_ai_recommendation(rec: AIRecommendation, name: str, icon: str):
    """Display a single AI's recommendation with error handling"""
    if not rec:
        st.warning(f"No {name} recommendation available")
        return
    
    try:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"#### {name} Strategy")
            
            # Confidence meter
            confidence_color = "green" if rec.confidence > 0.7 else "orange" if rec.confidence > 0.5 else "red"
            st.metric("Confidence", f"{rec.confidence:.0%}")
            st.progress(rec.confidence)
            
            if rec.narrative:
                st.info(f"**Narrative:** {rec.narrative[:200]}")
            
            if rec.captain_targets:
                st.markdown("**Captain Targets:**")
                for i, captain in enumerate(rec.captain_targets[:5], 1):
                    st.write(f"{i}. {captain}")
            
            if rec.must_play:
                st.markdown("**Must Play:**")
                st.write(", ".join(rec.must_play[:5]))
            
            if rec.never_play:
                st.markdown("**Fade:**")
                st.write(", ".join(rec.never_play[:5]))
        
        with col2:
            if rec.stacks:
                st.markdown("**Key Stacks:**")
                for stack in rec.stacks[:3]:
                    if isinstance(stack, dict):
                        p1 = stack.get('player1', '')
                        p2 = stack.get('player2', '')
                        if p1 and p2:
                            st.write(f"• {p1} + {p2}")
            
            if rec.enforcement_rules:
                hard_rules = len([r for r in rec.enforcement_rules if r.get('type') == 'hard'])
                soft_rules = len([r for r in rec.enforcement_rules if r.get('type') == 'soft'])
                st.markdown("**Enforcement:**")
                st.write(f"Hard: {hard_rules} | Soft: {soft_rules}")
            
            if hasattr(rec, 'contrarian_angles') and rec.contrarian_angles:
                st.markdown("**Contrarian Angles:**")
                for angle in rec.contrarian_angles[:2]:
                    if angle:
                        st.write(f"• {angle[:50]}")
                        
    except Exception as e:
        st.error(f"Error displaying {name} recommendation: {str(e)}")

def display_ai_synthesis(synthesis: Dict):
    """Display the synthesized AI strategy"""
    try:
        st.markdown("### AI Synthesis & Consensus")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            consensus_captains = len([
                c for c, l in synthesis.get('captain_strategy', {}).items()
                if l == 'consensus'
            ])
            st.metric("Consensus Captains", consensus_captains)
        
        with col2:
            majority_captains = len([
                c for c, l in synthesis.get('captain_strategy', {}).items()
                if l == 'majority'
            ])
            st.metric("Majority Captains", majority_captains)
        
        with col3:
            st.metric("Overall Confidence", f"{synthesis.get('confidence', 0):.0%}")
        
        with col4:
            st.metric("Enforcement Rules", len(synthesis.get('enforcement_rules', [])))
        
        # Show captain consensus details
        with st.expander("Captain Consensus Details"):
            captain_strategy = synthesis.get('captain_strategy', {})
            if captain_strategy:
                for captain, consensus_type in list(captain_strategy.items())[:10]:
                    icon = "✅" if consensus_type == 'consensus' else "🤝" if consensus_type == 'majority' else "💭"
                    st.write(f"{icon} **{captain}** - {consensus_type}")
            else:
                st.write("No captain consensus data available")
        
        # Show stacking consensus
        if synthesis.get('stacking_rules'):
            with st.expander("Stack Consensus"):
                for stack in synthesis['stacking_rules'][:5]:
                    strength = stack.get('strength', 'moderate')
                    icon = "💪" if strength == 'strong' else "👍"
                    st.write(f"{icon} {stack.get('player1', '')} + {stack.get('player2', '')} ({strength})")
                    
    except Exception as e:
        st.error(f"Error displaying synthesis: {str(e)}")

# ============================================================================
# LINEUP ANALYSIS - AI FOCUSED
# ============================================================================

def display_ai_lineup_analysis(lineups_df: pd.DataFrame, df: pd.DataFrame,
                              synthesis: Dict, field_size: str):
    """Display AI-driven lineup analysis with error handling"""
    if lineups_df.empty:
        st.warning("No lineups to analyze")
        return
    
    try:
        st.markdown("### AI-Driven Lineup Analysis")
        
        # Create safe visualizations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. AI Strategy Distribution
        ax1 = axes[0, 0]
        if 'AI_Strategy' in lineups_df.columns:
            strategy_counts = lineups_df['AI_Strategy'].value_counts()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']
            ax1.pie(strategy_counts.values, labels=strategy_counts.index,
                   autopct='%1.0f%%', colors=colors[:len(strategy_counts)])
            ax1.set_title('AI Strategy Distribution')
        else:
            ax1.text(0.5, 0.5, 'No Strategy Data', ha='center', va='center')
            ax1.set_title('AI Strategy Distribution')
        
        # 2. Captain Usage
        ax2 = axes[0, 1]
        if 'Captain' in lineups_df.columns:
            captain_usage = lineups_df['Captain'].value_counts().head(10)
            ax2.bar(range(len(captain_usage)), captain_usage.values, color='steelblue')
            ax2.set_xticks(range(len(captain_usage)))
            ax2.set_xticklabels(captain_usage.index, rotation=45, ha='right')
            ax2.set_title('Top 10 Captain Usage')
            ax2.set_ylabel('Times Used')
        else:
            ax2.text(0.5, 0.5, 'No Captain Data', ha='center', va='center')
        
        # 3. Ownership Distribution
        ax3 = axes[0, 2]
        if 'Total_Ownership' in lineups_df.columns:
            ax3.hist(lineups_df['Total_Ownership'], bins=20, alpha=0.7, color='green')
            ax3.axvline(lineups_df['Total_Ownership'].mean(), color='red',
                       linestyle='--', label=f"Mean: {lineups_df['Total_Ownership'].mean():.1f}%")
            ax3.set_xlabel('Total Ownership %')
            ax3.set_ylabel('Number of Lineups')
            ax3.set_title('Ownership Distribution')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No Ownership Data', ha='center', va='center')
        
        # 4. Salary Distribution
        ax4 = axes[1, 0]
        if 'Salary' in lineups_df.columns:
            ax4.hist(lineups_df['Salary'], bins=15, alpha=0.7, color='orange')
            ax4.axvline(lineups_df['Salary'].mean(), color='red',
                       linestyle='--', label=f"Mean: ${lineups_df['Salary'].mean():.0f}")
            ax4.set_xlabel('Salary Used')
            ax4.set_ylabel('Number of Lineups')
            ax4.set_title('Salary Distribution')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No Salary Data', ha='center', va='center')
        
        # 5. Projection Distribution
        ax5 = axes[1, 1]
        if 'Projected' in lineups_df.columns:
            ax5.hist(lineups_df['Projected'], bins=15, alpha=0.7, color='purple')
            ax5.axvline(lineups_df['Projected'].mean(), color='red',
                       linestyle='--', label=f"Mean: {lineups_df['Projected'].mean():.1f}")
            ax5.set_xlabel('Projected Points')
            ax5.set_ylabel('Number of Lineups')
            ax5.set_title('Projection Distribution')
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'No Projection Data', ha='center', va='center')
        
        # 6. Leverage Score Distribution
        ax6 = axes[1, 2]
        if 'Leverage_Score' in lineups_df.columns:
            ax6.hist(lineups_df['Leverage_Score'], bins=15, alpha=0.7, color='brown')
            ax6.axvline(lineups_df['Leverage_Score'].mean(), color='red',
                       linestyle='--', label=f"Mean: {lineups_df['Leverage_Score'].mean():.1f}")
            ax6.set_xlabel('Leverage Score')
            ax6.set_ylabel('Number of Lineups')
            ax6.set_title('Leverage Distribution')
            ax6.legend()
        else:
            ax6.text(0.5, 0.5, 'No Leverage Data', ha='center', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Summary statistics
        st.markdown("### Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Lineup Metrics**")
            st.write(f"Total Lineups: {len(lineups_df)}")
            if 'Projected' in lineups_df.columns:
                st.write(f"Avg Projection: {lineups_df['Projected'].mean():.1f}")
            if 'Salary' in lineups_df.columns:
                st.write(f"Avg Salary: ${lineups_df['Salary'].mean():.0f}")
        
        with col2:
            st.markdown("**Ownership Profile**")
            if 'Total_Ownership' in lineups_df.columns:
                st.write(f"Avg Ownership: {lineups_df['Total_Ownership'].mean():.1f}%")
                st.write(f"Min Ownership: {lineups_df['Total_Ownership'].min():.1f}%")
                st.write(f"Max Ownership: {lineups_df['Total_Ownership'].max():.1f}%")
        
        with col3:
            st.markdown("**Captain Analysis**")
            if 'Captain' in lineups_df.columns:
                unique_captains = lineups_df['Captain'].nunique()
                st.write(f"Unique Captains: {unique_captains}")
                if 'Captain_Own%' in lineups_df.columns:
                    st.write(f"Avg Captain Own: {lineups_df['Captain_Own%'].mean():.1f}%")
                    
    except Exception as e:
        st.error(f"Error in lineup analysis: {str(e)}")
        get_logger().log_exception(e, "display_ai_lineup_analysis")

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_lineups_draftkings(lineups_df: pd.DataFrame) -> str:
    """Export lineups in DraftKings format"""
    try:
        dk_lineups = []
        
        for idx, row in lineups_df.iterrows():
            flex_players = row['FLEX'] if isinstance(row['FLEX'], list) else []
            
            # Ensure we have exactly 5 FLEX players
            while len(flex_players) < 5:
                flex_players.append('')
            
            dk_lineups.append({
                'CPT': row.get('Captain', ''),
                'FLEX 1': flex_players[0] if len(flex_players) > 0 else '',
                'FLEX 2': flex_players[1] if len(flex_players) > 1 else '',
                'FLEX 3': flex_players[2] if len(flex_players) > 2 else '',
                'FLEX 4': flex_players[3] if len(flex_players) > 3 else '',
                'FLEX 5': flex_players[4] if len(flex_players) > 4 else ''
            })
        
        dk_df = pd.DataFrame(dk_lineups)
        return dk_df.to_csv(index=False)
        
    except Exception as e:
        get_logger().log(f"Export error: {e}", "ERROR")
        return ""

def export_detailed_lineups(lineups_df: pd.DataFrame) -> str:
    """Export detailed lineup information"""
    try:
        # Create expanded DataFrame with all details
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
                'Leverage_Score': row.get('Leverage_Score', 0),
                'AI_Enforced': row.get('AI_Enforced', False)
            }
            
            # Add FLEX players
            flex_players = row.get('FLEX', [])
            for i, player in enumerate(flex_players):
                lineup_detail[f'FLEX_{i+1}'] = player
            
            detailed.append(lineup_detail)
        
        detailed_df = pd.DataFrame(detailed)
        return detailed_df.to_csv(index=False)
        
    except Exception as e:
        get_logger().log(f"Detailed export error: {e}", "ERROR")
        return ""

# ============================================================================
# MAIN STREAMLIT UI
# ============================================================================

def main():
    """Main application entry point with comprehensive error handling"""
    
    # Page configuration
    st.set_page_config(
        page_title="NFL GPP AI-Chef Optimizer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏈 NFL GPP Tournament Optimizer - AI-as-Chef Edition")
    st.markdown("*Version 6.3 - Triple AI System with Enhanced Robustness*")
    
    # Initialize session state
    init_ai_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("AI-Chef Configuration")
        
        # AI Mode Selection
        st.markdown("### AI Enforcement Level")
        enforcement_level = st.select_slider(
            "How strictly to enforce AI decisions",
            options=['Advisory', 'Moderate', 'Strong', 'Mandatory'],
            value='Mandatory',
            help="Mandatory = AI decisions are hard constraints"
        )
        
        if enforcement_level != 'Mandatory':
            st.info("AI-as-Chef mode works best with Mandatory enforcement")
        
        # Contest Type
        st.markdown("### Contest Type")
        contest_type = st.selectbox(
            "Select GPP Type",
            list(OptimizerConfig.FIELD_SIZES.keys()),
            index=2,
            help="Different contests require different AI strategies"
        )
        field_size = OptimizerConfig.FIELD_SIZES[contest_type]
        
        # Display AI strategy for field
        ai_config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size, {})
        with st.expander("Field Configuration"):
            st.write(f"**Enforcement:** {ai_config.get('ai_enforcement', AIEnforcementLevel.MANDATORY).value}")
            st.write(f"**Min Unique Captains:** {ai_config.get('min_unique_captains', 10)}")
            st.write(f"**Max Chalk Players:** {ai_config.get('max_chalk_players', 2)}")
            st.write(f"**Min Leverage:** {ai_config.get('min_leverage_players', 2)}")
        
        st.markdown("---")
        
        # API Configuration
        st.markdown("### AI Connection")
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
                help="Get from console.anthropic.com"
            )
            
            if api_key:
                if st.button("Connect to Claude"):
                    try:
                        api_manager = ClaudeAPIManager(api_key)
                        if api_manager.validate_connection():
                            st.success("Connected to Claude AI")
                            st.session_state['api_manager'] = api_manager
                            use_api = True
                        else:
                            st.error("Failed to validate connection")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
            
            # Check existing connection
            if 'api_manager' in st.session_state and st.session_state['api_manager']:
                api_manager = st.session_state['api_manager']
                use_api = True
                st.success("Using existing connection")
        
        st.markdown("---")
        
        # Advanced Settings
        with st.expander("Advanced Settings"):
            st.markdown("### Optimization")
            
            force_unique_captains = st.checkbox(
                "Force Unique Captains",
                value=True,
                help="Each lineup gets a different captain"
            )
            
            parallel_generation = st.checkbox(
                "Enable Parallel Generation",
                value=True,
                help="Generate multiple lineups simultaneously"
            )
            
            show_diagnostics = st.checkbox(
                "Show diagnostic information",
                value=False,
                help="Display detailed information about constraint violations"
            )

            # Store in session state
            st.session_state['show_diagnostics'] = show_diagnostics
            
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
        
        # Debug Panel
        with st.expander("Debug & Monitoring"):
            if st.button("Show Performance Metrics"):
                perf = get_performance_monitor()
                perf.display_metrics()
            
            if st.button("Show Logs"):
                logger = get_logger()
                logger.display_log_summary()
            
            if st.button("Clear Cache"):
                st.cache_data.clear()
                st.success("Cache cleared")
    
    # Main Content Area
    st.markdown("## Data & Game Configuration")
    
    uploaded_file = st.file_uploader(
        "Upload DraftKings CSV",
        type="csv",
        help="Export from DraftKings Showdown contest"
    )
    
    if uploaded_file is not None:
        try:
            # Load and validate data
            raw_df = pd.read_csv(uploaded_file)
            df, validation = validate_and_process_dataframe(raw_df)
            
            # Display validation results
            if validation['errors']:
                st.error("Validation Errors:")
                for error in validation['errors']:
                    st.write(f"  - {error}")
            
            if validation['warnings']:
                st.warning("Warnings:")
                for warning in validation['warnings']:
                    st.write(f"  - {warning}")
            
            if validation['fixes_applied']:
                st.info("Fixes Applied:")
                for fix in validation['fixes_applied']:
                    st.write(f"  - {fix}")
            
            if not validation['is_valid']:
                st.error("Cannot proceed with optimization due to validation errors")
                st.stop()
            
            # Store in session state
            st.session_state['df'] = df
            
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
            
            # Display player pool
            st.markdown("### Player Pool")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Players", len(df))
            with col2:
                st.metric("Avg Salary", f"${df['Salary'].mean():.0f}")
            with col3:
                st.metric("Avg Projection", f"{df['Projected_Points'].mean():.1f}")
            with col4:
                st.metric("Avg Ownership", f"{df['Ownership'].mean():.1f}%")
            
            # Show player data
            with st.expander("View Player Pool"):
                display_cols = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 'Ownership']
                available_cols = [col for col in display_cols if col in df.columns]
                st.dataframe(
                    df[available_cols].sort_values('Projected_Points', ascending=False),
                    use_container_width='stretch'
                )
            
            # Optimization section
            st.markdown("---")
            st.markdown("## AI-Driven Lineup Generation")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_lineups = st.number_input(
                    "Number of Lineups",
                    min_value=1,
                    max_value=150,
                    value=20,
                    step=5,
                    help="AI will determine strategy distribution"
                )
            
            with col2:
                st.metric("Field Size", field_size.replace('_', ' ').title())
                st.metric("AI Mode", enforcement_level)
            
            with col3:
                generate_button = st.button(
                    "Generate AI-Driven Lineups",
                    type="primary",
                    use_container_width='stretch'
                )
            
            if generate_button:
                try:
                    # Initialize optimizer
                    optimizer = AIChefGPPOptimizer(df, game_info, field_size, api_manager)
                    
                    # Get AI strategies
                    with st.spinner("Consulting Triple AI System..."):
                        ai_recommendations = optimizer.get_triple_ai_strategies(use_api=use_api)
                    
                    if not ai_recommendations:
                        st.error("Failed to get AI recommendations")
                        st.stop()
                    
                    # Store recommendations
                    st.session_state['ai_recommendations'] = ai_recommendations
                    
                    # Display AI recommendations
                    display_ai_recommendations(ai_recommendations)
                    
                    # Synthesize strategies
                    with st.spinner("Synthesizing AI strategies..."):
                        ai_strategy = optimizer.synthesize_ai_strategies(ai_recommendations)
                    
                    # Store synthesis
                    st.session_state['ai_synthesis'] = ai_strategy['synthesis']
                    
                    # Display synthesis
                    display_ai_synthesis(ai_strategy['synthesis'])
                    
                    # Generate lineups
                    with st.spinner(f"Generating {num_lineups} AI-enforced lineups..."):
                        lineups_df = optimizer.generate_ai_driven_lineups(num_lineups, ai_strategy)
                    
                    if not lineups_df.empty:
                        # Store in session
                        st.session_state['lineups_df'] = lineups_df
                        
                        # Save session
                        save_ai_optimization_session(
                            lineups_df,
                            ai_strategy['synthesis'],
                            {
                                'num_lineups': num_lineups,
                                'field_size': field_size,
                                'enforcement': enforcement_level
                            }
                        )
                        
                        # Display enforcement results
                        st.markdown("---")
                        logger = get_logger()
                        logger.display_ai_enforcement()
                        
                except Exception as e:
                    st.error(f"Optimization error: {str(e)}")
                    get_logger().log_exception(e, "main_optimization", critical=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            get_logger().log_exception(e, "file_processing")
    
    # Display results if available
    if 'lineups_df' in st.session_state and not st.session_state['lineups_df'].empty:
        lineups_df = st.session_state['lineups_df']
        synthesis = st.session_state.get('ai_synthesis', {})
        df = st.session_state.get('df', pd.DataFrame())
        
        st.markdown("---")
        st.markdown("## AI Optimization Results")
        
        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Lineups", "Analysis", "Visualizations", "Export"
        ])
        
        with tab1:
            st.markdown("### AI-Generated Lineups")
            
            # Display options
            col1, col2 = st.columns(2)
            with col1:
                if len(lineups_df) <= 5:     
                   show_count = len(lineups_df)     
                   st.write(f"Showing all {show_count} lineups") 
                else:     
                    default_show = min(10, len(lineups_df))     
                    show_count = st.slider("Show lineups", 5, len(lineups_df), default_show)
            with col2:
                sort_by = st.selectbox("Sort by", ["Lineup", "Projected", "Total_Ownership", "Leverage_Score"])
            
            # Sort lineups
            display_df = lineups_df.sort_values(sort_by, ascending=(sort_by == "Lineup"))
            
            # Display lineups
            for idx, row in display_df.head(show_count).iterrows():
                with st.expander(
                    f"Lineup {row['Lineup']} - {row.get('AI_Strategy', 'Unknown')} "
                    f"(Proj: {row.get('Projected', 0):.1f}, Own: {row.get('Total_Ownership', 0):.1f}%)"
                ):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Roster:**")
                        st.write(f"CPT: {row['Captain']}")
                        for i, player in enumerate(row['FLEX']):
                            st.write(f"FLEX {i+1}: {player}")
                    
                    with col2:
                        st.markdown("**Metrics:**")
                        st.write(f"Projected: {row.get('Projected', 0):.1f}")
                        st.write(f"Salary: ${row.get('Salary', 0):,}")
                        st.write(f"Remaining: ${row.get('Salary_Remaining', 0):,}")
                        st.write(f"Ownership: {row.get('Total_Ownership', 0):.1f}%")
                    
                    with col3:
                        st.markdown("**AI Info:**")
                        st.write(f"Strategy: {row.get('AI_Strategy', 'N/A')}")
                        st.write(f"Confidence: {row.get('Confidence', 0):.0%}")
                        st.write(f"Leverage: {row.get('Leverage_Score', 0):.1f}")
                        if row.get('AI_Sources'):
                            st.write(f"Source: {', '.join(row['AI_Sources'])}")
        
        with tab2:
            display_ai_lineup_analysis(lineups_df, df, synthesis, field_size)
        
        with tab3:
            st.markdown("### Strategy Visualizations")
            
            # Create visualizations
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Captain consensus visualization
            ax1 = axes[0]
            if synthesis and 'captain_strategy' in synthesis:
                captain_strategy = synthesis['captain_strategy']
                if captain_strategy:
                    consensus_types = Counter(captain_strategy.values())
                    ax1.pie(consensus_types.values(), labels=consensus_types.keys(),
                           autopct='%1.0f%%', colors=['gold', 'silver', 'bronze', 'gray'])
                    ax1.set_title('Captain Consensus Distribution')
                else:
                    ax1.text(0.5, 0.5, 'No Captain Data', ha='center', va='center')
            
            # Ownership vs Projection scatter
            ax2 = axes[1]
            if not lineups_df.empty and 'Total_Ownership' in lineups_df.columns and 'Projected' in lineups_df.columns:
                ax2.scatter(lineups_df['Total_Ownership'], lineups_df['Projected'], alpha=0.6)
                ax2.set_xlabel('Total Ownership %')
                ax2.set_ylabel('Projected Points')
                ax2.set_title('Ownership vs Projection')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No Data', ha='center', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab4:
            st.markdown("### Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### DraftKings Format")
                dk_csv = export_lineups_draftkings(lineups_df)
                if dk_csv:
                    st.download_button(
                        label="Download DraftKings CSV",
                        data=dk_csv,
                        file_name=f"dk_lineups_{field_size}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                st.markdown("#### Detailed Format")
                detailed_csv = export_detailed_lineups(lineups_df)
                if detailed_csv:
                    st.download_button(
                        label="Download Detailed CSV",
                        data=detailed_csv,
                        file_name=f"detailed_lineups_{field_size}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
            
            # AI Strategy export
            st.markdown("#### AI Strategy Export")
            if synthesis:
                strategy_export = {
                    'timestamp': datetime.now().isoformat(),
                    'field_size': field_size,
                    'synthesis': synthesis,
                    'lineup_count': len(lineups_df)
                }
                
                strategy_json = json.dumps(strategy_export, default=str, indent=2)
                st.download_button(
                    label="Download AI Strategy",
                    data=strategy_json,
                    file_name=f"ai_strategy_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        get_logger().log_exception(e, "main_entry", critical=True)
