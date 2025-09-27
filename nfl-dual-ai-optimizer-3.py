# NFL GPP DUAL-AI OPTIMIZER - PART 1: CONFIGURATION & MONITORING
# Version 6.0 - AI-as-Chef Architecture with Triple AI System

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum
import json
from datetime import datetime
import streamlit as st

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class StrategyType(Enum):
    """GPP Strategy Types - Now AI-Driven"""
    AI_CONSENSUS = 'ai_consensus'        # All 3 AIs agree
    AI_MAJORITY = 'ai_majority'          # 2 of 3 AIs agree  
    AI_CONTRARIAN = 'ai_contrarian'      # Following contrarian AI specifically
    AI_CORRELATION = 'ai_correlation'    # Following correlation AI specifically
    AI_GAME_THEORY = 'ai_game_theory'    # Following game theory AI specifically
    AI_MIXED = 'ai_mixed'                # Mixing different AI perspectives
    LEVERAGE = 'leverage'                # Legacy - will be AI-driven
    CONTRARIAN = 'contrarian'            # Legacy - will be AI-driven

class AIEnforcementLevel(Enum):
    """How strictly to enforce AI decisions"""
    MANDATORY = 'mandatory'              # AI decisions are hard constraints
    STRONG = 'strong'                    # AI decisions heavily weighted
    MODERATE = 'moderate'                # AI decisions moderately weighted
    SUGGESTION = 'suggestion'            # AI decisions as hints only

class AIStrategistType(Enum):
    """Three AI Strategist Types"""
    GAME_THEORY = 'game_theory'
    CORRELATION = 'correlation'
    CONTRARIAN_NARRATIVE = 'contrarian_narrative'

# ============================================================================
# CONFIGURATION
# ============================================================================

class OptimizerConfig:
    """Core configuration for GPP optimizer - AI-as-Chef version"""
    
    # Salary and roster constraints
    SALARY_CAP = 50000
    MIN_SALARY = 3000
    MAX_PLAYERS_PER_TEAM = 5
    ROSTER_SIZE = 6  # 1 CPT + 5 FLEX
    
    # Default ownership when unknown (pre-lock)
    DEFAULT_OWNERSHIP = 5.0
    
    # AI Configuration
    AI_ENFORCEMENT_MODE = AIEnforcementLevel.MANDATORY
    REQUIRE_AI_FOR_GENERATION = True  # If True, won't generate without AI input
    MIN_AI_CONFIDENCE = 0.3  # Minimum confidence to use AI recommendations
    
    # Triple AI Weights (how much each AI contributes)
    AI_WEIGHTS = {
        AIStrategistType.GAME_THEORY: 0.35,
        AIStrategistType.CORRELATION: 0.35,
        AIStrategistType.CONTRARIAN_NARRATIVE: 0.30
    }
    
    # AI Consensus Requirements
    AI_CONSENSUS_THRESHOLDS = {
        'captain': 2,  # How many AIs must agree for consensus captain
        'must_play': 3,  # All 3 must agree for "must play"
        'never_play': 2,  # 2 must agree for "never play"
    }
    
    # Field size configurations - Now with AI strategy distribution
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
            }
        },
        'medium_field': {
            'min_unique_captains': 6,
            'max_chalk_players': 2,
            'min_leverage_players': 1,
            'max_ownership_per_player': 20,
            'min_total_ownership': 70,
            'max_total_ownership': 100,
            'ai_enforcement': AIEnforcementLevel.STRONG,
            'ai_strategy_distribution': {
                'ai_consensus': 0.2,
                'ai_majority': 0.3,
                'ai_contrarian': 0.2,
                'ai_correlation': 0.2,
                'ai_game_theory': 0.1
            }
        },
        'large_field': {
            'min_unique_captains': 10,
            'max_chalk_players': 1,
            'min_leverage_players': 2,
            'max_ownership_per_player': 15,
            'min_total_ownership': 60,
            'max_total_ownership': 90,
            'ai_enforcement': AIEnforcementLevel.MANDATORY,
            'ai_strategy_distribution': {
                'ai_consensus': 0.1,
                'ai_majority': 0.2,
                'ai_contrarian': 0.3,
                'ai_correlation': 0.2,
                'ai_game_theory': 0.2
            }
        },
        'milly_maker': {
            'min_unique_captains': 20,
            'max_chalk_players': 0,
            'min_leverage_players': 3,
            'max_ownership_per_player': 10,
            'min_total_ownership': 50,
            'max_total_ownership': 80,
            'ai_enforcement': AIEnforcementLevel.MANDATORY,
            'ai_strategy_distribution': {
                'ai_contrarian': 0.4,
                'ai_game_theory': 0.3,
                'ai_correlation': 0.2,
                'ai_mixed': 0.1
            }
        }
    }
    
    # AI Strategy Requirements - What each AI strategy type enforces
    AI_STRATEGY_REQUIREMENTS = {
        'ai_consensus': {
            'must_use_consensus_captain': True,
            'must_include_agreed_players': True,
            'must_avoid_fade_players': True
        },
        'ai_majority': {
            'must_use_majority_captain': True,
            'should_include_agreed_players': True
        },
        'ai_contrarian': {
            'must_use_contrarian_captain': True,
            'must_include_narrative_plays': True,
            'must_fade_chalk_narrative': True
        },
        'ai_correlation': {
            'must_use_correlation_captain': True,
            'must_include_primary_stack': True,
            'should_include_secondary_correlation': True
        },
        'ai_game_theory': {
            'must_use_leverage_captain': True,
            'must_meet_ownership_targets': True,
            'must_avoid_ownership_traps': True
        }
    }
    
    # Ownership bucket thresholds
    OWNERSHIP_BUCKETS = {
        'mega_chalk': 35,
        'chalk': 20,
        'pivot': 10,
        'leverage': 5,
        'super_leverage': 0
    }
    
    # Contest types to field sizes
    FIELD_SIZES = {
        'Single Entry': 'small_field',
        '3-Max Entry': 'small_field',
        '20-Max Entry': 'medium_field',
        '150-Max Entry': 'large_field',
        'Milly Maker': 'milly_maker'
    }
    
    # Correlation thresholds
    CORRELATION_THRESHOLDS = {
        'strong_positive': 0.6,
        'moderate_positive': 0.3,
        'weak': 0.0,
        'moderate_negative': -0.3,
        'strong_negative': -0.6
    }

# ============================================================================
# MONITORING AND LOGGING - AI-FOCUSED
# ============================================================================

class AIDecisionTracker:
    """Tracks AI decisions and enforcement throughout optimization"""
    
    def __init__(self):
        self.ai_decisions = []
        self.enforcement_stats = {
            'total_rules': 0,
            'enforced_rules': 0,
            'violated_rules': 0,
            'consensus_decisions': 0,
            'majority_decisions': 0,
            'single_ai_decisions': 0
        }
        self.ai_performance = {
            AIStrategistType.GAME_THEORY: {'suggestions': 0, 'used': 0},
            AIStrategistType.CORRELATION: {'suggestions': 0, 'used': 0},
            AIStrategistType.CONTRARIAN_NARRATIVE: {'suggestions': 0, 'used': 0}
        }
    
    def track_decision(self, decision_type: str, ai_source: str, 
                      enforced: bool, details: Dict):
        """Track an AI decision and whether it was enforced"""
        self.ai_decisions.append({
            'timestamp': datetime.now(),
            'type': decision_type,
            'source': ai_source,
            'enforced': enforced,
            'details': details
        })
        
        self.enforcement_stats['total_rules'] += 1
        if enforced:
            self.enforcement_stats['enforced_rules'] += 1
        else:
            self.enforcement_stats['violated_rules'] += 1
    
    def track_consensus(self, consensus_type: str, ais_agreeing: List[str]):
        """Track consensus between AIs"""
        if len(ais_agreeing) == 3:
            self.enforcement_stats['consensus_decisions'] += 1
        elif len(ais_agreeing) == 2:
            self.enforcement_stats['majority_decisions'] += 1
        else:
            self.enforcement_stats['single_ai_decisions'] += 1
    
    def get_enforcement_rate(self) -> float:
        """Get the rate of AI decision enforcement"""
        if self.enforcement_stats['total_rules'] == 0:
            return 0.0
        return self.enforcement_stats['enforced_rules'] / self.enforcement_stats['total_rules']
    
    def get_summary(self) -> Dict:
        """Get summary of AI decision tracking"""
        return {
            'enforcement_rate': self.get_enforcement_rate(),
            'stats': self.enforcement_stats,
            'ai_performance': self.ai_performance,
            'recent_decisions': self.ai_decisions[-10:]
        }

class GlobalLogger:
    """Enhanced logger with AI decision tracking"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        self.entries = []
        self.ai_tracker = AIDecisionTracker()
        self.verbose = False
        self.log_to_file = False
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with level"""
        entry = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message
        }
        self.entries.append(entry)
        
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = entry['timestamp'].strftime('%H:%M:%S')
            print(f"[{timestamp}] {level}: {message}")
            
        if self.log_to_file:
            self._write_to_file(entry)
    
    def log_ai_decision(self, decision_type: str, ai_source: str, 
                       enforced: bool, details: Dict):
        """Log an AI decision"""
        self.ai_tracker.track_decision(decision_type, ai_source, enforced, details)
        message = f"AI Decision [{ai_source}]: {decision_type} - {'ENFORCED' if enforced else 'VIOLATED'}"
        self.log(message, "AI_DECISION")
    
    def log_ai_consensus(self, consensus_type: str, ais_agreeing: List[str], 
                        decision: str):
        """Log AI consensus"""
        self.ai_tracker.track_consensus(consensus_type, ais_agreeing)
        message = f"AI Consensus ({len(ais_agreeing)}/3): {decision}"
        self.log(message, "AI_CONSENSUS")
    
    def log_optimization_start(self, num_lineups: int, field_size: str, 
                              settings: Dict):
        """Log optimization start with AI focus"""
        self.log(f"Starting AI-driven optimization: {num_lineups} lineups for {field_size}", "INFO")
        self.log(f"AI Enforcement: {settings.get('ai_enforcement', 'MANDATORY')}", "INFO")
        self.log(f"Settings: {settings}", "DEBUG")
    
    def log_lineup_generation(self, strategy: str, lineup_num: int, 
                             status: str, ai_rules_enforced: int = 0):
        """Log lineup generation with AI enforcement tracking"""
        message = f"Lineup {lineup_num} ({strategy}): {status}"
        if ai_rules_enforced > 0:
            message += f" - {ai_rules_enforced} AI rules enforced"
        self.log(message, "DEBUG" if status == "SUCCESS" else "WARNING")
    
    def get_ai_summary(self) -> Dict:
        """Get AI tracking summary"""
        return self.ai_tracker.get_summary()
    
    def display_ai_enforcement(self):
        """Display AI enforcement statistics in Streamlit"""
        summary = self.get_ai_summary()
        
        st.markdown("### 🤖 AI Decision Enforcement")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rate = summary['enforcement_rate'] * 100
            st.metric("Enforcement Rate", f"{rate:.1f}%")
        
        with col2:
            stats = summary['stats']
            st.metric("Rules Enforced", 
                     f"{stats['enforced_rules']}/{stats['total_rules']}")
        
        with col3:
            st.metric("Consensus Decisions", stats['consensus_decisions'])
        
        # Show AI performance
        st.markdown("#### AI Performance")
        for ai_type, perf in summary['ai_performance'].items():
            if perf['suggestions'] > 0:
                usage_rate = (perf['used'] / perf['suggestions']) * 100
                st.write(f"**{ai_type.value}**: {usage_rate:.1f}% usage rate")
    
    def export_logs(self) -> str:
        """Export logs including AI decisions"""
        output = "=== OPTIMIZATION LOG ===\n\n"
        for entry in self.entries:
            timestamp = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            output += f"[{timestamp}] {entry['level']}: {entry['message']}\n"
        
        output += "\n=== AI DECISION SUMMARY ===\n"
        summary = self.get_ai_summary()
        output += f"Enforcement Rate: {summary['enforcement_rate']*100:.1f}%\n"
        output += f"Total AI Rules: {summary['stats']['total_rules']}\n"
        output += f"Enforced: {summary['stats']['enforced_rules']}\n"
        output += f"Violated: {summary['stats']['violated_rules']}\n"
        
        return output

# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """Monitor performance with AI-specific metrics"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        self.timers = {}
        self.counters = {}
        self.ai_metrics = {
            'ai_api_calls': 0,
            'ai_response_time': [],
            'ai_cache_hits': 0,
            'ai_enforcement_time': []
        }
    
    def start_timer(self, name: str):
        self.timers[name] = {'start': datetime.now()}
    
    def stop_timer(self, name: str) -> float:
        if name in self.timers and 'start' in self.timers[name]:
            elapsed = (datetime.now() - self.timers[name]['start']).total_seconds()
            if 'total' not in self.timers[name]:
                self.timers[name]['total'] = 0
                self.timers[name]['count'] = 0
            self.timers[name]['total'] += elapsed
            self.timers[name]['count'] += 1
            self.timers[name]['average'] = self.timers[name]['total'] / self.timers[name]['count']
            
            # Track AI-specific timings
            if 'ai' in name.lower():
                self.ai_metrics['ai_response_time'].append(elapsed)
            
            return elapsed
        return 0.0
    
    def increment_counter(self, name: str, value: int = 1):
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
        
        # Track AI-specific counters
        if 'ai' in name.lower():
            if 'cache' in name.lower():
                self.ai_metrics['ai_cache_hits'] += value
            elif 'api' in name.lower():
                self.ai_metrics['ai_api_calls'] += value
    
    def get_metrics(self) -> Dict:
        """Get all metrics including AI-specific ones"""
        avg_ai_response = (
            np.mean(self.ai_metrics['ai_response_time']) 
            if self.ai_metrics['ai_response_time'] else 0
        )
        
        return {
            'timers': self.timers,
            'counters': self.counters,
            'ai_metrics': {
                'api_calls': self.ai_metrics['ai_api_calls'],
                'cache_hits': self.ai_metrics['ai_cache_hits'],
                'avg_response_time': avg_ai_response,
                'cache_hit_rate': (
                    self.ai_metrics['ai_cache_hits'] / 
                    (self.ai_metrics['ai_api_calls'] + self.ai_metrics['ai_cache_hits'])
                    if (self.ai_metrics['ai_api_calls'] + self.ai_metrics['ai_cache_hits']) > 0
                    else 0
                )
            }
        }
    
    def display_metrics(self):
        """Display performance metrics with AI focus"""
        metrics = self.get_metrics()
        
        st.markdown("### ⚡ Performance Metrics")
        
        # AI Metrics
        if metrics['ai_metrics']['api_calls'] > 0:
            st.markdown("#### AI Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("API Calls", metrics['ai_metrics']['api_calls'])
            with col2:
                st.metric("Avg Response", f"{metrics['ai_metrics']['avg_response_time']:.2f}s")
            with col3:
                st.metric("Cache Hit Rate", f"{metrics['ai_metrics']['cache_hit_rate']*100:.1f}%")
        
        # General Metrics
        st.markdown("#### Optimization Performance")
        for name, stats in metrics['timers'].items():
            if 'average' in stats:
                st.write(f"**{name}**: {stats['average']:.3f}s avg ({stats['count']} ops)")

# ============================================================================
# SINGLETON GETTERS
# ============================================================================

def get_logger() -> GlobalLogger:
    """Get the global logger instance"""
    return GlobalLogger()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return PerformanceMonitor()

# ============================================================================
# AI RECOMMENDATION DATACLASS - ENHANCED
# ============================================================================

@dataclass
class AIRecommendation:
    """Enhanced AI recommendation with enforcement rules"""
    captain_targets: List[str]
    must_play: List[str]  # Players that must be included
    never_play: List[str]  # Players to avoid
    stacks: List[Dict]
    key_insights: List[str]
    confidence: float
    enforcement_rules: List[Dict]  # Specific rules to enforce
    narrative: str  # AI's narrative/reasoning
    source_ai: AIStrategistType
    
    # Optional fields for specific AI types
    contrarian_angles: Optional[List[str]] = None
    correlation_matrix: Optional[Dict] = None
    ownership_leverage: Optional[Dict] = None
    
    def get_hard_constraints(self) -> List[str]:
        """Get constraints that MUST be enforced"""
        constraints = []
        for rule in self.enforcement_rules:
            if rule.get('type') == 'hard':
                constraints.append(rule['constraint'])
        return constraints
    
    def get_soft_constraints(self) -> List[Tuple[str, float]]:
        """Get constraints with weights for soft enforcement"""
        constraints = []
        for rule in self.enforcement_rules:
            if rule.get('type') == 'soft':
                constraints.append((rule['constraint'], rule.get('weight', 1.0)))
        return constraints

# NFL GPP DUAL-AI OPTIMIZER - PART 2: CORE COMPONENTS (AI-AS-CHEF VERSION)
# Version 6.0 - AI-Driven Core Components with Enforcement

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import streamlit as st
import json

# ============================================================================
# AI ENFORCEMENT ENGINE
# ============================================================================

class AIEnforcementEngine:
    """Core engine for enforcing AI decisions throughout optimization"""
    
    def __init__(self, enforcement_level: AIEnforcementLevel = AIEnforcementLevel.MANDATORY):
        self.enforcement_level = enforcement_level
        self.logger = get_logger()
        self.enforcement_history = []
    
    def create_enforcement_rules(self, ai_recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """Convert AI recommendations into enforceable optimization rules"""
        
        rules = {
            'hard_constraints': [],  # Must be satisfied
            'soft_constraints': [],  # Should be satisfied with weights
            'objective_modifiers': {},  # Modifications to objective function
            'variable_locks': {},  # Specific variable assignments
        }
        
        # Find consensus among AIs
        consensus = self._find_consensus(ai_recommendations)
        
        # Create rules based on consensus and enforcement level
        if self.enforcement_level == AIEnforcementLevel.MANDATORY:
            # All consensus decisions become hard constraints
            rules['hard_constraints'].extend(self._create_hard_constraints(consensus['must_play'], 'include'))
            rules['hard_constraints'].extend(self._create_hard_constraints(consensus['never_play'], 'exclude'))
            rules['variable_locks']['captain'] = consensus['consensus_captains']
            
        elif self.enforcement_level == AIEnforcementLevel.STRONG:
            # Consensus becomes hard, majority becomes soft
            rules['hard_constraints'].extend(self._create_hard_constraints(consensus['must_play'], 'include'))
            rules['soft_constraints'].extend(self._create_soft_constraints(consensus['should_play'], 'include', 0.8))
            
        # Add AI-specific modifiers
        for ai_type, rec in ai_recommendations.items():
            if rec.confidence > OptimizerConfig.MIN_AI_CONFIDENCE:
                rules['objective_modifiers'].update(self._create_objective_modifiers(rec))
        
        self.logger.log_ai_decision(
            "enforcement_rules_created",
            "AIEnforcementEngine",
            True,
            {'num_hard': len(rules['hard_constraints']), 
             'num_soft': len(rules['soft_constraints'])}
        )
        
        return rules
    
    def _find_consensus(self, ai_recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """Find consensus among AI recommendations"""
        
        consensus = {
            'consensus_captains': [],  # All 3 AIs agree
            'majority_captains': [],   # 2 of 3 agree
            'must_play': [],           # All agree these must play
            'should_play': [],         # 2 of 3 agree these should play
            'never_play': [],          # 2+ agree these should be avoided
        }
        
        # Aggregate captain recommendations
        captain_votes = {}
        for ai_type, rec in ai_recommendations.items():
            for captain in rec.captain_targets:
                if captain not in captain_votes:
                    captain_votes[captain] = []
                captain_votes[captain].append(ai_type)
        
        # Classify by agreement level
        for captain, voters in captain_votes.items():
            if len(voters) == 3:
                consensus['consensus_captains'].append(captain)
                self.logger.log_ai_consensus("captain", voters, f"Consensus captain: {captain}")
            elif len(voters) == 2:
                consensus['majority_captains'].append(captain)
                self.logger.log_ai_consensus("captain", voters, f"Majority captain: {captain}")
        
        # Aggregate must_play and never_play
        must_play_votes = {}
        never_play_votes = {}
        
        for ai_type, rec in ai_recommendations.items():
            for player in rec.must_play:
                if player not in must_play_votes:
                    must_play_votes[player] = []
                must_play_votes[player].append(ai_type)
            
            for player in rec.never_play:
                if player not in never_play_votes:
                    never_play_votes[player] = []
                never_play_votes[player].append(ai_type)
        
        # Classify must_play by agreement
        for player, voters in must_play_votes.items():
            if len(voters) == 3:
                consensus['must_play'].append(player)
            elif len(voters) == 2:
                consensus['should_play'].append(player)
        
        # Classify never_play (2+ agreement for fades)
        for player, voters in never_play_votes.items():
            if len(voters) >= 2:
                consensus['never_play'].append(player)
        
        return consensus
    
    def _create_hard_constraints(self, players: List[str], constraint_type: str) -> List[Dict]:
        """Create hard constraints for optimization"""
        constraints = []
        
        for player in players:
            if constraint_type == 'include':
                constraints.append({
                    'type': 'hard',
                    'player': player,
                    'rule': 'must_include',
                    'description': f"AI consensus: must include {player}"
                })
            elif constraint_type == 'exclude':
                constraints.append({
                    'type': 'hard',
                    'player': player,
                    'rule': 'must_exclude',
                    'description': f"AI consensus: must exclude {player}"
                })
        
        return constraints
    
    def _create_soft_constraints(self, players: List[str], constraint_type: str, weight: float) -> List[Dict]:
        """Create weighted soft constraints"""
        constraints = []
        
        for player in players:
            constraints.append({
                'type': 'soft',
                'player': player,
                'rule': f"should_{constraint_type}",
                'weight': weight,
                'description': f"AI majority: should {constraint_type} {player}"
            })
        
        return constraints
    
    def _create_objective_modifiers(self, recommendation: AIRecommendation) -> Dict:
        """Create objective function modifiers based on AI recommendation"""
        modifiers = {}
        
        # Boost recommended players
        for player in recommendation.captain_targets:
            modifiers[player] = modifiers.get(player, 1.0) * (1.0 + recommendation.confidence * 0.3)
        
        for player in recommendation.must_play:
            modifiers[player] = modifiers.get(player, 1.0) * (1.0 + recommendation.confidence * 0.2)
        
        # Penalize fade targets
        for player in recommendation.never_play:
            modifiers[player] = modifiers.get(player, 1.0) * (1.0 - recommendation.confidence * 0.3)
        
        return modifiers
    
    def validate_lineup_against_ai(self, lineup: Dict, ai_rules: Dict) -> Tuple[bool, List[str]]:
        """Validate that a lineup satisfies AI requirements"""
        violations = []
        
        lineup_players = [lineup.get('Captain')] + lineup.get('FLEX', [])
        
        # Check hard constraints
        for constraint in ai_rules.get('hard_constraints', []):
            player = constraint['player']
            rule = constraint['rule']
            
            if rule == 'must_include' and player not in lineup_players:
                violations.append(f"Missing required player: {player}")
            elif rule == 'must_exclude' and player in lineup_players:
                violations.append(f"Included banned player: {player}")
        
        # Check captain requirements
        required_captains = ai_rules.get('variable_locks', {}).get('captain', [])
        if required_captains and lineup.get('Captain') not in required_captains:
            violations.append(f"Captain {lineup.get('Captain')} not in AI requirements: {required_captains}")
        
        is_valid = len(violations) == 0
        
        # Track enforcement
        self.enforcement_history.append({
            'lineup': lineup.get('Lineup', 0),
            'valid': is_valid,
            'violations': violations
        })
        
        return is_valid, violations

# ============================================================================
# AI-DRIVEN OWNERSHIP BUCKET MANAGER
# ============================================================================

class AIOwnershipBucketManager:
    """Enhanced bucket manager that respects AI decisions"""
    
    def __init__(self, ai_enforcement_engine: Optional[AIEnforcementEngine] = None):
        self.thresholds = OptimizerConfig.OWNERSHIP_BUCKETS
        self.ai_engine = ai_enforcement_engine
        self.logger = get_logger()
    
    def categorize_players_with_ai(self, df: pd.DataFrame, 
                                   ai_recommendations: Dict[AIStrategistType, AIRecommendation]) -> Dict:
        """Categorize players considering AI input"""
        
        # Standard categorization first
        buckets = self.categorize_players(df)
        
        if self.ai_engine and ai_recommendations:
            # Apply AI overrides
            consensus = self.ai_engine._find_consensus(ai_recommendations)
            
            # Move consensus captains to their own category
            buckets['ai_consensus_captains'] = consensus['consensus_captains']
            
            # Remove them from other buckets
            for captain in consensus['consensus_captains']:
                for bucket_name, players in buckets.items():
                    if bucket_name != 'ai_consensus_captains' and captain in players:
                        players.remove(captain)
            
            # Create AI leverage category
            buckets['ai_leverage'] = []
            for ai_type, rec in ai_recommendations.items():
                if ai_type == AIStrategistType.CONTRARIAN_NARRATIVE:
                    # Contrarian AI's unique picks are leverage
                    buckets['ai_leverage'].extend([
                        p for p in rec.captain_targets 
                        if p not in consensus['consensus_captains']
                    ])
            
            self.logger.log(f"AI-enhanced buckets: {len(buckets['ai_consensus_captains'])} consensus captains", "DEBUG")
        
        return buckets
    
    def categorize_players(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Standard categorization by ownership"""
        buckets = {
            'mega_chalk': [],
            'chalk': [],
            'pivot': [],
            'leverage': [],
            'super_leverage': []
        }
        
        for _, row in df.iterrows():
            player = row['Player']
            ownership = row['Ownership']
            
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
        """Calculate leverage score with AI weighting"""
        
        base_score = self.calculate_gpp_leverage(lineup_players, df)
        
        if not ai_recommendations:
            return base_score
        
        # Bonus for using AI contrarian picks
        ai_bonus = 0
        for player in lineup_players:
            for ai_type, rec in ai_recommendations.items():
                if ai_type == AIStrategistType.CONTRARIAN_NARRATIVE:
                    if player in rec.captain_targets:
                        ai_bonus += 3
                    elif player in rec.must_play:
                        ai_bonus += 2
                elif player in rec.captain_targets:
                    ai_bonus += 1
        
        return base_score + ai_bonus
    
    def calculate_gpp_leverage(self, lineup_players: List[str], df: pd.DataFrame) -> float:
        """Calculate standard GPP leverage score"""
        score = 0
        
        for player in lineup_players:
            if player in df['Player'].values:
                ownership = df[df['Player'] == player]['Ownership'].values[0]
                
                if ownership < 5:
                    score += 3
                elif ownership < 10:
                    score += 2
                elif ownership < 15:
                    score += 1
                elif ownership > 30:
                    score -= 2
                elif ownership > 20:
                    score -= 1
        
        return score
    
    def get_gpp_summary(self, lineup_players: List[str], df: pd.DataFrame, 
                       field_size: str, ai_enforced: bool = False) -> str:
        """Get GPP summary with AI enforcement indicator"""
        
        ownership_counts = {'<5%': 0, '5-10%': 0, '10-20%': 0, '20-30%': 0, '30%+': 0}
        
        for player in lineup_players:
            if player in df['Player'].values:
                ownership = df[df['Player'] == player]['Ownership'].values[0]
                if ownership < 5:
                    ownership_counts['<5%'] += 1
                elif ownership < 10:
                    ownership_counts['5-10%'] += 1
                elif ownership < 20:
                    ownership_counts['10-20%'] += 1
                elif ownership < 30:
                    ownership_counts['20-30%'] += 1
                else:
                    ownership_counts['30%+'] += 1
        
        summary = ' | '.join([f"{k}:{v}" for k, v in ownership_counts.items() if v > 0])
        
        if ai_enforced:
            summary = "🤖 AI-ENFORCED | " + summary
        
        return summary

# ============================================================================
# AI-DRIVEN CONFIG VALIDATOR
# ============================================================================

class AIConfigValidator:
    """Enhanced validator that ensures AI requirements are feasible"""
    
    @staticmethod
    def validate_ai_requirements(ai_rules: Dict, player_pool: pd.DataFrame) -> Dict:
        """Validate that AI requirements can be satisfied"""
        
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'adjustments': []
        }
        
        available_players = set(player_pool['Player'].tolist())
        
        # Check hard constraints feasibility
        must_include = []
        must_exclude = []
        
        for constraint in ai_rules.get('hard_constraints', []):
            player = constraint['player']
            rule = constraint['rule']
            
            if player not in available_players:
                validation['errors'].append(f"AI required player '{player}' not in pool")
                validation['is_valid'] = False
                continue
            
            if rule == 'must_include':
                must_include.append(player)
            elif rule == 'must_exclude':
                must_exclude.append(player)
        
        # Check if we have enough players after exclusions
        available_after_exclusions = len(available_players) - len(must_exclude)
        required_spots = 6  # 1 CPT + 5 FLEX
        
        if len(must_include) > required_spots:
            validation['errors'].append(f"AI requires {len(must_include)} players but lineup has {required_spots} spots")
            validation['is_valid'] = False
        
        if available_after_exclusions < required_spots:
            validation['errors'].append(f"Not enough players after AI exclusions")
            validation['is_valid'] = False
        
        # Check captain requirements
        required_captains = ai_rules.get('variable_locks', {}).get('captain', [])
        if required_captains:
            valid_captains = [c for c in required_captains if c in available_players]
            if not valid_captains:
                validation['errors'].append("No valid AI captains in player pool")
                validation['is_valid'] = False
            elif len(valid_captains) < len(required_captains):
                validation['warnings'].append(f"Only {len(valid_captains)} of {len(required_captains)} AI captains available")
                ai_rules['variable_locks']['captain'] = valid_captains
                validation['adjustments'].append("Reduced captain pool to available players")
        
        # Check salary feasibility with requirements
        if must_include:
            must_include_df = player_pool[player_pool['Player'].isin(must_include)]
            min_required_salary = must_include_df['Salary'].sum()
            
            if min_required_salary > OptimizerConfig.SALARY_CAP:
                validation['errors'].append(f"AI required players cost ${min_required_salary} (exceeds cap)")
                validation['is_valid'] = False
        
        return validation
    
    @staticmethod
    def validate_field_config_with_ai(field_size: str, num_lineups: int, 
                                      ai_recommendations: Dict) -> Dict:
        """Validate configuration considering AI strategy distribution"""
        
        base_config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size, {})
        
        # Get AI strategy distribution
        ai_distribution = base_config.get('ai_strategy_distribution', {})
        
        # Adjust based on AI consensus
        if ai_recommendations:
            consensus_count = len([1 for rec in ai_recommendations.values() if rec.confidence > 0.8])
            
            if consensus_count == 3:
                # Strong consensus - increase consensus strategy allocation
                ai_distribution['ai_consensus'] = min(0.5, ai_distribution.get('ai_consensus', 0.2) * 2)
            
        return {
            **base_config,
            'ai_strategy_distribution': ai_distribution,
            'ai_enforcement': base_config.get('ai_enforcement', AIEnforcementLevel.MANDATORY)
        }
    
    @staticmethod
    def get_ai_strategy_distribution(field_size: str, num_lineups: int, 
                                     ai_consensus_level: str = 'mixed') -> Dict[StrategyType, int]:
        """Get lineup distribution across AI strategies"""
        
        config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size, {})
        distribution = config.get('ai_strategy_distribution', {})
        
        # Convert percentages to lineup counts
        strategy_counts = {}
        remaining = num_lineups
        
        # Prioritize by consensus level
        if ai_consensus_level == 'high':
            # More consensus lineups
            priority = ['ai_consensus', 'ai_majority', 'ai_mixed', 'ai_contrarian', 'ai_correlation', 'ai_game_theory']
        elif ai_consensus_level == 'low':
            # More diverse lineups
            priority = ['ai_contrarian', 'ai_game_theory', 'ai_correlation', 'ai_mixed', 'ai_majority', 'ai_consensus']
        else:
            # Balanced
            priority = list(distribution.keys())
        
        for strategy in priority:
            if strategy in distribution and remaining > 0:
                count = int(distribution[strategy] * num_lineups)
                count = min(count, remaining)
                if count > 0:
                    # Convert string to StrategyType enum
                    try:
                        strategy_enum = StrategyType[strategy.upper()]
                        strategy_counts[strategy_enum] = count
                        remaining -= count
                    except KeyError:
                        # New AI strategies not in enum yet, use string
                        strategy_counts[strategy] = count
                        remaining -= count
        
        # Distribute remaining lineups
        if remaining > 0:
            # Add to first strategy
            first_strategy = list(strategy_counts.keys())[0] if strategy_counts else StrategyType.AI_MIXED
            if first_strategy in strategy_counts:
                strategy_counts[first_strategy] += remaining
            else:
                strategy_counts[first_strategy] = remaining
        
        return strategy_counts
    
    @staticmethod
    def validate_player_pool_for_ai(df: pd.DataFrame, field_size: str, 
                                    ai_requirements: Dict) -> Dict:
        """Validate player pool can satisfy AI requirements"""
        
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'ai_feasibility': {}
        }
        
        # Standard validation
        base_validation = ConfigValidator.validate_player_pool(df, field_size)
        validation['errors'].extend(base_validation.get('errors', []))
        validation['warnings'].extend(base_validation.get('warnings', []))
        
        # Check AI-specific requirements
        if ai_requirements:
            ai_validation = AIConfigValidator.validate_ai_requirements(ai_requirements, df)
            validation['errors'].extend(ai_validation.get('errors', []))
            validation['warnings'].extend(ai_validation.get('warnings', []))
            validation['ai_feasibility'] = ai_validation
            
            if not ai_validation['is_valid']:
                validation['is_valid'] = False
        
        return validation

# ============================================================================
# STANDARD CONFIG VALIDATOR (Legacy compatibility)
# ============================================================================

class ConfigValidator:
    """Standard configuration validator for backward compatibility"""
    
    @staticmethod
    def validate_field_config(field_size: str, num_lineups: int) -> Dict:
        """Validate and return field configuration"""
        config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size)
        
        if not config:
            st.warning(f"Unknown field size '{field_size}', using large_field defaults")
            config = OptimizerConfig.FIELD_SIZE_CONFIGS['large_field']
        
        # Adjust unique captains if needed
        if num_lineups < config['min_unique_captains']:
            config['min_unique_captains'] = max(1, num_lineups // 2)
        
        return config
    
    @staticmethod
    def validate_player_pool(df: pd.DataFrame, field_size: str) -> Dict:
        """Validate player pool for optimization"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check minimum players
        if len(df) < 12:
            validation['errors'].append(f"Only {len(df)} players available (minimum 12 recommended)")
            if len(df) < 6:
                validation['is_valid'] = False
        
        # Check position distribution
        positions = df['Position'].value_counts()
        if 'QB' not in positions or positions.get('QB', 0) == 0:
            validation['warnings'].append("No QB in player pool")
        
        # Check team distribution
        teams = df['Team'].unique()
        if len(teams) != 2:
            validation['errors'].append(f"Expected 2 teams, found {len(teams)}")
            validation['is_valid'] = False
        
        # Check salary distribution
        avg_salary = df['Salary'].mean()
        if avg_salary < 5000:
            validation['warnings'].append(f"Low average salary (${avg_salary:.0f})")
        
        return validation
    
    @staticmethod
    def get_strategy_distribution(field_size: str, num_lineups: int) -> Dict:
        """Get standard strategy distribution (for fallback)"""
        
        # Use AI strategy distribution if available
        config = OptimizerConfig.FIELD_SIZE_CONFIGS.get(field_size, {})
        if 'ai_strategy_distribution' in config:
            return AIConfigValidator.get_ai_strategy_distribution(field_size, num_lineups)
        
        # Fallback to legacy distribution
        if field_size == 'milly_maker':
            return {
                StrategyType.CONTRARIAN: num_lineups // 2,
                StrategyType.LEVERAGE: num_lineups - (num_lineups // 2)
            }
        else:
            return {
                StrategyType.LEVERAGE: num_lineups
            }

# ============================================================================
# AI SYNTHESIS ENGINE
# ============================================================================

class AISynthesisEngine:
    """Synthesizes recommendations from multiple AI strategists"""
    
    def __init__(self):
        self.logger = get_logger()
        self.synthesis_history = []
    
    def synthesize_recommendations(self, 
                                  game_theory_rec: AIRecommendation,
                                  correlation_rec: AIRecommendation,
                                  contrarian_rec: AIRecommendation) -> Dict:
        """Combine three AI perspectives into unified strategy"""
        
        synthesis = {
            'captain_strategy': {},
            'player_rankings': {},
            'stacking_rules': [],
            'avoidance_rules': [],
            'narrative': "",
            'confidence': 0.0,
            'enforcement_rules': []
        }
        
        # Combine captain recommendations with weighted voting
        all_captains = set()
        captain_votes = {}
        
        for rec, weight in [(game_theory_rec, 0.35), (correlation_rec, 0.35), (contrarian_rec, 0.30)]:
            for captain in rec.captain_targets:
                all_captains.add(captain)
                if captain not in captain_votes:
                    captain_votes[captain] = {'score': 0, 'voters': []}
                captain_votes[captain]['score'] += weight * rec.confidence
                captain_votes[captain]['voters'].append(rec.source_ai)
        
        # Classify captains by consensus level
        for captain, data in captain_votes.items():
            if len(data['voters']) == 3:
                synthesis['captain_strategy'][captain] = 'consensus'
            elif len(data['voters']) == 2:
                synthesis['captain_strategy'][captain] = 'majority'
            else:
                synthesis['captain_strategy'][captain] = data['voters'][0].value
        
        # Rank all players
        all_players = set()
        player_scores = {}
        
        for rec in [game_theory_rec, correlation_rec, contrarian_rec]:
            weight = OptimizerConfig.AI_WEIGHTS[rec.source_ai]
            
            # Score captains
            for captain in rec.captain_targets:
                all_players.add(captain)
                player_scores[captain] = player_scores.get(captain, 0) + weight * 1.5
            
            # Score must plays
            for player in rec.must_play:
                all_players.add(player)
                player_scores[player] = player_scores.get(player, 0) + weight * 1.0
            
            # Penalize never plays
            for player in rec.never_play:
                all_players.add(player)
                player_scores[player] = player_scores.get(player, 0) - weight * 0.5
        
        # Create rankings
        synthesis['player_rankings'] = dict(sorted(player_scores.items(), key=lambda x: x[1], reverse=True))
        
        # Combine stacking rules
        stack_map = {}
        for rec in [game_theory_rec, correlation_rec, contrarian_rec]:
            for stack in rec.stacks:
                stack_key = f"{stack.get('player1')}_{stack.get('player2')}"
                if stack_key not in stack_map:
                    stack_map[stack_key] = {'stack': stack, 'support': []}
                stack_map[stack_key]['support'].append(rec.source_ai)
        
        # Prioritize stacks by support
        for stack_key, data in stack_map.items():
            if len(data['support']) >= 2:
                synthesis['stacking_rules'].append({
                    **data['stack'],
                    'strength': 'strong',
                    'support': data['support']
                })
            else:
                synthesis['stacking_rules'].append({
                    **data['stack'],
                    'strength': 'moderate',
                    'support': data['support']
                })
        
        # Combine avoidance rules
        avoid_players = set()
        for rec in [game_theory_rec, correlation_rec, contrarian_rec]:
            avoid_players.update(rec.never_play)
        synthesis['avoidance_rules'] = list(avoid_players)
        
        # Create narrative
        narratives = []
        if game_theory_rec.narrative:
            narratives.append(f"Game Theory: {game_theory_rec.narrative}")
        if correlation_rec.narrative:
            narratives.append(f"Correlation: {correlation_rec.narrative}")
        if contrarian_rec.narrative:
            narratives.append(f"Contrarian: {contrarian_rec.narrative}")
        synthesis['narrative'] = " | ".join(narratives)
        
        # Calculate combined confidence
        synthesis['confidence'] = (
            game_theory_rec.confidence * OptimizerConfig.AI_WEIGHTS[AIStrategistType.GAME_THEORY] +
            correlation_rec.confidence * OptimizerConfig.AI_WEIGHTS[AIStrategistType.CORRELATION] +
            contrarian_rec.confidence * OptimizerConfig.AI_WEIGHTS[AIStrategistType.CONTRARIAN_NARRATIVE]
        )
        
        # Create enforcement rules
        synthesis['enforcement_rules'] = self._create_enforcement_rules(synthesis)
        
        # Log synthesis
        self.logger.log(f"AI Synthesis complete: {len(synthesis['captain_strategy'])} captains, "
                       f"{len(synthesis['stacking_rules'])} stacks, "
                       f"confidence: {synthesis['confidence']:.2f}", "INFO")
        
        self.synthesis_history.append(synthesis)
        
        return synthesis
    
    def _create_enforcement_rules(self, synthesis: Dict) -> List[Dict]:
        """Convert synthesis into enforcement rules"""
        rules = []
        
        # Captain rules
        consensus_captains = [c for c, level in synthesis['captain_strategy'].items() if level == 'consensus']
        if consensus_captains:
            rules.append({
                'type': 'hard',
                'rule': 'captain_from_list',
                'players': consensus_captains,
                'description': 'Must use consensus captain'
            })
        
        # Stacking rules
        strong_stacks = [s for s in synthesis['stacking_rules'] if s['strength'] == 'strong']
        for stack in strong_stacks[:2]:  # Limit to top 2
            rules.append({
                'type': 'soft',
                'rule': 'include_stack',
                'players': [stack['player1'], stack['player2']],
                'weight': 0.8,
                'description': f"Include stack: {stack['player1']}-{stack['player2']}"
            })
        
        # Avoidance rules
        for player in synthesis['avoidance_rules']:
            rules.append({
                'type': 'soft',
                'rule': 'avoid_player',
                'player': player,
                'weight': 0.7,
                'description': f"Avoid: {player}"
            })
        
        return rules
    
    def get_synthesis_summary(self) -> Dict:
        """Get summary of synthesis history"""
        if not self.synthesis_history:
            return {}
        
        latest = self.synthesis_history[-1]
        return {
            'total_syntheses': len(self.synthesis_history),
            'latest_confidence': latest['confidence'],
            'consensus_captains': len([c for c, l in latest['captain_strategy'].items() if l == 'consensus']),
            'total_captains': len(latest['captain_strategy']),
            'strong_stacks': len([s for s in latest['stacking_rules'] if s['strength'] == 'strong'])
        }

# NFL GPP DUAL-AI OPTIMIZER - PART 3: AI STRATEGISTS
# Game Theory and Correlation AI with GPP Focus

# ============================================================================
# GPP GAME THEORY STRATEGIST
# ============================================================================

class GPPGameTheoryStrategist:
    """AI Strategist 1: GPP Game Theory and ownership leverage"""
    
    def __init__(self, api_manager: ClaudeAPIManager = None):
        self.api_manager = api_manager
    
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field') -> str:
        """Generate GPP-focused game theory prompt"""
        
        bucket_manager = OwnershipBucketManager()
        buckets = bucket_manager.categorize_players(df)
        
        # Get ownership targets for field size
        min_own, max_own = OptimizerConfig.GPP_OWNERSHIP_TARGETS[field_size]
        
        # Get key players by GPP categories
        mega_chalk = df[df['Player'].isin(buckets.get('mega_chalk', []))].nlargest(5, 'Ownership')[
            ['Player', 'Position', 'Team', 'Ownership', 'Projected_Points', 'Salary']]
        
        super_leverage = df[df['Player'].isin(buckets.get('super_leverage', []))].nlargest(
            10, 'Projected_Points')[['Player', 'Position', 'Team', 'Ownership', 'Projected_Points', 'Salary']]
        
        leverage_plays = df[df['Player'].isin(buckets.get('leverage', []))].nlargest(
            10, 'Projected_Points')[['Player', 'Position', 'Team', 'Ownership', 'Projected_Points', 'Salary']]
        
        # Calculate GPP value plays (high ceiling, low ownership)
        df['GPP_Value'] = (df['Projected_Points'] / (df['Salary'] / 1000)) * (20 / (df['Ownership'] + 5))
        top_gpp_values = df.nlargest(10, 'GPP_Value')[['Player', 'Position', 'GPP_Value', 'Ownership']]
        
        # Field size strategy
        field_strategy = {
            'small_field': "Focus on slightly contrarian plays with 80-120% cumulative ownership",
            'medium_field': "Target 70-100% ownership with at least 2 leverage plays",
            'large_field': "Maximize leverage with 60-90% ownership, fade mega chalk",
            'milly_maker': "Ultra-contrarian with 50-80% ownership, zero chalk tolerance"
        }
        
        return f"""
        As an expert GPP tournament strategist, analyze this NFL Showdown slate for {field_size.upper()} tournaments:
        
        GAME CONTEXT:
        Teams: {game_info.get('teams', 'Unknown')}
        Total: {game_info.get('total', 0)} | Spread: {game_info.get('spread', 0)}
        Field Type: {field_size} ({field_strategy.get(field_size, 'Standard GPP')})
        
        MEGA CHALK (35%+ ownership - AVOID IN GPP):
        {mega_chalk.to_string() if not mega_chalk.empty else 'None'}
        
        SUPER LEVERAGE (<5% ownership - GPP GOLD):
        {super_leverage.to_string() if not super_leverage.empty else 'None'}
        
        LEVERAGE PLAYS (5-10% ownership - GPP TARGETS):
        {leverage_plays.to_string() if not leverage_plays.empty else 'None'}
        
        TOP GPP VALUE PLAYS (Ceiling + Low Ownership):
        {top_gpp_values.to_string()}
        
        OWNERSHIP DISTRIBUTION:
        - Mega Chalk (35%+): {len(buckets.get('mega_chalk', []))} players
        - Chalk (20-35%): {len(buckets.get('chalk', []))} players  
        - Pivot (10-20%): {len(buckets.get('pivot', []))} players
        - Leverage (5-10%): {len(buckets.get('leverage', []))} players
        - Super Leverage (<5%): {len(buckets.get('super_leverage', []))} players
        
        GPP STRATEGY REQUIREMENTS:
        - Target cumulative ownership: {min_own}-{max_own}%
        - Prioritize ceiling over floor
        - Identify tournament-winning leverage
        
        Provide GPP-specific recommendations:
        1. Which chalk is actually bad chalk (low ceiling despite ownership)?
        2. Which low-owned players have tournament-winning upside?
        3. Optimal captain leverage plays (sub-15% ownership)?
        4. How to differentiate lineups while maintaining projection?
        5. Contrarian game theory for this specific field size?
        
        Return ONLY valid JSON:
        {{
            "gpp_captain_targets": ["leverage_captain1", "leverage_captain2", "leverage_captain3"],
            "super_leverage_plays": ["gem1", "gem2", "gem3"],
            "must_fade_chalk": ["overowned1", "overowned2"],
            "tournament_winners": ["upside1", "upside2"],
            "gpp_construction_rules": {{
                "max_chalk_players": 1,
                "min_sub10_ownership": 2,
                "target_total_ownership": {{"min": {min_own}, "max": {max_own}}},
                "required_leverage_captain": true
            }},
            "leverage_stacks": [
                {{"player1": "low_own1", "player2": "low_own2", "combined_ownership": 15}}
            ],
            "differentiation_strategy": "specific approach for uniqueness",
            "field_size_adjustments": "specific tweaks for {field_size}",
            "gpp_insights": [
                "Key tournament insight 1",
                "Key tournament insight 2"
            ],
            "win_probability_boosters": ["factor1", "factor2"],
            "confidence_score": 0.85
        }}
        """
    
    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Parse and validate GPP game theory response"""
        try:
            data = json.loads(response) if response else {}
        except:
            data = {}
        
        # Extract GPP construction rules
        rules = data.get('gpp_construction_rules', {})
        max_chalk = rules.get('max_chalk_players', 1)
        min_leverage = rules.get('min_sub10_ownership', 2)
        
        # Determine GPP strategy weights based on field size
        if field_size == 'milly_maker':
            strategy_weights = {
                StrategyType.SUPER_CONTRARIAN: 0.4,
                StrategyType.LEVERAGE: 0.35,
                StrategyType.CONTRARIAN: 0.2,
                StrategyType.STARS_SCRUBS: 0.05,
                StrategyType.CORRELATION: 0.0,
                StrategyType.GAME_STACK: 0.0
            }
        elif field_size == 'large_field':
            strategy_weights = {
                StrategyType.LEVERAGE: 0.35,
                StrategyType.CONTRARIAN: 0.25,
                StrategyType.SUPER_CONTRARIAN: 0.15,
                StrategyType.GAME_STACK: 0.15,
                StrategyType.STARS_SCRUBS: 0.1,
                StrategyType.CORRELATION: 0.0
            }
        else:
            strategy_weights = {
                StrategyType.LEVERAGE: 0.25,
                StrategyType.CONTRARIAN: 0.15,
                StrategyType.GAME_STACK: 0.25,
                StrategyType.CORRELATION: 0.2,
                StrategyType.STARS_SCRUBS: 0.15,
                StrategyType.SUPER_CONTRARIAN: 0.0
            }
        
        # GPP-specific rules
        gpp_rules = {
            'max_chalk': max_chalk,
            'min_leverage': min_leverage,
            'force_leverage_captain': rules.get('required_leverage_captain', True),
            'ownership_targets': rules.get('target_total_ownership', 
                                         OptimizerConfig.GPP_OWNERSHIP_TARGETS[field_size])
        }
        
        return AIRecommendation(
            strategist_name="GPP Game Theory AI",
            confidence=data.get('confidence_score', 0.80),
            captain_targets=data.get('gpp_captain_targets', []),
            stacks=[{'player1': s.get('player1'), 'player2': s.get('player2')} 
                   for s in data.get('leverage_stacks', []) if isinstance(s, dict)],
            fades=data.get('must_fade_chalk', []),
            boosts=data.get('super_leverage_plays', []) + data.get('tournament_winners', []),
            strategy_weights=strategy_weights,
            key_insights=data.get('gpp_insights', ["Using default GPP strategy"]),
            gpp_specific_rules=gpp_rules
        )

# ============================================================================
# GPP CORRELATION STRATEGIST
# ============================================================================

class GPPCorrelationStrategist:
    """AI Strategist 2: GPP correlation and game stack specialist"""
    
    def __init__(self, api_manager: ClaudeAPIManager = None):
        self.api_manager = api_manager
    
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field') -> str:
        """Generate GPP correlation analysis prompt"""
        
        teams = df['Team'].unique()[:2]
        team_breakdown = {}
        
        for team in teams:
            team_df = df[df['Team'] == team]
            
            # Get players with ownership context for GPP
            qbs = team_df[team_df['Position'] == 'QB'][
                ['Player', 'Salary', 'Projected_Points', 'Ownership']].to_dict('records')
            pass_catchers = team_df[team_df['Position'].isin(['WR', 'TE'])][
                ['Player', 'Position', 'Salary', 'Projected_Points', 'Ownership']
            ].sort_values('Ownership').to_dict('records')  # Sort by ownership for GPP
            rbs = team_df[team_df['Position'] == 'RB'][
                ['Player', 'Salary', 'Projected_Points', 'Ownership']].to_dict('records')
            
            team_breakdown[team] = {
                'QB': qbs,
                'Pass_Catchers': pass_catchers[:6],  # Include more for GPP variety
                'RB': rbs[:3]
            }
        
        # Determine correlation strategy based on total
        if game_info.get('total', 48) > 54:
            correlation_focus = "SHOOTOUT: Prioritize game stacks with 4+ players from game"
        elif game_info.get('total', 48) > 50:
            correlation_focus = "MODERATE: Mix of game stacks and single team stacks"
        else:
            correlation_focus = "LOW TOTAL: Focus on single team stacks, leverage RBs"
        
        return f"""
        As an expert GPP Correlation strategist, identify tournament-winning stacks for {field_size} GPPs:
        
        GAME CONTEXT:
        Teams: {game_info.get('teams', 'Unknown')}
        Total: {game_info.get('total', 0)} | Spread: {game_info.get('spread', 0)}
        Correlation Strategy: {correlation_focus}
        
        TEAM ROSTERS WITH OWNERSHIP:
        {json.dumps(team_breakdown, indent=2)}
        
        GPP CORRELATION PRIORITIES:
        1. Leverage stacks (combined ownership < 30%)
        2. Game stacks for ceiling in projected shootouts
        3. Contrarian bring-backs (opposing team correlation)
        4. Low-owned QB stacks (QB < 20% ownership)
        5. Secondary stacks that differentiate lineups
        6. Negative correlations to avoid (RB-RB, etc.)
        
        Field Size Considerations:
        - Small field: Can use one higher-owned stack
        - Large field: Need multiple low-owned correlations
        - Milly Maker: Only super-leverage stacks
        
        Return ONLY valid JSON:
        {{
            "primary_leverage_stacks": [
                {{"qb": "name", "receiver": "name", "correlation": 0.6, "combined_ownership": 20, "gpp_score": 85}}
            ],
            "game_stacks": [
                {{"players": ["p1", "p2", "p3", "p4"], "scenario": "shootout", "total_ownership": 60, "ceiling_boost": "high"}}
            ],
            "super_leverage_stacks": [
                {{"player1": "sub5own", "player2": "sub10own", "combined_ownership": 12, "tournament_equity": "elite"}}
            ],
            "contrarian_bringbacks": [
                {{"primary": "team1_qb", "bringback": "team2_wr2", "leverage": "high"}}
            ],
            "avoid_correlations": [
                {{"player1": "rb1", "player2": "rb2", "reason": "negative correlation"}}
            ],
            "gpp_game_script": {{
                "most_likely": "scenario",
                "leverage_scenario": "contrarian_scenario",
                "boost_players": ["player1", "player2"],
                "fade_players": ["chalk1", "chalk2"]
            }},
            "field_specific_stacks": {{
                "small_field": ["balanced_stack"],
                "large_field": ["leverage_stack"],
                "milly_maker": ["super_leverage_stack"]
            }},
            "correlation_insights": [
                "GPP correlation insight 1",
                "GPP correlation insight 2"
            ],
            "stack_differentiation": "how to make stacks unique",
            "confidence": 0.85
        }}
        """
    
    def parse_response(self, response: str, df: pd.DataFrame, field_size: str) -> AIRecommendation:
        """Parse and validate GPP correlation response"""
        try:
            data = json.loads(response) if response else {}
        except:
            data = {}
        
        # Build comprehensive GPP stacks list
        all_stacks = []
        
        # Primary leverage stacks
        for stack in data.get('primary_leverage_stacks', []):
            if isinstance(stack, dict):
                all_stacks.append({
                    'player1': stack.get('qb'),
                    'player2': stack.get('receiver'),
                    'type': 'leverage_primary',
                    'correlation': stack.get('correlation', 0.5),
                    'gpp_score': stack.get('gpp_score', 50)
                })
        
        # Game stacks
        for stack in data.get('game_stacks', []):
            if isinstance(stack, dict):
                players = stack.get('players', [])
                if len(players) >= 2:
                    all_stacks.append({
                        'player1': players[0],
                        'player2': players[1],
                        'type': 'game_stack',
                        'correlation': 0.35,
                        'additional_players': players[2:] if len(players) > 2 else []
                    })
        
        # Super leverage stacks
        for stack in data.get('super_leverage_stacks', []):
            if isinstance(stack, dict):
                all_stacks.append({
                    'player1': stack.get('player1'),
                    'player2': stack.get('player2'),
                    'type': 'super_leverage',
                    'correlation': 0.3,
                    'tournament_equity': stack.get('tournament_equity', 'high')
                })
        
        # Determine strategy weights based on game script and field size
        game_script = data.get('gpp_game_script', {}).get('most_likely', 'balanced')
        
        if field_size == 'milly_maker':
            strategy_weights = {
                StrategyType.SUPER_CONTRARIAN: 0.35,
                StrategyType.LEVERAGE: 0.35,
                StrategyType.GAME_STACK: 0.2 if game_script == 'shootout' else 0.1,
                StrategyType.STARS_SCRUBS: 0.1,
                StrategyType.CONTRARIAN: 0.0,
                StrategyType.CORRELATION: 0.0
            }
        elif game_script == 'shootout':
            strategy_weights = {
                StrategyType.GAME_STACK: 0.35,
                StrategyType.CORRELATION: 0.25,
                StrategyType.LEVERAGE: 0.2,
                StrategyType.CONTRARIAN: 0.15,
                StrategyType.STARS_SCRUBS: 0.05,
                StrategyType.SUPER_CONTRARIAN: 0.0
            }
        else:
            strategy_weights = {
                StrategyType.LEVERAGE: 0.3,
                StrategyType.CONTRARIAN: 0.25,
                StrategyType.CORRELATION: 0.2,
                StrategyType.GAME_STACK: 0.15,
                StrategyType.STARS_SCRUBS: 0.1,
                StrategyType.SUPER_CONTRARIAN: 0.0
            }
        
        # Extract captain targets from low-owned QBs
        captain_targets = []
        for stack in data.get('primary_leverage_stacks', []):
            if isinstance(stack, dict) and stack.get('qb'):
                qb = stack['qb']
                if qb in df['Player'].values:
                    qb_own = df[df['Player'] == qb]['Ownership'].values[0]
                    if qb_own < 20:  # Only low-owned QBs for GPP
                        captain_targets.append(qb)
        
        # GPP-specific rules
        gpp_rules = {
            'leverage_scenario': data.get('gpp_game_script', {}).get('leverage_scenario'),
            'field_stacks': data.get('field_specific_stacks', {}).get(field_size, []),
            'avoid_together': data.get('avoid_correlations', [])
        }
        
        return AIRecommendation(
            strategist_name="GPP Correlation AI",
            confidence=data.get('confidence', 0.80),
            captain_targets=captain_targets,
            stacks=all_stacks,
            fades=data.get('gpp_game_script', {}).get('fade_players', []),
            boosts=data.get('gpp_game_script', {}).get('boost_players', []),
            strategy_weights=strategy_weights,
            key_insights=data.get('correlation_insights', ["Using GPP correlation strategy"]),
            gpp_specific_rules=gpp_rules
        )

# NFL GPP DUAL-AI OPTIMIZER - PART 4: MAIN OPTIMIZER & LINEUP GENERATION (COMPLETE CORRECTED v5.4)
# With Automatic Ownership Adjustment and Relaxed Mode Fallback

# ============================================================================
# GPP DUAL AI OPTIMIZER
# ============================================================================

class GPPDualAIOptimizer:
    """Main GPP optimizer with dual AI strategy integration, auto-adjustment, and fallback mechanisms"""
    
    def __init__(self, df: pd.DataFrame, game_info: Dict, field_size: str = 'large_field', 
                 api_manager: ClaudeAPIManager = None):
        self.df = df
        self.game_info = game_info
        self.field_size = field_size
        self.api_manager = api_manager
        self.game_theory_ai = GPPGameTheoryStrategist(api_manager)
        self.correlation_ai = GPPCorrelationStrategist(api_manager)
        self.bucket_manager = OwnershipBucketManager()
        self.pivot_generator = GPPCaptainPivotGenerator()
        self.correlation_engine = GPPCorrelationEngine()
        self.tournament_sim = GPPTournamentSimulator()
        self.optimization_logger = []
        self.field_config = None
        self.logger = get_logger()
        self.perf_monitor = get_performance_monitor()
    
    def adjust_ownership_distribution(self, players: List[str], points: Dict, 
                                     salaries: Dict, positions: Dict, 
                                     original_ownership: Dict) -> Dict:
        """Automatically adjust ownership when distribution is too narrow"""
        ownership_values = list(original_ownership.values())
        max_ownership = max(ownership_values) if ownership_values else 5
        min_ownership = min(ownership_values) if ownership_values else 5
        
        # Check if adjustment is needed
        if max_ownership - min_ownership >= 15:
            return original_ownership  # Distribution is fine
        
        st.warning(f"⚠️ Ownership range too narrow ({min_ownership:.1f}% - {max_ownership:.1f}%). Redistributing for GPP...")
        self.logger.log(f"Redistributing ownership from {min_ownership:.1f}-{max_ownership:.1f}", "WARNING")
        
        # Calculate percentiles for projections
        proj_values = [points.get(p, 10) for p in players]
        if not proj_values:
            return original_ownership
        
        proj_median = np.percentile(proj_values, 50)
        proj_75th = np.percentile(proj_values, 75)
        proj_25th = np.percentile(proj_values, 25)
        proj_90th = np.percentile(proj_values, 90)
        
        adjusted_ownership = {}
        
        for player in players:
            proj = points.get(player, 10)
            salary = salaries.get(player, 5000)
            position = positions.get(player, 'FLEX')
            
            # Base ownership on projection tier
            if proj >= proj_90th:
                base_own = 35 + np.random.uniform(-5, 10)
            elif proj >= proj_75th:
                base_own = 25 + np.random.uniform(-5, 8)
            elif proj >= proj_median:
                base_own = 15 + np.random.uniform(-5, 5)
            elif proj >= proj_25th:
                base_own = 8 + np.random.uniform(-3, 3)
            else:
                base_own = 4 + np.random.uniform(-2, 2)
            
            # Position adjustments
            if position == 'QB':
                base_own *= 1.4  # QBs typically higher owned
            elif position == 'RB' and salary > 8000:
                base_own *= 1.3  # Expensive RBs popular
            elif position == 'WR' and salary > 8500:
                base_own *= 1.25  # Elite WRs popular
            elif position == 'TE' and proj >= proj_75th:
                base_own *= 1.15  # Good TEs get owned
            elif position == 'DST':
                base_own *= 0.7  # DSTs usually lower
            elif position == 'K':
                base_own *= 0.6  # Kickers lowest
            
            # Salary tier adjustments
            if salary >= 10000:
                base_own *= 1.3
            elif salary >= 8500:
                base_own *= 1.15
            elif salary <= 3500:
                base_own *= 0.8
            elif salary <= 3000:
                base_own *= 0.6
            
            # Value play boost (high proj/salary ratio)
            if salary > 0:
                value = (proj / (salary / 1000))
                if value > 3.0:  # Great value
                    base_own *= 1.2
            
            # Cap ownership between realistic bounds
            adjusted_ownership[player] = max(1.5, min(50, base_own))
        
        # Show the new distribution
        new_values = list(adjusted_ownership.values())
        st.write("**Adjusted Ownership Distribution:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("<10%", len([v for v in new_values if v < 10]))
        with col2:
            st.metric("10-20%", len([v for v in new_values if 10 <= v < 20]))
        with col3:
            st.metric("20-30%", len([v for v in new_values if 20 <= v < 30]))
        with col4:
            st.metric(">30%", len([v for v in new_values if v >= 30]))
        
        return adjusted_ownership
    
    def get_fallback_recommendations(self) -> Dict:
        """Generate fallback recommendations when AI is not available"""
        self.logger.log("Using fallback recommendations", "WARNING")
        st.info("📊 Using statistical analysis for lineup generation (no AI input detected)")
        
        df = self.df.copy()
        
        # Calculate value metrics
        df['Value_Score'] = df['Projected_Points'] / (df['Salary'] / 1000)
        df['GPP_Score'] = df['Value_Score'] * (30 / (df['Ownership'] + 10))
        
        # Find captain candidates
        captain_candidates = []
        
        # Strategy 1: Low-owned high projection
        low_own_high_proj = df[(df['Ownership'] < 20) & (df['Projected_Points'] > df['Projected_Points'].quantile(0.6))]
        if not low_own_high_proj.empty:
            captain_candidates.extend(low_own_high_proj.nlargest(5, 'Projected_Points')['Player'].tolist())
        
        # Strategy 2: Best GPP value
        captain_candidates.extend(df.nlargest(7, 'GPP_Score')['Player'].tolist())
        
        # Strategy 3: QBs
        captain_candidates.extend(df[df['Position'] == 'QB']['Player'].tolist())
        
        # Strategy 4: Top projected
        captain_candidates.extend(df.nlargest(5, 'Projected_Points')['Player'].tolist())
        
        # Remove duplicates
        captain_candidates = list(dict.fromkeys(captain_candidates))
        if not captain_candidates:
            captain_candidates = df.nlargest(10, 'Projected_Points')['Player'].tolist()
        
        captain_candidates = captain_candidates[:15]
        
        # Create captain scores
        captain_scores = {}
        for captain in captain_candidates:
            if captain in df['Player'].values:
                row = df[df['Player'] == captain].iloc[0]
                score = row['GPP_Score'] / 100
                captain_scores[captain] = min(1.0, max(0.1, score))
        
        # Identify stacks
        stacks = []
        teams = df['Team'].unique()
        
        for team in teams[:2]:
            team_df = df[df['Team'] == team]
            qbs = team_df[team_df['Position'] == 'QB']['Player'].tolist()
            pass_catchers = team_df[team_df['Position'].isin(['WR', 'TE'])].nlargest(4, 'Projected_Points')['Player'].tolist()
            
            for qb in qbs:
                for pc in pass_catchers:
                    stacks.append({'player1': qb, 'player2': pc, 'type': 'qb_stack'})
        
        # Identify fades and boosts
        high_owned = df[df['Ownership'] > 25]
        fades = high_owned.nsmallest(3, 'Value_Score')['Player'].tolist() if not high_owned.empty else []
        
        low_owned = df[df['Ownership'] < 15]
        boosts = low_owned.nlargest(5, 'Projected_Points')['Player'].tolist() if not low_owned.empty else []
        
        strategy_weights = ConfigValidator.get_strategy_distribution(self.field_size, 20)
        
        return {
            'captain_scores': captain_scores,
            'strategy_weights': strategy_weights,
            'consensus_fades': fades[:3],
            'all_boosts': boosts[:5],
            'combined_stacks': stacks[:15],
            'confidence': 0.5,
            'insights': ['Using statistical analysis (no AI input)', 
                        f'Found {len(captain_candidates)} captain candidates'],
            'gpp_rules': {},
            'field_size': self.field_size
        }
    
    def diagnose_optimization_issues(self, df: pd.DataFrame, field_size: str, 
                                    field_config: Dict, salaries: Dict, 
                                    ownership: Dict, points: Dict) -> Dict:
        """Comprehensive diagnostics for optimization failures"""
        st.warning("🔍 Running optimization diagnostics...")
        self.logger.log("Running optimization diagnostics", "WARNING")
        
        issues = []
        warnings = []
        
        # Check player count
        if len(df) < 6:
            issues.append(f"Only {len(df)} players available - need at least 6")
        elif len(df) < 12:
            warnings.append(f"Limited player pool ({len(df)} players)")
        
        # Check salary feasibility
        sorted_salaries = sorted(salaries.values())
        if len(sorted_salaries) >= 6:
            min_lineup_salary = sorted_salaries[0] * 1.5 + sum(sorted_salaries[1:6])
            max_lineup_salary = sorted_salaries[-1] * 1.5 + sum(sorted_salaries[-5:])
            
            if min_lineup_salary > OptimizerConfig.SALARY_CAP:
                issues.append(f"Minimum salary (${min_lineup_salary:.0f}) exceeds cap")
            
            st.write(f"**Salary Range:** ${min_lineup_salary:.0f} - ${max_lineup_salary:.0f}")
        
        # Check ownership distribution
        ownership_values = list(ownership.values())
        st.write("**Ownership Distribution:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("<5%", len([v for v in ownership_values if v < 5]))
        with col2:
            st.metric("5-10%", len([v for v in ownership_values if 5 <= v < 10]))
        with col3:
            st.metric("10-30%", len([v for v in ownership_values if 10 <= v < 30]))
        with col4:
            st.metric(">30%", len([v for v in ownership_values if v > 30]))
        
        # Test basic lineup
        st.write("**Testing Basic Lineup Generation...**")
        test_result = self.test_basic_lineup(df, salaries, points, ownership)
        
        if test_result['success']:
            st.success("✅ Basic lineup generation successful")
            if test_result.get('lineup'):
                st.write(f"Test Captain: {test_result['lineup']['captain']}")
                st.write(f"Test Ownership: {test_result['lineup']['ownership']:.1f}%")
        else:
            issues.append("Basic lineup generation failed")
            st.error(f"Reason: {test_result.get('reason', 'Unknown')}")
        
        if issues:
            with st.expander("❌ Critical Issues", expanded=True):
                for issue in issues:
                    st.error(issue)
        
        if warnings:
            with st.expander("⚠️ Warnings"):
                for warning in warnings:
                    st.warning(warning)
        
        return {'has_critical_issues': len(issues) > 0}
    
    def test_basic_lineup(self, df: pd.DataFrame, salaries: Dict, 
                         points: Dict, ownership: Dict) -> Dict:
        """Test if a basic lineup can be generated"""
        players = list(salaries.keys())
        
        if len(players) < 6:
            return {'success': False, 'reason': 'Not enough players'}
        
        try:
            import pulp
            
            model = pulp.LpProblem("Test_Basic", pulp.LpMaximize)
            flex = pulp.LpVariable.dicts("Flex", players, cat='Binary')
            captain = pulp.LpVariable.dicts("Captain", players, cat='Binary')
            
            model += pulp.lpSum([
                points.get(p, 0) * flex[p] + 1.5 * points.get(p, 0) * captain[p]
                for p in players
            ])
            
            model += pulp.lpSum(captain.values()) == 1
            model += pulp.lpSum(flex.values()) == 5
            
            for p in players:
                model += flex[p] + captain[p] <= 1
            
            model += pulp.lpSum([
                salaries[p] * flex[p] + 1.5 * salaries[p] * captain[p]
                for p in players
            ]) <= OptimizerConfig.SALARY_CAP + 500
            
            model.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if pulp.LpStatus[model.status] == 'Optimal':
                captain_pick = [p for p in players if captain[p].value() == 1][0]
                flex_picks = [p for p in players if flex[p].value() == 1]
                
                total_salary = salaries[captain_pick] * 1.5 + sum(salaries[p] for p in flex_picks)
                total_ownership = ownership.get(captain_pick, 5) * 1.5 + sum(ownership.get(p, 5) for p in flex_picks)
                
                return {
                    'success': True,
                    'lineup': {
                        'captain': captain_pick,
                        'flex': flex_picks,
                        'salary': total_salary,
                        'ownership': total_ownership
                    }
                }
            else:
                return {'success': False, 'reason': f'Solver status: {pulp.LpStatus[model.status]}'}
                
        except Exception as e:
            return {'success': False, 'reason': str(e)}
    
    def safe_api_call(self, prompt: str, strategist_name: str, fallback_response: str = '{}') -> str:
        """Safely make API calls with fallback"""
        if not self.api_manager or not self.api_manager.client:
            self.logger.log(f"{strategist_name}: No API client", "WARNING")
            return fallback_response
        
        try:
            response = self.api_manager.get_ai_response(prompt)
            json.loads(response)
            return response
        except Exception as e:
            self.logger.log(f"{strategist_name}: {e}", "ERROR")
            return fallback_response
    
    def get_ai_strategies(self, use_api: bool = True) -> Tuple[AIRecommendation, AIRecommendation]:
        """Get strategies from both GPP AIs"""
        self.logger.log("Getting AI strategies", "INFO")
        
        if use_api and self.api_manager and self.api_manager.client:
            with st.spinner("🎯 GPP Game Theory AI analyzing..."):
                gt_prompt = self.game_theory_ai.generate_prompt(self.df, self.game_info, self.field_size)
                gt_response = self.safe_api_call(gt_prompt, "Game Theory AI")
            
            with st.spinner("🔗 GPP Correlation AI analyzing..."):
                corr_prompt = self.correlation_ai.generate_prompt(self.df, self.game_info, self.field_size)
                corr_response = self.safe_api_call(corr_prompt, "Correlation AI")
        else:
            st.subheader("📝 Manual AI Strategy Input")
            
            tab1, tab2 = st.tabs(["🎯 Game Theory AI", "🔗 Correlation AI"])
            
            with tab1:
                with st.expander("View GPP Game Theory Prompt"):
                    st.text_area("Copy this prompt:", 
                               value=self.game_theory_ai.generate_prompt(self.df, self.game_info, self.field_size),
                               height=300, key="gt_prompt_display")
                gt_response = st.text_area("Paste Game Theory Response (JSON):", 
                                          height=200, key="gt_manual_input", value='{}')
            
            with tab2:
                with st.expander("View GPP Correlation Prompt"):
                    st.text_area("Copy this prompt:", 
                               value=self.correlation_ai.generate_prompt(self.df, self.game_info, self.field_size),
                               height=300, key="corr_prompt_display")
                corr_response = st.text_area("Paste Correlation Response (JSON):", 
                                            height=200, key="corr_manual_input", value='{}')
        
        rec1 = self.game_theory_ai.parse_response(gt_response, self.df, self.field_size)
        rec2 = self.correlation_ai.parse_response(corr_response, self.df, self.field_size)
        
        return rec1, rec2
    
    def combine_gpp_recommendations(self, rec1: AIRecommendation, rec2: AIRecommendation) -> Dict:
        """Combine recommendations with GPP-specific weighting"""
        self.logger.log("Combining AI recommendations", "DEBUG")
        
        total_confidence = rec1.confidence + rec2.confidence
        w1 = rec1.confidence / total_confidence if total_confidence > 0 else 0.5
        w2 = rec2.confidence / total_confidence if total_confidence > 0 else 0.5
        
        all_captains = set(rec1.captain_targets + rec2.captain_targets)
        captain_scores = {}
        ownership_dict = self.df.set_index('Player')['Ownership'].to_dict()
        
        for captain in all_captains:
            score = 0
            ownership = ownership_dict.get(captain, 5)
            
            if captain in rec1.captain_targets:
                score += w1
            if captain in rec2.captain_targets:
                score += w2
            
            if ownership < 5:
                score *= 2.0
            elif ownership < 10:
                score *= 1.5
            elif ownership < 15:
                score *= 1.2
            elif ownership > 30:
                score *= 0.5
            elif ownership > 20:
                score *= 0.7
            
            captain_scores[captain] = score
        
        strategy_weights = ConfigValidator.get_strategy_distribution(self.field_size, 20)
        
        combined = {
            'captain_scores': captain_scores,
            'strategy_weights': strategy_weights,
            'consensus_fades': list(set(rec1.fades) & set(rec2.fades)) if rec1.fades and rec2.fades else [],
            'all_boosts': list(set(rec1.boosts) | set(rec2.boosts)) if rec1.boosts or rec2.boosts else [],
            'combined_stacks': rec1.stacks + rec2.stacks,
            'confidence': (rec1.confidence + rec2.confidence) / 2,
            'insights': rec1.key_insights + rec2.key_insights,
            'field_size': self.field_size
        }
        
        self.logger.log(f"Combined: {len(captain_scores)} captains, {len(combined['combined_stacks'])} stacks", "DEBUG")
        
        return combined
    
    def validate_lineup_constraints(self, captain: str, flex_players: List[str], 
                                  salaries: Dict, ownership: Dict, teams: Dict) -> Tuple[bool, str]:
        """Validate lineup constraints"""
        if not captain or len(flex_players) != 5:
            return False, "Invalid structure"
        
        all_players = [captain] + flex_players
        if len(set(all_players)) != 6:
            return False, "Duplicate players"
        
        total_salary = salaries.get(captain, 0) * 1.5 + sum(salaries.get(p, 0) for p in flex_players)
        if total_salary > OptimizerConfig.SALARY_CAP:
            return False, f"Salary exceeds cap"
        
        team_counts = {}
        for player in all_players:
            team = teams.get(player, 'Unknown')
            team_counts[team] = team_counts.get(team, 0) + 1
        
        for team, count in team_counts.items():
            if count > OptimizerConfig.MAX_PLAYERS_PER_TEAM:
                return False, f"Too many from {team}"
        
        return True, "Valid"
    
    def generate_relaxed_lineups(self, num_lineups: int, players: List[str], 
                                salaries: Dict, points: Dict, ownership: Dict, 
                                teams: Dict, used_captains: Set[str]) -> List[Dict]:
        """Generate lineups with minimal constraints as fallback"""
        st.warning("Standard optimization failed. Trying relaxed mode...")
        self.logger.log("Attempting relaxed mode optimization", "WARNING")
        
        relaxed_lineups = []
        
        for i in range(min(num_lineups, 20)):
            try:
                import pulp
                
                model = pulp.LpProblem(f"Relaxed_{i}", pulp.LpMaximize)
                flex = pulp.LpVariable.dicts("Flex", players, cat='Binary')
                captain = pulp.LpVariable.dicts("Captain", players, cat='Binary')
                
                # Simple objective
                model += pulp.lpSum([
                    points[p] * flex[p] + 1.5 * points[p] * captain[p]
                    for p in players
                ])
                
                # Basic constraints
                model += pulp.lpSum(captain.values()) == 1
                model += pulp.lpSum(flex.values()) == 5
                
                for p in players:
                    model += flex[p] + captain[p] <= 1
                
                # Salary
                model += pulp.lpSum([
                    salaries[p] * flex[p] + 1.5 * salaries[p] * captain[p]
                    for p in players
                ]) <= OptimizerConfig.SALARY_CAP
                
                # Team limits
                for team in set(teams.values()):
                    team_players = [p for p in players if teams.get(p) == team]
                    if team_players:
                        model += pulp.lpSum([flex[p] + captain[p] for p in team_players]) <= OptimizerConfig.MAX_PLAYERS_PER_TEAM
                
                # Unique captains
                if used_captains and len(used_captains) < len(players):
                    for prev_captain in used_captains:
                        if prev_captain in players:
                            model += captain[prev_captain] == 0
                
                model.solve(pulp.PULP_CBC_CMD(msg=0))
                
                if pulp.LpStatus[model.status] == 'Optimal':
                    captain_pick = None
                    flex_picks = []
                    
                    for p in players:
                        if captain[p].value() == 1:
                            captain_pick = p
                            used_captains.add(p)
                        if flex[p].value() == 1:
                            flex_picks.append(p)
                    
                    if captain_pick and len(flex_picks) == 5:
                        total_salary = sum(salaries[p] for p in flex_picks) + 1.5 * salaries[captain_pick]
                        total_proj = sum(points[p] for p in flex_picks) + 1.5 * points[captain_pick]
                        total_ownership = ownership.get(captain_pick, 5) * 1.5 + sum(ownership.get(p, 5) for p in flex_picks)
                        
                        relaxed_lineups.append({
                            'Lineup': len(relaxed_lineups) + 1,
                            'Strategy': 'relaxed',
                            'Captain': captain_pick,
                            'Captain_Own%': ownership.get(captain_pick, 5),
                            'FLEX': flex_picks,
                            'Projected': round(total_proj, 2),
                            'Salary': int(total_salary),
                            'Salary_Remaining': int(OptimizerConfig.SALARY_CAP - total_salary),
                            'Total_Ownership': round(total_ownership, 1),
                            'Ownership_Tier': '🔧 Relaxed',
                            'GPP_Summary': 'Relaxed constraints',
                            'Leverage_Score': 0,
                            'Has_Stack': False,
                            'Stack_Details': 'None',
                            'Field_Size': self.field_size,
                            'Unique_Captain': True
                        })
                        
                        self.logger.log(f"Relaxed lineup {i+1} generated", "INFO")
                        
            except Exception as e:
                self.logger.log(f"Relaxed mode error: {e}", "ERROR")
                continue
        
        if relaxed_lineups:
            st.info(f"✅ Generated {len(relaxed_lineups)} lineups using relaxed constraints")
        
        return relaxed_lineups
    
    def generate_gpp_lineups(self, num_lineups: int, rec1: AIRecommendation, 
                            rec2: AIRecommendation, force_unique_captains: bool = True) -> pd.DataFrame:
        """Generate GPP-optimized lineups with auto-adjustment and fallback"""
        
        # Start monitoring
        self.perf_monitor.start_timer("total_optimization")
        self.logger.log_optimization_start(num_lineups, self.field_size, {
            'force_unique_captains': force_unique_captains
        })
        
        # Validate configuration
        self.logger.log("Validating configuration", "DEBUG")
        self.field_config = ConfigValidator.validate_field_config(self.field_size, num_lineups)
        self.logger.log(f"Configuration validated: {self.field_config}", "DEBUG")
        
        # Validate player pool
        self.logger.log("Validating player pool", "DEBUG")
        pool_validation = ConfigValidator.validate_player_pool(self.df, self.field_size)
        
        if not pool_validation['is_valid']:
            for error in pool_validation['errors']:
                st.error(f"Player pool issue: {error}")
            return pd.DataFrame()
        
        for warning in pool_validation.get('warnings', []):
            st.warning(warning)
        
        # Display config
        config_msg = (f"Optimization config for {self.field_size}: "
                     f"Max chalk={self.field_config['max_chalk_players']}, "
                     f"Min leverage={self.field_config['min_leverage_players']}, "
                     f"Target ownership={self.field_config['min_total_ownership']}-{self.field_config['max_total_ownership']}%")
        st.info(config_msg)
        self.logger.log(config_msg, "INFO")
        
        # Combine recommendations
        combined = self.combine_gpp_recommendations(rec1, rec2)
        
        # Check for valid recommendations
        if not combined.get('captain_scores') or len(combined['captain_scores']) == 0:
            st.warning("No AI recommendations available - using statistical analysis")
            combined = self.get_fallback_recommendations()
        
        # Get data
        players = self.df['Player'].tolist()
        positions = self.df.set_index('Player')['Position'].to_dict()
        teams = self.df.set_index('Player')['Team'].to_dict()
        salaries = self.df.set_index('Player')['Salary'].to_dict()
        points = self.df.set_index('Player')['Projected_Points'].to_dict()
        ownership = self.df.set_index('Player')['Ownership'].to_dict()
        
        # Check and adjust ownership if needed
        ownership = self.adjust_ownership_distribution(players, points, salaries, positions, ownership)
        
        # Update field config based on adjusted ownership
        max_possible_ownership = 1.5 * max(ownership.values()) + 5 * np.mean(list(ownership.values()))
        if max_possible_ownership < self.field_config['min_total_ownership']:
            st.info(f"📊 Adjusting targets for player pool (max possible: {max_possible_ownership:.0f}%)")
            self.field_config['min_total_ownership'] = max(30, max_possible_ownership - 30)
            self.field_config['max_total_ownership'] = min(200, max_possible_ownership + 50)
        
        # Check minimum requirements
        if len(players) < 6:
            st.error(f"Not enough players ({len(players)})")
            self.diagnose_optimization_issues(self.df, self.field_size, self.field_config,
                                             salaries, ownership, points)
            return pd.DataFrame()
        
        # Apply adjustments
        adjusted_points = points.copy()
        for player in combined.get('consensus_fades', []):
            if player in adjusted_points and ownership.get(player, 5) > 30:
                adjusted_points[player] *= 0.80
        
        for player in combined.get('all_boosts', []):
            if player in adjusted_points and ownership.get(player, 5) < 10:
                adjusted_points[player] *= 1.20
        
        # Get strategy distribution
        strategy_distribution = ConfigValidator.get_strategy_distribution(self.field_size, num_lineups)
        self.logger.log(f"Strategy distribution: {strategy_distribution}", "DEBUG")
        
        all_lineups = []
        used_captains = set()
        lineup_num = 0
        failed_attempts = 0
        max_failures = 200
        
        # Generate lineups by strategy
        for strategy, count in strategy_distribution.items():
            self.logger.log(f"Generating {count} lineups for strategy: {strategy.value}", "INFO")
            
            strategy_attempts = 0
            strategy_max_attempts = count * 15
            
            for i in range(count):
                if strategy_attempts > strategy_max_attempts or failed_attempts > max_failures:
                    break
                
                lineup_num += 1
                
                try:
                    import pulp
                    
                    model = pulp.LpProblem(f"GPP_{lineup_num}_{strategy.value}", pulp.LpMaximize)
                    flex = pulp.LpVariable.dicts("Flex", players, cat='Binary')
                    captain = pulp.LpVariable.dicts("Captain", players, cat='Binary')
                    
                    # Objective
                    if strategy == StrategyType.LEVERAGE:
                        model += pulp.lpSum([
                            adjusted_points[p] * flex[p] * (1 + max(0, 20 - ownership.get(p, 10))/30) +
                            1.5 * adjusted_points[p] * captain[p] * (1 + max(0, 15 - ownership.get(p, 10))/20)
                            for p in players
                        ])
                    else:
                        model += pulp.lpSum([
                            adjusted_points[p] * flex[p] + 1.5 * adjusted_points[p] * captain[p]
                            for p in players
                        ])
                    
                    # Basic constraints
                    model += pulp.lpSum(captain.values()) == 1
                    model += pulp.lpSum(flex.values()) == 5
                    
                    for p in players:
                        model += flex[p] + captain[p] <= 1
                    
                    # Salary
                    model += pulp.lpSum([
                        salaries[p] * flex[p] + 1.5 * salaries[p] * captain[p]
                        for p in players
                    ]) <= OptimizerConfig.SALARY_CAP
                    
                    # Team limits
                    for team in self.df['Team'].unique():
                        team_players = [p for p in players if teams.get(p) == team]
                        if team_players:
                            model += pulp.lpSum([flex[p] + captain[p] for p in team_players]) <= OptimizerConfig.MAX_PLAYERS_PER_TEAM
                    
                    # No ownership constraints - let natural distribution work
                    
                    # Unique captains
                    if force_unique_captains and used_captains:
                        for prev_captain in list(used_captains)[:100]:
                            if prev_captain in players:
                                model += captain[prev_captain] == 0
                    
                    # Light diversity
                    if all_lineups and len(all_lineups) > 0:
                        prev_lineup = all_lineups[-1]
                        prev_players = [prev_lineup['Captain']] + prev_lineup['FLEX']
                        model += pulp.lpSum([flex[p] + captain[p] for p in prev_players]) <= 5
                    
                    model.solve(pulp.PULP_CBC_CMD(msg=0))
                    
                    if pulp.LpStatus[model.status] == 'Optimal':
                        captain_pick = None
                        flex_picks = []
                        
                        for p in players:
                            if captain[p].value() == 1:
                                captain_pick = p
                                used_captains.add(p)
                            if flex[p].value() == 1:
                                flex_picks.append(p)
                        
                        if captain_pick and len(flex_picks) == 5:
                            is_valid, reason = self.validate_lineup_constraints(
                                captain_pick, flex_picks, salaries, ownership, teams
                            )
                            
                            if is_valid:
                                total_salary = sum(salaries[p] for p in flex_picks) + 1.5 * salaries[captain_pick]
                                total_proj = sum(points[p] for p in flex_picks) + 1.5 * points[captain_pick]
                                total_ownership = ownership.get(captain_pick, 5) * 1.5 + sum(ownership.get(p, 5) for p in flex_picks)
                                
                                # Ownership tier
                                if total_ownership < 60:
                                    ownership_tier = '💎 Elite'
                                elif total_ownership < 80:
                                    ownership_tier = '🟢 Optimal'
                                elif total_ownership < 100:
                                    ownership_tier = '🟡 Balanced'
                                else:
                                    ownership_tier = '⚠️ Chalky'
                                
                                all_lineups.append({
                                    'Lineup': len(all_lineups) + 1,
                                    'Strategy': strategy.value,
                                    'Captain': captain_pick,
                                    'Captain_Own%': ownership.get(captain_pick, 5),
                                    'FLEX': flex_picks,
                                    'Projected': round(total_proj, 2),
                                    'Salary': int(total_salary),
                                    'Salary_Remaining': int(OptimizerConfig.SALARY_CAP - total_salary),
                                    'Total_Ownership': round(total_ownership, 1),
                                    'Ownership_Tier': ownership_tier,
                                    'GPP_Summary': self.bucket_manager.get_gpp_summary([captain_pick] + flex_picks, self.df, self.field_size),
                                    'Leverage_Score': self.bucket_manager.calculate_gpp_leverage([captain_pick] + flex_picks, self.df),
                                    'Has_Stack': False,
                                    'Stack_Details': 'None',
                                    'Field_Size': self.field_size
                                })
                                
                                self.perf_monitor.increment_counter("successful_lineups")
                            else:
                                strategy_attempts += 1
                        else:
                            strategy_attempts += 1
                    else:
                        failed_attempts += 1
                        strategy_attempts += 1
                        
                except Exception as e:
                    self.logger.log(f"Error: {e}", "ERROR")
                    failed_attempts += 1
                    strategy_attempts += 1
        
        # If no lineups generated, try relaxed mode
        if len(all_lineups) == 0:
            relaxed_lineups = self.generate_relaxed_lineups(
                num_lineups, players, salaries, points, ownership, teams, used_captains
            )
            all_lineups = relaxed_lineups
        
        # End logging
        total_time = self.perf_monitor.stop_timer("total_optimization")
        self.logger.log_optimization_end(len(all_lineups), total_time)
        
        # Handle results
        if len(all_lineups) == 0:
            st.error("❌ No valid lineups generated")
            self.diagnose_optimization_issues(self.df, self.field_size, self.field_config,
                                             salaries, ownership, points)
            return pd.DataFrame()
        
        if len(all_lineups) < num_lineups:
            st.warning(f"Only generated {len(all_lineups)}/{num_lineups} valid lineups")
            self.logger.log(f"Partial generation: {len(all_lineups)}/{num_lineups} lineups", "WARNING")
        else:
            st.success(f"✅ Generated all {num_lineups} lineups!")
            self.logger.log(f"Successfully generated all {num_lineups} lineups", "INFO")
        
        return pd.DataFrame(all_lineups)

# NFL GPP DUAL-AI OPTIMIZER - PART 5: MAIN UI AND HELPER FUNCTIONS (COMPLETE CORRECTED)
# With Performance Optimizations and Caching

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, Counter
from datetime import datetime
import hashlib
import pickle

# ============================================================================
# PERFORMANCE OPTIMIZATION - CACHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def calculate_correlations_cached(df_json: str, game_info_json: str, field_size: str) -> Dict:
    """Cache correlation calculations"""
    df = pd.read_json(df_json)
    game_info = json.loads(game_info_json)
    
    correlation_engine = GPPCorrelationEngine()
    correlations = correlation_engine.calculate_gpp_correlations(df, game_info)
    stacks = correlation_engine.identify_gpp_stacks(df, correlations, field_size)
    
    return {
        'correlations': correlations,
        'stacks': stacks
    }

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def run_simulations_cached(lineup_json: str, df_json: str, correlations_json: str, 
                          n_sims: int, field_size: str) -> Dict:
    """Cache simulation results"""
    lineup = json.loads(lineup_json)
    df = pd.read_json(df_json)
    correlations = json.loads(correlations_json)
    
    # Convert correlation keys back to tuples
    correlations_fixed = {}
    for key, value in correlations.items():
        if isinstance(key, str) and ',' in key:
            players = key.strip('()').split(', ')
            correlations_fixed[tuple(players)] = value
        else:
            correlations_fixed[key] = value
    
    simulator = GPPTournamentSimulator()
    return simulator.simulate_gpp_tournament(lineup, df, correlations_fixed, n_sims, field_size)

@st.cache_data(ttl=3600)
def validate_player_pool_cached(df_json: str, field_size: str) -> Dict:
    """Cache player pool validation results"""
    df = pd.read_json(df_json)
    return ConfigValidator.validate_player_pool(df, field_size)

@st.cache_data(ttl=7200)  # Cache for 2 hours
def analyze_ownership_distribution(df_json: str) -> Dict:
    """Cache ownership distribution analysis"""
    df = pd.read_json(df_json)
    bucket_manager = OwnershipBucketManager()
    buckets = bucket_manager.categorize_players(df)
    
    return {
        'buckets': buckets,
        'distribution': {k: len(v) for k, v in buckets.items()},
        'total_players': len(df)
    }

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def init_session_state():
    """Initialize session state variables for persistence"""
    if 'optimization_history' not in st.session_state:
        st.session_state['optimization_history'] = []
    if 'saved_lineups' not in st.session_state:
        st.session_state['saved_lineups'] = []
    if 'api_cache' not in st.session_state:
        st.session_state['api_cache'] = {}
    if 'performance_metrics' not in st.session_state:
        st.session_state['performance_metrics'] = {}

def save_optimization_session(lineups_df: pd.DataFrame, settings: Dict):
    """Save optimization session to history"""
    session = {
        'timestamp': datetime.now(),
        'lineups': lineups_df.to_dict('records'),
        'settings': settings,
        'field_size': st.session_state.get('field_size', 'large_field')
    }
    st.session_state['optimization_history'].append(session)
    
    # Limit history to last 10 sessions
    if len(st.session_state['optimization_history']) > 10:
        st.session_state['optimization_history'].pop(0)

# ============================================================================
# DATA VALIDATION AND INTEGRITY CHECKS
# ============================================================================

def validate_lineup_data(df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Comprehensive data validation for lineup generation with performance optimization"""
    
    # Try to use cached validation if available
    df_json = df.to_json()
    df_hash = hashlib.md5(df_json.encode()).hexdigest()
    
    # Check if we've validated this data recently
    cache_key = f"validation_{df_hash}"
    if cache_key in st.session_state.get('api_cache', {}):
        cached_result = st.session_state['api_cache'][cache_key]
        if (datetime.now() - cached_result['timestamp']).seconds < 300:  # 5 minute cache
            return cached_result['result']
    
    issues = []
    warnings = []
    data_stats = {}
    
    # Basic size check
    data_stats['total_players'] = len(df)
    if len(df) < 12:
        issues.append(f"Critical: Only {len(df)} players found - need at least 12 for valid lineups")
    elif len(df) < 20:
        warnings.append(f"Warning: Only {len(df)} players - limited lineup diversity possible")
    
    # Check for required columns
    required_columns = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Critical: Missing required columns: {missing_columns}")
        return False, issues, data_stats
    
    # Check for missing values
    for col in required_columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            issues.append(f"Critical: {null_count} missing values in {col}")
    
    # Validate data types and ranges
    if not df['Salary'].isna().all():
        salary_issues = df[df['Salary'] < OptimizerConfig.MIN_SALARY]
        if len(salary_issues) > 0:
            issues.append(f"Critical: {len(salary_issues)} players with salary below minimum (${OptimizerConfig.MIN_SALARY})")
        
        data_stats['avg_salary'] = df['Salary'].mean()
        data_stats['min_salary'] = df['Salary'].min()
        data_stats['max_salary'] = df['Salary'].max()
    
    # Validate projections
    if not df['Projected_Points'].isna().all():
        negative_projections = df[df['Projected_Points'] < 0]
        if len(negative_projections) > 0:
            issues.append(f"Critical: {len(negative_projections)} players with negative projections")
        
        zero_projections = df[df['Projected_Points'] == 0]
        if len(zero_projections) > 0:
            warnings.append(f"Warning: {len(zero_projections)} players with zero projections")
        
        data_stats['avg_projection'] = df['Projected_Points'].mean()
        data_stats['max_projection'] = df['Projected_Points'].max()
    
    # Team validation
    teams = df['Team'].unique()
    data_stats['num_teams'] = len(teams)
    
    if len(teams) != 2:
        issues.append(f"Critical: Found {len(teams)} teams - Showdown requires exactly 2 teams")
    else:
        team_counts = df['Team'].value_counts()
        for team, count in team_counts.items():
            if count < 5:
                issues.append(f"Critical: Team {team} has only {count} players - need at least 5 per team")
            data_stats[f'{team}_players'] = count
    
    # Position distribution check
    position_counts = df['Position'].value_counts()
    data_stats['positions'] = position_counts.to_dict()
    
    essential_positions = ['QB', 'RB', 'WR']
    for pos in essential_positions:
        if pos not in position_counts or position_counts[pos] == 0:
            issues.append(f"Critical: No {pos} players in pool")
        elif position_counts[pos] < 2:
            warnings.append(f"Warning: Only {position_counts[pos]} {pos} player(s)")
    
    # Ownership validation
    if 'Ownership' in df.columns:
        ownership_sum = df['Ownership'].sum()
        if ownership_sum > 0:
            if ownership_sum < 400 or ownership_sum > 800:
                warnings.append(f"Warning: Total ownership sums to {ownership_sum:.1f}% (expected ~600%)")
            
            unrealistic_high = df[df['Ownership'] > 60]
            if len(unrealistic_high) > 0:
                warnings.append(f"Warning: {len(unrealistic_high)} players with >60% ownership")
            
            data_stats['avg_ownership'] = df['Ownership'].mean()
            data_stats['total_ownership'] = ownership_sum
    
    # Salary cap feasibility check
    min_possible_salary = df.nsmallest(6, 'Salary')['Salary'].sum()
    if min_possible_salary > OptimizerConfig.SALARY_CAP:
        issues.append(f"Critical: Impossible to create valid lineup - minimum salary {min_possible_salary} exceeds cap")
    
    # Check for duplicate players
    duplicate_players = df[df.duplicated(subset=['Player'], keep=False)]
    if len(duplicate_players) > 0:
        unique_dups = duplicate_players['Player'].unique()
        issues.append(f"Critical: {len(unique_dups)} duplicate player(s): {', '.join(unique_dups[:5])}")
    
    # Compile results
    is_valid = len(issues) == 0
    all_messages = issues + warnings
    
    # Cache the result
    result = (is_valid, all_messages, data_stats)
    if 'api_cache' not in st.session_state:
        st.session_state['api_cache'] = {}
    st.session_state['api_cache'][cache_key] = {
        'result': result,
        'timestamp': datetime.now()
    }
    
    return result

def auto_fix_common_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically fix common data issues with performance optimization"""
    logger = get_logger()
    df_fixed = df.copy()
    fixes_applied = []
    
    # Remove duplicates
    if df_fixed.duplicated(subset=['Player']).any():
        df_fixed = df_fixed.sort_values('Projected_Points', ascending=False).drop_duplicates(subset=['Player'])
        fixes_applied.append("Removed duplicate players (kept highest projection)")
        logger.log("Fixed duplicate players", "INFO")
    
    # Fill missing ownership
    if 'Ownership' in df_fixed.columns:
        missing_own = df_fixed['Ownership'].isna().sum()
        if missing_own > 0:
            df_fixed['Ownership'].fillna(OptimizerConfig.DEFAULT_OWNERSHIP, inplace=True)
            fixes_applied.append(f"Filled {missing_own} missing ownership values with {OptimizerConfig.DEFAULT_OWNERSHIP}%")
            logger.log(f"Fixed {missing_own} missing ownership values", "INFO")
    
    # Remove invalid salaries
    before_count = len(df_fixed)
    df_fixed = df_fixed[df_fixed['Salary'] >= OptimizerConfig.MIN_SALARY]
    removed = before_count - len(df_fixed)
    if removed > 0:
        fixes_applied.append(f"Removed {removed} players with invalid salaries")
        logger.log(f"Removed {removed} invalid salary players", "INFO")
    
    # Remove missing critical data
    before_count = len(df_fixed)
    df_fixed = df_fixed.dropna(subset=['Player', 'Position', 'Team', 'Salary', 'Projected_Points'])
    removed = before_count - len(df_fixed)
    if removed > 0:
        fixes_applied.append(f"Removed {removed} players with missing critical data")
        logger.log(f"Removed {removed} players with missing data", "INFO")
    
    if fixes_applied:
        st.info("Data fixes applied automatically:")
        for fix in fixes_applied:
            st.write(f"  ✓ {fix}")
    
    return df_fixed

def display_data_validation_results(is_valid: bool, messages: List[str], stats: Dict[str, Any]):
    """Display validation results with performance metrics"""
    logger = get_logger()
    
    if is_valid:
        st.success("✅ Data validation passed - ready to optimize!")
        logger.log("Data validation passed", "INFO")
    else:
        st.error("❌ Data validation failed - issues must be resolved")
        logger.log("Data validation failed", "ERROR")
    
    if messages:
        critical_issues = [msg for msg in messages if msg.startswith("Critical:")]
        warnings = [msg for msg in messages if msg.startswith("Warning:")]
        
        if critical_issues:
            with st.expander("🚨 Critical Issues (Must Fix)", expanded=True):
                for issue in critical_issues:
                    st.error(issue.replace("Critical: ", ""))
        
        if warnings:
            with st.expander("⚠️ Warnings (Should Review)", expanded=not critical_issues):
                for warning in warnings:
                    st.warning(warning.replace("Warning: ", ""))
    
    with st.expander("📊 Data Statistics", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Players", stats.get('total_players', 0))
            st.metric("Teams", stats.get('num_teams', 0))
            if 'avg_salary' in stats:
                st.metric("Avg Salary", f"${stats['avg_salary']:,.0f}")
        
        with col2:
            if 'avg_projection' in stats:
                st.metric("Avg Projection", f"{stats['avg_projection']:.1f}")
            if 'max_projection' in stats:
                st.metric("Max Projection", f"{stats['max_projection']:.1f}")
            if 'avg_ownership' in stats:
                st.metric("Avg Ownership", f"{stats['avg_ownership']:.1f}%")
        
        with col3:
            if 'positions' in stats:
                st.write("**Position Counts:**")
                for pos, count in stats['positions'].items():
                    st.write(f"{pos}: {count}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_validate_data(uploaded_file) -> pd.DataFrame:
    """Load and validate CSV with comprehensive checks and caching"""
    logger = get_logger()
    perf = get_performance_monitor()
    
    perf.start_timer("data_loading")
    logger.log("Starting data load and validation", "INFO")
    
    df = pd.read_csv(uploaded_file)
    
    # Check required columns
    required_cols = ['first_name', 'last_name', 'position', 'team', 'salary', 'ppg_projection']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.info("Required columns: first_name, last_name, position, team, salary, ppg_projection")
        logger.log(f"Missing columns: {missing_cols}", "ERROR")
        st.stop()
    
    # Create player names
    df['first_name'] = df['first_name'].fillna('')
    df['last_name'] = df['last_name'].fillna('')
    df['Player'] = (df['first_name'] + ' ' + df['last_name']).str.strip()
    
    # Remove empty players
    df = df[df['Player'].str.len() > 0]
    
    # Rename columns
    df = df.rename(columns={
        'position': 'Position',
        'team': 'Team',
        'salary': 'Salary',
        'ppg_projection': 'Projected_Points'
    })
    
    # Ensure numeric types
    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
    df['Projected_Points'] = pd.to_numeric(df['Projected_Points'], errors='coerce')
    
    # Add GPP value column
    df['Value'] = np.where(
        df['Salary'] > 0,
        df['Projected_Points'] / (df['Salary'] / 1000),
        0
    )
    
    # Run comprehensive validation
    is_valid, messages, stats = validate_lineup_data(df)
    
    # Try auto-fixing if there are issues
    if not is_valid:
        st.warning("Attempting to auto-fix common issues...")
        df = auto_fix_common_issues(df)
        
        # Re-validate after fixes
        is_valid, messages, stats = validate_lineup_data(df)
    
    # Display validation results
    display_data_validation_results(is_valid, messages, stats)
    
    # Stop if still invalid after auto-fix
    if not is_valid:
        st.error("Cannot proceed with optimization. Please fix the critical issues above.")
        logger.log("Data validation failed after auto-fix attempts", "ERROR")
        st.stop()
    
    elapsed = perf.stop_timer("data_loading")
    logger.log(f"Data loaded and validated in {elapsed:.2f}s", "INFO")
    perf.increment_counter("successful_data_loads")
    
    return df

def display_gpp_lineup_analysis(lineups_df: pd.DataFrame, df: pd.DataFrame, field_size: str):
    """Display GPP-specific lineup analysis with performance optimization"""
    
    if lineups_df.empty:
        st.warning("No lineups to analyze")
        return
    
    logger = get_logger()
    perf = get_performance_monitor()
    
    perf.start_timer("lineup_analysis")
    logger.log("Generating lineup analysis visualizations", "DEBUG")
    
    # Create GPP analysis visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Strategy Distribution for GPP
    ax1 = axes[0, 0]
    if 'Strategy' in lineups_df.columns:
        strategy_counts = lineups_df['Strategy'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A', '#98D8C8']
        ax1.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.0f%%',
               colors=colors[:len(strategy_counts)], startangle=90)
        ax1.set_title('GPP Strategy Distribution')
    else:
        ax1.text(0.5, 0.5, 'No Strategy Data', ha='center', va='center')
        ax1.set_title('Strategy Distribution')
    
    # 2. Ownership vs Ceiling (GPP Focus)
    ax2 = axes[0, 1]
    ceiling_col = 'Ceiling_99th' if 'Ceiling_99th' in lineups_df.columns else 'Projected'
    if 'Total_Ownership' in lineups_df.columns and ceiling_col in lineups_df.columns:
        scatter = ax2.scatter(lineups_df['Total_Ownership'], lineups_df[ceiling_col],
                            c=lineups_df.get('GPP_Score', lineups_df['Projected']), 
                            cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add optimal ownership zone
        min_own, max_own = OptimizerConfig.GPP_OWNERSHIP_TARGETS.get(field_size, (60, 90))
        ax2.axvspan(min_own, max_own, alpha=0.2, color='green', label=f'Optimal ({min_own}-{max_own}%)')
        
        ax2.set_xlabel('Total Ownership %')
        ax2.set_ylabel('99th Percentile Points')
        ax2.set_title(f'GPP Ownership vs Ceiling ({field_size})')
        ax2.legend()
        plt.colorbar(scatter, ax=ax2, label='GPP Score')
    
    # 3. Captain Ownership Distribution
    ax3 = axes[0, 2]
    if 'Captain_Own%' in lineups_df.columns:
        captain_owns = lineups_df['Captain_Own%'].values
        ax3.hist(captain_owns, bins=min(20, len(set(captain_owns))), alpha=0.7, color='#4ECDC4', edgecolor='black')
        ax3.axvline(15, color='red', linestyle='--', label='15% Threshold')
        ax3.set_xlabel('Captain Ownership %')
        ax3.set_ylabel('Number of Lineups')
        ax3.set_title('GPP Captain Ownership Distribution')
        ax3.legend()
    
    # 4. Leverage Score Distribution
    ax4 = axes[1, 0]
    if 'Leverage_Score' in lineups_df.columns:
        ax4.hist(lineups_df['Leverage_Score'], bins=15, alpha=0.7, color='#45B7D1', edgecolor='black')
        ax4.axvline(lineups_df['Leverage_Score'].mean(), color='red', linestyle='--', 
                   label=f"Mean: {lineups_df['Leverage_Score'].mean():.1f}")
        ax4.set_xlabel('GPP Leverage Score')
        ax4.set_ylabel('Number of Lineups')
        ax4.set_title('Leverage Distribution')
        ax4.legend()
    
    # 5. Ship Equity vs Tournament EV
    ax5 = axes[1, 1]
    if 'Ship_Equity' in lineups_df.columns and 'Tournament_EV' in lineups_df.columns:
        ax5.scatter(lineups_df['Ship_Equity'], lineups_df['Tournament_EV'],
                   c=lineups_df.get('Total_Ownership', 50), cmap='RdYlGn_r',
                   s=80, alpha=0.7)
        ax5.set_xlabel('Ship Equity (Win Probability)')
        ax5.set_ylabel('Tournament EV')
        ax5.set_title('GPP Tournament Equity Analysis')
    elif 'Total_Ownership' in lineups_df.columns and 'GPP_Score' in lineups_df.columns:
        ax5.scatter(lineups_df['Total_Ownership'], lineups_df['GPP_Score'], alpha=0.6)
        ax5.set_xlabel('Total Ownership %')
        ax5.set_ylabel('GPP Score')
        ax5.set_title('GPP Score Distribution')
    
    # 6. Top Captains for GPP
    ax6 = axes[1, 2]
    if 'Captain' in lineups_df.columns:
        captain_counts = lineups_df['Captain'].value_counts().head(10)
        
        if 'Captain_Own%' in lineups_df.columns:
            captain_ownership = lineups_df.groupby('Captain')['Captain_Own%'].first()
            colors = ['#FF6B6B' if captain_ownership.get(cap, 15) > 20 else 
                     '#4ECDC4' if captain_ownership.get(cap, 15) > 10 else '#45B7D1' 
                     for cap in captain_counts.index]
        else:
            colors = ['#4ECDC4'] * len(captain_counts)
        
        y_pos = np.arange(len(captain_counts))
        ax6.barh(y_pos, captain_counts.values, color=colors)
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(captain_counts.index, fontsize=8)
        ax6.set_xlabel('Times Used as Captain')
        ax6.set_title('Top 10 GPP Captains')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    elapsed = perf.stop_timer("lineup_analysis")
    logger.log(f"Analysis visualizations generated in {elapsed:.2f}s", "DEBUG")
    
    # Ownership Tier Distribution with caching
    st.markdown("### 🎯 GPP Ownership Tier Analysis")
    
    if 'Captain' in lineups_df.columns and 'FLEX' in lineups_df.columns:
        # Use cached ownership distribution if available
        df_json = df.to_json()
        ownership_dist = analyze_ownership_distribution(df_json)
        
        tier_data = []
        for idx, row in lineups_df.head(20).iterrows():
            bucket_counts = {'mega_chalk': 0, 'chalk': 0, 'pivot': 0, 'leverage': 0, 'super_leverage': 0}
            lineup_players = [row['Captain']] + (row['FLEX'] if isinstance(row['FLEX'], list) else [])
            
            for player in lineup_players:
                if player in df['Player'].values:
                    player_own = df[df['Player'] == player]['Ownership'].values[0]
                    bucket = OwnershipBucketManager.get_bucket(player_own)
                    bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            
            tier_data.append(list(bucket_counts.values()))
        
        if tier_data:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            cmap = plt.cm.RdYlGn_r
            im = ax.imshow(tier_data, cmap=cmap, aspect='auto', vmin=0, vmax=6)
            
            ax.set_xticks(range(5))
            ax.set_xticklabels(['Mega\nChalk\n(35%+)', 'Chalk\n(20-35%)', 'Pivot\n(10-20%)', 
                               'Leverage\n(5-10%)', 'Super\nLeverage\n(<5%)'])
            ax.set_yticks(range(len(tier_data)))
            ax.set_yticklabels([f'#{i+1}' for i in range(len(tier_data))])
            ax.set_title(f'GPP Ownership Distribution - {field_size.upper()} Field')
            ax.set_xlabel('Ownership Tier')
            ax.set_ylabel('Lineup Rank')
            
            # Add text annotations
            for i in range(len(tier_data)):
                for j in range(5):
                    value = tier_data[i][j]
                    color = "white" if value > 3 else "black"
                    if j < 2 and value > 2:  # Too much chalk for GPP
                        color = "red"
                    elif j >= 3 and value >= 2:  # Good leverage
                        color = "green"
                    text = ax.text(j, i, value, ha="center", va="center",
                                 color=color, fontweight='bold' if j >= 3 else 'normal')
            
            plt.colorbar(im, ax=ax, label='Player Count')
            st.pyplot(fig)

# ============================================================================
# MAIN STREAMLIT UI
# ============================================================================

# Initialize session state
init_session_state()

with st.sidebar:
    st.header("🏆 GPP Tournament Settings")
    
    # Debug panel
    with st.expander("🐛 Debug & Logging"):
        verbose_logging = st.checkbox("Verbose Logging", value=False)
        log_to_file = st.checkbox("Log to File", value=False)
        
        # Update logger settings
        logger = get_logger()
        logger.verbose = verbose_logging
        logger.log_to_file = log_to_file
        
        if st.button("Show Log Summary"):
            logger.display_log_summary()
        
        if st.button("Show Performance Metrics"):
            perf = get_performance_monitor()
            perf.display_metrics()
        
        if st.button("Export Logs"):
            log_text = logger.export_logs()
            st.download_button(
                "Download Logs",
                data=log_text,
                file_name=f"optimizer_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.session_state['api_cache'] = {}
            st.success("Cache cleared!")
    
    st.markdown("### 🎯 Contest Type")
    contest_type = st.selectbox(
        "Select GPP Type",
        list(OptimizerConfig.FIELD_SIZES.keys()),
        index=2,
        help="Different GPP types require different strategies"
    )
    field_size = OptimizerConfig.FIELD_SIZES[contest_type]
    
    # Display field size strategy
    if field_size == 'milly_maker':
        st.info("💎 **Milly Maker Strategy:**\nUltra-contrarian, <80% total ownership, zero chalk")
    elif field_size == 'large_field':
        st.info("🎯 **Large Field Strategy:**\n60-90% ownership, maximize leverage")
    elif field_size == 'medium_field':
        st.info("🔄 **Medium Field Strategy:**\n70-100% ownership, balanced approach")
    else:
        st.info("⚖️ **Small Field Strategy:**\n80-120% ownership, slight contrarian")
    
    st.markdown("---")
    
    st.markdown("### 🤖 AI Configuration")
    api_mode = st.radio(
        "Connection Mode",
        ["Manual (Free)", "API (Automated)"],
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
            if st.button("🔌 Connect API"):
                api_manager = ClaudeAPIManager(api_key)
                use_api = bool(api_manager.client)
        
        if use_api:
            st.success("✅ API Connected")
    else:
        st.info("📋 Manual mode: Copy prompts to Claude")
    
    st.markdown("---")
    
    # GPP-Specific Settings
    with st.expander("⚙️ GPP Advanced Settings"):
        st.markdown("### Optimization Parameters")
        
        force_unique_captains = st.checkbox(
            "Force Unique Captains", 
            value=True,
            help="Each lineup gets different captain (recommended for GPP)"
        )
        
        min_leverage_players = st.slider(
            "Min Sub-10% Players", 0, 4, 2,
            help="Minimum low-owned players per lineup"
        )
        
        st.markdown("### Simulation Settings")
        num_sims = st.slider(
            "Monte Carlo Simulations", 
            1000, 10000, 5000, 1000,
            help="More sims = better accuracy"
        )
        
        st.markdown("### Captain Settings")
        max_captain_ownership = st.slider(
            "Max Captain Ownership %", 
            5, 30, 15, 5,
            help="Maximum ownership for captains"
        )
        
        st.markdown("### Export Options")
        include_pivots = st.checkbox(
            "Generate Captain Pivots", 
            value=True,
            help="Create leverage captain swaps"
        )
        
        export_top_n = st.number_input(
            "Export Top N Lineups", 
            5, 150, 20, 5,
            help="Number of lineups to export"
        )

# Main Content Area
st.markdown("## 🎮 GPP Tournament Optimizer")

# Display current settings
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Contest Type", contest_type)
with col2:
    ownership_range = OptimizerConfig.GPP_OWNERSHIP_TARGETS[field_size]
    st.metric("Target Own%", f"{ownership_range[0]}-{ownership_range[1]}%")
with col3:
    st.metric("Field Size", field_size.replace('_', ' ').title())
with col4:
    st.metric("Mode", "🏆 GPP Only")

st.markdown("---")
st.markdown("## 📁 Data Upload & Game Configuration")

uploaded_file = st.file_uploader(
    "Upload DraftKings CSV",
    type="csv",
    help="Export player pool from DraftKings Showdown contest"
)

if uploaded_file is not None:
    # Load and validate data with caching
    df = load_and_validate_data(uploaded_file)
    
    # Game Configuration
    st.markdown("### ⚙️ Game Setup")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        teams = st.text_input("Teams", "BUF vs MIA", help="Team matchup")
    with col2:
        total = st.number_input("O/U Total", 30.0, 70.0, 48.5, 0.5,
                               help="Higher totals = more correlation plays")
    with col3:
        spread = st.number_input("Spread", -20.0, 20.0, -3.0, 0.5,
                                help="Large spreads = leverage on dogs")
    with col4:
        weather = st.selectbox("Weather", ["Clear", "Wind", "Rain", "Snow"],
                             help="Weather impacts volatility")
    
    game_info = {
        'teams': teams,
        'total': total,
        'spread': spread,
        'weather': weather,
        'field_size': field_size
    }
    
    # GPP-Specific Player Adjustments
    st.markdown("### 🎯 GPP Player Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Ownership Projections (CRITICAL FOR GPP)")
        ownership_text = st.text_area(
            "Format: Player: %",
            height=150,
            placeholder="Josh Allen: 45\nStefon Diggs: 35\nTyreek Hill: 40\nJaylen Waddle: 25",
            help="Accurate ownership projections are essential for GPP success"
        )
        
        # Parse ownership
        ownership_dict = {}
        for line in ownership_text.split('\n'):
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    try:
                        player = parts[0].strip()
                        own_pct = float(parts[1].strip())
                        if 0 <= own_pct <= 100:
                            ownership_dict[player] = own_pct
                    except:
                        pass
        
        # Apply ownership
        df['Ownership'] = df['Player'].map(ownership_dict).fillna(OptimizerConfig.DEFAULT_OWNERSHIP)
        
        # Show ownership distribution
        if ownership_dict:
            high_owned = len(df[df['Ownership'] > 30])
            low_owned = len(df[df['Ownership'] < 10])
            st.info(f"Chalk (30%+): {high_owned} | Leverage (<10%): {low_owned}")
    
    with col2:
        st.markdown("#### 🎲 GPP Ceiling Boosts")
        boost_text = st.text_area(
            "High Ceiling Players (Format: Player: Multiplier)",
            height=150,
            placeholder="Tyreek Hill: 1.2\nJosh Allen: 1.15\nRaheem Mostert: 1.25",
            help="Boost projections for boom/bust players"
        )
        
        # Apply ceiling boosts
        for line in boost_text.split('\n'):
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    try:
                        player = parts[0].strip()
                        multiplier = float(parts[1].strip())
                        if player in df['Player'].values and 0.5 <= multiplier <= 2.0:
                            df.loc[df['Player'] == player, 'Projected_Points'] *= multiplier
                            df.loc[df['Player'] == player, 'Ceiling_Boost'] = multiplier
                    except:
                        pass
    
    # Add GPP-specific columns
    df['Bucket'] = df['Ownership'].apply(OwnershipBucketManager.get_bucket)
    df['GPP_Value'] = np.where(
        df['Salary'] > 0,
        (df['Projected_Points'] / (df['Salary'] / 1000)) * (20 / (df['Ownership'] + 5)),
        0
    )
    df['Leverage_Score'] = df.apply(
        lambda x: 3 if x['Ownership'] < 5 else 2 if x['Ownership'] < 10 else 1 if x['Ownership'] < 15 else -1 if x['Ownership'] > 30 else 0, 
        axis=1
    )
    
    # Display GPP player pool
    st.markdown("### 💎 GPP Player Pool Analysis")
    
    # Use cached ownership distribution
    df_json = df.to_json()
    ownership_dist = analyze_ownership_distribution(df_json)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**GPP Ownership Tiers:**")
        tier_emojis = {
            'mega_chalk': '🔴',
            'chalk': '🟠', 
            'pivot': '🟡',
            'leverage': '🟢',
            'super_leverage': '💎'
        }
        
        for bucket in ['mega_chalk', 'chalk', 'pivot', 'leverage', 'super_leverage']:
            count = ownership_dist['distribution'].get(bucket, 0)
            emoji = tier_emojis.get(bucket, '')
            percentage = (count / ownership_dist['total_players'] * 100) if ownership_dist['total_players'] > 0 else 0
            st.write(f"{emoji} {bucket}: {count} ({percentage:.0f}%)")
        
        # GPP recommendations
        if ownership_dist['distribution'].get('super_leverage', 0) < 5:
            st.warning("⚠️ Limited super leverage plays available")
        if ownership_dist['distribution'].get('mega_chalk', 0) > 10:
            st.info("📊 Heavy chalk slate - focus on leverage")
    
    with col2:
        # Show player pool
        display_cols = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 
                       'Ownership', 'Bucket', 'GPP_Value', 'Leverage_Score']
        
        def highlight_gpp_rows(row):
            if row['Bucket'] == 'super_leverage':
                return ['background-color: #90EE90'] * len(row)
            elif row['Bucket'] == 'leverage':
                return ['background-color: #98FB98'] * len(row)
            elif row['Bucket'] == 'mega_chalk':
                return ['background-color: #FFB6C1'] * len(row)
            return [''] * len(row)
        
        styled_df = df[display_cols].sort_values('GPP_Value', ascending=False).head(20).style.apply(
            highlight_gpp_rows, axis=1
        )
        
        st.dataframe(
            styled_df,
            height=400,
            use_container_width=True
        )
    
    # Optimization Section
    st.markdown("---")
    st.markdown("## 🚀 GPP Lineup Generation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_lineups = st.number_input(
            "Number of Lineups", 
            5, 150, 20, 5,
            help="More lineups = better coverage"
        )
    
    with col2:
        correlation_emphasis = st.slider(
            "Correlation Focus", 
            0, 100, 50,
            help="Higher = more stacks"
        )
    
    with col3:
        if st.button("🎯 Generate GPP Lineups", type="primary", use_container_width=True):
            logger = get_logger()
            perf = get_performance_monitor()
            
            # Save current settings
            current_settings = {
                'num_lineups': num_lineups,
                'field_size': field_size,
                'force_unique_captains': force_unique_captains,
                'num_sims': num_sims,
                'correlation_emphasis': correlation_emphasis
            }
            
            # Initialize optimizer
            optimizer = GPPDualAIOptimizer(df, game_info, field_size, api_manager)
            
            # Get AI strategies
            with st.spinner("Getting GPP AI strategies..."):
                rec1, rec2 = optimizer.get_ai_strategies(use_api=use_api)
            
            # Display AI insights
            with st.expander("🧠 GPP AI Strategic Analysis", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Game Theory AI")
                    st.metric("Confidence", f"{rec1.confidence:.0%}")
                    
                    if rec1.captain_targets:
                        st.markdown("**Leverage Captains:**")
                        for captain in rec1.captain_targets[:5]:
                            own = df[df['Player'] == captain]['Ownership'].values[0] if captain in df['Player'].values else 5
                            emoji = "💎" if own < 5 else "🟢" if own < 10 else "🟡" if own < 15 else "⚠️"
                            st.write(f"{emoji} {captain} ({own:.0f}%)")
                    
                    if rec1.fades:
                        st.markdown("**Fade Targets (Chalk):**")
                        for fade in rec1.fades[:3]:
                            st.write(f"🔴 {fade}")
                    
                    if rec1.key_insights:
                        st.markdown("**GPP Insights:**")
                        for insight in rec1.key_insights[:3]:
                            st.info(insight)
                
                with col2:
                    st.markdown("### 🔗 Correlation AI")
                    st.metric("Confidence", f"{rec2.confidence:.0%}")
                    
                    if rec2.stacks:
                        st.markdown("**GPP Stacks:**")
                        for stack in rec2.stacks[:5]:
                            if isinstance(stack, dict):
                                p1 = stack.get('player1', '')
                                p2 = stack.get('player2', '')
                                stack_type = stack.get('type', '')
                                if p1 and p2:
                                    emoji = "💎" if 'leverage' in stack_type else "🔗"
                                    st.write(f"{emoji} {p1} + {p2}")
                    
                    if rec2.key_insights:
                        st.markdown("**Correlation Insights:**")
                        for insight in rec2.key_insights[:3]:
                            st.info(insight)
            
            # Generate GPP lineups
            with st.spinner(f"Generating {num_lineups} GPP lineups for {field_size}..."):
                lineups_df = optimizer.generate_gpp_lineups(
                    num_lineups, rec1, rec2, force_unique_captains=force_unique_captains
                )
            
            if lineups_df.empty:
                st.error("❌ No valid lineups generated. Try adjusting constraints.")
            else:
                st.success(f"✅ Generated {len(lineups_df)} GPP lineups!")
                
                # Calculate correlations with caching
                df_json = df.to_json()
                game_info_json = json.dumps(game_info)
                correlation_data = calculate_correlations_cached(df_json, game_info_json, field_size)
                correlations = correlation_data['correlations']
                
                # Run GPP simulations with caching
                with st.spinner("Running GPP tournament simulations..."):
                    perf.start_timer("simulations")
                    
                    for idx, row in lineups_df.iterrows():
                        lineup_json = json.dumps(row.to_dict())
                        correlations_json = json.dumps({str(k): v for k, v in correlations.items()})
                        
                        sim_results = run_simulations_cached(
                            lineup_json, df_json, correlations_json, num_sims, field_size
                        )
                        
                        for key, value in sim_results.items():
                            lineups_df.loc[idx, key] = value
                    
                    perf.stop_timer("simulations")
                
                # Calculate GPP scores
                lineups_df = calculate_gpp_scores(lineups_df, field_size)
                
                # Sort by GPP score
                lineups_df = lineups_df.sort_values('GPP_Score', ascending=False).reset_index(drop=True)
                
                # Store in session state
                st.session_state['lineups_df'] = lineups_df
                st.session_state['df'] = df
                st.session_state['correlations'] = correlations
                st.session_state['field_size'] = field_size
                
                # Save optimization session
                save_optimization_session(lineups_df, current_settings)
                
                # Generate captain pivots if enabled
                if include_pivots:
                    with st.spinner("Generating GPP captain pivots..."):
                        all_pivots = []
                        for i in range(min(3, len(lineups_df))):
                            pivots = optimizer.pivot_generator.find_optimal_gpp_pivots(
                                lineups_df.iloc[i].to_dict(), df, field_size
                            )
                            all_pivots.extend(pivots)
                        st.session_state['pivots_df'] = all_pivots
    
    # Display results if lineups exist
    if 'lineups_df' in st.session_state:
        lineups_df = st.session_state['lineups_df']
        df = st.session_state['df']
        field_size = st.session_state.get('field_size', 'large_field')
        
        st.markdown("---")
        st.markdown("## 📊 GPP Optimization Results")
        
        # GPP Summary metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Lineups", len(lineups_df))
        with col2:
            ceiling_col = 'Ceiling_99th' if 'Ceiling_99th' in lineups_df else 'Projected'
            st.metric("Avg 99th%", f"{lineups_df[ceiling_col].mean():.1f}")
        with col3:
            st.metric("Avg Own%", f"{lineups_df['Total_Ownership'].mean():.1f}%")
        with col4:
            st.metric("Avg Leverage", f"{lineups_df['Leverage_Score'].mean():.1f}")
        with col5:
            unique_captains = lineups_df['Captain'].nunique()
            st.metric("Unique CPT", f"{unique_captains}/{len(lineups_df)}")
        with col6:
            elite_lineups = len(lineups_df[lineups_df['Total_Ownership'] < 70])
            st.metric("Elite (<70%)", elite_lineups)
        
        # Results tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏆 GPP Lineups", "🔄 Captain Pivots", "📈 GPP Analysis", 
            "💎 Leverage Plays", "📊 Simulations", "💾 Export"
        ])
        
        with tab1:
            st.markdown("### 🏆 Top GPP Lineups")
            
            # Display options
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                display_format = st.radio(
                    "Display Format",
                    ["GPP Summary", "Detailed", "Compact"],
                    horizontal=True
                )
            with col2:
                max_display = len(lineups_df) if len(lineups_df) > 0 else 150
                default_display = min(10, max_display)
                show_top_n = st.number_input("Show Top", 5, max_display, default_display, 5)
            with col3:
                min_leverage = st.number_input("Min Leverage", 0, 20, 5)
                filtered_df = lineups_df[lineups_df['Leverage_Score'] >= min_leverage]
                if len(filtered_df) == 0:
                    filtered_df = lineups_df
            
            if display_format == "GPP Summary":
                display_cols = ['Lineup', 'Strategy', 'Captain', 'Captain_Own%', 'Projected', 
                              'Ceiling_99th', 'Total_Ownership', 'Leverage_Score', 
                              'GPP_Score', 'Ship_Equity']
                
                display_cols = [col for col in display_cols if col in filtered_df.columns]
                
                st.dataframe(
                    filtered_df[display_cols].head(show_top_n),
                    use_container_width=True
                )
            
            elif display_format == "Detailed":
                for i, (idx, lineup) in enumerate(filtered_df.head(show_top_n).iterrows(), 1):
                    tier_emoji = "💎" if lineup['Total_Ownership'] < 60 else "🟢" if lineup['Total_Ownership'] < 80 else "🟡"
                    
                    with st.expander(f"{tier_emoji} Lineup #{i} - {lineup['Strategy']} - GPP Score: {lineup.get('GPP_Score', 0):.1f}"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown("**Roster:**")
                            captain_own = lineup.get('Captain_Own%', 0)
                            st.write(f"🎯 **Captain:** {lineup['Captain']} ({captain_own:.1f}%)")
                            st.write("**FLEX:**")
                            flex_players = lineup['FLEX'] if isinstance(lineup['FLEX'], list) else []
                            for player in flex_players:
                                if player in df['Player'].values:
                                    pos = df[df['Player'] == player]['Position'].values[0]
                                    own = df[df['Player'] == player]['Ownership'].values[0]
                                    emoji = "💎" if own < 5 else "🟢" if own < 10 else ""
                                    st.write(f"{emoji} {player} ({pos}) - {own:.0f}%")
                        
                        with col2:
                            st.markdown("**GPP Projections:**")
                            st.metric("99th %ile", f"{lineup.get('Ceiling_99th', lineup['Projected']*1.8):.1f}")
                            st.metric("99.9th %ile", f"{lineup.get('Ceiling_99_9th', lineup['Projected']*2):.1f}")
                            st.metric("Ship Rate", f"{lineup.get('Ship_Rate', 0.1):.3f}%")
                            st.metric("Boom Rate", f"{lineup.get('Boom_Rate', 5):.1f}%")
                        
                        with col3:
                            st.markdown("**GPP Metrics:**")
                            st.write(f"💰 Salary: ${lineup['Salary']:,}")
                            st.write(f"📊 Total Own: {lineup['Total_Ownership']:.1f}%")
                            st.write(f"🎯 Leverage: {lineup['Leverage_Score']:.1f}")
                            st.write(f"🏆 GPP Score: {lineup.get('GPP_Score', 0):.1f}")
                        
                        with col4:
                            st.markdown("**Stack Info:**")
                            if lineup['Has_Stack']:
                                st.success(f"✅ {lineup['Stack_Details']}")
                            else:
                                st.info("No primary stack")
                            st.write(f"Field: {lineup['Field_Size']}")
                            st.write(f"Tier: {lineup.get('Ownership_Tier', 'Unknown')}")
            
            else:  # Compact view
                for i, (idx, lineup) in enumerate(filtered_df.head(show_top_n).iterrows(), 1):
                    emoji = "💎" if lineup['Total_Ownership'] < 60 else "🟢" if lineup['Total_Ownership'] < 80 else "🟡"
                    captain_own = lineup.get('Captain_Own%', 0)
                    flex_players = lineup['FLEX'] if isinstance(lineup['FLEX'], list) else []
                    flex_preview = ', '.join(flex_players[:3]) + ('...' if len(flex_players) > 3 else '')
                    st.write(f"{emoji} **#{i}:** CPT: {lineup['Captain']} ({captain_own:.0f}%) | FLEX: {flex_preview} | Own: {lineup['Total_Ownership']:.0f}% | GPP: {lineup.get('GPP_Score', 0):.0f}")
        
        with tab2:
            st.markdown("### 🔄 GPP Captain Pivots")
            
            if 'pivots_df' in st.session_state and st.session_state['pivots_df']:
                pivots_df = st.session_state['pivots_df']
                
                st.info(f"Generated {len(pivots_df)} GPP captain pivot variations")
                
                for i, pivot in enumerate(pivots_df[:10], 1):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f"**#{i}:** {pivot['Original_Captain']} → {pivot['Captain']}")
                    with col2:
                        st.write(f"Captain Own: {pivot['Captain_Own%']:.1f}%")
                    with col3:
                        st.write(f"Leverage: +{pivot['Leverage_Gain']:.1f}")
                    with col4:
                        st.write(f"{pivot['Pivot_Type']}")
            else:
                st.info("Enable captain pivots in settings to generate variations")
        
        with tab3:
            st.markdown("### 📈 GPP Tournament Analysis")
            display_gpp_lineup_analysis(lineups_df, df, field_size)
        
        with tab4:
            st.markdown("### 💎 GPP Leverage Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Low-Owned Captains (<15%)")
                if 'Captain_Own%' in lineups_df.columns:
                    low_captains = lineups_df[lineups_df['Captain_Own%'] < 15]['Captain'].value_counts()
                else:
                    low_captains = pd.Series()
                    
                for player, count in low_captains.items():
                    if player in df['Player'].values:
                        own = df[df['Player'] == player]['Ownership'].values[0]
                        emoji = "💎" if own < 5 else "🟢" if own < 10 else "🟡"
                        pct = count / len(lineups_df) * 100
                        st.write(f"{emoji} {player} ({own:.0f}%) - {count} lineups ({pct:.0f}%)")
            
            with col2:
                st.markdown("#### 🔗 Leverage Stacks")
                leverage_stacks = []
                
                for idx, row in lineups_df.iterrows():
                    if row['Has_Stack'] and row['Total_Ownership'] < 80:
                        stacks = row['Stack_Details'].split(', ')
                        for stack in stacks:
                            if stack != 'None':
                                leverage_stacks.append(stack)
                
                if leverage_stacks:
                    stack_counts = Counter(leverage_stacks)
                    for stack, count in stack_counts.most_common(10):
                        pct = count / len(lineups_df) * 100
                        st.write(f"🔗 {stack} - {count} lineups ({pct:.0f}%)")
            
            st.markdown("#### 💎 Super Leverage Plays (<5% ownership)")
            
            player_usage = defaultdict(int)
            for idx, row in lineups_df.iterrows():
                lineup_players = [row['Captain']] + (row['FLEX'] if isinstance(row['FLEX'], list) else [])
                for player in lineup_players:
                    player_usage[player] += 1
            
            super_leverage = []
            for player, usage in player_usage.items():
                if player in df['Player'].values:
                    own = df[df['Player'] == player]['Ownership'].values[0]
                    if own < 5 and usage >= 3:
                        super_leverage.append((player, own, usage))
            
            if super_leverage:
                super_leverage.sort(key=lambda x: x[2], reverse=True)
                for player, own, usage in super_leverage[:10]:
                    pct = usage / len(lineups_df) * 100
                    st.write(f"💎 {player} ({own:.0f}%) - {usage} lineups ({pct:.0f}%)")
        
        with tab5:
            st.markdown("### 📊 GPP Simulation Results")
            
            sim_cols = ['Lineup', 'Captain', 'Total_Ownership', 'Mean', 
                       'Ceiling_95th', 'Ceiling_99th', 'Ceiling_99_9th',
                       'Ship_Rate', 'Elite_Rate', 'Boom_Rate']
            
            sim_cols = [col for col in sim_cols if col in lineups_df.columns]
            
            if sim_cols:
                st.dataframe(
                    lineups_df[sim_cols].head(20),
                    use_container_width=True
                )
            
            if 'Ship_Rate' in lineups_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(lineups_df['Ship_Rate'], bins=20, alpha=0.7, color='purple', edgecolor='black')
                ax.set_xlabel('Ship Rate (%)')
                ax.set_ylabel('Number of Lineups')
                ax.set_title(f'Tournament Win Probability Distribution - {field_size}')
                st.pyplot(fig)
        
        with tab6:
            st.markdown("### 💾 Export GPP Lineups")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### DraftKings Upload Format")
                
                dk_lineups = []
                export_df = lineups_df.head(export_top_n)
                
                for idx, row in export_df.iterrows():
                    flex_players = row['FLEX'] if isinstance(row['FLEX'], list) else []
                    dk_lineups.append({
                        'CPT': row['Captain'],
                        'FLEX 1': flex_players[0] if len(flex_players) > 0 else '',
                        'FLEX 2': flex_players[1] if len(flex_players) > 1 else '',
                        'FLEX 3': flex_players[2] if len(flex_players) > 2 else '',
                        'FLEX 4': flex_players[3] if len(flex_players) > 3 else '',
                        'FLEX 5': flex_players[4] if len(flex_players) > 4 else ''
                    })
                
                dk_df = pd.DataFrame(dk_lineups)
                
                st.write(f"Preview (first 5 of {len(dk_df)}):")
                st.dataframe(dk_df.head(), use_container_width=True)
                
                csv = dk_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download DK CSV ({len(dk_df)} lineups)",
                    data=csv,
                    file_name=f"dk_gpp_{field_size}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.markdown("#### GPP Analysis Export")
                
                export_analysis = export_df.copy()
                export_analysis['FLEX'] = export_analysis['FLEX'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                )
                
                gpp_export_cols = ['Lineup', 'Strategy', 'Captain', 'Captain_Own%', 'FLEX', 
                                  'Projected', 'Salary', 'Total_Ownership', 'Leverage_Score',
                                  'Ceiling_95th', 'Ceiling_99th', 'Ceiling_99_9th',
                                  'Ship_Rate', 'Elite_Rate', 'Boom_Rate',
'GPP_Score', 'Tournament_EV', 'Has_Stack']
                
                gpp_export_cols = [col for col in gpp_export_cols if col in export_analysis.columns]
                
                final_export = export_analysis[gpp_export_cols]
                
                st.write(f"Preview (first 5 of {len(final_export)}):")
                st.dataframe(final_export.head(), use_container_width=True)
                
                csv_full = final_export.to_csv(index=False)
                st.download_button(
                    label=f"📊 Download GPP Analysis ({len(final_export)} lineups)",
                    data=csv_full,
                    file_name=f"gpp_analysis_{field_size}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            # Captain pivot export
            if include_pivots and 'pivots_df' in st.session_state and st.session_state['pivots_df']:
                st.markdown("#### 🔄 Captain Pivots Export")
                
                pivots_list = st.session_state['pivots_df']
                pivot_export = pd.DataFrame(pivots_list)
                
                csv_pivots = pivot_export.to_csv(index=False)
                st.download_button(
                    label=f"🔄 Download Captain Pivots ({len(pivot_export)} pivots)",
                    data=csv_pivots,
                    file_name=f"captain_pivots_{field_size}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            # Session history export
            st.markdown("#### 📜 Session History")
            if st.session_state.get('optimization_history'):
                st.info(f"You have {len(st.session_state['optimization_history'])} saved sessions")
                
                if st.button("Export Session History"):
                    history_data = []
                    for session in st.session_state['optimization_history']:
                        history_data.append({
                            'Timestamp': session['timestamp'],
                            'Field Size': session['field_size'],
                            'Lineups Generated': len(session['lineups']),
                            'Settings': str(session['settings'])
                        })
                    
                    history_df = pd.DataFrame(history_data)
                    csv_history = history_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📜 Download Session History",
                        data=csv_history,
                        file_name=f"session_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )

# Footer with GPP-specific tips
st.markdown("---")
st.markdown("""
### 🏆 NFL GPP Tournament Optimizer - Professional Edition v5.1

**New Performance Features:**
- ⚡ **Cached Calculations**: Correlations and simulations are cached for faster re-runs
- 💾 **Session History**: Automatically saves your optimization sessions
- 📊 **Performance Monitoring**: Track operation timings and bottlenecks
- 🔄 **Smart Caching**: Reuses validation and analysis results when possible

**GPP-Specific Features:**
- 💎 **Super Leverage Detection**: Identifies <5% owned tournament winners
- 🎯 **Field-Size Optimization**: Tailored strategies for different GPP types
- 🔄 **Captain Pivot Engine**: Creates unique lineups with leverage captains
- 📊 **Ship Equity Calculator**: Tournament win probability analysis
- 🤖 **Dual AI System**: Game theory + correlation for maximum edge

**GPP Strategy Tips:**
- **Milly Maker**: Target <80% total ownership with zero chalk tolerance
- **Large Field**: 60-90% ownership with 2+ leverage plays minimum
- **Small Field**: Can use slightly higher ownership (80-120%) 
- **Captain Selection**: Prioritize <15% owned captains for differentiation
- **Stacking**: Low-owned game stacks in projected shootouts (50+ total)

**Ownership Tier Guide:**
- 💎 Super Leverage (<5%): Maximum tournament equity
- 🟢 Leverage (5-10%): Strong GPP plays
- 🟡 Pivot (10-20%): Balanced risk/reward
- 🟠 Chalk (20-35%): Use sparingly
- 🔴 Mega Chalk (35%+): Avoid in large field GPPs

**Performance Tips:**
- Enable caching for repeated optimizations with same data
- Use session history to track successful configurations
- Check debug panel for performance metrics and bottlenecks
- Clear cache if experiencing issues with stale data

**Version:** 5.1 GPP Edition | **Focus:** Tournament Winning Upside with Performance Optimization

*Maximize your tournament equity with blazing fast optimization!* 🚀
""")

# Display performance summary if available
if 'lineups_df' in st.session_state:
    perf = get_performance_monitor()
    metrics = perf.get_metrics()
    
    if metrics['timers']:
        with st.expander("⚡ Performance Summary"):
            total_time = sum(timer['total'] for timer in metrics['timers'].values())
            st.write(f"**Total Time:** {total_time:.2f}s")
            
            for name, stats in metrics['timers'].items():
                st.write(f"- {name}: {stats['average']:.3f}s avg ({stats['count']} operations)")
            
            if metrics['counters']:
                st.write("**Operation Counts:**")
                for name, count in metrics['counters'].items():
                    st.write(f"- {name}: {count}")

# Display current timestamp and field size
if 'field_size' in st.session_state:
    current_field = st.session_state['field_size']
    st.caption(f"GPP Optimizer last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Field: {current_field} | Cache Status: Active")
else:
    st.caption(f"GPP Optimizer ready: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | No active session")


