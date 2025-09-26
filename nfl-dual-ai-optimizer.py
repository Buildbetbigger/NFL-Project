import streamlit as st
import pandas as pd
import pulp
import numpy as np
from scipy.stats import multivariate_normal, gamma
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Set
import json
from dataclasses import dataclass
from enum import Enum
import itertools
from collections import defaultdict

st.set_page_config(page_title="NFL Dual-AI Optimizer v2", page_icon="🏈", layout="wide")
st.title('🏈 NFL Showdown Optimizer - Phase 1 Enhanced')

# Configuration
class OptimizerConfig:
    SALARY_CAP = 50000
    ROSTER_SIZE = 6  # 1 Captain + 5 FLEX
    MAX_PLAYERS_PER_TEAM = 4
    CAPTAIN_MULTIPLIER = 1.5
    DEFAULT_OWNERSHIP = 5
    MIN_SALARY = 1000
    NUM_SIMS = 10000
    FIELD_SIZE = 100000
    
    # Ownership buckets
    OWNERSHIP_BUCKETS = {
        'mega_chalk': (40, 100),      # 40%+ ownership
        'chalk': (25, 40),             # 25-40% ownership
        'pivot': (10, 25),             # 10-25% ownership
        'leverage': (5, 10),           # 5-10% ownership
        'super_leverage': (0, 5)       # <5% ownership
    }
    
    # Bucket constraints for different strategies
    BUCKET_RULES = {
        'balanced': {
            'mega_chalk': (0, 2),      # Max 2 mega chalk
            'chalk': (1, 3),            # 1-3 chalk plays
            'pivot': (1, 3),            # 1-3 pivots
            'leverage': (0, 2),         # 0-2 leverage
            'super_leverage': (0, 1)    # Max 1 super leverage
        },
        'contrarian': {
            'mega_chalk': (0, 1),       # Max 1 mega chalk
            'chalk': (0, 2),            # Max 2 chalk
            'pivot': (2, 4),            # 2-4 pivots
            'leverage': (1, 3),         # 1-3 leverage
            'super_leverage': (0, 2)    # 0-2 super leverage
        },
        'leverage': {
            'mega_chalk': (0, 0),       # No mega chalk
            'chalk': (0, 1),            # Max 1 chalk
            'pivot': (1, 3),            # 1-3 pivots
            'leverage': (2, 4),         # 2-4 leverage
            'super_leverage': (1, 2)    # 1-2 super leverage
        }
    }

class StrategyType(Enum):
    LEVERAGE = "leverage"
    CORRELATION = "correlation"
    BALANCED = "balanced"
    CONTRARIAN = "contrarian"
    GAME_STACK = "game_stack"

@dataclass
class AIRecommendation:
    """Structure for AI strategist recommendations"""
    strategist_name: str
    confidence: float
    captain_targets: List[str]
    stacks: List[Dict[str, str]]
    fades: List[str]
    boosts: List[str]
    strategy_weights: Dict[StrategyType, float]
    key_insights: List[str]

class OwnershipBucketManager:
    """Manages ownership bucketing and constraints"""
    
    @staticmethod
    def get_bucket(ownership: float) -> str:
        """Determine which bucket a player belongs to"""
        for bucket_name, (min_own, max_own) in OptimizerConfig.OWNERSHIP_BUCKETS.items():
            if min_own <= ownership < max_own:
                return bucket_name
        return 'super_leverage'
    
    @staticmethod
    def categorize_players(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize all players into ownership buckets"""
        buckets = defaultdict(list)
        for _, row in df.iterrows():
            bucket = OwnershipBucketManager.get_bucket(row['Ownership'])
            buckets[bucket].append(row['Player'])
        return dict(buckets)
    
    @staticmethod
    def validate_lineup_buckets(lineup_players: List[str], df: pd.DataFrame, 
                               strategy: str = 'balanced') -> Tuple[bool, str]:
        """Check if lineup meets bucket constraints"""
        ownership_dict = df.set_index('Player')['Ownership'].to_dict()
        bucket_counts = defaultdict(int)
        
        for player in lineup_players:
            ownership = ownership_dict.get(player, OptimizerConfig.DEFAULT_OWNERSHIP)
            bucket = OwnershipBucketManager.get_bucket(ownership)
            bucket_counts[bucket] += 1
        
        rules = OptimizerConfig.BUCKET_RULES.get(strategy, OptimizerConfig.BUCKET_RULES['balanced'])
        
        for bucket_name, (min_count, max_count) in rules.items():
            count = bucket_counts.get(bucket_name, 0)
            if count < min_count:
                return False, f"Need at least {min_count} {bucket_name} players"
            if count > max_count:
                return False, f"Too many {bucket_name} players ({count}/{max_count})"
        
        return True, "Valid"
    
    @staticmethod
    def get_bucket_summary(lineup_players: List[str], df: pd.DataFrame) -> str:
        """Get a summary of lineup's ownership profile"""
        ownership_dict = df.set_index('Player')['Ownership'].to_dict()
        bucket_counts = defaultdict(int)
        total_ownership = 0
        
        for player in lineup_players:
            ownership = ownership_dict.get(player, OptimizerConfig.DEFAULT_OWNERSHIP)
            bucket = OwnershipBucketManager.get_bucket(ownership)
            bucket_counts[bucket] += 1
            total_ownership += ownership
        
        summary = f"Total: {total_ownership:.1f}% | "
        summary += " | ".join([f"{k}: {v}" for k, v in bucket_counts.items() if v > 0])
        return summary

class CaptainPivotGenerator:
    """Generates captain pivot variations of lineups"""
    
    @staticmethod
    def generate_pivots(lineup: Dict, df: pd.DataFrame, max_pivots: int = 3) -> List[Dict]:
        """Generate captain pivot variations of a lineup"""
        captain = lineup['Captain']
        flex_players = lineup['FLEX']
        all_players = [captain] + list(flex_players)
        
        salaries = df.set_index('Player')['Salary'].to_dict()
        points = df.set_index('Player')['Projected_Points'].to_dict()
        ownership = df.set_index('Player')['Ownership'].to_dict()
        
        pivot_lineups = []
        
        # Try each FLEX player as captain
        for new_captain in flex_players[:max_pivots]:
            # Calculate salary adjustment
            old_captain_salary = salaries.get(captain, 0)
            new_captain_salary = salaries.get(new_captain, 0)
            
            # Captain pays 1.5x, FLEX pays 1x
            salary_freed = old_captain_salary * 0.5  # Captain becomes FLEX (saves 0.5x)
            salary_needed = new_captain_salary * 0.5  # FLEX becomes captain (costs 0.5x more)
            
            if salary_freed >= salary_needed:
                # Simple swap is valid
                new_flex = [p for p in flex_players if p != new_captain] + [captain]
                
                pivot_lineup = lineup.copy()
                pivot_lineup['Captain'] = new_captain
                pivot_lineup['FLEX'] = new_flex
                pivot_lineup['Pivot_Type'] = 'Simple Swap'
                pivot_lineup['Original_Captain'] = captain
                
                # Recalculate metrics
                total_proj = points.get(new_captain, 0) * 1.5 + sum(points.get(p, 0) for p in new_flex)
                total_own = ownership.get(new_captain, 5) * 1.5 + sum(ownership.get(p, 5) for p in new_flex)
                
                pivot_lineup['Projected'] = round(total_proj, 2)
                pivot_lineup['Total_Ownership'] = round(total_own, 1)
                pivot_lineup['Ownership_Delta'] = round(total_own - lineup['Total_Ownership'], 1)
                
                pivot_lineups.append(pivot_lineup)
        
        return pivot_lineups
    
    @staticmethod
    def find_leverage_captain_swaps(lineup: Dict, df: pd.DataFrame, 
                                   ai_recommendations: Dict) -> List[Dict]:
        """Find captain swaps that create maximum leverage"""
        captain = lineup['Captain']
        flex_players = lineup['FLEX']
        
        ownership_dict = df.set_index('Player')['Ownership'].to_dict()
        captain_ownership = ownership_dict.get(captain, 10)
        
        leverage_pivots = []
        
        # Look for low-owned captains in FLEX
        for player in flex_players:
            player_own = ownership_dict.get(player, 10)
            
            # Significant leverage if swapping high-owned captain for low-owned
            if captain_ownership > 30 and player_own < 15:
                leverage_score = (captain_ownership - player_own) / captain_ownership
                
                pivot = {
                    'original_captain': captain,
                    'new_captain': player,
                    'leverage_score': leverage_score,
                    'ownership_swing': captain_ownership - player_own
                }
                leverage_pivots.append(pivot)
        
        return sorted(leverage_pivots, key=lambda x: x['leverage_score'], reverse=True)

class GameTheoryStrategist:
    """AI Strategist 1: Focus on ownership leverage and game theory"""
    
    @staticmethod
    def generate_prompt(df: pd.DataFrame, game_info: Dict) -> str:
        """Generate prompt for game theory analysis"""
        
        # Ownership bucket analysis
        bucket_manager = OwnershipBucketManager()
        buckets = bucket_manager.categorize_players(df)
        
        # Get key players by bucket
        mega_chalk = df[df['Player'].isin(buckets.get('mega_chalk', []))][['Player', 'Position', 'Ownership', 'Projected_Points']]
        leverage_plays = df[df['Player'].isin(buckets.get('leverage', []) + buckets.get('super_leverage', []))][['Player', 'Position', 'Ownership', 'Projected_Points']]
        
        return f"""
        As a DFS Game Theory Expert, analyze this NFL Showdown slate for LEVERAGE opportunities:
        
        Game: {game_info.get('teams', 'Unknown')} | O/U: {game_info.get('total', 0)} | Spread: {game_info.get('spread', 0)}
        
        MEGA CHALK (40%+ ownership):
        {mega_chalk.to_string() if not mega_chalk.empty else 'None'}
        
        LEVERAGE PLAYS (<10% ownership):
        {leverage_plays.to_string() if not leverage_plays.empty else 'None'}
        
        OWNERSHIP BUCKETS:
        - Mega Chalk (40%+): {len(buckets.get('mega_chalk', []))} players
        - Chalk (25-40%): {len(buckets.get('chalk', []))} players  
        - Pivot (10-25%): {len(buckets.get('pivot', []))} players
        - Leverage (<10%): {len(buckets.get('leverage', []) + buckets.get('super_leverage', []))} players
        
        Provide strategic recommendations:
        1. Which chalk plays are MUST FADES (bad projection/ownership)?
        2. Which leverage plays are MUST PLAYS (hidden upside)?
        3. Optimal lineup construction rules (e.g., "max 1 mega chalk")?
        4. Captain leverage opportunities (low-owned with ceiling)?
        
        Return JSON:
        {{
            "leverage_captains": ["player1", "player2"],
            "must_fades": ["overowned1"],
            "must_plays": ["leverage1"],
            "construction_rules": {{"max_mega_chalk": 1, "min_leverage": 2}},
            "ownership_targets": {{"min": 80, "max": 120}},
            "contrarian_stacks": [{{"player1": "name1", "player2": "name2"}}],
            "confidence_score": 0.85,
            "key_insights": ["insight1", "insight2"]
        }}
        """
    
    @staticmethod
    def parse_response(response: str, df: pd.DataFrame) -> AIRecommendation:
        """Parse game theory strategist response"""
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            else:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end] if start >= 0 else response
            
            data = json.loads(json_str)
            
            # Dynamic strategy weights based on construction rules
            rules = data.get('construction_rules', {})
            max_chalk = rules.get('max_mega_chalk', 2)
            min_leverage = rules.get('min_leverage', 1)
            
            if max_chalk <= 1 and min_leverage >= 2:
                strategy_weights = {
                    StrategyType.LEVERAGE: 0.5,
                    StrategyType.CONTRARIAN: 0.3,
                    StrategyType.BALANCED: 0.2,
                    StrategyType.CORRELATION: 0.0,
                    StrategyType.GAME_STACK: 0.0
                }
            else:
                strategy_weights = {
                    StrategyType.BALANCED: 0.4,
                    StrategyType.LEVERAGE: 0.3,
                    StrategyType.CONTRARIAN: 0.2,
                    StrategyType.CORRELATION: 0.1,
                    StrategyType.GAME_STACK: 0.0
                }
            
            return AIRecommendation(
                strategist_name="Game Theory AI",
                confidence=data.get('confidence_score', 0.7),
                captain_targets=data.get('leverage_captains', []),
                stacks=[{'player1': s.get('player1'), 'player2': s.get('player2')} 
                       for s in data.get('contrarian_stacks', [])],
                fades=data.get('must_fades', []),
                boosts=data.get('must_plays', []),
                strategy_weights=strategy_weights,
                key_insights=data.get('key_insights', [])
            )
            
        except Exception as e:
            st.warning(f"Error parsing Game Theory response: {e}")
            return AIRecommendation(
                strategist_name="Game Theory AI",
                confidence=0.5,
                captain_targets=[],
                stacks=[],
                fades=[],
                boosts=[],
                strategy_weights={s: 0.2 for s in StrategyType},
                key_insights=["Failed to parse - using defaults"]
            )

class CorrelationStrategist:
    """AI Strategist 2: Focus on correlation, stacking, and game flow"""
    
    @staticmethod
    def generate_prompt(df: pd.DataFrame, game_info: Dict) -> str:
        """Generate prompt for correlation analysis"""
        
        teams = df['Team'].unique()
        team_breakdown = {}
        
        for team in teams:
            team_df = df[df['Team'] == team]
            team_breakdown[team] = {
                'QB': team_df[team_df['Position'] == 'QB'][['Player', 'Salary', 'Projected_Points']].to_dict('records'),
                'Pass_Catchers': team_df[team_df['Position'].isin(['WR', 'TE'])][['Player', 'Position', 'Salary', 'Projected_Points']].to_dict('records'),
                'RB': team_df[team_df['Position'] == 'RB'][['Player', 'Salary', 'Projected_Points']].to_dict('records')
            }
        
        return f"""
        As a DFS Correlation Expert, identify optimal stacking patterns:
        
        Game: {game_info.get('teams', 'Unknown')} | O/U: {game_info.get('total', 0)} | Spread: {game_info.get('spread', 0)}
        
        TEAM BREAKDOWNS:
        {json.dumps(team_breakdown, indent=2)}
        
        Analyze:
        1. Primary stacks (QB + pass catchers with highest correlation)
        2. Game stacks (players from both teams that correlate)
        3. Leverage stacks (lower owned but high correlation)
        4. Anti-correlation plays (what to avoid together)
        5. Game script dependent correlations
        
        Consider:
        - Target share leaders
        - Red zone usage
        - Game script scenarios (blowout vs shootout)
        - Comeback potential
        
        Return JSON:
        {{
            "primary_stacks": [{{"qb": "name", "receiver": "name", "correlation": 0.6, "stack_ownership": 25}}],
            "game_stacks": [{{"team1_player": "name1", "team2_player": "name2", "scenario": "shootout"}}],
            "leverage_stacks": [{{"player1": "name1", "player2": "name2", "combined_own": 15}}],
            "avoid_together": [{{"player1": "name1", "player2": "name2", "reason": "negative correlation"}}],
            "game_script": {{
                "most_likely": "balanced",
                "correlations": {{"balanced": [{{"player1": "p1", "player2": "p2"}}]}}
            }},
            "confidence": 0.8,
            "insights": ["key insight 1", "key insight 2"]
        }}
        """
    
    @staticmethod
    def parse_response(response: str, df: pd.DataFrame) -> AIRecommendation:
        """Parse correlation strategist response"""
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            else:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end] if start >= 0 else response
            
            data = json.loads(json_str)
            
            # Build comprehensive stacks list
            all_stacks = []
            
            # Primary stacks
            for stack in data.get('primary_stacks', []):
                all_stacks.append({
                    'player1': stack.get('qb'),
                    'player2': stack.get('receiver')
                })
            
            # Game stacks
            for stack in data.get('game_stacks', []):
                all_stacks.append({
                    'player1': stack.get('team1_player'),
                    'player2': stack.get('team2_player')
                })
            
            # Leverage stacks
            for stack in data.get('leverage_stacks', []):
                all_stacks.append({
                    'player1': stack.get('player1'),
                    'player2': stack.get('player2')
                })
            
            # Determine strategy weights based on game script
            game_script = data.get('game_script', {}).get('most_likely', 'balanced')
            
            if game_script == 'shootout':
                strategy_weights = {
                    StrategyType.GAME_STACK: 0.5,
                    StrategyType.CORRELATION: 0.4,
                    StrategyType.BALANCED: 0.1,
                    StrategyType.LEVERAGE: 0.0,
                    StrategyType.CONTRARIAN: 0.0
                }
            elif game_script == 'blowout':
                strategy_weights = {
                    StrategyType.CONTRARIAN: 0.4,
                    StrategyType.LEVERAGE: 0.3,
                    StrategyType.BALANCED: 0.3,
                    StrategyType.CORRELATION: 0.0,
                    StrategyType.GAME_STACK: 0.0
                }
            else:
                strategy_weights = {
                    StrategyType.CORRELATION: 0.4,
                    StrategyType.BALANCED: 0.3,
                    StrategyType.GAME_STACK: 0.2,
                    StrategyType.LEVERAGE: 0.1,
                    StrategyType.CONTRARIAN: 0.0
                }
            
            return AIRecommendation(
                strategist_name="Correlation AI",
                confidence=data.get('confidence', 0.7),
                captain_targets=[s.get('qb') for s in data.get('primary_stacks', []) if s.get('qb')],
                stacks=all_stacks,
                fades=[],
                boosts=[],
                strategy_weights=strategy_weights,
                key_insights=data.get('insights', [])
            )
            
        except Exception as e:
            st.warning(f"Error parsing Correlation response: {e}")
            return AIRecommendation(
                strategist_name="Correlation AI",
                confidence=0.5,
                captain_targets=[],
                stacks=[],
                fades=[],
                boosts=[],
                strategy_weights={s: 0.2 for s in StrategyType},
                key_insights=["Failed to parse - using defaults"]
            )

class DualAIOptimizer:
    """Main optimizer combining both AI strategists with Phase 1 enhancements"""
    
    def __init__(self, df: pd.DataFrame, game_info: Dict):
        self.df = df
        self.game_info = game_info
        self.game_theory_ai = GameTheoryStrategist()
        self.correlation_ai = CorrelationStrategist()
        self.bucket_manager = OwnershipBucketManager()
        self.pivot_generator = CaptainPivotGenerator()
        self.recommendations = {}
    
    def combine_recommendations(self, rec1: AIRecommendation, rec2: AIRecommendation) -> Dict:
        """Combine recommendations from both AIs"""
        
        total_confidence = rec1.confidence + rec2.confidence
        w1 = rec1.confidence / total_confidence if total_confidence > 0 else 0.5
        w2 = rec2.confidence / total_confidence if total_confidence > 0 else 0.5
        
        # Combine captain targets
        all_captains = set(rec1.captain_targets + rec2.captain_targets)
        captain_scores = {}
        for captain in all_captains:
            score = 0
            if captain in rec1.captain_targets:
                score += w1
            if captain in rec2.captain_targets:
                score += w2
            captain_scores[captain] = score
        
        # Combine strategy weights
        combined_weights = {}
        for strategy in StrategyType:
            combined_weights[strategy] = (
                rec1.strategy_weights.get(strategy, 0) * w1 +
                rec2.strategy_weights.get(strategy, 0) * w2
            )
        
        # Normalize
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            combined_weights = {k: v/total_weight for k, v in combined_weights.items()}
        
        # Consensus fades (both agree)
        consensus_fades = set(rec1.fades) & set(rec2.fades) if rec1.fades and rec2.fades else set()
        
        # All boosts (either agrees)
        all_boosts = set(rec1.boosts) | set(rec2.boosts) if rec1.boosts or rec2.boosts else set()
        
        # Combine stacks
        all_stacks = rec1.stacks + rec2.stacks
        unique_stacks = []
        seen = set()
        for stack in all_stacks:
            key = tuple(sorted([stack.get('player1', ''), stack.get('player2', '')]))
            if key not in seen and key[0] and key[1]:
                unique_stacks.append(stack)
                seen.add(key)
        
        return {
            'captain_scores': captain_scores,
            'strategy_weights': combined_weights,
            'consensus_fades': list(consensus_fades),
            'all_boosts': list(all_boosts),
            'combined_stacks': unique_stacks,
            'confidence': (rec1.confidence + rec2.confidence) / 2,
            'insights': rec1.key_insights + rec2.key_insights
        }
    
    def generate_lineups_with_buckets(self, num_lineups: int, rec1: AIRecommendation, 
                                     rec2: AIRecommendation) -> pd.DataFrame:
        """Generate lineups with ownership bucket constraints"""
        
        combined = self.combine_recommendations(rec1, rec2)
        
        # Prepare data
        players = self.df['Player'].tolist()
        positions = self.df.set_index('Player')['Position'].to_dict()
        teams = self.df.set_index('Player')['Team'].to_dict()
        salaries = self.df.set_index('Player')['Salary'].to_dict()
        points = self.df.set_index('Player')['Projected_Points'].to_dict()
        ownership = self.df.set_index('Player')['Ownership'].to_dict()
        
        # Apply AI adjustments
        adjusted_points = points.copy()
        for player in combined['consensus_fades']:
            if player in adjusted_points:
                adjusted_points[player] *= 0.8
        
        for player in combined['all_boosts']:
            if player in adjusted_points:
                adjusted_points[player] *= 1.15
        
        # Get ownership buckets
        player_buckets = {}
        for player in players:
            player_buckets[player] = self.bucket_manager.get_bucket(
                ownership.get(player, OptimizerConfig.DEFAULT_OWNERSHIP)
            )
        
        # Determine lineup distribution
        lineups_per_strategy = {}
        for strategy, weight in combined['strategy_weights'].items():
            lineups_per_strategy[strategy] = max(1, int(num_lineups * weight))
        
        # Adjust to match exact number
        total_assigned = sum(lineups_per_strategy.values())
        if total_assigned < num_lineups:
            lineups_per_strategy[StrategyType.BALANCED] = lineups_per_strategy.get(StrategyType.BALANCED, 0) + (num_lineups - total_assigned)
        elif total_assigned > num_lineups:
            while total_assigned > num_lineups and total_assigned > 0:
                max_strategy = max(lineups_per_strategy, key=lineups_per_strategy.get)
                if lineups_per_strategy[max_strategy] > 1:
                    lineups_per_strategy[max_strategy] -= 1
                    total_assigned -= 1
        
        all_lineups = []
        lineup_num = 0
        
        for strategy, count in lineups_per_strategy.items():
            strategy_name = 'contrarian' if strategy == StrategyType.CONTRARIAN else 'balanced'
            if strategy == StrategyType.LEVERAGE:
                strategy_name = 'leverage'
            
            bucket_rules = OptimizerConfig.BUCKET_RULES[strategy_name]
            
            for i in range(count):
                lineup_num += 1
                model = pulp.LpProblem(f"Lineup_{lineup_num}_{strategy.value}", pulp.LpMaximize)
                
                flex = pulp.LpVariable.dicts("Flex", players, cat='Binary')
                captain = pulp.LpVariable.dicts("Captain", players, cat='Binary')
                
                # Objective function
                if strategy == StrategyType.LEVERAGE:
                    captain_multiplier = {}
                    for p in players:
                        mult = 1.0
                        if p in combined['captain_scores']:
                            mult += combined['captain_scores'][p] * 0.5
                        if ownership.get(p, 10) < 15:
                            mult += 0.3
                        captain_multiplier[p] = mult
                    
                    model += pulp.lpSum([
                        adjusted_points[p] * flex[p] +
                        1.5 * adjusted_points[p] * captain[p] * captain_multiplier[p]
                        for p in players
                    ])
                
                elif strategy == StrategyType.CONTRARIAN:
                    model += pulp.lpSum([
                        adjusted_points[p] * flex[p] * (1 - ownership.get(p, 5)/200) +
                        1.5 * adjusted_points[p] * captain[p] * (1 - ownership.get(p, 5)/150)
                        for p in players
                    ])
                
                else:
                    model += pulp.lpSum([
                        adjusted_points[p] * flex[p] +
                        1.5 * adjusted_points[p] * captain[p]
                        for p in players
                    ])
                
                # Standard constraints
                model += pulp.lpSum(captain.values()) == 1
                model += pulp.lpSum(flex.values()) == 5
                
                for p in players:
                    model += flex[p] + captain[p] <= 1
                
                model += pulp.lpSum([
                    salaries[p] * flex[p] + 1.5 * salaries[p] * captain[p]
                    for p in players
                ]) <= OptimizerConfig.SALARY_CAP
                
                for team in self.df['Team'].unique():
                    team_players = [p for p in players if teams.get(p) == team]
                    model += pulp.lpSum([flex[p] + captain[p] for p in team_players]) <= OptimizerConfig.MAX_PLAYERS_PER_TEAM
                
                # Ownership bucket constraints
                for bucket_name, (min_count, max_count) in bucket_rules.items():
                    bucket_players = [p for p in players if player_buckets[p] == bucket_name]
                    if bucket_players:
                        if min_count > 0:
                            model += pulp.lpSum([flex[p] + captain[p] for p in bucket_players]) >= min_count
                        if max_count < 6:
                            model += pulp.lpSum([flex[p] + captain[p] for p in bucket_players]) <= max_count
                
                # Diversity constraint
                if lineup_num > 1 and all_lineups:
                    for prev_lineup in all_lineups[-min(3, len(all_lineups)):]:
                        prev_players = [prev_lineup['Captain']] + prev_lineup['FLEX']
                        model += pulp.lpSum([flex[p] + captain[p] for p in prev_players]) <= 4
                
                # Solve
                model.solve(pulp.PULP_CBC_CMD(msg=0))
                
                if pulp.LpStatus[model.status] == 'Optimal':
                    captain_pick = [p for p in players if captain[p].value() == 1][0]
                    flex_picks = [p for p in players if flex[p].value() == 1]
                    
                    total_salary = sum(salaries[p] for p in flex_picks) + 1.5 * salaries[captain_pick]
                    total_proj = sum(adjusted_points[p] for p in flex_picks) + 1.5 * adjusted_points[captain_pick]
                    total_ownership = ownership.get(captain_pick, 5) * 1.5 + sum(ownership.get(p, 5) for p in flex_picks)
                    
                    # Check bucket validity
                    lineup_players = [captain_pick] + flex_picks
                    bucket_valid, bucket_msg = self.bucket_manager.validate_lineup_buckets(
                        lineup_players, self.df, strategy_name
                    )
                    bucket_summary = self.bucket_manager.get_bucket_summary(lineup_players, self.df)
                    
                    # Check for AI recommended stacks
                    has_ai_stack = False
                    stack_details = []
                    for stack in combined['combined_stacks']:
                        p1, p2 = stack.get('player1'), stack.get('player2')
                        if p1 in lineup_players and p2 in lineup_players:
                            has_ai_stack = True
                            stack_details.append(f"{p1}-{p2}")
                    
                    all_lineups.append({
                        'Lineup': lineup_num,
                        'Strategy': strategy.value,
                        'Captain': captain_pick,
                        'FLEX': flex_picks,
                        'Projected': round(total_proj, 2),
                        'Salary': int(total_salary),
                        'Total_Ownership': round(total_ownership, 1),
                        'Bucket_Summary': bucket_summary,
                        'Bucket_Valid': bucket_valid,
                        'Has_AI_Stack': has_ai_stack,
                        'Stack_Details': ', '.join(stack_details) if stack_details else 'None',
                        'AI1_Captain': captain_pick in rec1.captain_targets,
                        'AI2_Captain': captain_pick in rec2.captain_targets
                    })
        
        return pd.DataFrame(all_lineups)
    
    def generate_captain_pivots(self, lineups_df: pd.DataFrame) -> pd.DataFrame:
        """Generate captain pivot variations for top lineups"""
        all_pivots = []
        
        # Generate pivots for top lineups
        for idx, lineup in lineups_df.head(10).iterrows():
            pivots = self.pivot_generator.generate_pivots(lineup.to_dict(), self.df, max_pivots=2)
            
            for pivot in pivots:
                pivot['Parent_Lineup'] = lineup['Lineup']
                pivot['Lineup'] = f"{lineup['Lineup']}-P{len(all_pivots)+1}"
                all_pivots.append(pivot)
        
        if all_pivots:
            pivots_df = pd.DataFrame(all_pivots)
            
            # Add bucket summaries for pivots
            for idx, row in pivots_df.iterrows():
                lineup_players = [row['Captain']] + row['FLEX']
                bucket_summary = self.bucket_manager.get_bucket_summary(lineup_players, self.df)
                pivots_df.loc[idx, 'Bucket_Summary'] = bucket_summary
            
            return pivots_df
        
        return pd.DataFrame()

def load_and_validate_data(uploaded_file) -> pd.DataFrame:
    """Load and validate the uploaded CSV"""
    df = pd.read_csv(uploaded_file)
    
    required_cols = ['first_name', 'last_name', 'position', 'team', 'salary', 'ppg_projection']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()
    
    df['first_name'] = df['first_name'].fillna('')
    df['last_name'] = df['last_name'].fillna('')
    df['Player'] = (df['first_name'] + ' ' + df['last_name']).str.strip()
    
    df = df.rename(columns={
        'position': 'Position',
        'team': 'Team',
        'salary': 'Salary',
        'ppg_projection': 'Projected_Points'
    })
    
    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
    df = df[df['Salary'] >= OptimizerConfig.MIN_SALARY]
    
    return df[['Player', 'Position', 'Team', 'Salary', 'Projected_Points']]

def simulate_tournament_with_correlation(lineup: Dict, df: pd.DataFrame, 
                                        n_sims: int = 1000) -> Dict[str, float]:
    """Simulate tournament performance with player correlations"""
    captain = lineup['Captain']
    flex_players = lineup['FLEX'] if isinstance(lineup['FLEX'], list) else lineup['FLEX'].split(', ')
    all_players = [captain] + flex_players
    
    projections = df.set_index('Player')['Projected_Points'].to_dict()
    positions = df.set_index('Player')['Position'].to_dict()
    teams = df.set_index('Player')['Team'].to_dict()
    
    # Build simple correlation matrix
    n_players = len(all_players)
    means = np.array([projections.get(p, 0) for p in all_players])
    cov_matrix = np.eye(n_players) * 0.3  # Base variance
    
    # Add correlations
    for i in range(n_players):
        for j in range(i+1, n_players):
            p1, p2 = all_players[i], all_players[j]
            team1, team2 = teams.get(p1), teams.get(p2)
            pos1, pos2 = positions.get(p1), positions.get(p2)
            
            corr = 0
            if team1 == team2:
                if pos1 == 'QB' and pos2 in ['WR', 'TE']:
                    corr = 0.4
                elif pos1 in ['WR', 'TE'] and pos2 in ['WR', 'TE']:
                    corr = 0.2
            
            cov_matrix[i, j] = cov_matrix[j, i] = corr * 0.3
    
    # Simulate
    try:
        sims = multivariate_normal(mean=means, cov=cov_matrix).rvs(n_sims)
        sims = np.maximum(0, sims)
        
        captain_scores = sims[:, 0] * OptimizerConfig.CAPTAIN_MULTIPLIER
        flex_scores = np.sum(sims[:, 1:], axis=1)
        total_scores = captain_scores + flex_scores
        
        return {
            'Mean': round(np.mean(total_scores), 2),
            'Std': round(np.std(total_scores), 2),
            'Floor_10th': round(np.percentile(total_scores, 10), 2),
            'Ceiling_90th': round(np.percentile(total_scores, 90), 2),
            'Ceiling_95th': round(np.percentile(total_scores, 95), 2),
            'Ceiling_99th': round(np.percentile(total_scores, 99), 2)
        }
    except:
        # Fallback to simple calculation
        total_proj = projections.get(captain, 0) * 1.5 + sum(projections.get(p, 0) for p in flex_players)
        return {
            'Mean': round(total_proj, 2),
            'Std': round(total_proj * 0.25, 2),
            'Floor_10th': round(total_proj * 0.7, 2),
            'Ceiling_90th': round(total_proj * 1.4, 2),
            'Ceiling_95th': round(total_proj * 1.5, 2),
            'Ceiling_99th': round(total_proj * 1.7, 2)
        }

# STREAMLIT UI
st.markdown("""
<style>
.stAlert {
    padding: 0.5rem;
    margin: 0.5rem 0;
}
.bucket-badge {
    display: inline-block;
    padding: 2px 8px;
    margin: 2px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: bold;
}
.mega-chalk { background-color: #ff4444; color: white; }
.chalk { background-color: #ff8844; color: white; }
.pivot { background-color: #44ff44; color: black; }
.leverage { background-color: #4444ff; color: white; }
.super-leverage { background-color: #aa44ff; color: white; }
</style>
""", unsafe_allow_html=True)

# Sidebar for AI inputs
with st.sidebar:
    st.header("🤖 Dual AI Strategy System")
    st.markdown("""
    ### Phase 1 Enhancements Active:
    ✅ **Ownership Bucketing**
    - Automatic lineup categorization
    - Enforced diversity rules
    
    ✅ **Captain Pivoting**
    - Generate variants of top lineups
    - Maximize uniqueness
    
    ✅ **Dual AI Integration**
    - Game Theory strategist
    - Correlation strategist
    """)
    
    st.markdown("---")
    st.subheader("📊 Ownership Buckets")
    st.markdown("""
    - **Mega Chalk**: 40%+ 🔴
    - **Chalk**: 25-40% 🟠
    - **Pivot**: 10-25% 🟢
    - **Leverage**: 5-10% 🔵
    - **Super Leverage**: <5% 🟣
    """)

# Main content
uploaded_file = st.file_uploader("Upload DraftKings CSV", type="csv")

if uploaded_file is not None:
    df = load_and_validate_data(uploaded_file)
    
    # Game Information
    st.subheader("⚙️ Game Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        teams = st.text_input("Teams", "BUF vs MIA", key="teams_input")
    with col2:
        total = st.number_input("O/U Total", 30.0, 70.0, 48.5, 0.5)
    with col3:
        spread = st.number_input("Spread", -20.0, 20.0, -3.0, 0.5)
    
    game_info = {'teams': teams, 'total': total, 'spread': spread}
    
    # Data Input Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏥 Injuries")
        injury_text = st.text_area("Format: Player: Status", height=100)
        
        injuries = {}
        for line in injury_text.split('\n'):
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    injuries[parts[0].strip()] = parts[1].strip()
        
        injury_multipliers = {'OUT': 0.0, 'DOUBTFUL': 0.3, 'QUESTIONABLE': 0.75, 'PROBABLE': 0.95}
        for player, status in injuries.items():
            if player in df['Player'].values:
                mult = injury_multipliers.get(status.upper(), 1.0)
                df.loc[df['Player'] == player, 'Projected_Points'] *= mult
                df.loc[df['Player'] == player, 'Injury'] = status
    
    with col2:
        st.subheader("📊 Ownership")
        ownership_text = st.text_area("Format: Player: %", height=100)
        
        ownership_dict = {}
        for line in ownership_text.split('\n'):
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    try:
                        ownership_dict[parts[0].strip()] = float(parts[1].strip())
                    except:
                        pass
        
        df['Ownership'] = df['Player'].map(ownership_dict).fillna(OptimizerConfig.DEFAULT_OWNERSHIP)
    
    # Add ownership buckets to dataframe
    bucket_manager = OwnershipBucketManager()
    df['Bucket'] = df['Ownership'].apply(bucket_manager.get_bucket)
    
    # Display player pool with buckets
    st.subheader("Player Pool with Ownership Buckets")
    
    # Show bucket distribution
    bucket_counts = df['Bucket'].value_counts()
    bucket_summary = " | ".join([f"**{bucket}**: {count}" for bucket, count in bucket_counts.items()])
    st.markdown(f"Distribution: {bucket_summary}")
    
    # Display dataframe with color coding
    display_df = df[['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 'Ownership', 'Bucket']].copy()
    display_df = display_df.sort_values('Ownership', ascending=False)
    
    st.dataframe(
        display_df,
        height=250,
        use_container_width=True,
        column_config={
            "Ownership": st.column_config.NumberColumn(format="%.1f%%"),
            "Projected_Points": st.column_config.NumberColumn(format="%.1f"),
            "Salary": st.column_config.NumberColumn(format="$%d")
        }
    )
    
    # AI Strategy Section
    st.markdown("---")
    st.subheader("🧠 AI Strategy Configuration")
    
    ai_tabs = st.tabs(["🎯 Game Theory AI", "🔗 Correlation AI", "⚙️ Settings"])
    
    with ai_tabs[0]:
        st.markdown("### Game Theory Strategist Prompt")
        if st.button("Generate Game Theory Prompt"):
            prompt = GameTheoryStrategist.generate_prompt(df, game_info)
            st.text_area("Copy this to Claude:", prompt, height=300, key="gt_prompt")
        
        gt_response = st.text_area(
            "Paste Game Theory AI Response (JSON):",
            height=200,
            key="gt_response",
            placeholder='{"leverage_captains": [...], "must_fades": [...], ...}'
        )
    
    with ai_tabs[1]:
        st.markdown("### Correlation Strategist Prompt")
        if st.button("Generate Correlation Prompt"):
            prompt = CorrelationStrategist.generate_prompt(df, game_info)
            st.text_area("Copy this to Claude:", prompt, height=300, key="corr_prompt")
        
        corr_response = st.text_area(
            "Paste Correlation AI Response (JSON):",
            height=200,
            key="corr_response",
            placeholder='{"primary_stacks": [...], "game_stacks": [...], ...}'
        )
    
    with ai_tabs[2]:
        st.markdown("### Optimization Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            num_lineups = st.slider("Base Lineups to Generate", 10, 50, 20)
            enable_pivots = st.checkbox("Generate Captain Pivots", value=True)
            max_pivot_lineups = st.slider("Max Pivots per Lineup", 1, 5, 2) if enable_pivots else 0
        
        with col2:
            st.markdown("**Strategy Distribution:**")
            use_ai_weights = st.checkbox("Use AI Recommended Weights", value=True)
            if not use_ai_weights:
                st.info("Manual weight configuration available after AI parsing")
    
    # Generate Lineups Button
    if st.button("🚀 Generate Optimized Lineups", type="primary"):
        
        # Parse AI responses
        rec1 = GameTheoryStrategist.parse_response(gt_response if gt_response else "{}", df)
        rec2 = CorrelationStrategist.parse_response(corr_response if corr_response else "{}", df)
        
        # Show AI insights
        with st.expander("📝 AI Strategic Insights"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Game Theory AI:**")
                st.write(f"Confidence: {rec1.confidence:.0%}")
                if rec1.captain_targets:
                    st.write("Leverage Captains:", ', '.join(rec1.captain_targets[:3]))
                if rec1.fades:
                    st.write("Fades:", ', '.join(rec1.fades[:3]))
                if rec1.key_insights:
                    for insight in rec1.key_insights[:2]:
                        st.write(f"• {insight}")
            
            with col2:
                st.markdown("**Correlation AI:**")
                st.write(f"Confidence: {rec2.confidence:.0%}")
                if rec2.stacks:
                    st.write(f"Recommended Stacks: {len(rec2.stacks)}")
                if rec2.key_insights:
                    for insight in rec2.key_insights[:2]:
                        st.write(f"• {insight}")
        
        # Initialize optimizer
        optimizer = DualAIOptimizer(df, game_info)
        
        # Generate base lineups
        with st.spinner(f"Generating {num_lineups} optimized lineups with bucket constraints..."):
            lineups_df = optimizer.generate_lineups_with_buckets(num_lineups, rec1, rec2)
        
        if lineups_df.empty:
            st.error("No valid lineups generated. Try adjusting constraints.")
        else:
            # Add simulations
            st.info("Running tournament simulations...")
            for idx, row in lineups_df.iterrows():
                sim_results = simulate_tournament_with_correlation(row.to_dict(), df)
                for key, value in sim_results.items():
                    lineups_df.loc[idx, key] = value
            
            # Calculate composite score
            lineups_df['Composite_Score'] = (
                0.30 * lineups_df['Ceiling_90th'] +
                0.20 * lineups_df['Mean'] +
                0.20 * (150 - lineups_df['Total_Ownership']) +
                0.15 * lineups_df['Has_AI_Stack'].astype(int) * 100 +
                0.15 * (lineups_df['AI1_Captain'].astype(int) + lineups_df['AI2_Captain'].astype(int)) * 50
            )
            
            # Sort by composite score
            lineups_df = lineups_df.sort_values('Composite_Score', ascending=False)
            
            # Generate captain pivots
            pivots_df = pd.DataFrame()
            if enable_pivots:
                with st.spinner("Generating captain pivot variations..."):
                    pivots_df = optimizer.generate_captain_pivots(lineups_df)
                    
                    if not pivots_df.empty:
                        # Add simulations for pivots
                        for idx, row in pivots_df.iterrows():
                            sim_results = simulate_tournament_with_correlation(row.to_dict(), df)
                            for key, value in sim_results.items():
                                pivots_df.loc[idx, key] = value
                        
                        # Calculate scores for pivots
                        pivots_df['Composite_Score'] = (
                            0.30 * pivots_df.get('Ceiling_90th', 0) +
                            0.20 * pivots_df.get('Mean', 0) +
                            0.30 * (150 - pivots_df.get('Total_Ownership', 100))
                        )
            
            # Display Results
            st.markdown("---")
            st.subheader("🏆 Optimization Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Lineups Generated", len(lineups_df) + len(pivots_df))
            with col2:
                st.metric("Avg Ceiling (90th)", f"{lineups_df['Ceiling_90th'].mean():.1f}")
            with col3:
                st.metric("Avg Ownership", f"{lineups_df['Total_Ownership'].mean():.1f}%")
            with col4:
                valid_count = lineups_df['Bucket_Valid'].sum()
                st.metric("Bucket Valid", f"{valid_count}/{len(lineups_df)}")
            
            # Tabs for different views
            result_tabs = st.tabs(["📊 Base Lineups", "🔄 Captain Pivots", "📈 Analysis", "💾 Export"])
            
            with result_tabs[0]:
                st.markdown("### Base Lineups")
                
                # Display columns
                display_cols = ['Lineup', 'Strategy', 'Captain', 'Ceiling_90th', 'Total_Ownership', 
                              'Bucket_Summary', 'Composite_Score', 'Has_AI_Stack']
                
                st.dataframe(
                    lineups_df[display_cols].head(20),
                    use_container_width=True,
                    column_config={
                        "Ceiling_90th": st.column_config.NumberColumn(format="%.1f"),
                        "Total_Ownership": st.column_config.NumberColumn(format="%.1f%%"),
                        "Composite_Score": st.column_config.NumberColumn(format="%.1f")
                    }
                )
                
                # Detailed view of top 3
                st.markdown("### Top 3 Lineups - Detailed View")
                for i, (idx, lineup) in enumerate(lineups_df.head(3).iterrows(), 1):
                    with st.expander(f"#{i} - {lineup['Strategy']} - Score: {lineup['Composite_Score']:.1f}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Roster:**")
                            st.write(f"Captain: **{lineup['Captain']}**")
                            for player in lineup['FLEX']:
                                st.write(f"• {player}")
                        
                        with col2:
                            st.markdown("**Projections:**")
                            st.metric("Mean", f"{lineup['Mean']:.1f}")
                            st.metric("90th %ile", f"{lineup['Ceiling_90th']:.1f}")
                            st.metric("95th %ile", f"{lineup['Ceiling_95th']:.1f}")
                        
                        with col3:
                            st.markdown("**Ownership Profile:**")
                            st.write(lineup['Bucket_Summary'])
                            st.metric("Total Own%", f"{lineup['Total_Ownership']:.1f}%")
                            if lineup['Has_AI_Stack']:
                                st.success(f"✅ Stack: {lineup['Stack_Details']}")
            
            with result_tabs[1]:
                if not pivots_df.empty:
                    st.markdown("### Captain Pivot Variations")
                    st.info(f"Generated {len(pivots_df)} pivot lineups from top base lineups")
                    
                    pivot_cols = ['Lineup', 'Parent_Lineup', 'Captain', 'Original_Captain', 
                                'Ownership_Delta', 'Bucket_Summary', 'Composite_Score']
                    
                    # Filter columns that exist
                    available_pivot_cols = [col for col in pivot_cols if col in pivots_df.columns]
                    
                    st.dataframe(
                        pivots_df[available_pivot_cols].head(20),
                        use_container_width=True
                    )
                else:
                    st.info("Enable captain pivots in settings to generate variations")
            
            with result_tabs[2]:
                st.markdown("### Lineup Analysis")
                
                # Create visualizations
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # 1. Strategy distribution
                ax1 = axes[0, 0]
                strategy_counts = lineups_df['Strategy'].value_counts()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                ax1.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.0f%%',
                       colors=colors[:len(strategy_counts)], startangle=90)
                ax1.set_title('Strategy Distribution')
                
                # 2. Ownership vs Ceiling scatter
                ax2 = axes[0, 1]
                scatter = ax2.scatter(lineups_df['Total_Ownership'], lineups_df['Ceiling_90th'],
                                    c=lineups_df['Composite_Score'], cmap='viridis',
                                    s=100, alpha=0.6)
                ax2.set_xlabel('Total Ownership %')
                ax2.set_ylabel('90th Percentile Ceiling')
                ax2.set_title('Ownership vs Upside')
                plt.colorbar(scatter, ax=ax2, label='Composite Score')
                
                # Annotate top 3
                for i, (idx, row) in enumerate(lineups_df.head(3).iterrows(), 1):
                    ax2.annotate(f'#{i}', (row['Total_Ownership'], row['Ceiling_90th']),
                               fontsize=12, fontweight='bold', color='red')
                
                # 3. Bucket distribution heatmap
                ax3 = axes[1, 0]
                bucket_data = []
                for idx, row in lineups_df.head(10).iterrows():
                    bucket_counts = {'mega_chalk': 0, 'chalk': 0, 'pivot': 0, 
                                   'leverage': 0, 'super_leverage': 0}
                    for player in [row['Captain']] + row['FLEX']:
                        player_own = df[df['Player'] == player]['Ownership'].values
                        if len(player_own) > 0:
                            bucket = bucket_manager.get_bucket(player_own[0])
                            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
                    bucket_data.append(list(bucket_counts.values()))
                
                im = ax3.imshow(bucket_data, cmap='YlOrRd', aspect='auto')
                ax3.set_xticks(range(5))
                ax3.set_xticklabels(['Mega\nChalk', 'Chalk', 'Pivot', 'Leverage', 'Super\nLeverage'])
                ax3.set_yticks(range(len(bucket_data)))
                ax3.set_yticklabels([f'#{i+1}' for i in range(len(bucket_data))])
                ax3.set_title('Ownership Bucket Distribution (Top 10)')
                ax3.set_xlabel('Bucket Type')
                ax3.set_ylabel('Lineup Rank')
                
                # Add text annotations
                for i in range(len(bucket_data)):
                    for j in range(5):
                        text = ax3.text(j, i, bucket_data[i][j],
                                      ha="center", va="center", color="white" if bucket_data[i][j] > 2 else "black")
                
                # 4. Score components breakdown
                ax4 = axes[1, 1]
                top5 = lineups_df.head(5)
                x = np.arange(5)
                width = 0.15
                
                ax4.bar(x - 2*width, top5['Ceiling_90th']/3, width, label='Ceiling', alpha=0.8)
                ax4.bar(x - width, top5['Mean']/5, width, label='Mean', alpha=0.8)
                ax4.bar(x, 150 - top5['Total_Ownership'], width, label='Leverage', alpha=0.8)
                ax4.bar(x + width, top5['Has_AI_Stack'].astype(int) * 100, width, label='Stack', alpha=0.8)
                ax4.bar(x + 2*width, (top5['AI1_Captain'].astype(int) + top5['AI2_Captain'].astype(int)) * 50, 
                       width, label='AI Captain', alpha=0.8)
                
                ax4.set_xlabel('Lineup Rank')
                ax4.set_ylabel('Score Components')
                ax4.set_title('Composite Score Breakdown - Top 5')
                ax4.set_xticks(x)
                ax4.set_xticklabels([f'#{i+1}' for i in range(5)])
                ax4.legend(loc='upper right', fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with result_tabs[3]:
                st.markdown("### Export Lineups")
                
                # Combine base lineups and pivots for export
                export_df = lineups_df.copy()
                if not pivots_df.empty:
                    # Ensure consistent columns
                    common_cols = list(set(export_df.columns) & set(pivots_df.columns))
                    export_df = pd.concat([export_df[common_cols], pivots_df[common_cols]], ignore_index=True)
                
                # DraftKings format
                dk_lineups = []
                for idx, row in export_df.head(30).iterrows():
                    flex_players = row['FLEX'] if isinstance(row['FLEX'], list) else row['FLEX'].split(', ')
                    dk_lineups.append({
                        'Captain': row['Captain'],
                        'FLEX1': flex_players[0] if len(flex_players) > 0 else '',
                        'FLEX2': flex_players[1] if len(flex_players) > 1 else '',
                        'FLEX3': flex_players[2] if len(flex_players) > 2 else '',
                        'FLEX4': flex_players[3] if len(flex_players) > 3 else '',
                        'FLEX5': flex_players[4] if len(flex_players) > 4 else ''
                    })
                
                dk_df = pd.DataFrame(dk_lineups)
                csv_dk = dk_df.to_csv(index=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download DraftKings CSV",
                        data=csv_dk,
                        file_name="dk_lineups_optimized.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Full export with all metrics
                    export_df['FLEX'] = export_df['FLEX'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
                    csv_full = export_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Full Analysis",
                        data=csv_full,
                        file_name="full_lineup_analysis.csv",
                        mime="text/csv"
                    )
                
                # Show export preview
                st.markdown("### DraftKings Upload Preview")
                st.dataframe(dk_df.head(10), use_container_width=True)

# Footer with instructions
st.markdown("---")
st.markdown("""
### 📚 How to Use This Optimizer

**Phase 1 Features Active:**
1. **Ownership Bucketing** - Automatically categorizes players and enforces diversity
2. **Captain Pivoting** - Creates lineup variants with different captains  
3. **Dual AI System** - Two specialized AI strategists working together

**Quick Start:**
1. Upload your DraftKings CSV export
2. Enter game information (teams, total, spread)
3. Add any injury/ownership information
4. Generate prompts for both AI strategists
5. Copy prompts to Claude and paste JSON responses back
6. Click "Generate Optimized Lineups" 
7. Export your lineups in DraftKings format

**Strategy Guide:**
- **Balanced**: Mix of chalk and leverage plays
- **Leverage**: Focus on low-owned captain plays
- **Contrarian**: Fade chalk, embrace variance
- **Game Stack**: Correlate players from both teams
- **Correlation**: Stack teammates

**Ownership Buckets Explained:**
- 🔴 **Mega Chalk (40%+)**: Tournament favorites - limit exposure
- 🟠 **Chalk (25-40%)**: Popular plays - use selectively
- 🟢 **Pivot (10-25%)**: Differentiation plays
- 🔵 **Leverage (5-10%)**: Hidden upside plays
- 🟣 **Super Leverage (<5%)**: Tournament winners

💡 **Pro Tip**: The captain pivot feature can turn a good lineup into multiple unique entries by swapping the captain spot while maintaining the core build.
""")

st.markdown("---")
st.caption("NFL Showdown Optimizer v2.0 - Phase 1 Enhanced | Built with Dual AI Strategy System")