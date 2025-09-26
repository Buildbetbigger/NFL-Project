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

# For Google Colab compatibility
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    st.warning("Anthropic package not installed. Install with: pip install anthropic")

st.set_page_config(page_title="NFL Dual-AI Optimizer v3", page_icon="🏈", layout="wide")
st.title('🏈 NFL Showdown Optimizer - Cloud Ready Version')

# Test imports
try:
    import pulp
    st.success(f"✅ PuLP {pulp.__version__} loaded successfully")
except ImportError as e:
    st.error(f"❌ PuLP import failed: {e}")
    st.stop()

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
    
    # API Configuration
    CLAUDE_MODEL = "claude-3-sonnet-20241022"
    MAX_TOKENS = 2000
    TEMPERATURE = 0.7
    
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
            'mega_chalk': (0, 2),
            'chalk': (1, 3),
            'pivot': (1, 3),
            'leverage': (0, 2),
            'super_leverage': (0, 1)
        },
        'contrarian': {
            'mega_chalk': (0, 1),
            'chalk': (0, 2),
            'pivot': (2, 4),
            'leverage': (1, 3),
            'super_leverage': (0, 2)
        },
        'leverage': {
            'mega_chalk': (0, 0),
            'chalk': (0, 1),
            'pivot': (1, 3),
            'leverage': (2, 4),
            'super_leverage': (1, 2)
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

class ClaudeAPIManager:
    """Manages Claude API interactions"""
    
    def __init__(self, api_key: str):
        """Initialize with API key"""
        self.api_key = api_key
        self.client = None
        if ANTHROPIC_AVAILABLE and api_key:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
                self.test_connection()
            except Exception as e:
                st.error(f"Failed to initialize Claude API: {e}")
                self.client = None
    
    def test_connection(self):
        """Test API connection with a simple request"""
        try:
            response = self.client.messages.create(
                model=OptimizerConfig.CLAUDE_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception as e:
            st.error(f"API connection test failed: {e}")
            return False
    
    def get_ai_response(self, prompt: str, system_prompt: str = None) -> str:
        """Get response from Claude API"""
        if not self.client:
            return "{}"
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Add system message for better JSON formatting
            if system_prompt:
                system = system_prompt
            else:
                system = "You are a DFS expert AI. Always respond with valid JSON only, no additional text or markdown formatting."
            
            response = self.client.messages.create(
                model=OptimizerConfig.CLAUDE_MODEL,
                max_tokens=OptimizerConfig.MAX_TOKENS,
                temperature=OptimizerConfig.TEMPERATURE,
                system=system,
                messages=messages
            )
            
            # Extract text from response
            response_text = response.content[0].text
            
            # Clean up response to ensure valid JSON
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            return response_text.strip()
            
        except Exception as e:
            st.error(f"API call failed: {e}")
            return "{}"

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
        
        for new_captain in flex_players[:max_pivots]:
            old_captain_salary = salaries.get(captain, 0)
            new_captain_salary = salaries.get(new_captain, 0)
            
            salary_freed = old_captain_salary * 0.5
            salary_needed = new_captain_salary * 0.5
            
            if salary_freed >= salary_needed:
                new_flex = [p for p in flex_players if p != new_captain] + [captain]
                
                pivot_lineup = lineup.copy()
                pivot_lineup['Captain'] = new_captain
                pivot_lineup['FLEX'] = new_flex
                pivot_lineup['Pivot_Type'] = 'Simple Swap'
                pivot_lineup['Original_Captain'] = captain
                
                total_proj = points.get(new_captain, 0) * 1.5 + sum(points.get(p, 0) for p in new_flex)
                total_own = ownership.get(new_captain, 5) * 1.5 + sum(ownership.get(p, 5) for p in new_flex)
                
                pivot_lineup['Projected'] = round(total_proj, 2)
                pivot_lineup['Total_Ownership'] = round(total_own, 1)
                pivot_lineup['Ownership_Delta'] = round(total_own - lineup['Total_Ownership'], 1)
                
                pivot_lineups.append(pivot_lineup)
        
        return pivot_lineups

class GameTheoryStrategist:
    """AI Strategist 1: Focus on ownership leverage and game theory"""
    
    def __init__(self, api_manager: ClaudeAPIManager = None):
        self.api_manager = api_manager
    
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict) -> str:
        """Generate prompt for game theory analysis"""
        
        bucket_manager = OwnershipBucketManager()
        buckets = bucket_manager.categorize_players(df)
        
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
        
        Provide strategic recommendations. Return ONLY valid JSON with this structure:
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
    
    def get_ai_response(self, df: pd.DataFrame, game_info: Dict, use_api: bool = True) -> str:
        """Get AI response either from API or manual input"""
        prompt = self.generate_prompt(df, game_info)
        
        if use_api and self.api_manager and self.api_manager.client:
            with st.spinner("Getting Game Theory AI analysis..."):
                return self.api_manager.get_ai_response(prompt)
        else:
            # Manual input fallback
            st.info("Copy the prompt above to Claude and paste the JSON response below")
            return st.text_area(
                "Game Theory AI Response (JSON):",
                height=200,
                key="gt_manual_response",
                placeholder='{"leverage_captains": [...], ...}'
            )
    
    def parse_response(self, response: str, df: pd.DataFrame) -> AIRecommendation:
        """Parse game theory strategist response"""
        try:
            data = json.loads(response) if response else {}
            
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
    
    def __init__(self, api_manager: ClaudeAPIManager = None):
        self.api_manager = api_manager
    
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict) -> str:
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
        
        Return ONLY valid JSON with this structure:
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
    
    def get_ai_response(self, df: pd.DataFrame, game_info: Dict, use_api: bool = True) -> str:
        """Get AI response either from API or manual input"""
        prompt = self.generate_prompt(df, game_info)
        
        if use_api and self.api_manager and self.api_manager.client:
            with st.spinner("Getting Correlation AI analysis..."):
                return self.api_manager.get_ai_response(prompt)
        else:
            # Manual input fallback
            st.info("Copy the prompt above to Claude and paste the JSON response below")
            return st.text_area(
                "Correlation AI Response (JSON):",
                height=200,
                key="corr_manual_response",
                placeholder='{"primary_stacks": [...], ...}'
            )
    
    def parse_response(self, response: str, df: pd.DataFrame) -> AIRecommendation:
        """Parse correlation strategist response"""
        try:
            data = json.loads(response) if response else {}
            
            all_stacks = []
            
            for stack in data.get('primary_stacks', []):
                all_stacks.append({
                    'player1': stack.get('qb'),
                    'player2': stack.get('receiver')
                })
            
            for stack in data.get('game_stacks', []):
                all_stacks.append({
                    'player1': stack.get('team1_player'),
                    'player2': stack.get('team2_player')
                })
            
            for stack in data.get('leverage_stacks', []):
                all_stacks.append({
                    'player1': stack.get('player1'),
                    'player2': stack.get('player2')
                })
            
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
    """Main optimizer combining both AI strategists"""
    
    def __init__(self, df: pd.DataFrame, game_info: Dict, api_manager: ClaudeAPIManager = None):
        self.df = df
        self.game_info = game_info
        self.api_manager = api_manager
        self.game_theory_ai = GameTheoryStrategist(api_manager)
        self.correlation_ai = CorrelationStrategist(api_manager)
        self.bucket_manager = OwnershipBucketManager()
        self.pivot_generator = CaptainPivotGenerator()
        self.recommendations = {}
    
    def get_ai_strategies(self, use_api: bool = True) -> Tuple[AIRecommendation, AIRecommendation]:
        """Get strategies from both AIs"""
        
        if use_api and self.api_manager and self.api_manager.client:
            # API mode
            gt_response = self.game_theory_ai.get_ai_response(self.df, self.game_info, use_api=True)
            corr_response = self.correlation_ai.get_ai_response(self.df, self.game_info, use_api=True)
        else:
            # Manual mode
            st.subheader("Manual AI Strategy Input")
            
            tab1, tab2 = st.tabs(["Game Theory Prompt", "Correlation Prompt"])
            
            with tab1:
                st.text_area("Game Theory Prompt (copy to Claude):", 
                           value=self.game_theory_ai.generate_prompt(self.df, self.game_info),
                           height=200, key="gt_prompt_display")
                gt_response = st.text_area("Paste Game Theory Response:", 
                                          height=150, key="gt_manual_input")
            
            with tab2:
                st.text_area("Correlation Prompt (copy to Claude):", 
                           value=self.correlation_ai.generate_prompt(self.df, self.game_info),
                           height=200, key="corr_prompt_display")
                corr_response = st.text_area("Paste Correlation Response:", 
                                            height=150, key="corr_manual_input")
        
        rec1 = self.game_theory_ai.parse_response(gt_response, self.df)
        rec2 = self.correlation_ai.parse_response(corr_response, self.df)
        
        return rec1, rec2
    
    def combine_recommendations(self, rec1: AIRecommendation, rec2: AIRecommendation) -> Dict:
        """Combine recommendations from both AIs"""
        
        total_confidence = rec1.confidence + rec2.confidence
        w1 = rec1.confidence / total_confidence if total_confidence > 0 else 0.5
        w2 = rec2.confidence / total_confidence if total_confidence > 0 else 0.5
        
        all_captains = set(rec1.captain_targets + rec2.captain_targets)
        captain_scores = {}
        for captain in all_captains:
            score = 0
            if captain in rec1.captain_targets:
                score += w1
            if captain in rec2.captain_targets:
                score += w2
            captain_scores[captain] = score
        
        combined_weights = {}
        for strategy in StrategyType:
            combined_weights[strategy] = (
                rec1.strategy_weights.get(strategy, 0) * w1 +
                rec2.strategy_weights.get(strategy, 0) * w2
            )
        
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            combined_weights = {k: v/total_weight for k, v in combined_weights.items()}
        
        consensus_fades = set(rec1.fades) & set(rec2.fades) if rec1.fades and rec2.fades else set()
        all_boosts = set(rec1.boosts) | set(rec2.boosts) if rec1.boosts or rec2.boosts else set()
        
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
        
        players = self.df['Player'].tolist()
        positions = self.df.set_index('Player')['Position'].to_dict()
        teams = self.df.set_index('Player')['Team'].to_dict()
        salaries = self.df.set_index('Player')['Salary'].to_dict()
        points = self.df.set_index('Player')['Projected_Points'].to_dict()
        ownership = self.df.set_index('Player')['Ownership'].to_dict()
        
        adjusted_points = points.copy()
        for player in combined['consensus_fades']:
            if player in adjusted_points:
                adjusted_points[player] *= 0.8
        
        for player in combined['all_boosts']:
            if player in adjusted_points:
                adjusted_points[player] *= 1.15
        
        player_buckets = {}
        for player in players:
            player_buckets[player] = self.bucket_manager.get_bucket(
                ownership.get(player, OptimizerConfig.DEFAULT_OWNERSHIP)
            )
        
        lineups_per_strategy = {}
        for strategy, weight in combined['strategy_weights'].items():
            lineups_per_strategy[strategy] = max(1, int(num_lineups * weight))
        
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
                
                for bucket_name, (min_count, max_count) in bucket_rules.items():
                    bucket_players = [p for p in players if player_buckets[p] == bucket_name]
                    if bucket_players:
                        if min_count > 0:
                            model += pulp.lpSum([flex[p] + captain[p] for p in bucket_players]) >= min_count
                        if max_count < 6:
                            model += pulp.lpSum([flex[p] + captain[p] for p in bucket_players]) <= max_count
                
                if lineup_num > 1 and all_lineups:
                    for prev_lineup in all_lineups[-min(3, len(all_lineups)):]:
                        prev_players = [prev_lineup['Captain']] + prev_lineup['FLEX']
                        model += pulp.lpSum([flex[p] + captain[p] for p in prev_players]) <= 4
                
                # FIXED: Use default solver for Streamlit Cloud compatibility
                try:
                    # Try CBC solver first if available (faster)
                    solver_list = pulp.listSolvers(onlyAvailable=True)
                    if 'PULP_CBC_CMD' in solver_list:
                        model.solve(pulp.PULP_CBC_CMD(msg=0))
                    else:
                        model.solve()  # Use default solver
                except:
                    # Fallback to default solver if CBC fails
                    model.solve()
                
                if pulp.LpStatus[model.status] == 'Optimal':
                    captain_pick = [p for p in players if captain[p].value() == 1][0]
                    flex_picks = [p for p in players if flex[p].value() == 1]
                    
                    total_salary = sum(salaries[p] for p in flex_picks) + 1.5 * salaries[captain_pick]
                    total_proj = sum(adjusted_points[p] for p in flex_picks) + 1.5 * adjusted_points[captain_pick]
                    total_ownership = ownership.get(captain_pick, 5) * 1.5 + sum(ownership.get(p, 5) for p in flex_picks)
                    
                    lineup_players = [captain_pick] + flex_picks
                    bucket_valid, bucket_msg = self.bucket_manager.validate_lineup_buckets(
                        lineup_players, self.df, strategy_name
                    )
                    bucket_summary = self.bucket_manager.get_bucket_summary(lineup_players, self.df)
                    
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
        
        for idx, lineup in lineups_df.head(10).iterrows():
            pivots = self.pivot_generator.generate_pivots(lineup.to_dict(), self.df, max_pivots=2)
            
            for pivot in pivots:
                pivot['Parent_Lineup'] = lineup['Lineup']
                pivot['Lineup'] = f"{lineup['Lineup']}-P{len(all_pivots)+1}"
                all_pivots.append(pivot)
        
        if all_pivots:
            pivots_df = pd.DataFrame(all_pivots)
            
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
    
    n_players = len(all_players)
    means = np.array([projections.get(p, 0) for p in all_players])
    cov_matrix = np.eye(n_players) * 0.3
    
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
        total_proj = projections.get(captain, 0) * 1.5 + sum(projections.get(p, 0) for p in flex_players)
        return {
            'Mean': round(total_proj, 2),
            'Std': round(total_proj * 0.25, 2),
            'Floor_10th': round(total_proj * 0.7, 2),
            'Ceiling_90th': round(total_proj * 1.4, 2),
            'Ceiling_95th': round(total_proj * 1.5, 2),
            'Ceiling_99th': round(total_proj * 1.7, 2)
        }

# MAIN STREAMLIT UI
st.markdown("""
<style>
.stAlert { padding: 0.5rem; margin: 0.5rem 0; }
.api-status { 
    padding: 10px; 
    border-radius: 5px; 
    margin: 10px 0;
    font-weight: bold;
}
.api-connected { background-color: #d4edda; color: #155724; }
.api-disconnected { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Test which solvers are available
with st.expander("🔧 System Check"):
    st.write("Available PuLP Solvers:")
    available_solvers = pulp.listSolvers(onlyAvailable=True)
    for solver in available_solvers:
        st.write(f"  ✓ {solver}")
    if not available_solvers:
        st.warning("No specific solvers found - will use default")

# Sidebar for API Configuration
with st.sidebar:
    st.header("🤖 AI Configuration")
    
    st.subheader("API Setup")
    api_mode = st.radio(
        "Mode",
        ["Manual (Copy/Paste)", "API (Automated)"],
        help="API mode uses Claude API directly. Manual mode requires copying prompts to Claude."
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
            if st.button("Test API Connection"):
                api_manager = ClaudeAPIManager(api_key)
                if api_manager.client:
                    st.success("✅ API Connected Successfully!")
                    use_api = True
                else:
                    st.error("❌ API Connection Failed")
            
            # Initialize API manager for use
            if api_key and not api_manager:
                api_manager = ClaudeAPIManager(api_key)
                use_api = bool(api_manager.client)
        else:
            st.info("Enter your API key to enable automated AI analysis")
    else:
        st.info("Manual mode: Copy prompts and paste responses")
    
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
st.subheader("Data Upload")
uploaded_file = st.file_uploader("Upload DraftKings CSV", type="csv")

if uploaded_file is not None:
    df = load_and_validate_data(uploaded_file)
    
    # Game Information
    st.subheader("⚙️ Game Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        teams = st.text_input("Teams", "BUF vs MIA")
    with col2:
        total = st.number_input("O/U Total", 30.0, 70.0, 48.5, 0.5)
    with col3:
        spread = st.number_input("Spread", -20.0, 20.0, -3.0, 0.5)
    
    game_info = {'teams': teams, 'total': total, 'spread': spread}
    
    # Data Input
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
    
    # Add ownership buckets
    bucket_manager = OwnershipBucketManager()
    df['Bucket'] = df['Ownership'].apply(bucket_manager.get_bucket)
    
    # Display player pool
    st.subheader("Player Pool")
    st.dataframe(
        df[['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 'Ownership', 'Bucket']],
        height=250,
        use_container_width=True
    )
    
    # Optimization Settings
    st.subheader("Optimization Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        num_lineups = st.slider("Lineups to Generate", 10, 50, 20)
        enable_pivots = st.checkbox("Generate Captain Pivots", value=True)
    
    with col2:
        max_pivot_lineups = st.slider("Max Pivots per Lineup", 1, 5, 2) if enable_pivots else 0
        
        # Show API status
        if use_api and api_manager:
            st.markdown('<div class="api-status api-connected">🟢 API Connected</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="api-status api-disconnected">🔴 Manual Mode</div>', 
                       unsafe_allow_html=True)
    
    # Generate Lineups
    if st.button("🚀 Generate Optimized Lineups", type="primary"):
        
        # Initialize optimizer with API manager
        optimizer = DualAIOptimizer(df, game_info, api_manager)
        
        # Get AI strategies (will use API or manual based on settings)
        rec1, rec2 = optimizer.get_ai_strategies(use_api=use_api)
        
        # Show AI insights
        with st.expander("📝 AI Strategic Insights"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Game Theory AI:**")
                st.write(f"Confidence: {rec1.confidence:.0%}")
                if rec1.captain_targets:
                    st.write("Leverage Captains:", ', '.join(rec1.captain_targets[:3]))
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
        
        # Generate lineups
        with st.spinner(f"Generating {num_lineups} optimized lineups..."):
            lineups_df = optimizer.generate_lineups_with_buckets(num_lineups, rec1, rec2)
        
        if lineups_df.empty:
            st.error("No valid lineups generated. Check constraints.")
        else:
            # Add simulations
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
            
            lineups_df = lineups_df.sort_values('Composite_Score', ascending=False)
            
            # Generate pivots
            pivots_df = pd.DataFrame()
            if enable_pivots:
                pivots_df = optimizer.generate_captain_pivots(lineups_df)
            
            # Display Results
            st.subheader("🏆 Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Lineups", len(lineups_df) + len(pivots_df))
            with col2:
                st.metric("Avg Ceiling", f"{lineups_df['Ceiling_90th'].mean():.1f}")
            with col3:
                st.metric("Avg Own%", f"{lineups_df['Total_Ownership'].mean():.1f}%")
            with col4:
                st.metric("Valid", f"{lineups_df['Bucket_Valid'].sum()}/{len(lineups_df)}")
            
            # Display lineups
            st.dataframe(
                lineups_df[['Lineup', 'Strategy', 'Captain', 'Ceiling_90th', 
                          'Total_Ownership', 'Bucket_Summary', 'Composite_Score']].head(20),
                use_container_width=True
            )
            
            # Export
            st.subheader("📥 Export")
            
            dk_lineups = []
            for idx, row in lineups_df.head(30).iterrows():
                flex_players = row['FLEX']
                dk_lineups.append({
                    'Captain': row['Captain'],
                    'FLEX1': flex_players[0] if len(flex_players) > 0 else '',
                    'FLEX2': flex_players[1] if len(flex_players) > 1 else '',
                    'FLEX3': flex_players[2] if len(flex_players) > 2 else '',
                    'FLEX4': flex_players[3] if len(flex_players) > 3 else '',
                    'FLEX5': flex_players[4] if len(flex_players) > 4 else ''
                })
            
            dk_df = pd.DataFrame(dk_lineups)
            csv = dk_df.to_csv(index=False)
            
            st.download_button(
                "📥 Download DraftKings CSV",
                csv,
                "dk_lineups.csv",
                "text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
### 📚 Quick Guide

**This version is optimized for Streamlit Cloud deployment!**

**Manual Mode (Default - FREE):**
1. Upload CSV and configure settings
2. Click "Generate" - copy prompts when they appear
3. Paste prompts into Claude
4. Paste JSON responses back
5. Continue with optimization

**API Mode (Automated):**
1. Enter your Claude API key
2. Click "Test API Connection"
3. Upload CSV and configure
4. Click "Generate" - AI runs automatically!

**Solver Note:**
This version automatically uses the best available solver for your environment.
- Local: CBC (fast)
- Cloud: Default (compatible)

The optimizer will work on any platform without configuration changes!
""")

st.caption("NFL Optimizer v3.1 - Cloud Ready | Powered by Claude AI")