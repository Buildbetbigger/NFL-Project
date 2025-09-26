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
st.title('🏈 NFL Showdown Optimizer - Fixed Version')

# Configuration with RELAXED constraints
class OptimizerConfig:
    SALARY_CAP = 50000
    ROSTER_SIZE = 6  # 1 Captain + 5 FLEX
    MAX_PLAYERS_PER_TEAM = 4
    CAPTAIN_MULTIPLIER = 1.5
    DEFAULT_OWNERSHIP = 5
    MIN_SALARY = 1000
    NUM_SIMS = 1000  # Reduced for faster testing
    FIELD_SIZE = 100000
    
    # API Configuration - Using Haiku (most reliable)
    CLAUDE_MODEL = "claude-3-haiku-20240307"
    MAX_TOKENS = 2000
    TEMPERATURE = 0.7
    
    # Ownership buckets
    OWNERSHIP_BUCKETS = {
        'mega_chalk': (40, 100),
        'chalk': (25, 40),
        'pivot': (10, 25),
        'leverage': (5, 10),
        'super_leverage': (0, 5)
    }
    
    # RELAXED BUCKET RULES - Allow any combination
    BUCKET_RULES = {
        'balanced': {
            'mega_chalk': (0, 6),
            'chalk': (0, 6),
            'pivot': (0, 6),
            'leverage': (0, 6),
            'super_leverage': (0, 6)
        },
        'contrarian': {
            'mega_chalk': (0, 3),  # Slight preference away from chalk
            'chalk': (0, 6),
            'pivot': (0, 6),
            'leverage': (0, 6),
            'super_leverage': (0, 6)
        },
        'leverage': {
            'mega_chalk': (0, 2),  # Avoid too much chalk
            'chalk': (0, 6),
            'pivot': (0, 6),
            'leverage': (0, 6),
            'super_leverage': (0, 6)
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
                st.success("✅ API Client initialized")
            except Exception as e:
                st.error(f"Failed to initialize Claude API: {e}")
                self.client = None
    
    def get_ai_response(self, prompt: str, system_prompt: str = None) -> str:
        """Get response from Claude API"""
        if not self.client:
            return "{}"
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
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
        As a DFS Game Theory Expert, analyze this NFL Showdown slate:
        
        Game: {game_info.get('teams', 'Unknown')} | O/U: {game_info.get('total', 0)} | Spread: {game_info.get('spread', 0)}
        
        MEGA CHALK (40%+ ownership):
        {mega_chalk.head(5).to_string() if not mega_chalk.empty else 'None'}
        
        LEVERAGE PLAYS (<10% ownership):
        {leverage_plays.head(5).to_string() if not leverage_plays.empty else 'None'}
        
        Return ONLY valid JSON:
        {{
            "leverage_captains": ["player1", "player2"],
            "must_fades": ["overowned1"],
            "must_plays": ["leverage1"],
            "construction_rules": {{"max_mega_chalk": 2, "min_leverage": 1}},
            "ownership_targets": {{"min": 80, "max": 120}},
            "contrarian_stacks": [{{"player1": "name1", "player2": "name2"}}],
            "confidence_score": 0.85,
            "key_insights": ["insight1", "insight2"]
        }}
        """
    
    def parse_response(self, response: str, df: pd.DataFrame) -> AIRecommendation:
        """Parse game theory strategist response"""
        try:
            data = json.loads(response) if response else {}
        except:
            data = {}
        
        # Default strategy weights
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
                   for s in data.get('contrarian_stacks', []) if isinstance(s, dict)],
            fades=data.get('must_fades', []),
            boosts=data.get('must_plays', []),
            strategy_weights=strategy_weights,
            key_insights=data.get('key_insights', ["Using default strategy"])
        )

class CorrelationStrategist:
    """AI Strategist 2: Focus on correlation, stacking, and game flow"""
    
    def __init__(self, api_manager: ClaudeAPIManager = None):
        self.api_manager = api_manager
    
    def generate_prompt(self, df: pd.DataFrame, game_info: Dict) -> str:
        """Generate prompt for correlation analysis"""
        
        teams = df['Team'].unique()
        team_breakdown = {}
        
        for team in teams[:2]:  # Limit to 2 teams for prompt size
            team_df = df[df['Team'] == team]
            team_breakdown[team] = {
                'QB': team_df[team_df['Position'] == 'QB']['Player'].head(2).tolist(),
                'WR_TE': team_df[team_df['Position'].isin(['WR', 'TE'])]['Player'].head(4).tolist(),
                'RB': team_df[team_df['Position'] == 'RB']['Player'].head(2).tolist()
            }
        
        return f"""
        As a DFS Correlation Expert, identify stacking patterns:
        
        Game: {game_info.get('teams', 'Unknown')} | O/U: {game_info.get('total', 0)} | Spread: {game_info.get('spread', 0)}
        
        Teams: {json.dumps(team_breakdown, indent=2)}
        
        Return ONLY valid JSON:
        {{
            "primary_stacks": [{{"qb": "name", "receiver": "name", "correlation": 0.6}}],
            "game_stacks": [{{"team1_player": "name1", "team2_player": "name2"}}],
            "leverage_stacks": [{{"player1": "name1", "player2": "name2"}}],
            "game_script": {{"most_likely": "balanced"}},
            "confidence": 0.8,
            "insights": ["key insight"]
        }}
        """
    
    def parse_response(self, response: str, df: pd.DataFrame) -> AIRecommendation:
        """Parse correlation strategist response"""
        try:
            data = json.loads(response) if response else {}
        except:
            data = {}
        
        all_stacks = []
        for stack in data.get('primary_stacks', []):
            if isinstance(stack, dict):
                all_stacks.append({
                    'player1': stack.get('qb'),
                    'player2': stack.get('receiver')
                })
        
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
            captain_targets=[],
            stacks=all_stacks,
            fades=[],
            boosts=[],
            strategy_weights=strategy_weights,
            key_insights=data.get('insights', ["Using default correlations"])
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
    
    def get_ai_strategies(self, use_api: bool = True) -> Tuple[AIRecommendation, AIRecommendation]:
        """Get strategies from both AIs"""
        
        if use_api and self.api_manager and self.api_manager.client:
            # API mode
            gt_prompt = self.game_theory_ai.generate_prompt(self.df, self.game_info)
            gt_response = self.api_manager.get_ai_response(gt_prompt)
            
            corr_prompt = self.correlation_ai.generate_prompt(self.df, self.game_info)
            corr_response = self.api_manager.get_ai_response(corr_prompt)
        else:
            # Manual mode
            st.subheader("Manual AI Strategy Input")
            
            tab1, tab2 = st.tabs(["Game Theory Prompt", "Correlation Prompt"])
            
            with tab1:
                st.text_area("Game Theory Prompt:", 
                           value=self.game_theory_ai.generate_prompt(self.df, self.game_info),
                           height=200, key="gt_prompt_display")
                gt_response = st.text_area("Paste Game Theory Response (JSON):", 
                                          height=150, key="gt_manual_input",
                                          value='{}')
            
            with tab2:
                st.text_area("Correlation Prompt:", 
                           value=self.correlation_ai.generate_prompt(self.df, self.game_info),
                           height=200, key="corr_prompt_display")
                corr_response = st.text_area("Paste Correlation Response (JSON):", 
                                            height=150, key="corr_manual_input",
                                            value='{}')
        
        rec1 = self.game_theory_ai.parse_response(gt_response, self.df)
        rec2 = self.correlation_ai.parse_response(corr_response, self.df)
        
        return rec1, rec2
    
    def combine_recommendations(self, rec1: AIRecommendation, rec2: AIRecommendation) -> Dict:
        """Combine recommendations from both AIs"""
        
        total_confidence = rec1.confidence + rec2.confidence
        w1 = rec1.confidence / total_confidence if total_confidence > 0 else 0.5
        w2 = rec2.confidence / total_confidence if total_confidence > 0 else 0.5
        
        combined_weights = {}
        for strategy in StrategyType:
            combined_weights[strategy] = (
                rec1.strategy_weights.get(strategy, 0) * w1 +
                rec2.strategy_weights.get(strategy, 0) * w2
            )
        
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            combined_weights = {k: v/total_weight for k, v in combined_weights.items()}
        
        return {
            'captain_scores': {},
            'strategy_weights': combined_weights,
            'consensus_fades': list(set(rec1.fades) & set(rec2.fades)) if rec1.fades and rec2.fades else [],
            'all_boosts': list(set(rec1.boosts) | set(rec2.boosts)) if rec1.boosts or rec2.boosts else [],
            'combined_stacks': rec1.stacks + rec2.stacks,
            'confidence': (rec1.confidence + rec2.confidence) / 2,
            'insights': rec1.key_insights + rec2.key_insights
        }
    
    def generate_lineups_with_buckets(self, num_lineups: int, rec1: AIRecommendation, 
                                     rec2: AIRecommendation) -> pd.DataFrame:
        """Generate lineups with relaxed constraints"""
        
        combined = self.combine_recommendations(rec1, rec2)
        
        players = self.df['Player'].tolist()
        positions = self.df.set_index('Player')['Position'].to_dict()
        teams = self.df.set_index('Player')['Team'].to_dict()
        salaries = self.df.set_index('Player')['Salary'].to_dict()
        points = self.df.set_index('Player')['Projected_Points'].to_dict()
        ownership = self.df.set_index('Player')['Ownership'].to_dict()
        
        # Debug information
        with st.expander("Debug Information"):
            st.write(f"Total players available: {len(players)}")
            st.write(f"Salary range: ${min(salaries.values())} - ${max(salaries.values())}")
            st.write(f"Points range: {min(points.values()):.1f} - {max(points.values()):.1f}")
            st.write(f"Sample players: {players[:5]}")
            st.write(f"Strategy weights: {combined['strategy_weights']}")
        
        # Apply slight adjustments for AI recommendations
        adjusted_points = points.copy()
        for player in combined['consensus_fades']:
            if player in adjusted_points:
                adjusted_points[player] *= 0.9  # Slight penalty
        
        for player in combined['all_boosts']:
            if player in adjusted_points:
                adjusted_points[player] *= 1.1  # Slight boost
        
        all_lineups = []
        
        # Generate lineups with simpler strategy
        for i in range(num_lineups):
            model = pulp.LpProblem(f"Lineup_{i+1}", pulp.LpMaximize)
            
            flex = pulp.LpVariable.dicts("Flex", players, cat='Binary')
            captain = pulp.LpVariable.dicts("Captain", players, cat='Binary')
            
            # Objective: maximize points
            model += pulp.lpSum([
                adjusted_points[p] * flex[p] + 
                1.5 * adjusted_points[p] * captain[p]
                for p in players
            ])
            
            # Basic constraints
            model += pulp.lpSum(captain.values()) == 1
            model += pulp.lpSum(flex.values()) == 5
            
            # No duplicate players
            for p in players:
                model += flex[p] + captain[p] <= 1
            
            # Salary constraint
            model += pulp.lpSum([
                salaries[p] * flex[p] + 
                1.5 * salaries[p] * captain[p]
                for p in players
            ]) <= OptimizerConfig.SALARY_CAP
            
            # Team constraint
            for team in self.df['Team'].unique():
                team_players = [p for p in players if teams.get(p) == team]
                if team_players:  # Only add if team has players
                    model += pulp.lpSum([
                        flex[p] + captain[p] 
                        for p in team_players
                    ]) <= OptimizerConfig.MAX_PLAYERS_PER_TEAM
            
            # Diversity constraint for subsequent lineups
            if i > 0 and all_lineups:
                for prev_lineup in all_lineups[-min(2, len(all_lineups)):]:
                    prev_players = [prev_lineup['Captain']] + prev_lineup['FLEX']
                    model += pulp.lpSum([
                        flex[p] + captain[p] 
                        for p in prev_players
                    ]) <= 5  # Force at least 1 different player
            
            # Solve with default solver
            model.solve()
            
            status = pulp.LpStatus[model.status]
            
            if status == 'Optimal':
                # Extract solution
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
                    total_ownership = ownership.get(captain_pick, 5) * 1.5 + sum(ownership.get(p, 5) for p in flex_picks)
                    
                    lineup_players = [captain_pick] + flex_picks
                    bucket_summary = self.bucket_manager.get_bucket_summary(lineup_players, self.df)
                    
                    all_lineups.append({
                        'Lineup': i + 1,
                        'Captain': captain_pick,
                        'FLEX': flex_picks,
                        'Projected': round(total_proj, 2),
                        'Salary': int(total_salary),
                        'Total_Ownership': round(total_ownership, 1),
                        'Bucket_Summary': bucket_summary
                    })
            else:
                st.warning(f"Lineup {i+1} failed with status: {status}")
        
        if not all_lineups:
            st.error("No valid lineups could be generated. Check your data and constraints.")
            
            # Test with a simple lineup
            st.write("Testing simple lineup generation...")
            test_model = pulp.LpProblem("Test", pulp.LpMaximize)
            test_flex = pulp.LpVariable.dicts("Flex", players, cat='Binary')
            test_captain = pulp.LpVariable.dicts("Captain", players, cat='Binary')
            
            test_model += pulp.lpSum([points[p] * (test_flex[p] + 1.5 * test_captain[p]) for p in players])
            test_model += pulp.lpSum(test_captain.values()) == 1
            test_model += pulp.lpSum(test_flex.values()) == 5
            for p in players:
                test_model += test_flex[p] + test_captain[p] <= 1
            test_model += pulp.lpSum([salaries[p] * (test_flex[p] + 1.5 * test_captain[p]) for p in players]) <= 50000
            
            test_model.solve()
            if pulp.LpStatus[test_model.status] == 'Optimal':
                st.success("Basic lineup CAN be created - constraints might be too tight in main optimizer")
            else:
                st.error("Even basic lineup failed - check your CSV data")
        
        return pd.DataFrame(all_lineups)
    
    def generate_captain_pivots(self, lineups_df: pd.DataFrame) -> pd.DataFrame:
        """Generate captain pivot variations for top lineups"""
        all_pivots = []
        
        for idx, lineup in lineups_df.head(5).iterrows():
            pivots = self.pivot_generator.generate_pivots(lineup.to_dict(), self.df, max_pivots=2)
            
            for pivot in pivots:
                pivot['Parent_Lineup'] = lineup['Lineup']
                pivot['Lineup'] = f"{lineup['Lineup']}-P{len(all_pivots)+1}"
                all_pivots.append(pivot)
        
        return pd.DataFrame(all_pivots) if all_pivots else pd.DataFrame()

def load_and_validate_data(uploaded_file) -> pd.DataFrame:
    """Load and validate the uploaded CSV"""
    df = pd.read_csv(uploaded_file)
    
    required_cols = ['first_name', 'last_name', 'position', 'team', 'salary', 'ppg_projection']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()
    
    # Clean data
    df['first_name'] = df['first_name'].fillna('')
    df['last_name'] = df['last_name'].fillna('')
    df['Player'] = (df['first_name'] + ' ' + df['last_name']).str.strip()
    
    # Remove empty player names
    df = df[df['Player'].str.len() > 0]
    
    df = df.rename(columns={
        'position': 'Position',
        'team': 'Team',
        'salary': 'Salary',
        'ppg_projection': 'Projected_Points'
    })
    
    # Ensure numeric columns
    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
    df['Projected_Points'] = pd.to_numeric(df['Projected_Points'], errors='coerce')
    
    # Filter valid data
    df = df[df['Salary'] >= OptimizerConfig.MIN_SALARY]
    df = df[df['Projected_Points'] > 0]
    df = df.dropna(subset=['Salary', 'Projected_Points'])
    
    # Data validation info
    st.success(f"✅ Loaded {len(df)} valid players")
    
    with st.expander("Data Quality Check"):
        st.write(f"Players loaded: {len(df)}")
        st.write(f"Salary range: ${df['Salary'].min():.0f} - ${df['Salary'].max():.0f}")
        st.write(f"Projection range: {df['Projected_Points'].min():.1f} - {df['Projected_Points'].max():.1f}")
        st.write(f"Teams: {', '.join(df['Team'].unique())}")
        st.write(f"Positions: {', '.join(df['Position'].unique())}")
    
    return df[['Player', 'Position', 'Team', 'Salary', 'Projected_Points']]

# Simplified simulation function
def simple_simulation(lineup: Dict, df: pd.DataFrame) -> Dict[str, float]:
    """Simplified simulation for faster results"""
    captain = lineup['Captain']
    flex_players = lineup['FLEX'] if isinstance(lineup['FLEX'], list) else lineup['FLEX']
    
    projections = df.set_index('Player')['Projected_Points'].to_dict()
    
    captain_proj = projections.get(captain, 0) * 1.5
    flex_proj = sum(projections.get(p, 0) for p in flex_players)
    total_proj = captain_proj + flex_proj
    
    return {
        'Mean': round(total_proj, 2),
        'Ceiling_90th': round(total_proj * 1.3, 2),  # Simple 30% upside
        'Floor_10th': round(total_proj * 0.7, 2)     # Simple 30% downside
    }

# MAIN UI
with st.sidebar:
    st.header("🤖 AI Configuration")
    
    api_mode = st.radio(
        "Mode",
        ["Manual (Free)", "API (Automated)"],
        help="Manual mode for testing, API for automation"
    )
    
    api_manager = None
    use_api = False
    
    if api_mode == "API (Automated)":
        api_key = st.text_input(
            "Claude API Key",
            type="password",
            placeholder="sk-ant-api03-..."
        )
        
        if api_key:
            api_manager = ClaudeAPIManager(api_key)
            use_api = bool(api_manager.client)
    
    st.markdown("---")
    st.info("""
    **Quick Start:**
    1. Upload CSV
    2. Add injuries/ownership
    3. Generate lineups
    4. Export to DraftKings
    """)

# Main content
st.subheader("📁 Data Upload")
uploaded_file = st.file_uploader("Upload DraftKings CSV", type="csv")

if uploaded_file is not None:
    df = load_and_validate_data(uploaded_file)
    
    # Game setup
    st.subheader("⚙️ Game Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        teams = st.text_input("Teams", "BUF vs MIA")
    with col2:
        total = st.number_input("O/U", 30.0, 70.0, 48.5, 0.5)
    with col3:
        spread = st.number_input("Spread", -20.0, 20.0, -3.0, 0.5)
    
    game_info = {'teams': teams, 'total': total, 'spread': spread}
    
    # Player adjustments
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏥 Injuries")
        injury_text = st.text_area(
            "Format: Player: Status",
            height=100,
            help="Enter injuries like: Player Name: OUT"
        )
        
        injuries = {}
        for line in injury_text.split('\n'):
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    injuries[parts[0].strip()] = parts[1].strip()
        
        injury_multipliers = {'OUT': 0.0, 'DOUBTFUL': 0.3, 'QUESTIONABLE': 0.75}
        for player, status in injuries.items():
            if player in df['Player'].values:
                mult = injury_multipliers.get(status.upper(), 1.0)
                df.loc[df['Player'] == player, 'Projected_Points'] *= mult
    
    with col2:
        st.subheader("📊 Ownership")
        ownership_text = st.text_area(
            "Format: Player: %",
            height=100,
            help="Enter ownership like: Player Name: 35"
        )
        
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
    
    # Display player pool
    st.subheader("Player Pool")
    st.dataframe(
        df.sort_values('Projected_Points', ascending=False),
        height=250,
        use_container_width=True
    )
    
    # Optimization
    st.subheader("🚀 Optimization")
    col1, col2 = st.columns(2)
    
    with col1:
        num_lineups = st.slider("Number of Lineups", 5, 30, 10)
    with col2:
        enable_pivots = st.checkbox("Generate Captain Pivots", value=False)
    
    if st.button("Generate Optimized Lineups", type="primary"):
        
        optimizer = DualAIOptimizer(df, game_info, api_manager)
        
        # Get strategies
        with st.spinner("Getting AI strategies..."):
            rec1, rec2 = optimizer.get_ai_strategies(use_api=use_api)
        
        # Generate lineups
        with st.spinner(f"Generating {num_lineups} lineups..."):
            lineups_df = optimizer.generate_lineups_with_buckets(num_lineups, rec1, rec2)
        
        if not lineups_df.empty:
            # Add simple metrics
            for idx, row in lineups_df.iterrows():
                sim_results = simple_simulation(row.to_dict(), df)
                for key, value in sim_results.items():
                    lineups_df.loc[idx, key] = value
            
            # Sort by projection
            lineups_df = lineups_df.sort_values('Mean', ascending=False)
            
            # Generate pivots
            if enable_pivots:
                pivots_df = optimizer.generate_captain_pivots(lineups_df)
                if not pivots_df.empty:
                    st.success(f"Generated {len(pivots_df)} captain pivots")
            
            # Display results
            st.subheader("📊 Generated Lineups")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lineups Generated", len(lineups_df))
            with col2:
                st.metric("Avg Projection", f"{lineups_df['Mean'].mean():.1f}")
            with col3:
                st.metric("Avg Ownership", f"{lineups_df['Total_Ownership'].mean():.1f}%")
            
            # Show lineups
            display_cols = ['Lineup', 'Captain', 'Mean', 'Ceiling_90th', 'Total_Ownership', 'Salary']
            st.dataframe(lineups_df[display_cols], use_container_width=True)
            
            # Detailed view of top lineup
            if len(lineups_df) > 0:
                st.subheader("🏆 Top Lineup Detail")
                top = lineups_df.iloc[0]
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Captain:** {top['Captain']}")
                    st.write("**FLEX Players:**")
                    for player in top['FLEX']:
                        st.write(f"- {player}")
                
                with col2:
                    st.metric("Projected", f"{top['Mean']:.1f}")
                    st.metric("Salary Used", f"${top['Salary']:,}")
                    st.write(f"**Ownership:** {top['Bucket_Summary']}")
            
            # Export
            st.subheader("💾 Export")
            
            dk_lineups = []
            for idx, row in lineups_df.iterrows():
                flex_players = row['FLEX']
                dk_lineups.append({
                    'CPT': row['Captain'],
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

st.markdown("---")
st.caption("NFL Optimizer v3.2 - Fixed constraints for reliable lineup generation")
