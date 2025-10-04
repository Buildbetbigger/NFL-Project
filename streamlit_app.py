"""
NFL DFS AI-Driven Optimizer - Streamlit Interface
Version: 2.1.1
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback
import io
import time

st.set_page_config(
    page_title="NFL DFS AI Optimizer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# OPTIMIZER IMPORTS
# ============================================================================

try:
    import nfl_dfs_optimizer as optimizer
    
    OptimizerConfig = optimizer.OptimizerConfig
    
    try:
        AIEnforcementLevel = optimizer.AIEnforcementLevel
    except AttributeError:
        from enum import Enum
        class AIEnforcementLevel(Enum):
            ADVISORY = "Advisory"
            MODERATE = "Moderate"
            STRONG = "Strong"
            MANDATORY = "Mandatory"
    
    try:
        OptimizationMode = optimizer.OptimizationMode
    except AttributeError:
        from enum import Enum
        class OptimizationMode(Enum):
            BALANCED = "balanced"
            CEILING = "ceiling"
            FLOOR = "floor"
            BOOM_OR_BUST = "boom_or_bust"
    
    try:
        FieldSize = optimizer.FieldSize
    except AttributeError:
        from enum import Enum
        class FieldSize(Enum):
            SMALL = "small_field"
            MEDIUM = "medium_field"
            LARGE = "large_field"
            LARGE_AGGRESSIVE = "large_field_aggressive"
            MILLY_MAKER = "milly_maker"
    
    try:
        AIStrategistType = optimizer.AIStrategistType
    except AttributeError:
        from enum import Enum
        class AIStrategistType(Enum):
            GAME_THEORY = "Game Theory"
            CORRELATION = "Correlation"
            CONTRARIAN_NARRATIVE = "Contrarian Narrative"
    
    AIRecommendation = optimizer.AIRecommendation
    LineupConstraints = optimizer.LineupConstraints
    SimulationResults = optimizer.SimulationResults
    GeneticConfig = optimizer.GeneticConfig
    MonteCarloSimulationEngine = optimizer.MonteCarloSimulationEngine
    GeneticAlgorithmOptimizer = optimizer.GeneticAlgorithmOptimizer
    AnthropicAPIManager = optimizer.AnthropicAPIManager
    GameTheoryStrategist = optimizer.GameTheoryStrategist
    CorrelationStrategist = optimizer.CorrelationStrategist
    ContrarianNarrativeStrategist = optimizer.ContrarianNarrativeStrategist
    AIEnforcementEngine = optimizer.AIEnforcementEngine
    AISynthesisEngine = optimizer.AISynthesisEngine
    AIOwnershipBucketManager = optimizer.AIOwnershipBucketManager
    AIConfigValidator = optimizer.AIConfigValidator
    OptimizedDataProcessor = optimizer.OptimizedDataProcessor
    get_logger = optimizer.get_logger
    get_performance_monitor = optimizer.get_performance_monitor
    get_ai_tracker = optimizer.get_ai_tracker
    ValidationError = optimizer.ValidationError
    OptimizationError = optimizer.OptimizationError
    __version__ = optimizer.__version__
    
    OPTIMIZER_AVAILABLE = True
    
except ImportError as e:
    st.error(f"Failed to import optimizer: {str(e)}")
    st.error("Ensure nfl_dfs_optimizer.py is in the same directory")
    st.stop()

except AttributeError as e:
    st.error(f"Missing optimizer component: {str(e)}")
    st.error("Ensure all 7 parts are combined in nfl_dfs_optimizer.py")
    st.stop()

APP_VERSION = "2.1.1"
OPTIMIZER_VERSION = __version__

def safe_session_state_get(key: str, default: Any = None) -> Any:
    try:
        return st.session_state.get(key, default)
    except Exception:
        return default

def safe_session_state_set(key: str, value: Any) -> None:
    try:
        st.session_state[key] = value
    except Exception as e:
        st.error(f"Error saving to session state: {e}")

def initialize_session_state():
    defaults = {
        'df': None,
        'uploaded_file_name': None,
        'game_total': 47.0,
        'spread': 0.0,
        'home_team': '',
        'away_team': '',
        'salary_cap': OptimizerConfig.SALARY_CAP,
        'num_lineups': 20,
        'field_size': 'large_field',
        'ai_enforcement': 'Strong',
        'optimization_mode': 'balanced',
        'use_monte_carlo': True,
        'use_genetic': False,
        'anthropic_api_key': '',
        'use_api': False,
        'lineups': None,
        'ai_recommendations': None,
        'optimization_complete': False,
        'last_optimization_time': None,
        'min_salary_pct': 90,
        'max_ownership': 200,
        'max_exposure': None,
        'locked_players': [],
        'banned_players': [],
        'show_debug': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def validate_dataframe(df: pd.DataFrame) -> tuple:
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    required_columns = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    if len(df) < 6:
        return False, f"Need at least 6 players, found {len(df)}"
    
    try:
        df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
        df['Projected_Points'] = pd.to_numeric(df['Projected_Points'], errors='coerce')
        
        if df['Salary'].isna().any() or df['Projected_Points'].isna().any():
            return False, "Found non-numeric values in Salary or Projected_Points"
    except Exception as e:
        return False, f"Error validating numeric columns: {e}"
    
    invalid_salaries = (
        (df['Salary'] < OptimizerConfig.MIN_SALARY) |
        (df['Salary'] > OptimizerConfig.MAX_SALARY * 1.2)
    ).sum()
    
    if invalid_salaries > 0:
        return False, f"Found {invalid_salaries} players with invalid salaries"
    
    if 'Ownership' not in df.columns:
        df['Ownership'] = 10.0
        st.info("Ownership column not found - using default 10%")
    
    return True, ""

def format_currency(value: float) -> str:
    return f"${value:,.0f}"

def format_percentage(value: float) -> str:
    return f"{value:.1f}%"

def create_download_csv(df: pd.DataFrame, filename: str) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def main():
    initialize_session_state()
    
    st.title("🏈 NFL DFS AI-Driven Optimizer")
    st.markdown(f"**Version:** {APP_VERSION} | **Optimizer:** {OPTIMIZER_VERSION}")
    
    render_sidebar()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Data Upload",
        "🎯 Optimization",
        "📊 Results",
        "⚙️ Advanced Settings"
    ])
    
    with tab1:
        render_data_upload_tab()
    
    with tab2:
        render_optimization_tab()
    
    with tab3:
        render_results_tab()
    
    with tab4:
        render_advanced_settings_tab()
    
    render_footer()

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        with st.expander("🤖 AI Settings", expanded=False):
            use_api = st.checkbox(
                "Use Anthropic AI",
                value=safe_session_state_get('use_api', False),
                help="Enable AI-powered recommendations"
            )
            safe_session_state_set('use_api', use_api)
            
            if use_api:
                api_key = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    value=safe_session_state_get('anthropic_api_key', ''),
                    help="Get your API key from console.anthropic.com"
                )
                safe_session_state_set('anthropic_api_key', api_key)
                
                if api_key and not api_key.startswith('sk-ant-'):
                    st.warning("API key should start with 'sk-ant-'")
            else:
                st.info("Using statistical fallback mode")
        
        with st.expander("🎯 Optimization Settings", expanded=True):
            num_lineups = st.number_input(
                "Number of Lineups",
                min_value=1,
                max_value=150,
                value=safe_session_state_get('num_lineups', 20)
            )
            safe_session_state_set('num_lineups', num_lineups)
            
            field_size = st.selectbox(
                "Contest Type",
                options=list(OptimizerConfig.FIELD_SIZES.keys()),
                index=list(OptimizerConfig.FIELD_SIZES.keys()).index('Large GPP (1000+)')
            )
            safe_session_state_set('field_size', OptimizerConfig.FIELD_SIZES[field_size])
            
            ai_enforcement = st.selectbox(
                "AI Enforcement Level",
                options=['Advisory', 'Moderate', 'Strong', 'Mandatory'],
                index=2
            )
            safe_session_state_set('ai_enforcement', ai_enforcement)
            
            optimization_mode = st.selectbox(
                "Optimization Mode",
                options=['balanced', 'ceiling', 'floor', 'boom_or_bust'],
                index=0
            )
            safe_session_state_set('optimization_mode', optimization_mode)
        
        with st.expander("🧬 Algorithm Settings", expanded=False):
            use_monte_carlo = st.checkbox(
                "Enable Monte Carlo Simulation",
                value=safe_session_state_get('use_monte_carlo', True)
            )
            safe_session_state_set('use_monte_carlo', use_monte_carlo)
            
            use_genetic = st.checkbox(
                "Use Genetic Algorithm",
                value=safe_session_state_get('use_genetic', False)
            )
            safe_session_state_set('use_genetic', use_genetic)
        
        with st.expander("🐛 Debug", expanded=False):
            show_debug = st.checkbox(
                "Show Debug Info",
                value=safe_session_state_get('show_debug', False)
            )
            safe_session_state_set('show_debug', show_debug)

def render_data_upload_tab():
    st.header("📤 Upload Player Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV with player projections",
        type=['csv'],
        help="CSV must contain: Player, Position, Team, Salary, Projected_Points"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            is_valid, error_message = validate_dataframe(df)
            
            if not is_valid:
                st.error(f"❌ Validation Error: {error_message}")
                return
            
            safe_session_state_set('df', df)
            safe_session_state_set('uploaded_file_name', uploaded_file.name)
            
            teams = sorted(df['Team'].unique())
            if len(teams) >= 2:
                safe_session_state_set('home_team', teams[0])
                safe_session_state_set('away_team', teams[1])
            
            st.success(f"✅ Successfully loaded {len(df)} players")
            
            st.subheader("Data Preview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Players", len(df))
            with col2:
                st.metric("Teams", len(teams))
            with col3:
                st.metric("Avg Salary", format_currency(df['Salary'].mean()))
            with col4:
                st.metric("Avg Projection", f"{df['Projected_Points'].mean():.2f}")
            
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            
            st.subheader("Position Breakdown")
            position_counts = df['Position'].value_counts()
            
            cols = st.columns(len(position_counts))
            for i, (pos, count) in enumerate(position_counts.items()):
                with cols[i]:
                    st.metric(pos, count)
            
            if 'Ownership' in df.columns:
                st.subheader("Ownership Distribution")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Ownership", format_percentage(df['Ownership'].mean()))
                with col2:
                    st.metric("Max Ownership", format_percentage(df['Ownership'].max()))
        
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")
            if safe_session_state_get('show_debug', False):
                st.code(traceback.format_exc())
    
    else:
        st.info("📋 Upload a CSV file to get started")

def render_optimization_tab():
    st.header("🎯 Lineup Optimization")
    
    df = safe_session_state_get('df')
    if df is None:
        st.warning("⚠️ Please upload player data first")
        return
    
    st.subheader("📊 Game Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.text_input(
            "Home Team",
            value=safe_session_state_get('home_team', '')
        )
        safe_session_state_set('home_team', home_team)
        
        game_total = st.number_input(
            "Game Total",
            min_value=30.0,
            max_value=70.0,
            value=safe_session_state_get('game_total', 47.0),
            step=0.5
        )
        safe_session_state_set('game_total', game_total)
    
    with col2:
        away_team = st.text_input(
            "Away Team",
            value=safe_session_state_get('away_team', '')
        )
        safe_session_state_set('away_team', away_team)
        
        spread = st.number_input(
            "Spread",
            min_value=-20.0,
            max_value=20.0,
            value=safe_session_state_get('spread', 0.0),
            step=0.5
        )
        safe_session_state_set('spread', spread)
    
    st.subheader("💰 Salary Cap Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        salary_cap = st.number_input(
            "Salary Cap",
            min_value=30000,
            max_value=100000,
            value=safe_session_state_get('salary_cap', OptimizerConfig.SALARY_CAP),
            step=1000
        )
        
        is_valid, error_msg = OptimizerConfig.validate_salary_cap(salary_cap)
        if not is_valid:
            st.error(f"❌ {error_msg}")
        else:
            safe_session_state_set('salary_cap', salary_cap)
    
    with col2:
        min_salary_pct = st.slider(
            "Minimum Salary % of Cap",
            min_value=80,
            max_value=100,
            value=safe_session_state_get('min_salary_pct', 90)
        )
        safe_session_state_set('min_salary_pct', min_salary_pct)
        
        min_salary = int(salary_cap * (min_salary_pct / 100))
        st.info(f"Min Salary: {format_currency(min_salary)}")
    
    st.subheader("🔒 Player Constraints")
    
    col1, col2 = st.columns(2)
    
    with col1:
        locked_players = st.multiselect(
            "Locked Players",
            options=sorted(df['Player'].tolist()),
            default=safe_session_state_get('locked_players', [])
        )
        safe_session_state_set('locked_players', locked_players)
    
    with col2:
        banned_players = st.multiselect(
            "Banned Players",
            options=sorted(df['Player'].tolist()),
            default=safe_session_state_get('banned_players', [])
        )
        safe_session_state_set('banned_players', banned_players)
    
    conflicts = set(locked_players) & set(banned_players)
    if conflicts:
        st.error(f"❌ Conflicts: {', '.join(conflicts)}")
        return
    
    if locked_players:
        locked_df = df[df['Player'].isin(locked_players)]
        locked_salary = locked_df['Salary'].sum()
        
        if locked_salary > salary_cap:
            st.error(f"❌ Locked players exceed cap")
            return
        
        if len(locked_players) > 6:
            st.error("❌ Cannot lock more than 6 players")
            return
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        optimize_button = st.button(
            "🚀 Generate Lineups",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            reset_optimization()
            st.rerun()
    
    if optimize_button:
        execute_optimization(df, salary_cap, min_salary)

def reset_optimization():
    safe_session_state_set('lineups', None)
    safe_session_state_set('ai_recommendations', None)
    safe_session_state_set('optimization_complete', False)
    safe_session_state_set('last_optimization_time', None)

def execute_optimization(df: pd.DataFrame, salary_cap: int, min_salary: int):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("⚙️ Initializing optimizer...")
        progress_bar.progress(10)
        
        logger = get_logger()
        perf_monitor = get_performance_monitor()
        
        num_lineups = safe_session_state_get('num_lineups', 20)
        field_size = safe_session_state_get('field_size', 'large_field')
        ai_enforcement = safe_session_state_get('ai_enforcement', 'Strong')
        use_api = safe_session_state_get('use_api', False)
        use_monte_carlo = safe_session_state_get('use_monte_carlo', True)
        use_genetic = safe_session_state_get('use_genetic', False)
        
        game_info = {
            'game_total': safe_session_state_get('game_total', 47.0),
            'spread': safe_session_state_get('spread', 0.0),
            'home_team': safe_session_state_get('home_team', ''),
            'away_team': safe_session_state_get('away_team', ''),
            'teams': [safe_session_state_get('home_team', ''), 
                     safe_session_state_get('away_team', '')]
        }
        
        field_config = OptimizerConfig.get_field_config(field_size)
        
        status_text.text("🤖 Running AI analysis...")
        progress_bar.progress(20)
        
        api_key = safe_session_state_get('anthropic_api_key', '') if use_api else None
        api_manager = AnthropicAPIManager(api_key=api_key, fallback_mode=not use_api)
        
        game_theory = GameTheoryStrategist(api_manager)
        correlation = CorrelationStrategist(api_manager)
        contrarian = ContrarianNarrativeStrategist(api_manager)
        
        gt_rec = game_theory.analyze(df, game_info, field_config)
        progress_bar.progress(35)
        
        corr_rec = correlation.analyze(df, game_info, field_config)
        progress_bar.progress(50)
        
        contra_rec = contrarian.analyze(df, game_info, field_config)
        progress_bar.progress(60)
        
        ai_recommendations = {
            'game_theory': gt_rec,
            'correlation': corr_rec,
            'contrarian': contra_rec
        }
        safe_session_state_set('ai_recommendations', ai_recommendations)
        
        status_text.text("🔄 Synthesizing recommendations...")
        progress_bar.progress(65)
        
        synthesis_engine = AISynthesisEngine()
        synthesis = synthesis_engine.synthesize_recommendations(gt_rec, corr_rec, contra_rec)
        
        status_text.text("⚖️ Configuring constraints...")
        progress_bar.progress(70)
        
        enforcement_level = AIEnforcementLevel[ai_enforcement.upper()]
        enforcement_engine = AIEnforcementEngine(enforcement_level)
        
        enforcement_rules = enforcement_engine.create_enforcement_rules({
            AIStrategistType.GAME_THEORY: gt_rec,
            AIStrategistType.CORRELATION: corr_rec,
            AIStrategistType.CONTRARIAN_NARRATIVE: contra_rec
        })
        
        validation = AIConfigValidator.validate_ai_requirements(enforcement_rules, df)
        
        if not validation['is_valid']:
            st.error("❌ Configuration Validation Failed")
            for error in validation['errors']:
                st.error(f"  • {error}")
            progress_bar.empty()
            status_text.empty()
            return
        
        status_text.text("🎯 Generating optimal lineups...")
        progress_bar.progress(75)
        
        if use_genetic:
            status_text.text("🧬 Running genetic algorithm...")
            
            mc_engine = None
            if use_monte_carlo:
                mc_engine = MonteCarloSimulationEngine(df, game_info, n_simulations=1000)
            
            ga_optimizer = GeneticAlgorithmOptimizer(
                df=df,
                game_info=game_info,
                mc_engine=mc_engine,
                config=GeneticConfig(population_size=100, generations=50),
                salary_cap=salary_cap
            )
            
            ga_results = ga_optimizer.optimize(num_lineups=num_lineups, verbose=False)
            
            lineups = []
            for result in ga_results:
                lineup = {
                    'Captain': result['captain'],
                    'FLEX': ', '.join(result['flex']),
                    'Total_Salary': 0,
                    'Projected': 0,
                    'Total_Ownership': 0
                }
                
                all_players = [result['captain']] + result['flex']
                player_data = df[df['Player'].isin(all_players)]
                
                capt_data = player_data[player_data['Player'] == result['captain']].iloc[0]
                flex_data = player_data[player_data['Player'].isin(result['flex'])]
                
                lineup['Total_Salary'] = capt_data['Salary'] * 1.5 + flex_data['Salary'].sum()
                lineup['Projected'] = capt_data['Projected_Points'] * 1.5 + flex_data['Projected_Points'].sum()
                lineup['Total_Ownership'] = capt_data['Ownership'] * 1.5 + flex_data['Ownership'].sum()
                
                if result.get('sim_results'):
                    lineup['Ceiling_90th'] = result['sim_results'].ceiling_90th
                    lineup['Floor_10th'] = result['sim_results'].floor_10th
                
                lineups.append(lineup)
        else:
            st.warning("⚠️ Use Genetic Algorithm for lineup generation")
            progress_bar.empty()
            status_text.empty()
            return
        
        progress_bar.progress(90)
        
        status_text.text("📊 Finalizing results...")
        
        lineups_df = pd.DataFrame(lineups)
        lineups_df.insert(0, 'Lineup', range(1, len(lineups_df) + 1))
        
        safe_session_state_set('lineups', lineups_df)
        safe_session_state_set('optimization_complete', True)
        safe_session_state_set('last_optimization_time', datetime.now())
        
        progress_bar.progress(100)
        status_text.text("✅ Optimization complete!")
        
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Successfully generated {len(lineups_df)} unique lineups!")
        st.balloons()
        st.info("👉 View results in the 'Results' tab")
    
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        
        st.error(f"❌ Optimization failed: {str(e)}")
        
        if safe_session_state_get('show_debug', False):
            st.code(traceback.format_exc())

def render_results_tab():
    st.header("📊 Optimization Results")
    
    lineups_df = safe_session_state_get('lineups')
    
    if lineups_df is None:
        st.info("ℹ️ No results yet. Run optimization first.")
        return
    
    st.subheader("📈 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Lineups", len(lineups_df))
    
    with col2:
        st.metric("Avg Projection", f"{lineups_df['Projected'].mean():.2f}")
    
    with col3:
        st.metric("Avg Salary", format_currency(lineups_df['Total_Salary'].mean()))
    
    with col4:
        st.metric("Avg Ownership", format_percentage(lineups_df['Total_Ownership'].mean()))
    
    st.subheader("🎯 Generated Lineups")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        show_columns = st.multiselect(
            "Columns to display",
            options=list(lineups_df.columns),
            default=['Lineup', 'Captain', 'FLEX', 'Total_Salary', 'Projected', 'Total_Ownership']
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            options=['Lineup', 'Projected', 'Total_Salary', 'Total_Ownership'],
            index=1
        )
    
    display_df = lineups_df[show_columns].sort_values(sort_by, ascending=False)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
    
    st.subheader("💾 Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = create_download_csv(lineups_df, "lineups.csv")
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"lineups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        dk_format = convert_to_dk_format(lineups_df, safe_session_state_get('df'))
        if dk_format is not None:
            csv_data = create_download_csv(dk_format, "dk_upload.csv")
            st.download_button(
                label="📥 Download DK Format",
                data=csv_data,
                file_name=f"dk_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    ai_recommendations = safe_session_state_get('ai_recommendations')
    
    if ai_recommendations:
        st.subheader("🤖 AI Recommendations")
        
        tab1, tab2, tab3 = st.tabs(["Game Theory", "Correlation", "Contrarian"])
        
        with tab1:
            display_ai_recommendation(ai_recommendations.get('game_theory'))
        
        with tab2:
            display_ai_recommendation(ai_recommendations.get('correlation'))
        
        with tab3:
            display_ai_recommendation(ai_recommendations.get('contrarian'))

def convert_to_dk_format(lineups_df: pd.DataFrame, player_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        dk_lineups = []
        
        for _, lineup in lineups_df.iterrows():
            captain = lineup['Captain']
            flex_str = lineup['FLEX']
            
            flex = [p.strip() for p in flex_str.split(',')]
            
            dk_lineup = {
                'CPT': captain,
                'FLEX1': flex[0] if len(flex) > 0 else '',
                'FLEX2': flex[1] if len(flex) > 1 else '',
                'FLEX3': flex[2] if len(flex) > 2 else '',
                'FLEX4': flex[3] if len(flex) > 3 else '',
                'FLEX5': flex[4] if len(flex) > 4 else '',
            }
            
            dk_lineups.append(dk_lineup)
        
        return pd.DataFrame(dk_lineups)
    
    except Exception as e:
        st.error(f"Error converting to DK format: {e}")
        return None

def display_ai_recommendation(recommendation: Optional[AIRecommendation]):
    if recommendation is None:
        st.info("No recommendation available")
        return
    
    st.metric("Confidence", format_percentage(recommendation.confidence * 100))
    
    if recommendation.narrative:
        st.info(f"**Strategy:** {recommendation.narrative}")
    
    if recommendation.captain_targets:
        st.markdown("**Captain Targets:**")
        st.write(", ".join(recommendation.captain_targets[:5]))
    
    if recommendation.must_play:
        st.markdown("**Must Play:**")
        st.write(", ".join(recommendation.must_play))
    
    if recommendation.never_play:
        st.markdown("**Never Play:**")
        st.write(", ".join(recommendation.never_play))
    
    if recommendation.stacks:
        st.markdown("**Recommended Stacks:**")
        for i, stack in enumerate(recommendation.stacks[:3], 1):
            if 'player1' in stack and 'player2' in stack:
                st.write(f"{i}. {stack['player1']} + {stack['player2']}")
    
    if recommendation.key_insights:
        st.markdown("**Key Insights:**")
        for insight in recommendation.key_insights[:5]:
            st.write(f"• {insight}")

def render_advanced_settings_tab():
    st.header("⚙️ Advanced Settings")
    
    st.subheader("👥 Ownership Constraints")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_ownership = st.number_input(
            "Max Total Ownership %",
            min_value=50,
            max_value=300,
            value=safe_session_state_get('max_ownership', 200)
        )
        safe_session_state_set('max_ownership', max_ownership)
    
    with col2:
        max_exposure = st.slider(
            "Max Player Exposure %",
            min_value=10,
            max_value=100,
            value=25
        )
        safe_session_state_set('max_exposure', max_exposure / 100)
    
    st.info("Advanced settings will be applied on next optimization run")

def render_footer():
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**App Version:** {APP_VERSION}")
    
    with col2:
        st.markdown(f"**Optimizer:** v{OPTIMIZER_VERSION}")
    
    with col3:
        last_opt_time = safe_session_state_get('last_optimization_time')
        if last_opt_time:
            st.markdown(f"**Last Run:** {last_opt_time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
