"""
NFL DFS Optimizer - Complete Streamlit Application
Version: 5.0.1 - Production Ready with Debugging Fixes

COMPLETE REWRITE - Incorporates all debugging recommendations

FEATURES:
✅ Granular import checking with fallbacks
✅ Thread-safe optimization with proper locking
✅ MasterOptimizer integration
✅ Enhanced error messages with user guidance
✅ Robust session state management
✅ Pre-flight constraint checking
✅ Progress tracking with thread safety
✅ Comprehensive error handling
✅ Performance optimizations
✅ Memory-efficient file handling

COMPATIBLE WITH:
- NFL DFS Optimizer v5.0.0 (Parts 1-13)
- UPDATE 2: Advanced AI System
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict
import traceback
import io

# ============================================================================
# GRANULAR OPTIMIZER IMPORTS WITH FALLBACKS
# ============================================================================

# Track what's available
OPTIMIZER_STATUS = {
    'core': False,
    'algorithms': False,
    'ai': False,
    'advanced': False,
    'master': False
}

# Import Core Components
try:
    from nfl_dfs_optimizer import (
        # Data Processing
        OptimizedDataProcessor,
        safe_load_csv,
        
        # Core Classes
        DraftKingsRules,
        LineupConstraints,
        ValidationResult,
        GeneticConfig,
        SimulationResults,
        
        # Validation
        ConstraintFeasibilityChecker,
        BatchLineupValidator,
        
        # Utilities
        OptimizerLogger,
        PerformanceMonitor,
        UnifiedCache,
        get_logger,
        get_performance_monitor,
        get_unified_cache,
        
        # Configuration
        OptimizerConfig,
        CONTEST_TYPE_MAPPING,
        
        # Helper functions
        calculate_lineup_metrics,
        validate_lineup_with_context,
    )
    OPTIMIZER_STATUS['core'] = True
except ImportError as e:
    st.error(f"❌ Core optimizer imports failed: {e}")
    st.info("Please ensure nfl_dfs_optimizer.py is in the same directory")

# Import Algorithms
try:
    from nfl_dfs_optimizer import (
        GeneticAlgorithmOptimizer,
        SimulatedAnnealingOptimizer,
        SmartGreedyOptimizer,
        StandardLineupOptimizer,
    )
    OPTIMIZER_STATUS['algorithms'] = True
except ImportError as e:
    st.warning(f"⚠️ Some algorithms unavailable: {e}")

# Import AI Components
try:
    from nfl_dfs_optimizer import (
        AIStrategistType,
        AIEnforcementLevel,
        GameTheoryStrategist,
        CorrelationStrategist,
        ContrarianStrategist,
        StackingExpertStrategist,
        LeverageSpecialist,
    )
    OPTIMIZER_STATUS['ai'] = True
except ImportError as e:
    st.info(f"ℹ️ AI features unavailable: {e}")

# Import Advanced Features
try:
    from nfl_dfs_optimizer import (
        PlayerPoolAnalyzer,
        StackDetector,
        OwnershipAnalyzer,
        MonteCarloSimulationEngine,
        DiversityTracker,
        ExposureTracker,
    )
    OPTIMIZER_STATUS['advanced'] = True
except ImportError as e:
    st.info(f"ℹ️ Advanced features unavailable: {e}")

# Import Master Optimizer
try:
    from nfl_dfs_optimizer import (
        MasterOptimizer,
        optimize_showdown,
    )
    OPTIMIZER_STATUS['master'] = True
except ImportError as e:
    st.warning(f"⚠️ Master optimizer unavailable: {e}")

# Import Output
try:
    from nfl_dfs_optimizer import (
        CSVExporter,
        OutputManager,
    )
except ImportError:
    pass

# Overall status
OPTIMIZER_AVAILABLE = OPTIMIZER_STATUS['core']

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NFL DFS Optimizer v5.0",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================

class StreamlitConfig:
    """Streamlit-specific configuration"""
    
    # Timeouts
    OPTIMIZATION_TIMEOUT_SEC = 300
    THREAD_JOIN_SEC = 0.5
    
    # Display
    MAX_WARNINGS_SHOW = 5
    MAX_LINEUPS_DISPLAY = 50
    
    # Progress
    INITIAL_PROGRESS_THRESHOLD_SEC = 5
    PROGRESS_UPDATE_INTERVAL_SEC = 2
    
    # Cache
    CSV_CACHE_TTL_SEC = 3600  # 1 hour
    MAX_CACHE_ENTRIES = 5

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        # Core data
        'df': None,
        'processed_df': None,
        'lineups': [],
        'game_info': {},
        
        # Settings
        'num_lineups': 20,
        'game_total': 50.0,
        'spread': 0.0,
        'contest_type': 'Large GPP (1000+)',
        'optimization_mode': 'balanced',
        'min_salary_pct': 95,
        
        # AI Settings
        'use_ai': False,
        'ai_enforcement': 'moderate',
        'api_key': '',
        'use_iterative_refinement': True,
        'use_bayesian_synthesis': True,
        
        # Advanced Options
        'use_monte_carlo': True,
        'num_simulations': 10000,
        'diversity_threshold': 0.5,
        'randomness_factor': 0.15,
        'use_ensemble': False,
        
        # UI State
        'show_advanced': False,
        'show_debug': False,
        'show_profiling': False,
        'show_ai_details': False,
        
        # Processing flags
        'data_validated': False,
        'optimization_complete': False,
        'optimization_running': False,
        
        # Results
        'warnings': [],
        'errors': [],
        'last_optimization_time': None,
        'optimizer_summary': None,
        'pool_analysis': None,
        
        # File upload
        'last_file_hash': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize
init_session_state()

# ============================================================================
# STATE SNAPSHOT
# ============================================================================

class StateSnapshot:
    """Preserve state across errors"""
    
    def __init__(self):
        self.snapshot = {}
    
    def capture(self):
        """Capture current state"""
        self.snapshot = {
            k: v for k, v in st.session_state.items()
            if not k.startswith('_') and not k.startswith('FormSubmitter')
        }
    
    def restore(self):
        """Restore captured state"""
        for k, v in self.snapshot.items():
            if k in st.session_state:
                st.session_state[k] = v

# ============================================================================
# CACHING
# ============================================================================

@st.cache_data(
    ttl=StreamlitConfig.CSV_CACHE_TTL_SEC,
    max_entries=StreamlitConfig.MAX_CACHE_ENTRIES,
    show_spinner=False
)
def cached_csv_load(
    file_bytes: bytes,
    file_name: str,
    file_hash: str
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Cache CSV processing"""
    
    if not OPTIMIZER_AVAILABLE:
        return None, ["Optimizer not available"]
    
    logger = get_logger()
    warnings = []
    
    try:
        # Load CSV from bytes
        file_obj = io.BytesIO(file_bytes)
        df = pd.read_csv(file_obj)
        
        if df is None or df.empty:
            return None, ["CSV is empty"]
        
        # Process
        processor = OptimizedDataProcessor(logger)
        df_processed, proc_warnings = processor.process_dataframe(df)
        warnings.extend(proc_warnings)
        
        logger.log(f"Processed {len(df_processed)} players", "INFO")
        
        return df_processed, warnings
        
    except Exception as e:
        logger.log_exception(e, "cached_csv_load")
        return None, [f"Error: {str(e)}"]

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_player_pool(df_hash: str, df: pd.DataFrame) -> Dict:
    """Cache player pool analysis"""
    if not OPTIMIZER_STATUS['advanced']:
        return {}
    
    try:
        analyzer = PlayerPoolAnalyzer(df)
        return analyzer.analyze()
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# STYLING
# ============================================================================

def apply_css():
    """Apply custom CSS"""
    st.markdown("""
        <style>
        .main {padding-top: 1rem;}
        .stAlert {margin: 0.5rem 0;}
        h1 {color: #1e3a8a; font-weight: 700;}
        h2 {color: #2563eb; margin-top: 1.5rem;}
        h3 {color: #3b82f6; margin-top: 1rem;}
        .stButton>button {
            width: 100%;
            background-color: #3b82f6;
            color: white;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #2563eb;
        }
        .stButton>button:disabled {
            background-color: #94a3b8;
            cursor: not-allowed;
        }
        .success-box {
            background-color: #dcfce7;
            border-left: 4px solid #22c55e;
            padding: 1rem;
            border-radius: 0.25rem;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fef3c7;
            border-left: 4px solid #eab308;
            padding: 1rem;
            border-radius: 0.25rem;
            margin: 1rem 0;
        }
        .error-box {
            background-color: #fee2e2;
            border-left: 4px solid #ef4444;
            padding: 1rem;
            border-radius: 0.25rem;
            margin: 1rem 0;
        }
        .info-box {
            background-color: #dbeafe;
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            border-radius: 0.25rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CONSTRAINT DIAGNOSTICS
# ============================================================================

def run_constraint_check(
    df: pd.DataFrame,
    min_salary_pct: int
) -> Dict[str, Any]:
    """Run constraint feasibility check with validation"""
    
    # Validate inputs
    if df is None or df.empty:
        return {
            'is_feasible': False,
            'error': 'No player data available',
            'suggestions': ['Upload a valid CSV file first'],
            'constraints': None
        }
    
    if not OPTIMIZER_AVAILABLE:
        return {
            'is_feasible': False,
            'error': 'Optimizer not available',
            'suggestions': ['Check that nfl_dfs_optimizer.py is in the same directory'],
            'constraints': None
        }
    
    try:
        constraints = LineupConstraints(
            min_salary=int(DraftKingsRules.SALARY_CAP * (min_salary_pct / 100)),
            max_salary=DraftKingsRules.SALARY_CAP
        )
        
        is_feasible, error_msg, suggestions = ConstraintFeasibilityChecker.check(
            df,
            constraints
        )
        
        return {
            'is_feasible': is_feasible,
            'error': error_msg,
            'suggestions': suggestions,
            'constraints': constraints
        }
        
    except Exception as e:
        return {
            'is_feasible': False,
            'error': f'Constraint check failed: {str(e)}',
            'suggestions': [
                'Check player data format',
                'Ensure all required columns exist',
                'Verify salary values are valid'
            ],
            'constraints': None
        }

def render_constraint_diagnostics():
    """Render constraint diagnostics with auto-fix"""
    
    if st.session_state.processed_df is None:
        return
    
    df = st.session_state.processed_df
    
    with st.expander("🔍 Pre-Flight Constraint Check", expanded=False):
        st.subheader("Constraint Feasibility Analysis")
        
        # Run check
        diag = run_constraint_check(df, st.session_state.min_salary_pct)
        
        if diag['is_feasible']:
            st.markdown(
                '<div class="success-box">✅ All constraints feasible - optimization should succeed</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="error-box">❌ Constraint Issue: {diag["error"]}</div>',
                unsafe_allow_html=True
            )
            
            if diag['suggestions']:
                st.markdown("**Suggested Fixes:**")
                
                for i, suggestion in enumerate(diag['suggestions'], 1):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.write(f"{i}. {suggestion}")
                    
                    with col2:
                        # Auto-fix buttons
                        import re
                        match = re.search(r'(\d+)%', suggestion)
                        if match and ("lower" in suggestion.lower() or "reduce" in suggestion.lower()):
                            pct = int(match.group(1))
                            if st.button(f"Fix: {pct}%", key=f"fix_{i}"):
                                st.session_state.min_salary_pct = pct
                                st.rerun()
        
        # Pool statistics
        st.subheader("Player Pool Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        # Calculate feasibility bounds
        cheapest_6 = df.nsmallest(6, 'Salary')['Salary'].sum()
        cheapest_6 += df['Salary'].min() * 0.5  # Captain multiplier
        
        expensive_6 = df.nlargest(6, 'Salary')['Salary'].sum()
        expensive_6 += df['Salary'].max() * 0.5
        
        min_threshold = diag['constraints'].min_salary if diag['constraints'] else 0
        max_threshold = diag['constraints'].max_salary if diag['constraints'] else 50000
        
        with col1:
            st.metric("Players", len(df))
            st.metric("Teams", df['Team'].nunique())
        
        with col2:
            st.metric("Cheapest 6", f"${cheapest_6:,.0f}")
            st.metric("Most Expensive 6", f"${expensive_6:,.0f}")
        
        with col3:
            st.metric("Min Threshold", f"${min_threshold:,.0f}")
            st.metric("Max Threshold", f"${max_threshold:,.0f}")
        
        # Visual feasibility check
        if cheapest_6 > max_threshold:
            st.error("⚠️ IMPOSSIBLE: Even cheapest lineup exceeds cap!")
        elif expensive_6 < min_threshold:
            st.error("⚠️ IMPOSSIBLE: Can't reach minimum threshold!")
        elif cheapest_6 > min_threshold * 0.95:
            st.warning("⚠️ TIGHT: Minimal feasible range")
        else:
            st.success("✅ Good feasible range")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_process_data(uploaded_file):
    """Load and process uploaded CSV with improved error handling"""
    
    if not OPTIMIZER_AVAILABLE:
        st.error("Optimizer core not available")
        return None, ["Optimizer not loaded"]
    
    logger = get_logger()
    
    try:
        # Read file bytes ONCE (more efficient)
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        file_hash = hashlib.md5(file_bytes).hexdigest()[:8]
        
        # Check if same file
        if file_hash == st.session_state.last_file_hash:
            st.info("Using cached data (same file)")
            return st.session_state.processed_df, st.session_state.warnings
        
        # Process with caching
        df, warnings = cached_csv_load(file_bytes, file_name, file_hash)
        
        if df is None:
            return None, warnings
        
        # Update hash
        st.session_state.last_file_hash = file_hash
        
        # Show preview
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Showing 10 of {len(df)} rows")
        
        # Show warnings
        if warnings:
            with st.expander(f"⚠️ Warnings ({len(warnings)})", expanded=False):
                for w in warnings[:StreamlitConfig.MAX_WARNINGS_SHOW]:
                    st.warning(w)
                
                if len(warnings) > StreamlitConfig.MAX_WARNINGS_SHOW:
                    st.info(f"...and {len(warnings) - StreamlitConfig.MAX_WARNINGS_SHOW} more")
        
        return df, warnings
        
    except Exception as e:
        logger.log_exception(e, "load_and_process_data")
        st.error(f"Error loading data: {str(e)}")
        
        if st.session_state.show_debug:
            with st.expander("Debug Trace"):
                st.code(traceback.format_exc())
        
        return None, [str(e)]

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar configuration"""
    
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # 1. DATA UPLOAD
        st.header("1️⃣ Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload Player CSV",
            type=['csv'],
            help="CSV with player projections",
            key="file_uploader"
        )
        
        if uploaded_file:
            if st.button("📥 Load Data", type="primary", use_container_width=True):
                with st.spinner("Loading and validating..."):
                    df, warnings = load_and_process_data(uploaded_file)
                    
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.processed_df = df
                        st.session_state.warnings = warnings
                        st.session_state.data_validated = True
                        st.session_state.optimization_complete = False
                        st.session_state.lineups = []
                        
                        # Run pool analysis
                        if OPTIMIZER_STATUS['advanced']:
                            try:
                                df_hash = hashlib.md5(
                                    pd.util.hash_pandas_object(df).values
                                ).hexdigest()
                                st.session_state.pool_analysis = analyze_player_pool(df_hash, df)
                            except:
                                pass
                        
                        st.success(f"✅ Loaded {len(df)} players")
                        st.rerun()
                    else:
                        st.error("❌ Load failed")
        
        st.divider()
        
        # 2. GAME SETTINGS
        st.header("2️⃣ Game Settings")
        
        st.session_state.game_total = st.number_input(
            "Game Total (O/U)",
            min_value=30.0,
            max_value=70.0,
            value=st.session_state.game_total,
            step=0.5,
            key="game_total_input"
        )
        
        st.session_state.spread = st.number_input(
            "Spread",
            min_value=-20.0,
            max_value=20.0,
            value=st.session_state.spread,
            step=0.5,
            key="spread_input"
        )
        
        st.divider()
        
        # 3. CONTEST SETTINGS
        st.header("3️⃣ Contest Settings")
        
        st.session_state.contest_type = st.selectbox(
            "Contest Type",
            options=list(CONTEST_TYPE_MAPPING.keys()),
            index=list(CONTEST_TYPE_MAPPING.keys()).index(
                st.session_state.contest_type
            ),
            key="contest_type_select"
        )
        
        st.session_state.num_lineups = st.slider(
            "Number of Lineups",
            min_value=1,
            max_value=150,
            value=st.session_state.num_lineups,
            key="num_lineups_slider"
        )
        
        st.session_state.optimization_mode = st.selectbox(
            "Optimization Mode",
            options=['balanced', 'ceiling', 'floor', 'sharpe'],
            index=0,
            help="balanced: All-around | ceiling: High upside | floor: Consistent | sharpe: Risk-adjusted",
            key="opt_mode_select"
        )
        
        st.divider()
        
        # 4. AI SETTINGS
        st.header("4️⃣ AI Settings")
        
        if not OPTIMIZER_STATUS['ai'] or not OPTIMIZER_STATUS['master']:
            st.info("ℹ️ AI features unavailable (module not loaded)")
            st.session_state.use_ai = False
        else:
            st.session_state.use_ai = st.checkbox(
                "Enable AI Analysis",
                value=st.session_state.use_ai,
                help="Use 5 AI strategists for insights (requires API key)",
                key="use_ai_checkbox"
            )
            
            if st.session_state.use_ai:
                st.session_state.api_key = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    value=st.session_state.api_key,
                    key="api_key_input"
                )
                
                st.session_state.ai_enforcement = st.select_slider(
                    "AI Enforcement",
                    options=['advisory', 'moderate', 'strong', 'mandatory'],
                    value=st.session_state.ai_enforcement,
                    help="How strictly to apply AI recommendations",
                    key="ai_enforcement_slider"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.use_bayesian_synthesis = st.checkbox(
                        "Bayesian Synthesis",
                        value=st.session_state.use_bayesian_synthesis,
                        key="bayesian_checkbox"
                    )
                with col2:
                    st.session_state.use_iterative_refinement = st.checkbox(
                        "Iterative Refinement",
                        value=st.session_state.use_iterative_refinement,
                        key="refinement_checkbox"
                    )
                
                if not st.session_state.api_key:
                    st.warning("⚠️ API key required for AI features")
        
        st.divider()
        
        # 5. ADVANCED OPTIONS
        with st.expander("🔧 Advanced Options"):
            st.session_state.min_salary_pct = st.slider(
                "Min Salary %",
                min_value=50,
                max_value=100,
                value=st.session_state.min_salary_pct,
                step=5,
                key="min_salary_slider"
            )
            st.caption(
                f"${int(DraftKingsRules.SALARY_CAP * st.session_state.min_salary_pct / 100):,}"
            )
            
            if OPTIMIZER_STATUS['advanced']:
                st.session_state.use_monte_carlo = st.checkbox(
                    "Monte Carlo Simulation",
                    value=st.session_state.use_monte_carlo,
                    key="monte_carlo_checkbox"
                )
                
                if st.session_state.use_monte_carlo:
                    st.session_state.num_simulations = st.select_slider(
                        "Simulations",
                        options=[1000, 5000, 10000, 20000],
                        value=st.session_state.num_simulations,
                        key="simulations_slider"
                    )
            
            if OPTIMIZER_STATUS['algorithms']:
                st.session_state.use_ensemble = st.checkbox(
                    "Force Ensemble Mode",
                    value=st.session_state.use_ensemble,
                    help="Run all algorithms in parallel",
                    key="ensemble_checkbox"
                )
            
            st.session_state.diversity_threshold = st.slider(
                "Diversity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.diversity_threshold,
                step=0.05,
                help="Higher = more diverse lineups",
                key="diversity_slider"
            )
            
            st.session_state.show_debug = st.checkbox(
                "Debug Mode",
                value=st.session_state.show_debug,
                key="debug_checkbox"
            )

# ============================================================================
# DATA OVERVIEW
# ============================================================================

def render_data_overview():
    """Render data overview tab"""
    
    if st.session_state.processed_df is None:
        st.info("📤 Upload a CSV file to begin")
        
        # Show optimizer status
        with st.expander("Optimizer Status", expanded=True):
            st.write("**Component Availability:**")
            
            status_icons = {True: "✅", False: "❌"}
            
            st.write(f"{status_icons[OPTIMIZER_STATUS['core']]} Core Components")
            st.write(f"{status_icons[OPTIMIZER_STATUS['algorithms']]} Optimization Algorithms")
            st.write(f"{status_icons[OPTIMIZER_STATUS['ai']]} AI Strategists")
            st.write(f"{status_icons[OPTIMIZER_STATUS['advanced']]} Advanced Features")
            st.write(f"{status_icons[OPTIMIZER_STATUS['master']]} Master Optimizer")
            
            if not all(OPTIMIZER_STATUS.values()):
                st.warning("Some features unavailable - check console for import errors")
        
        return
    
    df = st.session_state.processed_df
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Players", len(df))
    
    with col2:
        st.metric("Teams", df['Team'].nunique())
    
    with col3:
        st.metric("Avg Salary", f"${df['Salary'].mean():,.0f}")
    
    with col4:
        st.metric("Avg Projection", f"{df['Projected_Points'].mean():.1f}")
    
    # Charts
    st.subheader("Pool Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**By Position**")
        st.bar_chart(df['Position'].value_counts())
    
    with col2:
        st.write("**By Team**")
        st.bar_chart(df['Team'].value_counts())
    
    # Top players
    st.subheader("Top 10 Players by Projection")
    
    top10 = df.nlargest(10, 'Projected_Points')[
        ['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 'Ownership']
    ].copy()
    
    st.dataframe(top10, use_container_width=True, hide_index=True)
    
    # Diagnostics
    render_constraint_diagnostics()
    
    # Pool analysis
    if st.session_state.pool_analysis and 'error' not in st.session_state.pool_analysis:
        with st.expander("📈 Pool Analysis", expanded=False):
            analysis = st.session_state.pool_analysis
            
            st.write(f"**Pool Quality:** {analysis.get('pool_quality', 'Unknown').upper()}")
            
            recs = analysis.get('recommendations', {})
            if recs:
                st.write("**Recommendations:**")
                st.write(f"- Algorithm: {recs.get('recommended_algorithm', 'N/A')}")
                st.write(f"- Randomness: {recs.get('suggested_randomness', 0):.2f}")
                st.write(f"- Max Lineups: {recs.get('max_reasonable_lineups', 0)}")

# ============================================================================
# OPTIMIZATION
# ============================================================================

def render_optimization():
    """Render optimization tab"""
    
    if st.session_state.processed_df is None:
        st.warning("⚠️ Load data first")
        return
    
    # Prevent double-click
    if st.session_state.optimization_running:
        st.warning("⏳ Optimization in progress...")
        return
    
    # Settings summary
    with st.expander("Current Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Game:**")
            st.write(f"- Total: {st.session_state.game_total}")
            st.write(f"- Spread: {st.session_state.spread}")
            st.write(f"- Contest: {st.session_state.contest_type}")
        
        with col2:
            st.write("**Optimization:**")
            st.write(f"- Lineups: {st.session_state.num_lineups}")
            st.write(f"- Mode: {st.session_state.optimization_mode}")
            st.write(f"- AI: {'Enabled' if st.session_state.use_ai else 'Disabled'}")
            st.write(f"- Monte Carlo: {'Enabled' if st.session_state.use_monte_carlo else 'Disabled'}")
    
    # Optimize button
    button_disabled = st.session_state.optimization_running
    
    if st.button(
        "🚀 Generate Lineups",
        type="primary",
        use_container_width=True,
        disabled=button_disabled
    ):
        run_optimization()

def run_optimization():
    """Execute optimization with full error handling and thread safety"""
    
    # Prevent concurrent runs
    if st.session_state.optimization_running:
        st.warning("Optimization already running")
        return
    
    st.session_state.optimization_running = True
    
    snapshot = StateSnapshot()
    snapshot.capture()
    
    logger = get_logger()
    perf = get_performance_monitor()
    
    # Progress tracking
    progress_container = st.empty()
    status_container = st.empty()
    
    progress = None
    status = None
    
    try:
        # Setup progress
        with progress_container.container():
            progress = st.progress(0)
        with status_container.container():
            status = st.empty()
        
        # Pre-flight check
        status.text("🔍 Running pre-flight check...")
        
        diag = run_constraint_check(
            st.session_state.processed_df,
            st.session_state.min_salary_pct
        )
        
        if not diag['is_feasible']:
            st.error(f"❌ Pre-flight Failed: {diag['error']}")
            
            if diag['suggestions']:
                st.warning("**Suggestions:**")
                for s in diag['suggestions']:
                    st.write(f"- {s}")
            
            return
        
        st.success("✅ Pre-flight passed")
        progress.progress(10)
        
        # Snapshot settings (avoid race conditions with session state)
        settings = {
            'df': st.session_state.processed_df.copy(),
            'num_lineups': st.session_state.num_lineups,
            'game_total': st.session_state.game_total,
            'spread': st.session_state.spread,
            'contest_type': st.session_state.contest_type,
            'optimization_mode': st.session_state.optimization_mode,
            'min_salary_pct': st.session_state.min_salary_pct,
            'use_ai': st.session_state.use_ai and bool(st.session_state.api_key),
            'api_key': st.session_state.api_key,
            'ai_enforcement': st.session_state.ai_enforcement,
            'use_monte_carlo': st.session_state.use_monte_carlo,
            'num_simulations': st.session_state.num_simulations,
            'use_ensemble': st.session_state.use_ensemble,
            'use_iterative_refinement': st.session_state.use_iterative_refinement,
            'use_bayesian_synthesis': st.session_state.use_bayesian_synthesis,
        }
        
        progress.progress(20)
        status.text("⚙️ Initializing optimizer...")
        
        # Result container (thread-safe)
        result = {'lineups': None, 'error': None, 'summary': None}
        result_lock = threading.Lock()
        progress_lock = threading.Lock()
        
        # Progress callback (thread-safe)
        def update_progress(message: str, pct: float):
            """Thread-safe progress update"""
            with progress_lock:
                try:
                    if progress:
                        progress.progress(min(int(pct * 100), 99))
                    if status:
                        status.text(message)
                except:
                    pass  # Streamlit widgets may fail from thread
        
        # Optimization thread
        def opt_thread():
            try:
                # Use MasterOptimizer if available, otherwise fallback
                if OPTIMIZER_STATUS['master']:
                    # Use the complete optimize_showdown function
                    lineups, _ = optimize_showdown(
                        csv_path_or_df=settings['df'],
                        num_lineups=settings['num_lineups'],
                        game_total=settings['game_total'],
                        spread=settings['spread'],
                        contest_type=settings['contest_type'],
                        api_key=settings['api_key'] if settings['use_ai'] else None,
                        use_ai=settings['use_ai'],
                        optimization_mode=settings['optimization_mode'],
                        ai_enforcement=settings['ai_enforcement'],
                        use_ensemble=settings['use_ensemble'],
                        use_iterative_refinement=settings['use_iterative_refinement'],
                        use_bayesian_synthesis=settings['use_bayesian_synthesis'],
                        progress_callback=update_progress
                    )
                    
                elif OPTIMIZER_STATUS['algorithms']:
                    # Fallback to genetic algorithm
                    update_progress("Using Genetic Algorithm fallback", 0.3)
                    
                    # Create game info
                    processor = OptimizedDataProcessor()
                    game_info = processor.infer_game_info(
                        settings['df'],
                        settings['game_total'],
                        settings['spread']
                    )
                    
                    # Create constraints
                    constraints = LineupConstraints(
                        min_salary=int(DraftKingsRules.SALARY_CAP * settings['min_salary_pct'] / 100),
                        max_salary=DraftKingsRules.SALARY_CAP
                    )
                    
                    # Create Monte Carlo engine
                    mc_engine = None
                    if settings['use_monte_carlo'] and OPTIMIZER_STATUS['advanced']:
                        mc_engine = MonteCarloSimulationEngine(
                            settings['df'],
                            game_info,
                            settings['num_simulations']
                        )
                    
                    # Create optimizer
                    genetic_config = OptimizerConfig.get_genetic_config(
                        len(settings['df']),
                        settings['num_lineups'],
                        time_budget_seconds=180
                    )
                    
                    optimizer = GeneticAlgorithmOptimizer(
                        df=settings['df'],
                        game_info=game_info,
                        constraints=constraints,
                        config=genetic_config,
                        mc_engine=mc_engine
                    )
                    
                    # Generate lineups
                    def ga_progress(gen, total, fitness):
                        pct = 0.4 + (gen / total) * 0.4
                        update_progress(f"GA Generation {gen}/{total}", pct)
                    
                    ga_lineups = optimizer.generate_lineups(
                        settings['num_lineups'],
                        progress_callback=ga_progress
                    )
                    
                    # Convert to dicts
                    lineups = []
                    for i, lineup in enumerate(ga_lineups, 1):
                        metrics = calculate_lineup_metrics(
                            lineup.captain,
                            lineup.flex,
                            settings['df'],
                            mc_engine
                        )
                        metrics['Lineup'] = i
                        lineups.append(metrics)
                    
                else:
                    raise Exception("No optimization algorithms available")
                
                # Store result
                with result_lock:
                    result['lineups'] = lineups
                    
            except Exception as e:
                with result_lock:
                    result['error'] = e
        
        # Start thread
        thread = threading.Thread(target=opt_thread, daemon=True)
        thread.start()
        
        # Wait with timeout
        perf.start_timer('optimization')
        start_time = time.time()
        timeout = StreamlitConfig.OPTIMIZATION_TIMEOUT_SEC
        
        elapsed = 0
        last_update = 0
        
        while thread.is_alive() and elapsed < timeout:
            thread.join(timeout=StreamlitConfig.THREAD_JOIN_SEC)
            elapsed = time.time() - start_time
            
            # Update progress periodically
            if elapsed - last_update >= StreamlitConfig.PROGRESS_UPDATE_INTERVAL_SEC:
                pct = min(90, 20 + int((elapsed / timeout) * 70))
                
                with progress_lock:
                    try:
                        if progress and elapsed > StreamlitConfig.INITIAL_PROGRESS_THRESHOLD_SEC:
                            progress.progress(pct)
                        if status:
                            status.text(f"🔄 Optimizing... ({int(elapsed)}s)")
                    except:
                        pass
                
                last_update = elapsed
        
        # Check timeout
        if thread.is_alive():
            st.error(f"⏱️ Timeout after {timeout}s")
            st.info("**Try:** Lower min salary %, reduce lineups, or disable AI/Monte Carlo")
            return
        
        # Check error
        with result_lock:
            if result['error']:
                raise result['error']
            lineups = result['lineups']
        
        elapsed_time = perf.stop_timer('optimization')
        
        # Update UI
        progress.progress(100)
        status.text("✅ Complete!")
        time.sleep(0.5)
        
        if not lineups:
            st.warning("⚠️ No lineups generated")
            st.info("Check constraints and try adjusting settings")
            return
        
        # Store results
        st.session_state.lineups = lineups
        st.session_state.optimization_complete = True
        st.session_state.last_optimization_time = elapsed_time
        
        # Success message
        st.success(f"✅ Generated {len(lineups)} lineups in {elapsed_time:.1f}s")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            projections = [l.get('Projected', 0) for l in lineups if 'Projected' in l]
            if projections:
                avg_proj = np.mean(projections)
                st.metric("Avg Projection", f"{avg_proj:.2f}")
        
        with col2:
            salaries = [l.get('Total_Salary', 0) for l in lineups if 'Total_Salary' in l]
            if salaries:
                avg_sal = np.mean(salaries)
                st.metric("Avg Salary", f"${avg_sal:,.0f}")
        
        with col3:
            ceilings = [l.get('Ceiling_90th', 0) for l in lineups if 'Ceiling_90th' in l]
            if ceilings:
                avg_ceil = np.mean(ceilings)
                st.metric("Avg Ceiling", f"{avg_ceil:.2f}")
        
        with col4:
            if projections:
                best = max(projections)
                st.metric("Best", f"{best:.2f}")
        
        # Trigger rerun to show results
        time.sleep(0.5)
        st.rerun()
        
    except Exception as e:
        # Restore state
        snapshot.restore()
        
        logger.log_exception(e, "run_optimization")
        
        # Enhanced error messaging
        error_type = type(e).__name__
        error_msg = str(e).lower()
        
        if "salary" in error_msg or "constraint" in error_msg:
            st.error(f"❌ Constraint Error: {str(e)}")
            st.info("💡 **Try:** Lower minimum salary % in Advanced Options")
            st.info("💡 **Or:** Run Pre-Flight Check in Data Overview tab")
            
        elif "infeasible" in error_msg:
            st.error(f"❌ Infeasible Constraints: {str(e)}")
            st.info("💡 **Try:** Check Pre-Flight diagnostics in Data Overview")
            
        elif "timeout" in error_msg:
            st.error(f"❌ Timeout: {str(e)}")
            st.info("💡 **Try:** Reduce lineups or disable Monte Carlo simulation")
            
        elif "api" in error_msg or "anthropic" in error_msg:
            st.error(f"❌ API Error: {str(e)}")
            st.info("💡 **Try:** Check API key or disable AI features")
            
        elif "import" in error_msg or "module" in error_msg:
            st.error(f"❌ Module Error: {str(e)}")
            st.info("💡 **Try:** Check that nfl_dfs_optimizer.py is in the same directory")
            
        else:
            st.error(f"❌ {error_type}: {str(e)}")
            st.info("💡 Enable Debug Mode in sidebar for detailed trace")
        
        if st.session_state.show_debug:
            with st.expander("🐛 Debug Trace", expanded=True):
                st.code(traceback.format_exc())
                
    finally:
        # Always cleanup
        st.session_state.optimization_running = False
        
        try:
            if progress:
                progress_container.empty()
            if status:
                status_container.empty()
        except:
            pass

# ============================================================================
# RESULTS
# ============================================================================

def render_results():
    """Render results tab with safe metric access"""
    
    if not st.session_state.optimization_complete:
        st.info("💡 Complete optimization to view results")
        return
    
    lineups = st.session_state.lineups
    
    if not lineups:
        st.warning("⚠️ No lineups were generated")
        st.info("Try adjusting settings and re-running optimization")
        return
    
    df = st.session_state.processed_df
    
    # Summary with safe access
    st.subheader("Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Safe metric extraction
    projections = [l.get('Projected', 0) for l in lineups if 'Projected' in l]
    salaries = [l.get('Total_Salary', 0) for l in lineups if 'Total_Salary' in l]
    ceilings = [l.get('Ceiling_90th', 0) for l in lineups if 'Ceiling_90th' in l]
    floors = [l.get('Floor_10th', 0) for l in lineups if 'Floor_10th' in l]
    
    with col1:
        st.metric("Total Lineups", len(lineups))
    
    with col2:
        if projections:
            best = max(projections)
            st.metric("Best Projection", f"{best:.2f}")
        else:
            st.metric("Best Projection", "N/A")
    
    with col3:
        if projections:
            avg = np.mean(projections)
            st.metric("Avg Projection", f"{avg:.2f}")
        else:
            st.metric("Avg Projection", "N/A")
    
    with col4:
        if salaries:
            avg_sal = np.mean(salaries)
            st.metric("Avg Salary", f"${avg_sal:,.0f}")
        else:
            st.metric("Avg Salary", "N/A")
    
    # Display options
    st.subheader("Lineup Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_count = st.selectbox(
            "Show:",
            options=[5, 10, 20, 50, "All"],
            index=1,
            key="show_count_select"
        )
    
    with col2:
        # Available sort options
        sort_options = ['Projected']
        if ceilings:
            sort_options.append('Ceiling_90th')
        if salaries:
            sort_options.append('Total_Salary')
        
        sort_by = st.selectbox(
            "Sort by:",
            options=sort_options,
            index=0,
            key="sort_by_select"
        )
    
    # Filter and sort
    display_count = len(lineups) if show_count == "All" else int(show_count)
    
    try:
        sorted_lineups = sorted(
            lineups,
            key=lambda x: x.get(sort_by, 0),
            reverse=True
        )[:display_count]
    except:
        sorted_lineups = lineups[:display_count]
    
    # Display lineups
    for i, lineup in enumerate(sorted_lineups, 1):
        with st.container():
            st.markdown(f"### Lineup {lineup.get('Lineup', i)}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                captain = lineup.get('Captain', 'Unknown')
                st.write(f"**Captain:** {captain}")
                st.write("**FLEX:**")
                
                flex = lineup.get('FLEX', [])
                if isinstance(flex, str):
                    flex = [p.strip() for p in flex.split(',') if p.strip()]
                
                for player in flex:
                    player_info = df[df['Player'] == player]
                    if not player_info.empty:
                        row = player_info.iloc[0]
                        st.write(
                            f"  - {player} ({row['Position']}, {row['Team']}) - "
                            f"${row['Salary']:,.0f} | {row['Projected_Points']:.1f} pts"
                        )
                    else:
                        st.write(f"  - {player}")
            
            with col2:
                if 'Projected' in lineup:
                    st.metric("Projected", f"{lineup['Projected']:.2f}")
                
                if 'Total_Salary' in lineup:
                    st.metric("Salary", f"${lineup['Total_Salary']:,.0f}")
                
                if 'Ceiling_90th' in lineup:
                    st.metric("Ceiling", f"{lineup['Ceiling_90th']:.2f}")
                
                if 'Floor_10th' in lineup:
                    st.metric("Floor", f"{lineup['Floor_10th']:.2f}")
            
            st.divider()

# ============================================================================
# EXPORT & ANALYTICS
# ============================================================================

def render_export_analytics():
    """Render export and analytics"""
    
    if not st.session_state.optimization_complete:
        st.info("💡 Complete optimization first")
        return
    
    lineups = st.session_state.lineups
    
    if not lineups:
        st.warning("⚠️ No lineups to export")
        return
    
    # EXPORT SECTION
    st.header("📥 Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        format_choice = st.selectbox(
            "Format",
            options=['Standard', 'DraftKings', 'Detailed'],
            key="format_select"
        )
    
    # Generate CSV
    try:
        if format_choice == 'DraftKings':
            # DK format
            rows = []
            for lineup in lineups:
                flex = lineup.get('FLEX', [])
                if isinstance(flex, str):
                    flex = [p.strip() for p in flex.split(',') if p.strip()]
                
                row = {
                    'CPT': lineup.get('Captain', ''),
                    'FLEX': flex[0] if len(flex) > 0 else '',
                    'FLEX_2': flex[1] if len(flex) > 1 else '',
                    'FLEX_3': flex[2] if len(flex) > 2 else '',
                    'FLEX_4': flex[3] if len(flex) > 3 else '',
                    'FLEX_5': flex[4] if len(flex) > 4 else '',
                }
                rows.append(row)
        else:
            # Standard format
            rows = []
            for i, lineup in enumerate(lineups, 1):
                flex = lineup.get('FLEX', [])
                if isinstance(flex, str):
                    flex = [p.strip() for p in flex.split(',') if p.strip()]
                
                row = {
                    'Lineup': lineup.get('Lineup', i),
                    'Captain': lineup.get('Captain', ''),
                    'FLEX_1': flex[0] if len(flex) > 0 else '',
                    'FLEX_2': flex[1] if len(flex) > 1 else '',
                    'FLEX_3': flex[2] if len(flex) > 2 else '',
                    'FLEX_4': flex[3] if len(flex) > 3 else '',
                    'FLEX_5': flex[4] if len(flex) > 4 else '',
                    'Salary': lineup.get('Total_Salary', 0),
                    'Projected': lineup.get('Projected', 0),
                }
                
                if format_choice == 'Detailed':
                    row.update({
                        'Ownership': lineup.get('Avg_Ownership', 0),
                        'Ceiling': lineup.get('Ceiling_90th', 0),
                        'Floor': lineup.get('Floor_10th', 0),
                        'Sharpe': lineup.get('Sharpe_Ratio', 0),
                    })
                
                rows.append(row)
        
        export_df = pd.DataFrame(rows)
        csv = export_df.to_csv(index=False)
        
        # Preview
        with st.expander("👁️ Preview"):
            st.dataframe(export_df.head(10), use_container_width=True)
        
        # Download
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"lineups_{format_choice.lower()}_{timestamp}.csv"
        
        st.download_button(
            label=f"📥 Download {format_choice} CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
            key="download_button"
        )
        
    except Exception as e:
        st.error(f"❌ Export failed: {e}")
        
        if st.session_state.show_debug:
            with st.expander("Debug"):
                st.code(traceback.format_exc())
    
    st.divider()
    
    # ANALYTICS SECTION
    st.header("📊 Analytics")
    
    # Player exposure
    st.subheader("Player Exposure")
    
    exposure = defaultdict(int)
    for lineup in lineups:
        captain = lineup.get('Captain', '')
        flex = lineup.get('FLEX', [])
        if isinstance(flex, str):
            flex = [p.strip() for p in flex.split(',') if p.strip()]
        
        for player in [captain] + flex:
            if player:
                exposure[player] += 1
    
    if exposure:
        exposure_pct = {
            p: (c / len(lineups)) * 100
            for p, c in exposure.items()
        }
        
        top_exposure = sorted(
            exposure_pct.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]
        
        exp_df = pd.DataFrame(top_exposure, columns=['Player', 'Exposure %'])
        
        st.bar_chart(exp_df.set_index('Player'))
    
    # Captain usage
    st.subheader("Captain Usage")
    
    captain_counts = defaultdict(int)
    for lineup in lineups:
        captain = lineup.get('Captain', '')
        if captain:
            captain_counts[captain] += 1
    
    if captain_counts:
        captain_pct = {
            c: (count / len(lineups)) * 100
            for c, count in captain_counts.items()
        }
        
        capt_df = pd.DataFrame(
            list(captain_pct.items()),
            columns=['Captain', 'Usage %']
        ).sort_values('Usage %', ascending=False)
        
        st.bar_chart(capt_df.set_index('Captain'))
    
    # Ownership distribution
    st.subheader("Ownership Distribution")
    
    ownerships = [l.get('Avg_Ownership', 0) for l in lineups if 'Avg_Ownership' in l]
    if ownerships:
        own_df = pd.DataFrame({'Avg Ownership %': ownerships})
        st.line_chart(own_df)
        
        st.write(f"**Average:** {np.mean(ownerships):.1f}%")
        st.write(f"**Range:** {np.min(ownerships):.1f}% - {np.max(ownerships):.1f}%")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    
    # Apply styling
    apply_css()
    
    # Header
    st.title("🏈 NFL DFS AI-Powered Optimizer")
    st.markdown("*Advanced DraftKings Showdown optimization with AI and Monte Carlo simulation*")
    
    version_info = "Version 5.0.1 - Production Ready"
    if OPTIMIZER_STATUS['master']:
        version_info += " | UPDATE 2 Compatible"
    
    st.caption(version_info)
    
    # Status indicator
    if not all(OPTIMIZER_STATUS.values()):
        with st.expander("⚠️ Component Status", expanded=False):
            for component, available in OPTIMIZER_STATUS.items():
                status = "✅" if available else "❌"
                st.write(f"{status} {component.title()}")
    
    # Sidebar
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview",
        "🚀 Optimization",
        "📈 Results",
        "💾 Export & Analytics"
    ])
    
    with tab1:
        render_data_overview()
    
    with tab2:
        render_optimization()
    
    with tab3:
        render_results()
    
    with tab4:
        render_export_analytics()
    
    # Footer
    st.divider()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.caption("NFL DFS Optimizer v5.0 | Built with Claude AI")
    
    with col2:
        if st.session_state.last_optimization_time:
            st.caption(f"⏱️ Last: {st.session_state.last_optimization_time:.1f}s")
    
    with col3:
        if st.session_state.optimization_complete:
            st.caption(f"✅ {len(st.session_state.lineups)} lineups")

if __name__ == "__main__":
    main()
