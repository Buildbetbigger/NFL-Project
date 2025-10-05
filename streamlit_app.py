"""
NFL DFS AI-Driven Optimizer - Streamlit Application
Version: 3.0.0 - Fully Refactored

FIXES APPLIED:
- FIX #17: State recovery on errors
- FIX #10: CSV encoding detection
- FIX #11: Enhanced error messages
- Better validation feedback
- Improved UI/UX
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import traceback
import io

# Import from the unified optimizer
from nfl_dfs_optimizer import (
    # Core functions
    optimize_showdown,
    safe_load_csv,
    validate_and_normalize_dataframe,
    validate_lineup_with_context,
    format_lineup_for_export,
    validate_export_format,
    calculate_lineup_similarity,
    ensure_lineup_diversity,
    
    # Classes
    OptimizedDataProcessor,
    MasterOptimizer,
    ValidationResult,
    GlobalLogger,
    PerformanceMonitor,
    
    # Enums
    ExportFormat,
    ValidationLevel,
    OptimizationMode,
    
    # Constants
    OptimizerConfig,
    DraftKingsRules,
    CONTEST_TYPE_MAPPING,
    FIELD_SIZE_CONFIGS,
    
    # Utilities
    get_logger,
    get_performance_monitor,
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NFL DFS Optimizer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FIX #17: STATE SNAPSHOT FOR ERROR RECOVERY
# ============================================================================

class StreamlitStateSnapshot:
    """
    FIX #17: Preserve Streamlit state across errors
    """
    
    def __init__(self):
        self.snapshot: Dict[str, Any] = {}
    
    def capture(self, keys: Optional[List[str]] = None) -> None:
        """Capture current session state"""
        if keys is None:
            # Capture all non-internal state
            keys = [k for k in st.session_state.keys() if not k.startswith('_')]
        
        self.snapshot = {
            key: st.session_state.get(key)
            for key in keys
        }
    
    def restore(self) -> None:
        """Restore captured state"""
        for key, value in self.snapshot.items():
            st.session_state[key] = value
    
    def preserve_on_error(self):
        """Context manager to auto-restore on error"""
        from contextlib import contextmanager
        
        @contextmanager
        def _context():
            self.capture()
            try:
                yield
            except Exception as e:
                self.restore()
                raise
        
        return _context()


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    
    # Core data
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    
    if 'lineups' not in st.session_state:
        st.session_state.lineups = []
    
    if 'game_info' not in st.session_state:
        st.session_state.game_info = {}
    
    # Settings
    if 'num_lineups' not in st.session_state:
        st.session_state.num_lineups = 20
    
    if 'game_total' not in st.session_state:
        st.session_state.game_total = 50.0
    
    if 'spread' not in st.session_state:
        st.session_state.spread = 0.0
    
    if 'contest_type' not in st.session_state:
        st.session_state.contest_type = 'Large GPP (1000+)'
    
    if 'optimization_mode' not in st.session_state:
        st.session_state.optimization_mode = 'balanced'
    
    if 'use_ai' not in st.session_state:
        st.session_state.use_ai = False
    
    if 'ai_enforcement' not in st.session_state:
        st.session_state.ai_enforcement = 'Moderate'
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    
    # UI state
    if 'show_advanced' not in st.session_state:
        st.session_state.show_advanced = False
    
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False
    
    # Processing flags
    if 'data_validated' not in st.session_state:
        st.session_state.data_validated = False
    
    if 'optimization_complete' not in st.session_state:
        st.session_state.optimization_complete = False
    
    # Warnings and errors
    if 'warnings' not in st.session_state:
        st.session_state.warnings = []
    
    if 'errors' not in st.session_state:
        st.session_state.errors = []


# ============================================================================
# STYLING
# ============================================================================

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
        <style>
        .main {
            padding-top: 2rem;
        }
        
        .stAlert {
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        
        .success-box {
            padding: 1rem;
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            margin: 1rem 0;
        }
        
        .warning-box {
            padding: 1rem;
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
        }
        
        .error-box {
            padding: 1rem;
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            margin: 1rem 0;
        }
        
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
        }
        
        .lineup-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
            margin-bottom: 1rem;
        }
        
        h1 {
            color: #1f4788;
        }
        
        h2 {
            color: #2c5aa0;
            margin-top: 2rem;
        }
        
        h3 {
            color: #4a90e2;
        }
        
        .stButton>button {
            width: 100%;
        }
        
        .stProgress > div > div > div {
            background-color: #4a90e2;
        }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

def load_and_validate_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    FIX #10: Load and validate uploaded CSV with encoding detection
    """
    warnings = []
    
    try:
        # FIX #10: Safe CSV loading with encoding detection
        logger = get_logger()
        df_raw, encoding_info = safe_load_csv(uploaded_file, logger)
        
        if df_raw is None:
            st.error(f"❌ Failed to load CSV: {encoding_info}")
            return None, [encoding_info]
        
        if encoding_info != 'utf-8':
            warnings.append(f"ℹ️ File loaded with {encoding_info} encoding")
        
        # Show raw data preview
        with st.expander("📋 Raw Data Preview", expanded=False):
            st.dataframe(df_raw.head(10))
            st.caption(f"Showing 10 of {len(df_raw)} rows")
        
        # Process and validate
        processor = OptimizedDataProcessor()
        df_processed, process_warnings = processor.process_dataframe(
            df_raw,
            ValidationLevel.MODERATE
        )
        
        warnings.extend(process_warnings)
        
        # Show processed data
        with st.expander("✅ Processed Data Preview", expanded=False):
            st.dataframe(df_processed.head(10))
            st.caption(f"Processed: {len(df_processed)} players")
        
        # Show warnings if any
        if warnings:
            with st.expander("⚠️ Processing Warnings", expanded=True):
                for warning in warnings:
                    st.warning(warning)
        
        return df_processed, warnings
        
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        st.error(f"❌ {error_msg}")
        
        if st.session_state.get('show_debug', False):
            st.code(traceback.format_exc())
        
        return None, [error_msg]


# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

def render_sidebar():
    """Render sidebar with configuration options"""
    
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # File Upload
        st.header("1️⃣ Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Player CSV",
            type=['csv'],
            help="Upload a CSV with player projections"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Load & Validate Data"):
                with st.spinner("Loading and validating data..."):
                    df, warnings = load_and_validate_data(uploaded_file)
                    
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.processed_df = df
                        st.session_state.warnings = warnings
                        st.session_state.data_validated = True
                        st.success(f"✅ Loaded {len(df)} players")
                    else:
                        st.session_state.data_validated = False
                        st.error("❌ Data validation failed")
        
        st.divider()
        
        # Game Settings
        st.header("2️⃣ Game Settings")
        
        game_total = st.number_input(
            "Game Total (O/U)",
            min_value=30.0,
            max_value=70.0,
            value=st.session_state.game_total,
            step=0.5,
            help="Expected total points in the game"
        )
        st.session_state.game_total = game_total
        
        spread = st.number_input(
            "Spread",
            min_value=-20.0,
            max_value=20.0,
            value=st.session_state.spread,
            step=0.5,
            help="Point spread (negative = favorite)"
        )
        st.session_state.spread = spread
        
        st.divider()
        
        # Contest Settings
        st.header("3️⃣ Contest Settings")
        
        contest_type = st.selectbox(
            "Contest Type",
            options=list(CONTEST_TYPE_MAPPING.keys()),
            index=list(CONTEST_TYPE_MAPPING.keys()).index(st.session_state.contest_type),
            help="Select your contest size"
        )
        st.session_state.contest_type = contest_type
        
        num_lineups = st.slider(
            "Number of Lineups",
            min_value=1,
            max_value=150,
            value=st.session_state.num_lineups,
            help="How many lineups to generate"
        )
        st.session_state.num_lineups = num_lineups
        
        optimization_mode = st.selectbox(
            "Optimization Mode",
            options=['balanced', 'ceiling', 'floor', 'boom_or_bust'],
            index=['balanced', 'ceiling', 'floor', 'boom_or_bust'].index(
                st.session_state.optimization_mode
            ),
            help="Strategy for lineup optimization"
        )
        st.session_state.optimization_mode = optimization_mode
        
        st.divider()
        
        # AI Settings
        st.header("4️⃣ AI Settings (Optional)")
        
        use_ai = st.checkbox(
            "Enable AI Analysis",
            value=st.session_state.use_ai,
            help="Use Claude AI for strategic recommendations"
        )
        st.session_state.use_ai = use_ai
        
        if use_ai:
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.api_key,
                help="Your Anthropic API key (starts with 'sk-ant-')"
            )
            st.session_state.api_key = api_key
            
            ai_enforcement = st.select_slider(
                "AI Enforcement Level",
                options=['Advisory', 'Moderate', 'Strong', 'Mandatory'],
                value=st.session_state.ai_enforcement,
                help="How strictly to enforce AI recommendations"
            )
            st.session_state.ai_enforcement = ai_enforcement
            
            if not api_key:
                st.warning("⚠️ API key required for AI features")
        
        st.divider()
        
        # Advanced Options
        if st.checkbox("🔧 Advanced Options", value=st.session_state.show_advanced):
            st.session_state.show_advanced = True
            
            st.subheader("Debug Options")
            show_debug = st.checkbox(
                "Show Debug Info",
                value=st.session_state.show_debug,
                help="Display detailed error traces"
            )
            st.session_state.show_debug = show_debug
        else:
            st.session_state.show_advanced = False


# ============================================================================
# MAIN CONTENT - DATA OVERVIEW
# ============================================================================

def render_data_overview():
    """Render data overview section"""
    
    st.header("📊 Data Overview")
    
    if st.session_state.processed_df is None:
        st.info("👆 Upload a CSV file in the sidebar to get started")
        return
    
    df = st.session_state.processed_df
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Players", len(df))
    
    with col2:
        teams = df['Team'].nunique()
        st.metric("Teams", teams)
    
    with col3:
        avg_salary = df['Salary'].mean()
        st.metric("Avg Salary", f"${avg_salary:,.0f}")
    
    with col4:
        avg_proj = df['Projected_Points'].mean()
        st.metric("Avg Projection", f"{avg_proj:.1f}")
    
    # Position breakdown
    st.subheader("Position Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pos_counts = df['Position'].value_counts()
        st.bar_chart(pos_counts)
    
    with col2:
        team_counts = df['Team'].value_counts()
        st.bar_chart(team_counts)
    
    # Top players
    st.subheader("Top 10 Players by Projection")
    
    top_players = df.nlargest(10, 'Projected_Points')[
        ['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 'Ownership']
    ].copy()
    
    top_players['Salary'] = top_players['Salary'].apply(lambda x: f"${x:,.0f}")
    top_players['Projected_Points'] = top_players['Projected_Points'].apply(lambda x: f"{x:.1f}")
    top_players['Ownership'] = top_players['Ownership'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(top_players, use_container_width=True, hide_index=True)


# ============================================================================
# MAIN CONTENT - OPTIMIZATION
# ============================================================================

def render_optimization_section():
    """Render optimization section"""
    
    st.header("🎯 Optimization")
    
    if st.session_state.processed_df is None:
        st.warning("⚠️ Please load and validate data first")
        return
    
    # Show current settings
    with st.expander("📋 Current Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Game Settings:**")
            st.write(f"- Total: {st.session_state.game_total}")
            st.write(f"- Spread: {st.session_state.spread}")
            st.write(f"- Contest: {st.session_state.contest_type}")
        
        with col2:
            st.write("**Optimization Settings:**")
            st.write(f"- Lineups: {st.session_state.num_lineups}")
            st.write(f"- Mode: {st.session_state.optimization_mode}")
            st.write(f"- AI Enabled: {st.session_state.use_ai}")
    
    # Optimization button
    if st.button("🚀 Generate Lineups", type="primary", use_container_width=True):
        run_optimization()


def run_optimization():
    """
    FIX #17: Run optimization with state preservation
    """
    snapshot = StreamlitStateSnapshot()
    
    try:
        with snapshot.preserve_on_error():
            # Validate prerequisites
            if st.session_state.use_ai and not st.session_state.api_key:
                st.error("❌ AI enabled but no API key provided")
                return
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Phase 1: Setup
            status_text.text("⚙️ Initializing optimizer...")
            progress_bar.progress(10)
            time.sleep(0.3)
            
            df = st.session_state.processed_df
            
            # Build game info
            processor = OptimizedDataProcessor()
            game_info = processor.infer_game_info(
                df,
                st.session_state.game_total,
                st.session_state.spread
            )
            st.session_state.game_info = game_info
            
            # Phase 2: Optimization
            status_text.text("🔄 Running optimization...")
            progress_bar.progress(30)
            
            start_time = time.time()
            
            try:
                lineups, _ = optimize_showdown(
                    csv_path_or_df=df,
                    num_lineups=st.session_state.num_lineups,
                    game_total=st.session_state.game_total,
                    spread=st.session_state.spread,
                    contest_type=st.session_state.contest_type,
                    api_key=st.session_state.api_key if st.session_state.use_ai else None,
                    use_ai=st.session_state.use_ai,
                    optimization_mode=st.session_state.optimization_mode,
                    ai_enforcement=st.session_state.ai_enforcement
                )
                
                progress_bar.progress(90)
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                
                # FIX #11: Enhanced error message
                st.error(f"❌ Optimization failed: {str(e)}")
                
                # Get suggestions from logger
                logger = get_logger()
                error_summary = logger.get_error_summary()
                
                if error_summary.get('recent_errors'):
                    recent = error_summary['recent_errors'][-1]
                    if recent.get('suggestions'):
                        st.info("💡 Suggestions:")
                        for suggestion in recent['suggestions'][:3]:
                            st.write(f"• {suggestion}")
                
                if st.session_state.show_debug:
                    with st.expander("🐛 Debug Info"):
                        st.code(traceback.format_exc())
                
                return
            
            elapsed_time = time.time() - start_time
            
            # Phase 3: Complete
            status_text.text("✅ Optimization complete!")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            
            # Store results
            st.session_state.lineups = lineups
            st.session_state.optimization_complete = True
            
            # Success message
            st.success(
                f"✅ Successfully generated {len(lineups)} lineups in {elapsed_time:.1f} seconds"
            )
            
            # Show summary
            if lineups:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_proj = np.mean([l.get('Projected', 0) for l in lineups])
                    st.metric("Avg Projection", f"{avg_proj:.2f}")
                
                with col2:
                    avg_salary = np.mean([l.get('Total_Salary', 0) for l in lineups])
                    st.metric("Avg Salary", f"${avg_salary:,.0f}")
                
                with col3:
                    if 'Ceiling_90th' in lineups[0]:
                        avg_ceiling = np.mean([l.get('Ceiling_90th', 0) for l in lineups])
                        st.metric("Avg Ceiling", f"{avg_ceiling:.2f}")
                    else:
                        st.metric("Lineups", len(lineups))
            
            # Rerun to show results
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        
        if st.session_state.show_debug:
            st.code(traceback.format_exc())
        
        st.info("⚠️ Your settings have been preserved - adjust and try again")


# ============================================================================
# MAIN CONTENT - RESULTS
# ============================================================================

def render_results_section():
    """Render optimization results"""
    
    if not st.session_state.optimization_complete or not st.session_state.lineups:
        return
    
    st.header("📈 Results")
    
    lineups = st.session_state.lineups
    df = st.session_state.processed_df
    
    # Results summary
    st.subheader("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    projections = [l.get('Projected', 0) for l in lineups]
    salaries = [l.get('Total_Salary', 0) for l in lineups]
    
    with col1:
        st.metric("Total Lineups", len(lineups))
    
    with col2:
        st.metric("Best Projection", f"{max(projections):.2f}")
    
    with col3:
        st.metric("Avg Projection", f"{np.mean(projections):.2f}")
    
    with col4:
        st.metric("Avg Salary", f"${np.mean(salaries):,.0f}")
    
    # Lineup display options
    st.subheader("Lineup Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_count = st.selectbox(
            "Show lineups:",
            options=[5, 10, 20, 50, "All"],
            index=1
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            options=['Projected', 'Ceiling_90th', 'Sharpe_Ratio', 'Total_Salary'],
            index=0
        )
    
    with col3:
        show_details = st.checkbox("Show simulation details", value=False)
    
    # Determine how many to show
    if show_count == "All":
        display_lineups = lineups
    else:
        display_lineups = lineups[:int(show_count)]
    
    # Sort lineups
    if sort_by in display_lineups[0]:
        display_lineups = sorted(
            display_lineups,
            key=lambda x: x.get(sort_by, 0),
            reverse=True
        )
    
    # Display lineups
    for i, lineup in enumerate(display_lineups, 1):
        with st.container():
            st.markdown(f"### Lineup {i}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Players
                st.write(f"**Captain:** {lineup.get('Captain', 'N/A')}")
                
                flex = lineup.get('FLEX', [])
                if isinstance(flex, str):
                    flex = [p.strip() for p in flex.split(',') if p.strip()]
                
                st.write("**FLEX:**")
                for player in flex:
                    # Get player info
                    player_info = df[df['Player'] == player]
                    if not player_info.empty:
                        pos = player_info.iloc[0]['Position']
                        salary = player_info.iloc[0]['Salary']
                        proj = player_info.iloc[0]['Projected_Points']
                        st.write(f"  • {player} ({pos}) - ${salary:,.0f} | {proj:.1f} pts")
                    else:
                        st.write(f"  • {player}")
            
            with col2:
                # Metrics
                st.metric("Projected", f"{lineup.get('Projected', 0):.2f}")
                st.metric("Salary", f"${lineup.get('Total_Salary', 0):,.0f}")
                st.metric("Ownership", f"{lineup.get('Total_Ownership', 0):.1f}%")
                
                if show_details and 'Ceiling_90th' in lineup:
                    st.metric("Ceiling (90th)", f"{lineup.get('Ceiling_90th', 0):.2f}")
                    st.metric("Floor (10th)", f"{lineup.get('Floor_10th', 0):.2f}")
                    st.metric("Sharpe Ratio", f"{lineup.get('Sharpe_Ratio', 0):.2f}")
                    st.metric("Win Prob", f"{lineup.get('Win_Probability', 0):.1%}")
            
            st.divider()


# ============================================================================
# MAIN CONTENT - EXPORT
# ============================================================================

def render_export_section():
    """Render export section"""
    
    if not st.session_state.optimization_complete or not st.session_state.lineups:
        return
    
    st.header("💾 Export")
    
    lineups = st.session_state.lineups
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Export Format",
            options=['Standard', 'DraftKings', 'Detailed'],
            help="Choose export format"
        )
    
    # Convert to enum
    format_map = {
        'Standard': ExportFormat.STANDARD,
        'DraftKings': ExportFormat.DRAFTKINGS,
        'Detailed': ExportFormat.DETAILED
    }
    selected_format = format_map[export_format]
    
    # Validate format
    is_valid, error_msg = validate_export_format(lineups, selected_format)
    
    if not is_valid:
        st.error(f"❌ {error_msg}")
        return
    
    # Generate export
    try:
        export_df = format_lineup_for_export(lineups, selected_format)
        
        # Preview
        with st.expander("📋 Export Preview", expanded=False):
            st.dataframe(export_df.head(10), use_container_width=True)
        
        # Download button
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label=f"⬇️ Download {export_format} CSV",
            data=csv,
            file_name=f"optimized_lineups_{export_format.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.success(f"✅ Ready to export {len(export_df)} lineups in {export_format} format")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")
        
        if st.session_state.show_debug:
            st.code(traceback.format_exc())


# ============================================================================
# MAIN CONTENT - ANALYTICS
# ============================================================================

def render_analytics_section():
    """Render analytics section"""
    
    if not st.session_state.optimization_complete or not st.session_state.lineups:
        return
    
    st.header("📊 Analytics")
    
    lineups = st.session_state.lineups
    df = st.session_state.processed_df
    
    # Player exposure
    st.subheader("Player Exposure")
    
    exposure = {}
    for lineup in lineups:
        captain = lineup.get('Captain', '')
        flex = lineup.get('FLEX', [])
        
        if isinstance(flex, str):
            flex = [p.strip() for p in flex.split(',') if p.strip()]
        
        all_players = [captain] + flex
        
        for player in all_players:
            exposure[player] = exposure.get(player, 0) + 1
    
    # Convert to percentage
    exposure_pct = {
        player: (count / len(lineups)) * 100
        for player, count in exposure.items()
    }
    
    # Sort and display top 15
    top_exposure = sorted(exposure_pct.items(), key=lambda x: x[1], reverse=True)[:15]
    
    exposure_df = pd.DataFrame(top_exposure, columns=['Player', 'Exposure %'])
    
    st.bar_chart(exposure_df.set_index('Player'))
    
    # Captain usage
    st.subheader("Captain Usage")
    
    captain_counts = {}
    for lineup in lineups:
        captain = lineup.get('Captain', '')
        captain_counts[captain] = captain_counts.get(captain, 0) + 1
    
    captain_pct = {
        captain: (count / len(lineups)) * 100
        for captain, count in captain_counts.items()
    }
    
    captain_df = pd.DataFrame(
        list(captain_pct.items()),
        columns=['Captain', 'Usage %']
    ).sort_values('Usage %', ascending=False)
    
    st.bar_chart(captain_df.set_index('Captain'))
    
    # Salary distribution
    st.subheader("Salary Distribution")
    
    salaries = [l.get('Total_Salary', 0) for l in lineups]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Min Salary", f"${min(salaries):,.0f}")
    
    with col2:
        st.metric("Max Salary", f"${max(salaries):,.0f}")
    
    with col3:
        st.metric("Avg Salary", f"${np.mean(salaries):,.0f}")
    
    # Histogram
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.hist(salaries, bins=20, edgecolor='black')
    ax.set_xlabel('Total Salary')
    ax.set_ylabel('Frequency')
    ax.set_title('Salary Distribution')
    
    st.pyplot(fig)
    
    # Team stacking
    st.subheader("Team Stacking Analysis")
    
    team_counts_dist = []
    for lineup in lineups:
        captain = lineup.get('Captain', '')
        flex = lineup.get('FLEX', [])
        
        if isinstance(flex, str):
            flex = [p.strip() for p in flex.split(',') if p.strip()]
        
        all_players = [captain] + flex
        
        teams = df[df['Player'].isin(all_players)]['Team'].value_counts()
        team_counts_dist.append(teams.max())
    
    stack_df = pd.DataFrame({
        'Max Players from Same Team': team_counts_dist
    })
    
    st.bar_chart(stack_df['Max Players from Same Team'].value_counts())


# ============================================================================
# FOOTER
# ============================================================================

def render_footer():
    """Render footer"""
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("NFL DFS Optimizer v3.0.0")
    
    with col2:
        st.caption("Built with Streamlit & Claude AI")
    
    with col3:
        if st.session_state.optimization_complete:
            perf_monitor = get_performance_monitor()
            stats = perf_monitor.get_operation_stats('lineup_generation')
            
            if stats:
                st.caption(f"Last optimization: {stats.get('avg_time', 0):.1f}s")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point"""
    
    # Initialize
    init_session_state()
    apply_custom_css()
    
    # Header
    st.title("🏈 NFL DFS AI-Driven Optimizer")
    st.markdown("*Optimize your DraftKings Showdown lineups with advanced AI and Monte Carlo simulation*")
    
    # Sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview",
        "🎯 Optimization",
        "📈 Results",
        "💾 Export & Analytics"
    ])
    
    with tab1:
        render_data_overview()
    
    with tab2:
        render_optimization_section()
    
    with tab3:
        render_results_section()
    
    with tab4:
        if st.session_state.optimization_complete:
            render_export_section()
            st.divider()
            render_analytics_section()
        else:
            st.info("💡 Complete optimization to view export and analytics")
    
    # Footer
    render_footer()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
