import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Tuple, Optional
import random
import requests
from openai import OpenAI
import anthropic
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import sys

# Set page configuration
st.set_page_config(
    page_title="NFL DFS Dual AI Optimizer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    .lineup-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #e9ecef;
    }
    .player-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #dee2e6;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .optimization-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'players_data' not in st.session_state:
    st.session_state.players_data = None
if 'optimized_lineup' not in st.session_state:
    st.session_state.optimized_lineup = None
if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []
if 'gpt_lineup' not in st.session_state:
    st.session_state.gpt_lineup = None
if 'claude_lineup' not in st.session_state:
    st.session_state.claude_lineup = None

# Title Section
st.markdown("""
    <div class="optimization-header">
        <h1>🏈 NFL DFS Dual AI Optimizer</h1>
        <p style="font-size: 18px; margin-top: 10px;">Leverage GPT-4 and Claude AI for Advanced Lineup Optimization</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # AI Model Selection
    st.subheader("🤖 AI Model Settings")
    ai_models = st.multiselect(
        "Select AI Models to Use",
        ["GPT-4 Turbo", "Claude 3.5 Sonnet"],
        default=["GPT-4 Turbo", "Claude 3.5 Sonnet"]
    )
    
    # API Keys
    with st.expander("🔑 API Configuration", expanded=False):
        openai_key = st.text_input("OpenAI API Key", type="password", value="")
        anthropic_key = st.text_input("Anthropic API Key", type="password", value="")
        
        if openai_key:
            st.session_state.openai_key = openai_key
        if anthropic_key:
            st.session_state.anthropic_key = anthropic_key
    
    # Contest Settings
    st.subheader("🎯 Contest Settings")
    contest_type = st.selectbox(
        "Contest Type",
        ["DraftKings", "FanDuel"],
        index=0
    )
    
    if contest_type == "DraftKings":
        salary_cap = 50000
        roster_spots = {
            "QB": 1, "RB": 2, "WR": 3, "TE": 1, 
            "FLEX": 1, "DST": 1
        }
    else:  # FanDuel
        salary_cap = 60000
        roster_spots = {
            "QB": 1, "RB": 2, "WR": 3, "TE": 1,
            "FLEX": 1, "DEF": 1
        }
    
    game_mode = st.selectbox(
        "Game Mode",
        ["Cash Games", "GPP Tournament", "Balanced"],
        index=0
    )
    
    # Optimization Parameters
    st.subheader("🎛️ Optimization Parameters")
    
    ownership_weight = st.slider(
        "Ownership Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3 if game_mode == "GPP Tournament" else 0.0,
        step=0.1,
        help="Higher values prioritize lower ownership players"
    )
    
    upside_weight = st.slider(
        "Upside Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.7 if game_mode == "GPP Tournament" else 0.3,
        step=0.1,
        help="Higher values prioritize ceiling over floor"
    )
    
    # Team Stacking
    st.subheader("🏟️ Team Stacking Rules")
    enable_stacking = st.checkbox("Enable Team Stacking", value=True)
    
    if enable_stacking:
        stack_type = st.selectbox(
            "Stack Type",
            ["QB + 2 Pass Catchers", "QB + 1 Pass Catcher", "Game Stack", "Custom"]
        )
        
        if stack_type == "Game Stack":
            bring_back = st.checkbox("Include Bring-back Player", value=True)
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings", expanded=False):
        max_players_per_team = st.number_input(
            "Max Players per Team",
            min_value=1,
            max_value=4,
            value=3
        )
        
        min_salary_usage = st.slider(
            "Minimum Salary Usage %",
            min_value=90,
            max_value=100,
            value=95,
            help="Minimum percentage of salary cap to use"
        )
        
        enable_late_swap = st.checkbox(
            "Enable Late Swap Suggestions",
            value=False,
            help="Identify pivot options for late games"
        )
        
        enable_captain_mode = st.checkbox(
            "Captain Mode (Showdown)",
            value=False,
            help="Optimize for single-game showdown slates"
        )

# Data Processing Functions
def load_and_process_data(file):
    """Load and process uploaded CSV data"""
    try:
        df = pd.read_csv(file)
        required_columns = ['Name', 'Position', 'Team', 'Salary', 'Projection']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Add additional calculated fields
        df['Value'] = df['Projection'] / (df['Salary'] / 1000)
        df['Ownership'] = np.random.uniform(5, 35, size=len(df))  # Simulated ownership
        df['Ceiling'] = df['Projection'] * 1.3
        df['Floor'] = df['Projection'] * 0.7
        df['Upside'] = df['Ceiling'] - df['Projection']
        
        return df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

# GPT Optimization Function
def optimize_with_gpt(players_df, constraints, api_key):
    """Use GPT-4 to generate optimized lineup"""
    if not api_key:
        st.error("Please provide OpenAI API key")
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Prepare player data for GPT
        player_summary = players_df.to_json(orient='records')
        
        prompt = f"""
        As an expert DFS analyst, create an optimal {constraints['contest_type']} lineup.
        
        Constraints:
        - Salary Cap: ${constraints['salary_cap']}
        - Roster Requirements: {constraints['roster_spots']}
        - Game Mode: {constraints['game_mode']}
        - Stacking: {constraints.get('stack_type', 'None')}
        
        Players Available:
        {player_summary[:10000]}  # Truncate if too long
        
        Please provide:
        1. Complete lineup with positions
        2. Total salary used
        3. Projected points
        4. Key strategic insights
        5. Alternative pivot players
        
        Return as JSON with structure:
        {{
            "lineup": [
                {{"name": "", "position": "", "team": "", "salary": 0, "projection": 0}}
            ],
            "total_salary": 0,
            "total_projection": 0,
            "insights": [],
            "pivots": []
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert DFS analyst with deep knowledge of NFL player performance, matchups, and optimization strategies."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Parse response
        result_text = response.choices[0].message.content
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            st.error("Could not parse GPT response")
            return None
            
    except Exception as e:
        st.error(f"GPT Optimization Error: {str(e)}")
        return None

# Claude Optimization Function
def optimize_with_claude(players_df, constraints, api_key):
    """Use Claude to generate optimized lineup"""
    if not api_key:
        st.error("Please provide Anthropic API key")
        return None
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Prepare player data for Claude
        player_summary = players_df.to_json(orient='records')
        
        prompt = f"""
        As an expert DFS analyst, create an optimal {constraints['contest_type']} lineup.
        
        Constraints:
        - Salary Cap: ${constraints['salary_cap']}
        - Roster Requirements: {constraints['roster_spots']}
        - Game Mode: {constraints['game_mode']}
        - Stacking: {constraints.get('stack_type', 'None')}
        
        Players Available:
        {player_summary[:10000]}  # Truncate if too long
        
        Please provide:
        1. Complete lineup with positions
        2. Total salary used
        3. Projected points
        4. Key strategic insights
        5. Alternative pivot players
        
        Return as JSON with structure:
        {{
            "lineup": [
                {{"name": "", "position": "", "team": "", "salary": 0, "projection": 0}}
            ],
            "total_salary": 0,
            "total_projection": 0,
            "insights": [],
            "pivots": []
        }}
        """
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Parse response
        result_text = message.content[0].text
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            st.error("Could not parse Claude response")
            return None
            
    except Exception as e:
        st.error(f"Claude Optimization Error: {str(e)}")
        return None

# Ensemble Optimization Function
def ensemble_optimization(gpt_result, claude_result, players_df):
    """Combine GPT and Claude results for ensemble optimization"""
    if not gpt_result or not claude_result:
        return None
    
    # Extract players from both lineups
    gpt_players = {p['name'] for p in gpt_result.get('lineup', [])}
    claude_players = {p['name'] for p in claude_result.get('lineup', [])}
    
    # Find consensus players (in both lineups)
    consensus_players = gpt_players.intersection(claude_players)
    
    # Calculate confidence scores
    all_players = gpt_players.union(claude_players)
    player_scores = {}
    
    for player in all_players:
        score = 0
        if player in gpt_players:
            score += 1
        if player in claude_players:
            score += 1
        if player in consensus_players:
            score += 0.5  # Bonus for consensus
        player_scores[player] = score
    
    # Create ensemble lineup prioritizing consensus players
    ensemble_lineup = []
    used_positions = {}
    total_salary = 0
    
    # Sort players by confidence score
    sorted_players = sorted(player_scores.items(), key=lambda x: x[1], reverse=True)
    
    for player_name, score in sorted_players:
        player_data = players_df[players_df['Name'] == player_name].iloc[0] if not players_df[players_df['Name'] == player_name].empty else None
        if player_data is not None:
            position = player_data['Position']
            salary = player_data['Salary']
            
            # Check if we can add this player
            if position not in used_positions:
                used_positions[position] = 0
            
            # Simple position limit check (would need refinement for actual use)
            if used_positions[position] < 3 and total_salary + salary <= 50000:
                ensemble_lineup.append({
                    'name': player_name,
                    'position': position,
                    'team': player_data['Team'],
                    'salary': salary,
                    'projection': player_data['Projection'],
                    'confidence': score
                })
                used_positions[position] += 1
                total_salary += salary
    
    return {
        'lineup': ensemble_lineup,
        'total_salary': total_salary,
        'consensus_players': list(consensus_players),
        'gpt_unique': list(gpt_players - consensus_players),
        'claude_unique': list(claude_players - consensus_players)
    }

# Main Application
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Upload", "🤖 AI Optimization", "📈 Analysis", "📜 History", "⚙️ Settings"])

with tab1:
    st.header("Data Upload & Preview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Player Projections CSV",
            type=['csv'],
            help="CSV should contain: Name, Position, Team, Salary, Projection columns"
        )
        
        if uploaded_file is not None:
            players_df = load_and_process_data(uploaded_file)
            if players_df is not None:
                st.session_state.players_data = players_df
                st.success(f"✅ Loaded {len(players_df)} players")
                
                # Display data preview
                st.subheader("Data Preview")
                
                # Filters
                col1_filter, col2_filter, col3_filter = st.columns(3)
                with col1_filter:
                    position_filter = st.multiselect(
                        "Filter by Position",
                        players_df['Position'].unique(),
                        default=players_df['Position'].unique()
                    )
                with col2_filter:
                    team_filter = st.multiselect(
                        "Filter by Team",
                        players_df['Team'].unique(),
                        default=players_df['Team'].unique()
                    )
                with col3_filter:
                    salary_range = st.slider(
                        "Salary Range",
                        min_value=int(players_df['Salary'].min()),
                        max_value=int(players_df['Salary'].max()),
                        value=(int(players_df['Salary'].min()), int(players_df['Salary'].max()))
                    )
                
                # Apply filters
                filtered_df = players_df[
                    (players_df['Position'].isin(position_filter)) &
                    (players_df['Team'].isin(team_filter)) &
                    (players_df['Salary'] >= salary_range[0]) &
                    (players_df['Salary'] <= salary_range[1])
                ]
                
                # Display filtered data
                st.dataframe(
                    filtered_df[['Name', 'Position', 'Team', 'Salary', 'Projection', 'Value', 'Ownership']],
                    use_container_width=True,
                    height=400
                )
                
                # Quick Stats
                st.subheader("Quick Statistics")
                col1_stats, col2_stats, col3_stats, col4_stats = st.columns(4)
                
                with col1_stats:
                    st.metric("Total Players", len(filtered_df))
                with col2_stats:
                    st.metric("Avg Projection", f"{filtered_df['Projection'].mean():.1f}")
                with col3_stats:
                    st.metric("Avg Salary", f"${filtered_df['Salary'].mean():.0f}")
                with col4_stats:
                    st.metric("Best Value", f"{filtered_df['Value'].max():.2f}")
    
    with col2:
        if st.session_state.players_data is not None:
            st.subheader("Top Projected Players")
            top_players = st.session_state.players_data.nlargest(10, 'Projection')[['Name', 'Position', 'Projection']]
            st.dataframe(top_players, hide_index=True, use_container_width=True)
            
            st.subheader("Best Values")
            best_values = st.session_state.players_data.nlargest(10, 'Value')[['Name', 'Position', 'Value']]
            st.dataframe(best_values, hide_index=True, use_container_width=True)

with tab2:
    st.header("AI-Powered Optimization")
    
    if st.session_state.players_data is None:
        st.warning("⚠️ Please upload player data first")
    else:
        # Optimization Controls
        col1_opt, col2_opt, col3_opt = st.columns([1, 1, 1])
        
        with col1_opt:
            if st.button("🚀 Run GPT Optimization", use_container_width=True, type="primary"):
                if 'openai_key' not in st.session_state or not st.session_state.openai_key:
                    st.error("Please provide OpenAI API key in the sidebar")
                else:
                    with st.spinner("GPT is optimizing lineup..."):
                        constraints = {
                            'contest_type': contest_type,
                            'salary_cap': salary_cap,
                            'roster_spots': roster_spots,
                            'game_mode': game_mode,
                            'stack_type': stack_type if enable_stacking else None
                        }
                        
                        gpt_result = optimize_with_gpt(
                            st.session_state.players_data,
                            constraints,
                            st.session_state.openai_key
                        )
                        
                        if gpt_result:
                            st.session_state.gpt_lineup = gpt_result
                            st.success("✅ GPT Optimization Complete!")
        
        with col2_opt:
            if st.button("🤖 Run Claude Optimization", use_container_width=True, type="primary"):
                if 'anthropic_key' not in st.session_state or not st.session_state.anthropic_key:
                    st.error("Please provide Anthropic API key in the sidebar")
                else:
                    with st.spinner("Claude is optimizing lineup..."):
                        constraints = {
                            'contest_type': contest_type,
                            'salary_cap': salary_cap,
                            'roster_spots': roster_spots,
                            'game_mode': game_mode,
                            'stack_type': stack_type if enable_stacking else None
                        }
                        
                        claude_result = optimize_with_claude(
                            st.session_state.players_data,
                            constraints,
                            st.session_state.anthropic_key
                        )
                        
                        if claude_result:
                            st.session_state.claude_lineup = claude_result
                            st.success("✅ Claude Optimization Complete!")
        
        with col3_opt:
            if st.button("🎯 Run Ensemble Optimization", use_container_width=True, type="primary"):
                if st.session_state.gpt_lineup and st.session_state.claude_lineup:
                    ensemble_result = ensemble_optimization(
                        st.session_state.gpt_lineup,
                        st.session_state.claude_lineup,
                        st.session_state.players_data
                    )
                    st.session_state.ensemble_lineup = ensemble_result
                    st.success("✅ Ensemble Optimization Complete!")
                else:
                    st.warning("Please run both GPT and Claude optimizations first")
        
        # Display Results
        st.divider()
        
        # GPT Results
        if st.session_state.gpt_lineup:
            with st.expander("🚀 GPT-4 Optimized Lineup", expanded=True):
                col1_gpt, col2_gpt = st.columns([2, 1])
                
                with col1_gpt:
                    st.subheader("Lineup")
                    gpt_lineup_df = pd.DataFrame(st.session_state.gpt_lineup['lineup'])
                    st.dataframe(gpt_lineup_df, use_container_width=True, hide_index=True)
                
                with col2_gpt:
                    st.subheader("Summary")
                    st.metric("Total Salary", f"${st.session_state.gpt_lineup.get('total_salary', 0):,}")
                    st.metric("Projected Points", f"{st.session_state.gpt_lineup.get('total_projection', 0):.1f}")
                    
                    if 'insights' in st.session_state.gpt_lineup:
                        st.subheader("Insights")
                        for insight in st.session_state.gpt_lineup['insights'][:3]:
                            st.write(f"• {insight}")
        
        # Claude Results
        if st.session_state.claude_lineup:
            with st.expander("🤖 Claude Optimized Lineup", expanded=True):
                col1_claude, col2_claude = st.columns([2, 1])
                
                with col1_claude:
                    st.subheader("Lineup")
                    claude_lineup_df = pd.DataFrame(st.session_state.claude_lineup['lineup'])
                    st.dataframe(claude_lineup_df, use_container_width=True, hide_index=True)
                
                with col2_claude:
                    st.subheader("Summary")
                    st.metric("Total Salary", f"${st.session_state.claude_lineup.get('total_salary', 0):,}")
                    st.metric("Projected Points", f"{st.session_state.claude_lineup.get('total_projection', 0):.1f}")
                    
                    if 'insights' in st.session_state.claude_lineup:
                        st.subheader("Insights")
                        for insight in st.session_state.claude_lineup['insights'][:3]:
                            st.write(f"• {insight}")
        
        # Ensemble Results
        if 'ensemble_lineup' in st.session_state and st.session_state.ensemble_lineup:
            with st.expander("🎯 Ensemble Lineup", expanded=True):
                col1_ens, col2_ens = st.columns([2, 1])
                
                with col1_ens:
                    st.subheader("Consensus Lineup")
                    ensemble_lineup_df = pd.DataFrame(st.session_state.ensemble_lineup['lineup'])
                    st.dataframe(ensemble_lineup_df, use_container_width=True, hide_index=True)
                
                with col2_ens:
                    st.subheader("Agreement Analysis")
                    st.write(f"**Consensus Players:** {len(st.session_state.ensemble_lineup['consensus_players'])}")
                    st.write(f"**GPT Unique:** {len(st.session_state.ensemble_lineup['gpt_unique'])}")
                    st.write(f"**Claude Unique:** {len(st.session_state.ensemble_lineup['claude_unique'])}")
                    
                    st.subheader("Consensus Players")
                    for player in st.session_state.ensemble_lineup['consensus_players'][:5]:
                        st.write(f"✅ {player}")
        
        # Export Lineups
        st.divider()
        col1_export, col2_export, col3_export = st.columns(3)
        
        with col1_export:
            if st.session_state.gpt_lineup:
                gpt_csv = pd.DataFrame(st.session_state.gpt_lineup['lineup']).to_csv(index=False)
                st.download_button(
                    label="📥 Download GPT Lineup",
                    data=gpt_csv,
                    file_name=f"gpt_lineup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2_export:
            if st.session_state.claude_lineup:
                claude_csv = pd.DataFrame(st.session_state.claude_lineup['lineup']).to_csv(index=False)
                st.download_button(
                    label="📥 Download Claude Lineup",
                    data=claude_csv,
                    file_name=f"claude_lineup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3_export:
            if 'ensemble_lineup' in st.session_state and st.session_state.ensemble_lineup:
                ensemble_csv = pd.DataFrame(st.session_state.ensemble_lineup['lineup']).to_csv(index=False)
                st.download_button(
                    label="📥 Download Ensemble Lineup",
                    data=ensemble_csv,
                    file_name=f"ensemble_lineup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Captain Mode Pivots
        if enable_captain_mode:
            st.divider()
            st.subheader("Captain Mode Pivots")
            
            if st.button("Generate Captain Pivots", use_container_width=True):
                # Simulate captain pivots
                all_pivots = []
                
                if st.session_state.gpt_lineup:
                    for player in st.session_state.gpt_lineup.get('pivots', [])[:3]:
                        all_pivots.append({
                            'Source': 'GPT',
                            'Player': player.get('name', 'Unknown'),
                            'Position': player.get('position', 'Unknown'),
                            'Reason': player.get('reason', 'High upside pivot')
                        })
                
                if st.session_state.claude_lineup:
                    for player in st.session_state.claude_lineup.get('pivots', [])[:3]:
                        all_pivots.append({
                            'Source': 'Claude',
                            'Player': player.get('name', 'Unknown'),
                            'Position': player.get('position', 'Unknown'),
                            'Reason': player.get('reason', 'Strategic pivot option')
                        })
                
                # Store pivots in session state
                pivots_df = pd.DataFrame(all_pivots)
                if not pivots_df.empty:
                    st.session_state.pivots_df = pivots_df
                else:
                    st.session_state.pivots_df = pd.DataFrame()
                
                st.info(f"Generated {len(pivots_df)} captain pivot variations")
    
    # Display individual pivot explanations
    if hasattr(st.session_state, 'pivots_df') and st.session_state.pivots_df is not None:
        if not st.session_state.pivots_df.empty:
            st.subheader("Pivot Explanations")
            for idx, pivot in st.session_state.pivots_df.iterrows():
                with st.expander(f"{pivot['Source']} - {pivot['Player']} ({pivot['Position']})"):
                    st.write(pivot['Reason'])
    else:
        if enable_captain_mode:
            st.info("Enable captain pivots in settings to generate variations")

with tab3:
    st.markdown("### Lineup Analysis & Visualization")
    
    if st.session_state.players_data is not None and (st.session_state.gpt_lineup or st.session_state.claude_lineup):
        
        # Create comparison visualizations
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Salary Distribution', 'Position Distribution', 'Team Stacking',
                          'Projected Points by Position', 'Ownership Distribution', 'Risk/Reward Matrix'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'histogram'}, {'type': 'scatter'}]]
        )
        
        # Prepare data for visualization
        lineup_data = []
        if st.session_state.gpt_lineup:
            for player in st.session_state.gpt_lineup['lineup']:
                player['model'] = 'GPT'
                lineup_data.append(player)
        
        if st.session_state.claude_lineup:
            for player in st.session_state.claude_lineup['lineup']:
                player['model'] = 'Claude'
                lineup_data.append(player)
        
        if lineup_data:
            lineup_df = pd.DataFrame(lineup_data)
            
            # 1. Salary Distribution
            salary_by_model = lineup_df.groupby('model')['salary'].sum().reset_index()
            fig.add_trace(
                go.Bar(x=salary_by_model['model'], y=salary_by_model['salary'], name='Salary'),
                row=1, col=1
            )
            
            # 2. Position Distribution
            position_counts = lineup_df['position'].value_counts()
            fig.add_trace(
                go.Pie(labels=position_counts.index, values=position_counts.values, name='Positions'),
                row=1, col=2
            )
            
            # 3. Team Stacking
            team_counts = lineup_df['team'].value_counts().head(5)
            fig.add_trace(
                go.Bar(x=team_counts.index, y=team_counts.values, name='Team Stack'),
                row=1, col=3
            )
            
            # 4. Projected Points by Position
            pts_by_position = lineup_df.groupby('position')['projection'].mean().reset_index()
            fig.add_trace(
                go.Bar(x=pts_by_position['position'], y=pts_by_position['projection'], name='Avg Projection'),
                row=2, col=1
            )
            
            # 5. Ownership Distribution (simulated)
            ownership_vals = np.random.normal(15, 5, len(lineup_df))
            fig.add_trace(
                go.Histogram(x=ownership_vals, name='Ownership', nbinsx=10),
                row=2, col=2
            )
            
            # 6. Risk/Reward Matrix
            lineup_df['risk'] = np.random.uniform(0.3, 0.8, len(lineup_df))
            lineup_df['reward'] = lineup_df['projection'] / lineup_df['projection'].max()
            fig.add_trace(
                go.Scatter(
                    x=lineup_df['risk'], 
                    y=lineup_df['reward'],
                    mode='markers+text',
                    text=lineup_df['name'],
                    textposition='top center',
                    marker=dict(size=10),
                    name='Risk/Reward'
                ),
                row=2, col=3
            )
            
            # Update layout
            fig.update_layout(height=800, showlegend=False, title_text="Lineup Analysis Dashboard")
            fig.update_xaxes(title_text="Model", row=1, col=1)
            fig.update_xaxes(title_text="Team", row=1, col=3)
            fig.update_xaxes(title_text="Position", row=2, col=1)
            fig.update_xaxes(title_text="Ownership %", row=2, col=2)
            fig.update_xaxes(title_text="Risk", row=2, col=3)
            fig.update_yaxes(title_text="Salary", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=3)
            fig.update_yaxes(title_text="Avg Points", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=2)
            fig.update_yaxes(title_text="Reward", row=2, col=3)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Analysis
        st.subheader("Player Correlation Matrix")
        
        if st.session_state.players_data is not None:
            # Create correlation matrix for top players
            top_players = st.session_state.players_data.nlargest(20, 'Projection')
            correlation_data = top_players[['Salary', 'Projection', 'Value', 'Ownership', 'Upside']]
            
            fig_corr = px.imshow(
                correlation_data.corr(),
                labels=dict(color="Correlation"),
                x=correlation_data.columns,
                y=correlation_data.columns,
                color_continuous_scale="RdBu",
                title="Player Metrics Correlation"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("📊 Run optimizations to see analysis visualizations")

with tab4:
    st.header("Optimization History")
    
    if st.session_state.optimization_history:
        # Display history table
        history_df = pd.DataFrame(st.session_state.optimization_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        # History analytics
        st.subheader("Historical Performance")
        
        col1_hist, col2_hist, col3_hist = st.columns(3)
        
        with col1_hist:
            avg_score = history_df['projected_score'].mean()
            st.metric("Average Projected Score", f"{avg_score:.1f}")
        
        with col2_hist:
            best_lineup = history_df.loc[history_df['projected_score'].idxmax()]
            st.metric("Best Lineup Score", f"{best_lineup['projected_score']:.1f}")
        
        with col3_hist:
            win_rate = (history_df['actual_score'] > history_df['projected_score']).mean() * 100
            st.metric("Beat Projection Rate", f"{win_rate:.1f}%")
        
        # Trend chart
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df['projected_score'],
            mode='lines+markers',
            name='Projected',
            line=dict(color='blue')
        ))
        
        if 'actual_score' in history_df.columns:
            fig_trend.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df['actual_score'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='green')
            ))
        
        fig_trend.update_layout(
            title="Score Trends Over Time",
            xaxis_title="Lineup #",
            yaxis_title="Score",
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No optimization history yet. Run some optimizations to see historical data.")
    
    # Add manual result entry
    st.divider()
    st.subheader("Add Contest Result")
    
    with st.form("add_result"):
        col1_form, col2_form = st.columns(2)
        
        with col1_form:
            contest_date = st.date_input("Contest Date")
            contest_name = st.text_input("Contest Name")
            actual_score = st.number_input("Actual Score", min_value=0.0, step=0.1)
        
        with col2_form:
            finish_position = st.number_input("Finish Position", min_value=1, step=1)
            total_entries = st.number_input("Total Entries", min_value=1, step=1)
            winnings = st.number_input("Winnings ($)", min_value=0.0, step=0.01)
        
        if st.form_submit_button("Add Result"):
            new_result = {
                'date': contest_date,
                'contest': contest_name,
                'actual_score': actual_score,
                'finish': finish_position,
                'entries': total_entries,
                'winnings': winnings,
                'roi': (winnings / 10 - 1) * 100  # Assuming $10 entry
            }
            st.session_state.optimization_history.append(new_result)
            st.success("Result added successfully!")
            st.rerun()

with tab5:
    st.header("Advanced Settings & Tools")
    
    # Correlation Settings
    st.subheader("📊 Correlation Rules")
    
    col1_corr, col2_corr = st.columns(2)
    
    with col1_corr:
        st.write("**Positive Correlations**")
        qb_stack_weight = st.slider("QB Stack Correlation", 0.0, 1.0, 0.7)
        game_stack_weight = st.slider("Game Stack Correlation", 0.0, 1.0, 0.5)
        team_defense_weight = st.slider("Team Defense Correlation", 0.0, 1.0, 0.3)
    
    with col2_corr:
        st.write("**Negative Correlations**")
        rb_dst_negative = st.slider("RB vs Opposing DST", -1.0, 0.0, -0.4)
        qb_dst_negative = st.slider("QB vs Opposing DST", -1.0, 0.0, -0.3)
    
    # Variance Settings
    st.subheader("📈 Variance & Diversity")
    
    num_lineups = st.number_input("Number of Lineups to Generate", min_value=1, max_value=150, value=20)
    min_player_diff = st.slider("Minimum Player Difference", min_value=1, max_value=5, value=2)
    
    if st.button("Generate Multiple Lineups", use_container_width=True):
        if st.session_state.players_data is not None:
            with st.spinner(f"Generating {num_lineups} unique lineups..."):
                # Simulate multiple lineup generation
                progress_bar = st.progress(0)
                generated_lineups = []
                
                for i in range(num_lineups):
                    # Simulate lineup generation with variance
                    time.sleep(0.1)  # Simulate processing
                    progress_bar.progress((i + 1) / num_lineups)
                    
                    # Create mock lineup with variance
                    sample_players = st.session_state.players_data.sample(9)
                    lineup = {
                        'lineup_id': i + 1,
                        'players': sample_players['Name'].tolist(),
                        'salary': sample_players['Salary'].sum(),
                        'projection': sample_players['Projection'].sum(),
                        'variance': np.random.uniform(0.1, 0.3)
                    }
                    generated_lineups.append(lineup)
                
                st.success(f"✅ Generated {num_lineups} unique lineups!")
                
                # Display lineup summaries
                lineups_df = pd.DataFrame(generated_lineups)
                st.dataframe(lineups_df[['lineup_id', 'salary', 'projection', 'variance']], use_container_width=True)
        else:
            st.warning("Please upload player data first")
    
    # Late Swap Optimizer
    st.divider()
    st.subheader("🔄 Late Swap Optimizer")
    
    if enable_late_swap:
        st.write("Identify optimal late swap opportunities based on:")
        st.write("• Injury news")
        st.write("• Weather updates")
        st.write("• Inactive reports")
        
        late_games = st.multiselect(
            "Select Late Game Teams",
            ["KC", "BUF", "DAL", "PHI", "SF", "SEA"],
            default=["KC", "BUF"]
        )
        
        if st.button("Find Late Swap Options"):
            st.info(f"Analyzing players from: {', '.join(late_games)}")
            # Would implement actual late swap logic here
    
    # Export Settings
    st.divider()
    st.subheader("📤 Export Settings")
    
    export_format = st.selectbox(
        "Export Format",
        ["CSV", "JSON", "DraftKings CSV", "FanDuel CSV"]
    )
    
    include_alternates = st.checkbox("Include alternate lineups in export")
    include_analysis = st.checkbox("Include analysis data in export")

# Display Pivots DataFrame
if hasattr(st.session_state, 'pivots_df') and st.session_state.pivots_df is not None:
    if not st.session_state.pivots_df.empty:
        st.subheader("Player Pivots Analysis")
        st.dataframe(st.session_state.pivots_df, use_container_width=True)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>NFL DFS Dual AI Optimizer v3.0 | Powered by GPT-4 & Claude AI</p>
        <p style='font-size: 12px;'>Always gamble responsibly. This tool is for entertainment and educational purposes only.</p>
    </div>
""", unsafe_allow_html=True)
