import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pulp

st.set_page_config(page_title="NFL Showdown Optimizer", layout="wide")

st.title("NFL Showdown Optimizer")

def detect_columns(df):
    """Auto-detect column names"""
    cols = df.columns.str.lower()
    
    # Find player name column
    player_col = None
    for possible in ['player', 'name', 'playername', 'player name']:
        matches = [c for c in df.columns if possible in c.lower()]
        if matches:
            player_col = matches[0]
            break
    
    # Find salary column
    salary_col = None
    for possible in ['salary', 'sal', 'cost']:
        matches = [c for c in df.columns if possible in c.lower()]
        if matches:
            salary_col = matches[0]
            break
    
    # Find projection column
    proj_col = None
    for possible in ['proj', 'fpts', 'points', 'projected']:
        matches = [c for c in df.columns if possible in c.lower()]
        if matches:
            proj_col = matches[0]
            break
    
    return player_col, salary_col, proj_col

def optimize_lineups(df, num_lineups=20):
    """Generate lineups using linear programming"""
    
    # Detect column names
    player_col, salary_col, proj_col = detect_columns(df)
    
    if not all([player_col, salary_col, proj_col]):
        missing = []
        if not player_col: missing.append("Player/Name")
        if not salary_col: missing.append("Salary")
        if not proj_col: missing.append("Projection")
        raise ValueError(f"Could not find columns: {', '.join(missing)}")
    
    lineups = []
    
    for i in range(num_lineups):
        prob = pulp.LpProblem(f"Showdown_{i}", pulp.LpMaximize)
        
        # Variables
        player_vars = {p: pulp.LpVariable(f"p_{j}_{i}", cat='Binary') 
                      for j, p in enumerate(df[player_col])}
        captain_vars = {p: pulp.LpVariable(f"c_{j}_{i}", cat='Binary') 
                       for j, p in enumerate(df[player_col])}
        
        # Objective with randomness
        proj_dict = dict(zip(df[player_col], 
                            df[proj_col] * (1 + np.random.uniform(-0.1, 0.1, len(df)))))
        
        prob += pulp.lpSum([
            captain_vars[p] * proj_dict[p] * 1.5 + player_vars[p] * proj_dict[p]
            for p in player_vars
        ])
        
        # Constraints
        prob += pulp.lpSum(captain_vars.values()) == 1
        prob += pulp.lpSum(player_vars.values()) == 5
        
        for p in player_vars:
            prob += captain_vars[p] + player_vars[p] <= 1
        
        # Salary cap
        salary_dict = dict(zip(df[player_col], df[salary_col]))
        prob += pulp.lpSum([
            captain_vars[p] * salary_dict[p] * 1.5 + player_vars[p] * salary_dict[p]
            for p in player_vars
        ]) <= 50000
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            captain = next((p for p in captain_vars if captain_vars[p].varValue > 0.5), None)
            flex = [p for p in player_vars if player_vars[p].varValue > 0.5]
            
            if captain and len(flex) == 5:
                # Calculate lineup stats
                lineup_players = [captain] + flex
                total_salary = (salary_dict[captain] * 1.5 + 
                               sum(salary_dict[p] for p in flex))
                total_proj = (proj_dict[captain] * 1.5 + 
                             sum(proj_dict[p] for p in flex))
                
                lineups.append({
                    'Lineup': i + 1,
                    'CPT': captain,
                    'FLEX1': flex[0],
                    'FLEX2': flex[1],
                    'FLEX3': flex[2],
                    'FLEX4': flex[3],
                    'FLEX5': flex[4],
                    'Total_Salary': int(total_salary),
                    'Projected': round(total_proj, 2)
                })
    
    return pd.DataFrame(lineups)

# UI
uploaded_file = st.file_uploader("Upload Player Projections CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Show detected columns
    player_col, salary_col, proj_col = detect_columns(df)
    
    st.success(f"Detected columns: Player='{player_col}', Salary='{salary_col}', Projection='{proj_col}'")
    
    st.subheader("Player Pool Preview")
    st.dataframe(df.head(10))
    
    num_lineups = st.slider("Number of Lineups", 5, 150, 20)
    
    if st.button("Generate Lineups", type="primary"):
        try:
            with st.spinner("Optimizing..."):
                lineups = optimize_lineups(df, num_lineups)
                
                st.success(f"Generated {len(lineups)} lineups")
                
                # Display metrics
                col1, col2 = st.columns(2)
                col1.metric("Avg Projection", f"{lineups['Projected'].mean():.2f}")
                col2.metric("Avg Salary", f"${lineups['Total_Salary'].mean():,.0f}")
                
                st.dataframe(lineups)
                
                csv = lineups.to_csv(index=False)
                st.download_button(
                    "Download CSV", 
                    csv, 
                    f"lineups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure your CSV has columns for player names, salaries, and projections")
else:
    st.info("Upload a CSV file with player data to begin")
    st.markdown("""
    **Required columns:**
    - Player names (e.g., 'Player', 'Name')
    - Salaries (e.g., 'Salary', 'Sal')
    - Projections (e.g., 'Proj', 'FPTS', 'Projected_Points')
    """)
