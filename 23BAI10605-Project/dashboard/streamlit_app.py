import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import time
import json
import numpy as np

# ── Project root resolution ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
EVAL_SCRIPT = os.path.join(PROJECT_ROOT, "evaluation", "evaluate.py")
CHECK_JUNCTIONS = os.path.join(PROJECT_ROOT, "check_junctions.py")
LIVE_CSV = os.path.join(LOGS_DIR, "live_metrics.csv")
JUNCTION_JSON = os.path.join(LOGS_DIR, "junction_states.json")
MANUAL_ACTIONS = os.path.join(LOGS_DIR, "manual_actions.json")
COMPARISON_IMG = os.path.join(PLOTS_DIR, "comparison_results.png")
EVAL_CSV = os.path.join(PROJECT_ROOT, "evaluation", "evaluation_results.csv")

st.set_page_config(page_title="Smart Urban Traffic Control", layout="wide", page_icon="🚥")

st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("🚥 Smart Urban Traffic Signal Control")
st.subheader("Interactive Multi-Agent Deep RL Simulation (6x6 Grid, 25 Intersections)")

# Sidebar
st.sidebar.header("🕹️ Simulation Control")
mode = st.sidebar.radio("Navigation", ["Overview", "Live Dashboard", "Manual Override", "Performance Comparison"])

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Environment Settings")
density = st.sidebar.selectbox("Traffic Density", ["low", "medium", "high"], index=1)
if st.sidebar.button("🔄 Regenerate Network & Traffic"):
    with st.spinner("Generating realistic 6x6 grid with buildings..."):
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "environment", "generate_grid.py")], cwd=PROJECT_ROOT)
        if 'junction_ids' in st.session_state: del st.session_state.junction_ids
        st.sidebar.success("Environment reset complete!")

if mode == "Overview":
    st.markdown("""
    ### 🏙️ Advanced Multi-Agent Traffic Management
    This system manages a **6x6 urban grid** with **25 intelligent intersections** using decentralized Deep Reinforcement Learning (PPO).
    
    #### 🚀 Realistic Simulation Features:
    - **Complex Grid:** 6x6 network with 3 lanes per edge, sidewalks, and crossings.
    - **Randomized Blocks:** Varied street lengths to simulate an organic city layout.
    - **Visual Realism:** 3D buildings (polygons) and diverse vehicle types (Bus, Truck, Moto, Cars).
    - **Intelligent Agents:** Multi-Agent PPO controllers that learn to minimize global waiting time.
    - **Safe Transitions:** Automatic 3-second yellow light logic for all phase switches.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Did you know?** The PPO agent here handles 37-dimensional state inputs for each junction, ensuring it captures traffic across all 12 potential lanes.")
    with col2:
        if os.path.exists(COMPARISON_IMG):
            st.image(COMPARISON_IMG, caption="Last Performance Evaluation", use_container_width=True)

elif mode == "Live Dashboard":
    st.markdown("### 📈 Live Performance Metrics")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        sim_steps = st.slider("Simulation Steps", 100, 2000, 500)
    with col2:
        launch_gui = st.checkbox("Open SUMO-GUI", value=True)
    with col3:
        st.write("Launch the simulation to see real-time performance of the PPO agents.")

    if st.button("🚥 Start Visual Simulation"):
        if os.path.exists(LIVE_CSV): os.remove(LIVE_CSV)
        
        # Build command — only add --gui when actually requested (avoids empty-string arg bug)
        cmd = [sys.executable, EVAL_SCRIPT, "--steps", str(sim_steps), "--live"]
        if launch_gui:
            cmd.append("--gui")
        subprocess.Popen(cmd, cwd=PROJECT_ROOT)
        st.success("Simulation started!")
        
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        st.write("---")
        st.subheader("🏙️ City Junction Status (Live Grid)")
        grid_placeholder = st.empty()
        
        start_time = time.time()
        while time.time() - start_time < (sim_steps / 1.5): # Rough timeout
            # Update metrics
            if os.path.exists(LIVE_CSV):
                try:
                    df = pd.read_csv(LIVE_CSV)
                    if not df.empty:
                        last = df.iloc[-1]
                        with metrics_placeholder.container():
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Avg Waiting Time", f"{last['avg_wait']:.1f}s")
                            m2.metric("Avg Queue Length", f"{last['avg_queue']:.1f}")
                            m3.metric("Throughput", int(last['throughput']))
                        
                        with chart_placeholder.container():
                            fig, ax = plt.subplots(figsize=(12, 4))
                            ax.plot(df['step'], df['avg_wait'], label='Wait (s)', color='#FF4B4B', linewidth=2)
                            ax.plot(df['step'], df['avg_queue'], label='Queue', color='#00CC96', linewidth=2)
                            ax.set_facecolor('#f5f7f9')
                            ax.set_title("Real-Time Simulation Performance")
                            ax.legend()
                            st.pyplot(fig)
                except: pass
            
            # Update grid view
            if os.path.exists(JUNCTION_JSON):
                try:
                    with open(JUNCTION_JSON, "r") as f:
                        j_states = json.load(f)
                    
                    if j_states:
                        with grid_placeholder.container():
                            # Organize into rows (assuming 5x5 or similar)
                            j_ids = sorted(list(j_states.keys()))
                            num_j = len(j_ids)
                            cols_per_row = 5
                            
                            for i in range(0, num_j, cols_per_row):
                                row_cols = st.columns(cols_per_row)
                                for idx, jid in enumerate(j_ids[i:i+cols_per_row]):
                                    state = j_states[jid]
                                    phase = state['phase']
                                    queue = state['queue']
                                    
                                    # Simple visual: color based on phase and size based on queue
                                    color = "green" if phase % 2 == 0 else "blue" # Simplified
                                    if phase == 1 or phase == 3: color = "orange" # Yellow phases often odd
                                    
                                    with row_cols[idx]:
                                        st.markdown(f"""
                                        <div style="background-color: white; border: 1px solid #ddd; padding: 10px; border-radius: 5px; text-align: center;">
                                            <div style="font-weight: bold; font-size: 0.8em;">{jid}</div>
                                            <div style="font-size: 1.5em;">{'🚥' if phase % 2 == 0 else '🚦'}</div>
                                            <div style="color: {'red' if queue > 5 else 'gray'};">🚗 {queue}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                except: pass
            
            time.sleep(1)


elif mode == "Manual Override":
    st.markdown("### 🎮 Remote Junction Control")
    st.write("Take control of individual junctions in the 6x6 grid.")
    
    if 'junction_ids' not in st.session_state:
        try:
            res = subprocess.check_output([sys.executable, CHECK_JUNCTIONS], cwd=PROJECT_ROOT).decode().strip()
            st.session_state.junction_ids = res.split(",") if res else []
        except:
            st.session_state.junction_ids = []

    if st.session_state.junction_ids:
        jid = st.selectbox("Select Junction to Control", st.session_state.junction_ids)
        
        manual_actions_path = MANUAL_ACTIONS
        
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔴 Switch Phase Now", use_container_width=True):
                actions = {}
                if os.path.exists(manual_actions_path):
                    with open(manual_actions_path, 'r') as f: actions = json.load(f)
                actions[jid] = 1
                with open(manual_actions_path, 'w') as f: json.dump(actions, f)
                st.toast(f"Phase Switch command sent to {jid}")
        with c2:
            if st.button("🟢 Keep Phase Fixed", use_container_width=True):
                actions = {}
                if os.path.exists(manual_actions_path):
                    with open(manual_actions_path, 'r') as f: actions = json.load(f)
                actions[jid] = 0
                with open(manual_actions_path, 'w') as f: json.dump(actions, f)
                st.toast(f"Keep Phase command sent to {jid}")
        with c3:
            if st.button("🔄 Auto (PPO Control)", use_container_width=True):
                actions = {}
                if os.path.exists(manual_actions_path):
                    with open(manual_actions_path, 'r') as f: actions = json.load(f)
                if jid in actions: del actions[jid]
                with open(manual_actions_path, 'w') as f: json.dump(actions, f)
                st.toast(f"Junction {jid} reset to RL control")
    else:
        st.warning("⚠️ No active simulation detected or junctions not loaded. Start a simulation first.")

elif mode == "Performance Comparison":
    st.markdown("### 📊 Baseline Comparison")
    st.write("Compare the PPO Agent against Fixed-Time and Random controllers.")
    
    if st.button("🚀 Run Comprehensive Evaluation"):
        with st.spinner("Simulating all policies... This may take 30-60 seconds."):
            subprocess.run([sys.executable, EVAL_SCRIPT, "--steps", "500"], cwd=PROJECT_ROOT)
            st.success("Evaluation Complete!")
            st.rerun()
            
    if os.path.exists(EVAL_CSV):
        df = pd.read_csv(EVAL_CSV)
        st.table(df)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        df.plot(kind='bar', x='Policy', y='Avg Waiting Time', ax=axes[0], color=['#FF4B4B', '#FFA15A', '#19D3F3'])
        axes[0].set_title("Avg Waiting Time (Lower is Better)")
        df.plot(kind='bar', x='Policy', y='Total Throughput', ax=axes[1], color=['#00CC96', '#636EFA', '#AB63FA'])
        axes[1].set_title("Total Throughput (Higher is Better)")
        st.pyplot(fig)
    else:
        st.info("No evaluation results yet. Click the button above to run the comparison.")
