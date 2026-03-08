import sys
import requests
import json
import re
from typing import List
from alfworld.agents.environment import get_environment
from alfworld.agents.environment.alfred_tw_env import TASK_TYPES
import alfworld.agents.modules.generic as generic

# --- THE JUPYTER BYPASS ---
sys.argv = ['jupyter_notebook.py', 'configs/base_config.yaml']

# --- API SETUP ---
API_KEY = "sk-SO0OE_L0z_JJPscdQSO2jg"
URL = "https://tritonai-api.ucsd.edu/v1/chat/completions"

def call_llm_api(prompt):
    """Helper function to keep API calls clean and handle JSON extraction."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {"model": "api-llama-4-scout", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
    
    response = requests.post(URL, headers=headers, json=payload)
    if response.status_code == 200:
        raw_content = response.json()["choices"][0]["message"]["content"].strip()
        start_idx = raw_content.find('{')
        end_idx = raw_content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            try:
                return json.loads(raw_content[start_idx:end_idx+1])
            except json.JSONDecodeError:
                pass
    else:
        print(f"API Error {response.status_code}: {response.text}")
    return None

# --- AGENT FUNCTIONS ---

def decompose_task(task_desc):
    """LLM Manager: Splits the main task into Target, Quantity, and Execution Goal."""
    prompt = f"""You are a task manager for a household AI. 
Task: {task_desc}

Identify the item to find, HOW MANY are needed, and the final goal.
CRITICAL PHYSICS RULE: The agent can only hold ONE item at a time. Multi-item tasks require moving them one by one.

Respond ONLY in valid JSON:
{{
    "search_target": "alarmclock",
    "target_quantity": 2,
    "execution_goal": "Take the first alarmclock, put it in the desk. Then take the second alarmclock and put it in the desk."
}}"""
    
    result = call_llm_api(prompt)
    return result if result else {"search_target": "item", "target_quantity": 1, "execution_goal": task_desc}

def get_programmatic_search_action(observation, valid_commands, search_state, target_item, target_qty):
    """Programmatic Search Drone: 100% code-based BFS algorithm. No LLM required."""
    
    # 1. Initialize the queue of locations on the very first step
    if search_state["unvisited"] is None:
        search_state["unvisited"] = [cmd for cmd in valid_commands if cmd.startswith("go to ")]

    # 2. Scan the observation for the target item using Regex
    # Matches items like "apple 1", "alarmclock 2", etc.
    found_matches = re.findall(rf"\b{target_item} \d+\b", observation)
    
    for match in found_matches:
        if match not in search_state["found_instances"]:
            search_state["found_instances"].append(match)
            # Record exactly where we saw it
            search_state["found_locations"].append(f"{match} at {search_state['current_loc']}")

    # 3. Check if we reached our target quantity
    if len(search_state["found_instances"]) >= target_qty:
        return "look", search_state, True # Phase complete!

    # 4. Handle closed receptacles
    if "is closed" in observation:
        open_cmd = f"open {search_state['current_loc']}"
        if open_cmd in valid_commands:
            return open_cmd, search_state, False

    # 5. Move to the next location in the queue
    if search_state["unvisited"]:
        next_cmd = search_state["unvisited"].pop(0)
        # Keep track of what furniture we are standing in front of
        search_state["current_loc"] = next_cmd.replace("go to ", "").strip()
        return next_cmd, search_state, False

    # 6. Fallback if exhausted (searched whole room, didn't find enough)
    return "look", search_state, False 

def get_execute_action(goal, observation, valid_commands, history):
    """The reasoning agent that takes over once items are found."""
    prompt = f"""You are an Execution Agent. Your job is to manipulate items to finish the goal. 
CRITICAL PHYSICS RULES: 
1. You can only hold ONE item at a time. 
2. You CANNOT interact with a receptacle (take, put, open) unless you are physically standing in front of it. If you need to put an item in a desk, but you are at a dresser, your next action MUST be 'go to desk 1'.

Goal: {goal}
Transcript: {history}
Current Observation: {observation}
Available Commands: {valid_commands}

Respond ONLY with ONE valid JSON object:
1. "current_location": Read the Current Observation or Transcript and state exactly what furniture you are currently facing.
2. "thought": Your step-by-step reasoning. Compare your "current_location" to your Goal to determine if you need to travel ('go to') first.
3. "action": The exact text of your chosen command VERBATIM from the Available Commands list.
4. "phase_complete": A boolean. Set to true ONLY if you have completed the final step of your goal.
"""
    
    result = call_llm_api(prompt)
    if result:
        print(f"  [Thought]: {result.get('thought', '...')}")
        return result.get("action", "look").strip(), result.get("phase_complete", False)
    return "look", False

# --- GAME FILTERING LOGIC ---
def detect_task_type(gamefile_path: str) -> str:
    for task_type in TASK_TYPES.values():
        if task_type in gamefile_path:
            return task_type
    return "unknown"

def load_games(game_files: List[str], num_games: int) -> List[str]:
    files = sorted(game_files)
    if num_games <= 0 or num_games >= len(files):
        return files
    groups = {}
    for f in files:
        groups.setdefault(detect_task_type(f), []).append(f)
    types = sorted(groups.keys())
    per_type = num_games // len(types)
    remainder = num_games % len(types)
    selected = []
    for i, tt in enumerate(types):
        take = per_type + (1 if i < remainder else 0)
        selected.extend(groups[tt][:take])
    return sorted(selected)

# --- GAME SETUP ---
config = generic.load_config()
env_type = config['env']['type']
AlfredEnvClass = get_environment(env_type)

eval_split = 'train'
num_games_to_load = 50

alfred_env = AlfredEnvClass(config, train_eval=eval_split)
alfred_env.game_files = load_games(alfred_env.game_files, num_games_to_load)
alfred_env.num_games = len(alfred_env.game_files)

env = alfred_env.init_env(batch_size=1)

print(f"=== ALFWorld Neuro-Symbolic Agent Started ({alfred_env.num_games} games loaded) ===\n")

# --- TRACKING METRICS ---
total_games_played = 0
successful_games = 0
failed_games = 0

# --- MAIN EVALUATION LOOP ---
for game_idx in range(alfred_env.num_games):
    obs, info = env.reset()
    total_games_played += 1
    
    print(f"\n{'='*60}")
    print(f"Starting Game {game_idx + 1} of {alfred_env.num_games}")
    print(f"{'='*60}")
    
    # 1. Extract Task
    raw_initial_obs = obs[0]
    match = re.search(r"Your task is to:\s*(.*)", raw_initial_obs)
    task_desc = match.group(1).strip() if match else raw_initial_obs 
    print(f"Goal: {task_desc}\n" + "-"*40)
    
    # 2. Decompose Task (LLM)
    print("Decomposing task...")
    task_plan = decompose_task(task_desc)
    target_item = task_plan.get("search_target", "item")
    target_qty = task_plan.get("target_quantity", 1)
    execute_goal = task_plan.get("execution_goal", task_desc)
    
    print(f"  [Target]: Find {target_qty} '{target_item}'")
    print(f"  [Execute Goal]: {execute_goal}\n" + "-"*40)

    # Initialize States
    search_state = {
        "unvisited": None, 
        "current_loc": "center of room", 
        "found_instances": [], 
        "found_locations": []
    }
    
    chat_history = f"Game Started. Initial observation: {raw_initial_obs}\n"
    current_phase = "search"
    
    MAX_STEPS = 50 # Increased significantly because programmatic search is extremely fast
    game_done = False

    # 3. Execution Loop
    for step in range(MAX_STEPS):
        raw_commands = list(info['admissible_commands'][0])
        
        if current_phase == "search":
            # FAST PYTHON SEARCH (Instantaneous)
            action, search_state, is_complete = get_programmatic_search_action(
                obs[0], raw_commands, search_state, target_item, target_qty
            )
            print(f"Step {step + 1} | [Programmatic Search] Executing: {action}")
            
        else:
            print(f"Search State: {search_state}")
            # LLM EXECUTOR (Reasoning required)
            print(f"Step {step + 1} | [LLM Executor] Thinking...")
            print(f"Raw commands: {raw_commands}")
            action, is_complete = get_execute_action(
                execute_goal, obs[0], raw_commands, chat_history
            )
            
            # Fallback for invalid commands
            if action not in raw_commands:
                print(f"  [Warning]: Invalid action '{action}'. Overriding to 'look'.")
                action = "look"

        print(f"LLM/Script Action: {action}")
        
        # Take the step
        obs, scores, dones, infos = env.step([action])
        print(f"Observation: {obs[0]}\n" + "-"*40)
        
        # We ONLY append Execution steps to the LLM's chat history to keep it perfectly clean
        if current_phase == "execute":
            chat_history += f"\n> Action: {action}\nObservation: {obs[0]}"
            
        info = {k: v for k, v in infos.items()}
        
        # Phase Handoff Logic
        if is_complete and current_phase == "search":
            found_str = ", ".join(search_state["found_locations"])
            
            print(f"\n>>> PROGRAMMATIC SEARCH COMPLETE. FOUND: {found_str} <<<")
            print(">>> HANDING OFF TO LLM EXECUTOR. <<<\n")
            current_phase = "execute"
            
            # Inject the finding directly into the Executor's blank history
            chat_history += f"\n\n>>> SYSTEM ALERT: SEARCH PHASE COMPLETE. TARGETS LOCATED AT: {found_str}. COMMENCE EXECUTION PHASE TO COMPLETE GOAL: {execute_goal} <<<"
            
        if dones[0]:
            print(f"\nGame Over! The Agent finished with a score of: {scores[0]}")
            if scores[0] > 0:
                successful_games += 1
            else:
                failed_games += 1
            game_done = True
            break

    if not game_done:
        print("\nHit max steps. The Agent didn't finish the task in time.")
        failed_games += 1

# --- FINAL SUMMARY ---
accuracy = (successful_games / total_games_played) * 100 if total_games_played > 0 else 0
print("\n\n" + "="*60)
print(" " * 20 + "EVALUATION SUMMARY")
print("="*60)
print(f"Total Games Played : {total_games_played}")
print(f"Successful Games   : {successful_games}")
print(f"Failed/Timed Out   : {failed_games}")
print(f"Overall Accuracy   : {accuracy:.2f}%")
print("="*60 + "\n")