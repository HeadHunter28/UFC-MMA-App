# UFC Fight Simulator Enhancement Plan
## Reinforcement Learning Approach for Realistic Fight Simulation

This document outlines a comprehensive plan to transform the fight simulator into a more realistic,
RL-based system that learns from historical fight data.

---

## Table of Contents
1. [Current State Analysis](#current-state-analysis)
2. [How UFC Games Approach Simulation](#how-ufc-games-approach-simulation)
3. [Academic Research Insights](#academic-research-insights)
4. [Feature Importance Analysis](#feature-importance-analysis)
5. [Proposed RL Architecture](#proposed-rl-architecture)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Data Requirements](#data-requirements)

---

## Current State Analysis

### What We Have
- Multi-model simulation with statistical, momentum, and stylistic models
- Fighter profiles with comprehensive stats
- Fighter tendencies analysis (striker/wrestler/grappler classification)
- Round-by-round simulation with strikes, takedowns, control time
- Sanity checks for matchup realism
- Realism scoring for simulation outputs

### Limitations
- Probabilistic/heuristic-based rather than learned behavior
- No learning from actual fight outcomes
- Limited style matchup dynamics
- No adaptive opponent modeling
- Static finish probabilities by round

---

## How UFC Games Approach Simulation

Based on research into EA Sports UFC 5:

### 1. Real Impact System
- **Dynamic Damage Accumulation**: Cuts, bruising, swelling in accurate locations
- **Gameplay Impact**: Damage affects fighter attributes (eye cut = lower defense on that side)
- **Stamina Decay**: Realistic stamina drain based on output and damage taken
- **Doctor Stoppages**: Automatic checks when fighters take serious damage

### 2. Fighter Attribute System
UFC games use 100-point scales for individual attributes:

```
STRIKING:
- Stand-up (overall striking skill)
- Punch Speed, Punch Power
- Kick Speed, Kick Power
- Strike Accuracy, Strike Defense

GRAPPLING:
- Takedown Offense, Takedown Defense
- Top Control, Bottom Control
- Submission Offense, Submission Defense
- Clinch Work

PHYSICAL:
- Health (chin durability)
- Stamina, Stamina Recovery
- Speed, Agility
- Strength

INTANGIBLES:
- Heart (comeback ability)
- Fight IQ
- Composure under pressure
```

### 3. Game Style Modes
- **Simulation**: Real-time rounds, increased damage/stamina drain
- **Knockout**: No health regen, unlimited stamina
- **Stand and Bang**: Striking only

### Key Insight
UFC games model fights as a **state machine** where:
- Fighter attributes determine action success probabilities
- Damage accumulates and affects future actions
- Stamina gates what actions are available
- AI difficulty affects reaction time and decision quality

---

## Academic Research Insights

### Key Papers

1. **Creating Pro-Level AI for Fighting Games (2019)**
   - [arXiv:1904.03821](https://arxiv.org/abs/1904.03821)
   - Achieved 62% win rate vs professional gamers
   - Key innovation: **Self-Play Curriculum Learning**
   - Creates agents with different combat styles (attack, defense, balance)
   - Agents compete in shared pool to improve generalization

2. **Mastering Fighting Game with Deep RL and Self-Play (IEEE 2020)**
   - [IEEE Paper](https://ieeexplore.ieee.org/document/9231639/)
   - Uses **PPO (Proximal Policy Optimization)** for stable learning
   - Incorporates **Monte Carlo Tree Search (MCTS)** for planning
   - Handles 0.2 second action delay (simulating human reaction time)

3. **DDQN for Fighting Games (2022)**
   - [Wiley Paper](https://onlinelibrary.wiley.com/doi/10.1155/2022/9984617)
   - Double Deep Q-Network achieved 95% win rate after 2590 training rounds
   - End-to-end system applicable to multiple fighting games

### Key Techniques from Research

```
1. SELF-PLAY TRAINING
   - Agent plays against copies of itself
   - Creates diverse opponent pool with different styles
   - Prevents overfitting to specific strategies

2. HIERARCHICAL RL
   - High-level: Strategy selection (aggressive, defensive, etc.)
   - Low-level: Action selection (which strike, takedown timing)

3. REWARD SHAPING
   - Primary: Win/lose outcome
   - Secondary: Damage dealt, takedowns landed, control time
   - Tertiary: Efficient energy use, successful defense

4. STATE REPRESENTATION
   - Fighter health/stamina
   - Position (standing, clinch, ground)
   - Recent action history
   - Opponent tendencies observed
```

---

## Feature Importance Analysis

Based on MMA prediction research and our data:

### Top Features from Academic Studies

1. **Differential Features** (Fighter A - Fighter B)
   - Significant Strike Accuracy Difference
   - Takedown Defense Difference
   - Reach Advantage
   - Age Difference

2. **Style Matchup Features**
   - Striker vs Wrestler matchup indicator
   - Pressure fighter vs Counter striker
   - Grappler vulnerability indicators

3. **Momentum/Form Features**
   - Win/Loss streak
   - Days since last fight
   - Recent fight duration (cardio indicator)
   - Recent performance trend

4. **Historical Performance**
   - Finish rate
   - Times finished (durability)
   - Performance in later rounds

### Proposed SHAP Analysis Pipeline

```python
# Feature importance using SHAP
import shap

# 1. Train model on historical fights
model = XGBClassifier()
model.fit(X_train, y_train)

# 2. Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 3. Identify top features
shap.summary_plot(shap_values, X_test)

# Key features typically found:
# - Significant strike differential
# - Takedown defense vs opponent's TD offense
# - Recent form (last 3 fights)
# - Experience in championship rounds
# - Finish rate vs times been finished
```

---

## Proposed RL Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FIGHT SIMULATION ENGINE                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Fighter A  │    │    Fight     │    │   Fighter B  │  │
│  │    Agent     │◄───│    State     │───►│    Agent     │  │
│  │   (Policy)   │    │   Manager    │    │   (Policy)   │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │          │
│         │            ┌──────┴───────┐           │          │
│         │            │   Reward     │           │          │
│         └───────────►│   Function   │◄──────────┘          │
│                      └──────┬───────┘                      │
│                             │                              │
│                      ┌──────┴───────┐                      │
│                      │  Experience  │                      │
│                      │    Replay    │                      │
│                      └──────────────┘                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### State Space

```python
@dataclass
class FightState:
    # Time context
    current_round: int  # 1-5
    time_in_round: float  # 0-300 seconds

    # Position
    position: str  # "standing", "clinch", "ground_top", "ground_bottom", "ground_guard"
    cage_position: str  # "center", "cage", "corner"

    # Fighter A state
    fighter_a_health: float  # 0-100
    fighter_a_stamina: float  # 0-100
    fighter_a_damage: Dict[str, float]  # {"head": 0-100, "body": 0-100, "legs": 0-100}
    fighter_a_cuts: List[str]  # ["left_eye", "forehead"]

    # Fighter B state (same structure)
    fighter_b_health: float
    fighter_b_stamina: float
    fighter_b_damage: Dict[str, float]
    fighter_b_cuts: List[str]

    # Round stats so far
    fighter_a_round_strikes: int
    fighter_b_round_strikes: int
    fighter_a_round_takedowns: int
    fighter_b_round_takedowns: int
    fighter_a_round_control: int  # seconds
    fighter_b_round_control: int

    # Recent action history (last 10 actions each)
    fighter_a_action_history: List[str]
    fighter_b_action_history: List[str]
```

### Action Space

```python
class ActionSpace:
    STANDING_ACTIONS = [
        # Strikes
        "jab", "cross", "hook", "uppercut",
        "front_kick", "roundhouse_head", "roundhouse_body", "roundhouse_leg",
        "knee", "elbow",

        # Combos
        "1_2_combo", "1_2_hook", "1_2_kick",

        # Movement
        "advance", "retreat", "circle_left", "circle_right",

        # Defense
        "block_high", "block_low", "head_movement", "check_kick",

        # Grappling initiations
        "single_leg", "double_leg", "body_lock", "clinch_entry",

        # Special
        "feint", "wait"
    ]

    CLINCH_ACTIONS = [
        "dirty_boxing", "knee_clinch", "elbow_clinch",
        "trip", "throw", "cage_press",
        "break_clinch", "defend_takedown"
    ]

    GROUND_TOP_ACTIONS = [
        "ground_and_pound", "advance_position", "submission_attempt",
        "maintain_control", "stand_up"
    ]

    GROUND_BOTTOM_ACTIONS = [
        "sweep", "submission_from_bottom", "wall_walk", "defend",
        "create_space", "stand_up_attempt"
    ]
```

### Reward Function

```python
def calculate_reward(state: FightState, action: str, outcome: Dict) -> float:
    """
    Multi-component reward function for realistic behavior.
    """
    reward = 0.0

    # Primary rewards (end of fight)
    if outcome.get("fight_ended"):
        if outcome.get("winner") == "self":
            reward += 100.0
            if outcome.get("method") == "KO/TKO":
                reward += 20.0  # Bonus for finish
            elif outcome.get("method") == "Submission":
                reward += 20.0
        else:
            reward -= 50.0

    # Secondary rewards (per-action)
    if outcome.get("strike_landed"):
        damage = outcome.get("damage_dealt", 0)
        reward += damage * 0.5  # Reward proportional to damage

    if outcome.get("strike_missed"):
        reward -= 0.1  # Small penalty for missing

    if outcome.get("got_hit"):
        damage = outcome.get("damage_taken", 0)
        reward -= damage * 0.3  # Penalty for taking damage

    if outcome.get("takedown_landed"):
        reward += 5.0

    if outcome.get("takedown_defended"):
        reward += 3.0

    if outcome.get("control_time"):
        reward += outcome.get("control_time") * 0.02  # Per second

    # Efficiency penalties
    stamina_used = outcome.get("stamina_cost", 0)
    reward -= stamina_used * 0.01  # Encourage efficiency

    # Strategic rewards
    if outcome.get("round_won"):
        reward += 10.0

    return reward
```

### Fighter-Specific Policy

```python
class FighterPolicy:
    """
    Fighter-specific policy that encodes their style.
    """

    def __init__(self, fighter_profile: FighterProfile, tendencies: FighterTendencies):
        self.profile = fighter_profile
        self.tendencies = tendencies

        # Initialize action preferences based on fighter style
        self.action_weights = self._initialize_weights()

    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize action weights based on fighter tendencies."""
        weights = {action: 1.0 for action in ActionSpace.STANDING_ACTIONS}

        # Strikers prefer striking actions
        if self.tendencies.primary_style == "Striker":
            for action in ["jab", "cross", "hook", "1_2_combo"]:
                weights[action] *= 1.5

        # Wrestlers prefer takedowns
        if self.tendencies.primary_style == "Wrestler":
            for action in ["single_leg", "double_leg"]:
                weights[action] *= 2.0

        # Power punchers prefer power shots
        if self.tendencies.power_puncher:
            weights["cross"] *= 1.3
            weights["hook"] *= 1.3

        # Volume strikers throw more
        if self.tendencies.volume_striker:
            for strike in ["jab", "1_2_combo", "1_2_kick"]:
                weights[strike] *= 1.4

        # Counter strikers are more patient
        if self.tendencies.counter_striker:
            weights["wait"] *= 1.5
            weights["head_movement"] *= 1.3

        return weights
```

---

## Implementation Roadmap

### Phase 1: Data Enhancement (2-3 weeks)
1. **Enhance fight_stats table**
   - Add round-by-round stats where available
   - Track position changes during fights
   - Record significant moments (knockdowns, near-finishes)

2. **Feature engineering**
   - Implement SHAP analysis on current model
   - Create style matchup features
   - Add head-to-head historical features

3. **Outcome labeling**
   - Label fight outcomes with detail (early KO, late sub, decision)
   - Track momentum shifts in fights

### Phase 2: Simulation Engine Refactor (3-4 weeks)
1. **Implement state machine**
   - Define FightState class
   - Implement position transitions
   - Add damage accumulation system

2. **Action system**
   - Define action space
   - Implement action outcomes based on stats
   - Add stamina costs per action

3. **Basic simulation loop**
   - Time-based simulation (second-by-second or action-by-action)
   - Referee logic (stoppages, position resets)
   - Round scoring

### Phase 3: RL Training Infrastructure (4-6 weeks)
1. **Environment wrapper**
   - OpenAI Gym-compatible environment
   - Observation/action space definitions
   - Reward function implementation

2. **Agent implementation**
   - PPO or DQN base agent
   - Fighter-specific policy initialization
   - Self-play training loop

3. **Training pipeline**
   - Curriculum learning (easy → hard opponents)
   - Style-based opponent pool
   - Model checkpointing and evaluation

### Phase 4: Integration & Testing (2-3 weeks)
1. **Integrate with existing UI**
   - Replace/augment current simulation
   - Add RL-based predictions
   - Show learned insights

2. **Validation**
   - Compare predictions with actual outcomes
   - A/B test against current system
   - Expert review of simulation quality

3. **Iterate**
   - Fine-tune reward function
   - Adjust training parameters
   - Add more nuanced behaviors

---

## Data Requirements

### Minimum Data Needed
- 3000+ UFC fights with round-by-round stats
- Fighter attribute snapshots at time of fight
- Detailed finish information (time, method, damage)

### Ideal Data
- Second-by-second strike data
- Position tracking (available from UFC stats)
- Video analysis features (head movement, footwork)
- Betting line movements (market sentiment)

### Data Sources
1. **UFCStats.com** - Official stats (already integrated)
2. **ESPN MMA** - Additional fight data
3. **Fight Metric** - Detailed striking data
4. **Tapology** - Historical records
5. **The Odds API** - Betting odds (already integrated)

---

## Key Metrics for Success

1. **Prediction Accuracy**
   - Target: 65-70% winner prediction (vs 70% Vegas baseline)
   - Method prediction: 50-55%

2. **Simulation Realism**
   - Expert rating: 4/5 or higher on fight flow
   - Statistical distribution matching actual UFC stats

3. **Learning Quality**
   - Converging training loss
   - Emergent strategic behavior (e.g., leg kick setups)
   - Style-appropriate fight patterns

---

## References

- [EA Sports UFC 5 Gameplay Deep Dive](https://www.ea.com/games/ufc/ufc-5/news/ufc-5-gameplay)
- [Creating Pro-Level AI for Fighting Games (arXiv)](https://arxiv.org/abs/1904.03821)
- [Mastering Fighting Game with Deep RL (IEEE)](https://ieeexplore.ieee.org/document/9231639/)
- [MMA Prediction with Machine Learning (ACM)](https://dl.acm.org/doi/10.1145/3696952.3696966)
- [Data-Driven MMA Prediction (IEEE MLISE 2024)](https://ieeexplore.ieee.org/document/10674447/)

---

## Next Steps

1. Review this plan and prioritize phases
2. Set up experiment tracking (MLflow/Weights & Biases)
3. Create prototype environment with simplified state space
4. Start with DQN on small action space, scale up
5. Integrate self-play training once basic agent works
