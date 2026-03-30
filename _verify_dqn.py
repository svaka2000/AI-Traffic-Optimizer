"""DQN stability verification after hyperparameter fixes.

Trains a DQN for 500 episodes with the corrected configuration and verifies
that average wait time beats fixed_timing in the single-intersection
training domain (the domain the policy was trained on).
"""
from __future__ import annotations

from traffic_ai.rl_models.environment import EnvConfig, SignalControlEnv
from traffic_ai.rl_models.dqn import train_dqn
from traffic_ai.simulation_engine import SimulatorConfig, TrafficNetworkSimulator
from traffic_ai.controllers.rl_controller import RLPolicyController
from traffic_ai.controllers.fixed_timing import FixedTimingController
from traffic_ai.metrics import simulation_result_to_summary_row

SEED = 42
EPISODES = 500
SIM_STEPS = 500  # 1-intersection training domain (where fixed_timing ≈ 26s)

env = SignalControlEnv(EnvConfig(seed=SEED))
print(
    f"Training DQN for {EPISODES} episodes "
    f"(lr=5e-4, gamma=0.99, warmup=1000, grad_clip=1.0, "
    f"reward_scale=3.0, switch_penalty=0.1) ..."
)
policy, reward_history, _ = train_dqn(env, episodes=EPISODES, seed=SEED)

sim_cfg = SimulatorConfig(steps=SIM_STEPS, intersections=1, seed=SEED)

simulator = TrafficNetworkSimulator(sim_cfg)
dqn_ctrl = RLPolicyController(policy=policy, name="rl_dqn", min_green=6)
result_dqn = simulator.run(dqn_ctrl, steps=SIM_STEPS)
dqn_wait = simulation_result_to_summary_row(result_dqn)["average_wait_time"]

simulator2 = TrafficNetworkSimulator(sim_cfg)
result_fixed = simulator2.run(FixedTimingController(), steps=SIM_STEPS)
fixed_wait = simulation_result_to_summary_row(result_fixed)["average_wait_time"]

print(f"\nfixed_timing  avg_wait = {fixed_wait:.2f}s")
print(f"DQN (500 ep)  avg_wait = {dqn_wait:.2f}s")
if dqn_wait < fixed_wait:
    print(
        f"PASS: DQN beats fixed_timing by {fixed_wait - dqn_wait:.2f}s "
        f"({(fixed_wait - dqn_wait) / fixed_wait * 100:.1f}% reduction)"
    )
else:
    print(f"FAIL: DQN still worse than fixed_timing by {dqn_wait - fixed_wait:.2f}s")
