# ============================================================================
# MA-DTSR Simulation — Step 4: RL-Augmented Routing
# ============================================================================
# Run after Steps 1, 2, and 3. Call main_step4(net, step2, step3) from Colab.
#
# What this step produces:
#   1. LinearApproximator — theta_i, phi(o,a), linear TD update (eq. 13)
#   2. AgentMemory        — per-agent weights and cooperative exchange (eq. 14)
#   3. RLRouter           — MA-DTSR with RL, supports two modes:
#                             use_wait=False : Forward-only (§3.8.2 Forward branch)
#                             use_wait=True  : Forward + Wait (full §3.8.2)
#   4. Training loop      — episodes, TD updates, rho decay
#   5. Convergence plots  — learning curves
#   6. 5-protocol comparison when Wait is enabled
#
# Action space:
#   Forward(j) : send RSM to neighbour j, consume 1 TTL, 1 energy unit
#   Wait(delta): hold message for delta steps, anticipate better contact
#                — only included when use_wait=True and TTL > delta
#
# All equation references are to Section 3 of the paper.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from tqdm import tqdm


# ============================================================================
# SECTION 1: Linear Function Approximator
# ============================================================================

class LinearApproximator:
    """
    Linear function approximator Q_i(o,a) = theta_i^T * phi(o,a).
    Implements Section 3.8.1 and eq. (13).

    Feature vector phi(o, a) in R^4:
      [0] normalised semantic score        (lower = better)
      [1] normalised remaining TTL
      [2] normalised remaining energy
      [3] wait flag (0 = Forward, 1 = Wait)
    """

    N_FEATURES = 4

    def __init__(self, lr=0.05, gamma=0.9):
        self.lr          = lr
        self.gamma       = gamma
        self.theta       = np.zeros(self.N_FEATURES)
        self.visit_count = 0

    def build_features(self, score, ttl, ttl0, e_rem, e0,
                       is_wait=False):
        """Build phi(o, a) from local state and action type."""
        return np.array([
            score / (score + 1e-6),        # semantic score [0,1)
            ttl   / max(ttl0, 1),          # TTL fraction   [0,1]
            e_rem / max(e0,   1e-6),       # energy fraction[0,1]
            float(is_wait),                # wait flag
        ])

    def q_value(self, phi):
        return float(self.theta @ phi)

    def update(self, phi, reward, phi_next=None):
        """
        Linear TD update (eq. 13):
        theta <- theta + lr * [r + gamma*max_a'Q(s',a') - Q(s,a)] * phi
        """
        q_now   = self.q_value(phi)
        q_next  = self.q_value(phi_next) if phi_next is not None else 0.0
        td_err  = reward + self.gamma * q_next - q_now
        self.theta      += self.lr * td_err * phi
        self.visit_count += 1

    def merge(self, other_theta, alpha_merge=0.3):
        """Cooperative weight merge (eq. 14)."""
        self.theta = ((1 - alpha_merge) * self.theta
                      + alpha_merge * other_theta)

    def copy_weights(self):
        return self.theta.copy()

    def warm_start(self, scores=None):
        """
        Initialise theta so Q ≈ -score (higher score = lower Q-value),
        matching the softmin heuristic direction before any training.
        """
        self.theta = np.array([-1.0, 0.1, 0.1, -0.5])
        # wait flag coefficient -0.5: slight initial penalty on Wait
        # so the agent prefers Forward early in training


# ============================================================================
# SECTION 2: Agent Memory
# ============================================================================

class AgentMemory:
    """One LinearApproximator per agent, with cooperative exchange."""

    def __init__(self, agents, lr=0.05, gamma=0.9):
        self.approximators = {
            a.agent_id: LinearApproximator(lr=lr, gamma=gamma)
            for a in agents
        }
        for approx in self.approximators.values():
            approx.warm_start()

    def get(self, agent_id):
        return self.approximators[agent_id]

    def exchange(self, i_id, j_id, n_min=5, b_q=1, alpha_merge=0.3):
        """
        Cooperative weight exchange (eq. 14, Algorithm 3 lines 14-17).
        Only exchanges if the sending agent has >= n_min updates.
        """
        ai = self.approximators[i_id]
        aj = self.approximators[j_id]
        if ai.visit_count >= n_min:
            aj.merge(ai.copy_weights(), alpha_merge)
        if aj.visit_count >= n_min:
            ai.merge(aj.copy_weights(), alpha_merge)

    def total_updates(self):
        return sum(a.visit_count for a in self.approximators.values())


# ============================================================================
# SECTION 3: Reward Function (Section 3.8.3)
# ============================================================================

REWARD_PARAMS = {
    'U_max'  : 1.0,
    'gamma_r': 0.05,
    'eta_r'  : 0.1,
    'c_hop'  : 0.01,
    'c_wait' : 0.02,    # cost per wait step (TTL + idle energy)
    'c_ttl'  : 0.5,     # penalty for TTL exhaustion
}

def compute_reward(success, hops, energy_used,
                   wait_steps=0, params=None):
    """
    RL reward r(s, a, s') from Section 3.8.3.

    On success  : U_max * exp(-gamma_r*H) * exp(-eta_r*e) - c_hop*H
    On failure  : -c_ttl
    Wait cost   : -c_wait * delta (applied per wait step)
    """
    if params is None:
        params = REWARD_PARAMS
    if success:
        r = (params['U_max']
             * np.exp(-params['gamma_r'] * hops)
             * np.exp(-params['eta_r']   * energy_used))
    else:
        r = -params['c_ttl']
    r -= params['c_hop']  * hops
    r -= params['c_wait'] * wait_steps
    return r


# ============================================================================
# SECTION 4: Contact Frequency Estimator
# ============================================================================

class ContactFrequency:
    """
    Tracks empirical inter-contact times per (agent_i, agent_j) pair.
    Used to estimate expected semantic gain from Wait action (§3.8.2).
    """

    def __init__(self):
        self._last_contact = {}   # (i,j) -> last contact time
        self._intervals    = {}   # (i,j) -> list of intervals

    def record_contact(self, i_id, j_id, t):
        key = (min(i_id, j_id), max(i_id, j_id))
        if key in self._last_contact:
            interval = t - self._last_contact[key]
            if interval > 0:
                self._intervals.setdefault(key, []).append(interval)
        self._last_contact[key] = t

    def mean_interval(self, i_id, j_id):
        """Mean inter-contact time in simulation steps. Returns inf if unknown."""
        key = (min(i_id, j_id), max(i_id, j_id))
        ivs = self._intervals.get(key, [])
        if not ivs:
            return float('inf')
        return float(np.mean(ivs[-10:]))   # sliding window of last 10

    def contact_rate(self, i_id, j_id):
        """Contacts per step (lambda_ij). Returns 0 if unknown."""
        mi = self.mean_interval(i_id, j_id)
        return 0.0 if np.isinf(mi) else 1.0 / mi


# ============================================================================
# SECTION 5: RL Router (Forward + Wait)
# ============================================================================

class RLRouter:
    """
    MA-DTSR with RL — full action space (Forward + Wait).

    When use_wait=False: only Forward actions, matching Step 4 original.
    When use_wait=True : Forward and Wait actions per §3.8.2.

    Wait action:
      - Agent holds message for delta steps (delta in WAIT_DELTAS)
      - Gated by TTL > delta (Property 3, §4.7)
      - Rational when expected semantic gain > c_wait * delta
      - Q-function learns when Wait is worth it from experience
      - During exploration: only Forward is sampled (ensures progress)
      - During exploitation: Wait can be chosen if Q-value is highest

    Parameters
    ----------
    use_wait    : bool  — enable Wait action (False = forward-only)
    wait_deltas : list  — discretised wait durations (Delta, §3.8.2)
    (all other params same as original RLRouter)
    """

    WAIT_DELTAS = [1, 2, 3]    # delta in {1,2,3} steps

    def __init__(self, agents, epsilon=1.0, alpha=0.01, beta=2.0,
                 rho_start=0.8, rho_end=0.05, rho_decay=0.995,
                 lr=0.05, gamma_q=0.9,
                 alpha_merge=0.3, n_min=5,
                 lambda_E=0.0, lambda_A=0.0,
                 use_wait=False):

        self.name        = 'RL-MADTSR-Wait' if use_wait else 'RL-MADTSR'
        self.epsilon     = epsilon
        self.alpha       = alpha
        self.beta        = beta
        self.rho         = rho_start
        self.rho_end     = rho_end
        self.rho_decay   = rho_decay
        self.alpha_merge = alpha_merge
        self.n_min       = n_min
        self.lambda_E    = lambda_E
        self.lambda_A    = lambda_A
        self.use_wait    = use_wait

        self.memory        = AgentMemory(agents, lr=lr, gamma=gamma_q)
        self.contact_freq  = ContactFrequency()
        self.episode_count = 0
        self.training_log  = []

    # ── Scoring ───────────────────────────────────────────────────────────────

    def get_distance(self, query, descriptor, mask=None):
        if mask is None:
            mask = np.ones(len(query))
        return float(np.sum(mask * np.abs(query - descriptor)))

    def check_local_match(self, agent, rsm):
        if not agent.resource or agent.descriptor is None:
            return False, None
        dist = self.get_distance(rsm.query, agent.descriptor, rsm.mask)
        return (True, dist) if dist <= self.epsilon else (False, None)

    def _get_score(self, query, descriptor, timestamp,
                   current_time, mask=None):
        """Budget-aware score from eq. (12)."""
        age = current_time - timestamp
        return (np.exp(self.alpha * age)
                * self.get_distance(query, descriptor, mask)
                + self.lambda_E + self.lambda_A)

    def _forward_candidates(self, agent, rsm, net):
        """
        Returns list of (candidate, score, phi) for all Forward actions.
        """
        ttl0   = rsm.hops + rsm.ttl
        e_rem  = agent.energy
        items  = []
        for j_id in net.neighbours.get(agent.agent_id, []):
            if j_id in rsm.path:
                continue
            cand  = net.agent_map[j_id]
            entry = net.contact_db.get_entry(agent.agent_id, j_id)
            score = (self._get_score(rsm.query, entry['descriptor'],
                                     entry['timestamp'], net.time, rsm.mask)
                     if entry is not None else 10.0)
            approx = self.memory.get(agent.agent_id)
            phi    = approx.build_features(
                score, rsm.ttl, ttl0, e_rem, e_rem, is_wait=False)
            items.append((cand, score, phi))
        return items

    def _wait_actions(self, agent, rsm, net):
        """
        Returns list of (delta, expected_gain, phi) for all Wait actions.
        Only included when use_wait=True and TTL > delta (§4.7 Property 3).

        Expected gain per wait step:
          gain = sum_j lambda_ij * (score_min - score[j])
          where lambda_ij is the contact rate to neighbour j
        """
        if not self.use_wait:
            return []

        ttl0    = rsm.hops + rsm.ttl
        e_rem   = agent.energy
        items   = []

        # Compute min score among current candidates for gain baseline
        neighbours = [j for j in net.neighbours.get(agent.agent_id, [])
                      if j not in rsm.path]
        if not neighbours:
            return []

        scores = []
        for j_id in neighbours:
            entry = net.contact_db.get_entry(agent.agent_id, j_id)
            scores.append(
                self._get_score(rsm.query, entry['descriptor'],
                                entry['timestamp'], net.time, rsm.mask)
                if entry is not None else 10.0)
        score_min = min(scores) if scores else 10.0

        # Expected semantic gain from waiting (contact frequency model)
        expected_gain = 0.0
        for j_id in net.agent_map:
            if j_id == agent.agent_id or j_id in rsm.path:
                continue
            lam = self.contact_freq.contact_rate(agent.agent_id, j_id)
            if lam > 0:
                entry = net.contact_db.get_entry(agent.agent_id, j_id)
                if entry is not None:
                    s = self._get_score(
                        rsm.query, entry['descriptor'],
                        entry['timestamp'], net.time, rsm.mask)
                    expected_gain += lam * max(0.0, score_min - s)

        approx = self.memory.get(agent.agent_id)
        for delta in self.WAIT_DELTAS:
            if rsm.ttl <= delta:          # TTL gate (Property 3)
                continue
            gain_per_step = expected_gain / max(delta, 1)
            phi = approx.build_features(
                max(0.0, score_min - gain_per_step),  # effective score
                rsm.ttl - delta,                       # residual TTL
                ttl0, e_rem, e_rem, is_wait=True)
            items.append((delta, gain_per_step, phi))

        return items

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, agent, rsm, net, is_training=True):
        """
        Select action from full action space A_i(t) (§3.8.2).

        Returns one of:
          ('forward', Agent,  phi, score)   — Forward(j*)
          ('wait',    delta,  phi, gain)    — Wait(delta)
          ('none',    None,   None, None)   — no candidates
        """
        forward_items = self._forward_candidates(agent, rsm, net)
        wait_items    = self._wait_actions(agent, rsm, net)

        if not forward_items and not wait_items:
            return 'none', None, None, None

        # ── Exploration: only sample Forward actions ──────────────────────────
        # During exploration Wait is excluded to ensure routing progress
        # while theta_i is still unreliable (§4.3 lines 16-22)
        if is_training and np.random.random() < self.rho:
            if not forward_items:
                return 'none', None, None, None
            candidates, scores, phis = zip(*forward_items)
            scores  = np.array(scores)
            shifted = scores - scores.min()
            weights = np.exp(-self.beta * shifted)
            weights /= weights.sum()
            idx = np.random.choice(len(candidates), p=weights)
            return ('forward', candidates[idx],
                    phis[idx], float(scores[idx]))

        # ── Exploitation: evaluate Q for all actions ──────────────────────────
        approx  = self.memory.get(agent.agent_id)
        best_q  = -np.inf
        best    = ('none', None, None, None)

        for cand, score, phi in forward_items:
            q = approx.q_value(phi)
            if q > best_q:
                best_q = q
                best   = ('forward', cand, phi, score)

        for delta, gain, phi in wait_items:
            q = approx.q_value(phi)
            if q > best_q:
                best_q = q
                best   = ('wait', delta, phi, gain)

        return best

    # ── Cooperative exchange ──────────────────────────────────────────────────

    def _maybe_exchange(self, agent_i_id, net):
        for j_id in net.neighbours.get(agent_i_id, []):
            # Record contact for frequency estimation
            self.contact_freq.record_contact(agent_i_id, j_id, net.time)
            self.memory.exchange(agent_i_id, j_id,
                                 n_min=self.n_min,
                                 alpha_merge=self.alpha_merge)

    # ── Episode runner ────────────────────────────────────────────────────────

    def run_episode(self, net, step2_module, ttl, rng,
                    alpha=0.01, energy_cost_per_hop=0.02,
                    is_training=True):
        """
        Run one complete routing episode with RL updates.

        Handles both Forward and Wait actions when use_wait=True.
        Returns EpisodeResult_RL.
        """
        source      = net.agents[rng.integers(0, len(net.agents))]
        query, mask = step2_module.generate_query(rng)
        rsm         = RSM(query, mask, ttl, source.agent_id)

        messages_sent = 0
        energy_used   = 0.0
        wait_steps    = 0
        current_agent = source
        trajectory    = []   # list of (agent_id, phi_taken) for TD update

        # Check source immediately
        match, dist = self.check_local_match(current_agent, rsm)
        if match:
            r = compute_reward(True, 0, 0.0)
            u = compute_utility_rl(True, 0, 1, dist, 0.0)
            self._log_and_decay(r, True, is_training)
            return EpisodeResult_RL(True, 0, 1, dist, u, self.name)

        # ── Main routing loop ─────────────────────────────────────────────────
        while rsm.ttl > 0:

            if is_training:
                self._maybe_exchange(current_agent.agent_id, net)

            action_type, target, phi_taken, value = self.select_action(
                current_agent, rsm, net, is_training=is_training)

            # ── No action available ───────────────────────────────────────────
            if action_type == 'none':
                break

            # ── Wait action ───────────────────────────────────────────────────
            elif action_type == 'wait':
                delta       = target
                wait_steps += delta
                # Advance network time by delta steps
                for _ in range(delta):
                    net.step()
                    self._maybe_exchange(current_agent.agent_id, net)
                # Decrement TTL by delta (Property 3: TTL gate already checked)
                # We simulate TTL decrement by reducing rsm.ttl directly
                rsm.ttl -= delta
                energy_used += energy_cost_per_hop * delta * 0.3
                # (idle energy = 30% of transmission energy per step)
                if is_training and phi_taken is not None:
                    trajectory.append(
                        (current_agent.agent_id, phi_taken))
                continue   # re-evaluate action after waiting

            # ── Forward action ────────────────────────────────────────────────
            else:
                next_agent = target
                if is_training and phi_taken is not None:
                    trajectory.append(
                        (current_agent.agent_id, phi_taken))

                rsm           = rsm.copy_to(next_agent.agent_id)
                messages_sent += 1
                energy_used   += energy_cost_per_hop
                current_agent  = next_agent

                match, dist = self.check_local_match(current_agent, rsm)
                if match:
                    total_msgs = messages_sent + 1
                    r = compute_reward(True, rsm.hops, energy_used,
                                       wait_steps)
                    u = compute_utility_rl(
                        True, rsm.hops, total_msgs, dist, energy_used)
                    result = EpisodeResult_RL(
                        True, rsm.hops, total_msgs, dist, u, self.name)
                    if is_training:
                        self._update_trajectory(trajectory, r, None)
                    self._log_and_decay(r, True, is_training)
                    return result

        # ── Episode failed ────────────────────────────────────────────────────
        r = compute_reward(False, rsm.hops, energy_used, wait_steps)
        result = EpisodeResult_RL(
            False, rsm.hops, messages_sent, None, 0.0, self.name)
        if is_training:
            self._update_trajectory(trajectory, r, None)
        self._log_and_decay(r, False, is_training)
        return result

    def _update_trajectory(self, trajectory, terminal_reward, next_phi):
        """Retrospective Monte Carlo credit assignment."""
        for agent_id, phi in reversed(trajectory):
            self.memory.get(agent_id).update(
                phi, terminal_reward, phi_next=next_phi)

    def _log_and_decay(self, reward, success, is_training):
        self.episode_count += 1
        self.training_log.append({
            'episode': self.episode_count,
            'reward' : reward,
            'success': int(success),
            'rho'    : self.rho,
        })
        if is_training:
            self.rho = max(self.rho_end, self.rho * self.rho_decay)


# ============================================================================
# SECTION 6: RSM import (from Step 3)
# ============================================================================

try:
    from MA_DTSR_Step3_Baselines import (
        RSM, EpisodeResult, compute_utility, UTILITY_PARAMS,
        EpidemicRouter, RandomWalkRouter, HeuristicRouter,
        run_comparison, summarise_results,
    )
except ImportError:
    pass   # injected by Colab runner


# ============================================================================
# SECTION 7: Episode Result and Utility (RL version)
# ============================================================================

class EpisodeResult_RL:
    def __init__(self, success, hops, messages,
                 match_error, utility, protocol):
        self.success     = success
        self.hops        = hops
        self.messages    = messages
        self.match_error = match_error
        self.utility     = utility
        self.protocol    = protocol

    def to_dict(self):
        return {
            'protocol'   : self.protocol,
            'success'    : int(self.success),
            'hops'       : self.hops,
            'messages'   : self.messages,
            'match_error': self.match_error,
            'utility'    : self.utility,
        }


def compute_utility_rl(success, hops, messages,
                        match_error, energy_used):
    if not success:
        return 0.0
    from MA_DTSR_Step3_Baselines import compute_utility, UTILITY_PARAMS
    return compute_utility(success, hops, messages,
                           match_error, energy_used, UTILITY_PARAMS)


# ============================================================================
# SECTION 8: Training and Evaluation
# ============================================================================

def train_rl_router(rl_router, net, step2_module,
                    ttl, n_train_episodes, alpha=0.01, seed=123):
    """Train the RL router for n_train_episodes episodes."""
    rng = np.random.default_rng(seed)
    print(f"  Training {'(Forward+Wait)' if rl_router.use_wait else '(Forward-only)'}: "
          f"{n_train_episodes} episodes, TTL={ttl}, "
          f"rho_start={rl_router.rho:.2f}")
    for _ in tqdm(range(n_train_episodes),
                  desc='  Training', unit='ep', leave=False):
        rl_router.run_episode(
            net, step2_module, ttl, rng,
            alpha=alpha, is_training=True)
    log_df = pd.DataFrame(rl_router.training_log)
    print(f"  Done. rho={rl_router.rho:.4f}, "
          f"updates={rl_router.memory.total_updates()}")
    return log_df


def evaluate_rl_router(rl_router, net, step2_module,
                       ttl_values, n_eval_episodes,
                       alpha=0.01, seed=456):
    """Evaluate trained router with greedy policy (no updates)."""
    rng     = np.random.default_rng(seed)
    records = []
    for ttl in ttl_values:
        for _ in range(n_eval_episodes):
            result = rl_router.run_episode(
                net, step2_module, ttl, rng,
                alpha=alpha, is_training=False)
            row        = result.to_dict()
            row['ttl'] = ttl
            records.append(row)
    return pd.DataFrame(records)


# ============================================================================
# SECTION 9: Visualisations
# ============================================================================

PROTOCOL_STYLES_4 = {
    'Epidemic'         : {'color': '#e63946', 'ls': '-',  'marker': 'o'},
    'RandomWalk'       : {'color': '#adb5bd', 'ls': '--', 'marker': 's'},
    'Heuristic-MADTSR' : {'color': '#457b9d', 'ls': '--', 'marker': '^'},
    'RL-MADTSR'        : {'color': '#1d3557', 'ls': '-',  'marker': 'D'},
    'RL-MADTSR-Wait'   : {'color': '#2a9d8f', 'ls': '-',  'marker': 'P'},
}
PROTOCOL_LABELS_4 = {
    'Epidemic'         : 'Epidemic routing',
    'RandomWalk'       : 'Random walk',
    'Heuristic-MADTSR' : 'MA-DTSR (heuristic)',
    'RL-MADTSR'        : 'MA-DTSR (RL, forward-only)',
    'RL-MADTSR-Wait'   : 'MA-DTSR (RL, forward+wait)',
}


def _sty(name, key):
    return PROTOCOL_STYLES_4.get(name, {}).get(key, '#999999')


def plot_learning_curves(log_df, title_suffix='', window=20):
    """Learning curves — smoothed success rate and reward."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'MA-DTSR Step 4: RL Training Curves {title_suffix}',
                 fontsize=11)

    sr_smooth  = log_df['success'].rolling(window, min_periods=1).mean()*100
    rew_smooth = log_df['reward'].rolling(window, min_periods=1).mean()

    ax1.plot(log_df['episode'], sr_smooth, color='#1d3557', linewidth=1.5)
    ax1.fill_between(log_df['episode'],
                     sr_smooth - sr_smooth.std(),
                     sr_smooth + sr_smooth.std(),
                     alpha=0.15, color='#1d3557')
    ax1.set_xlabel('Training episode', fontsize=9)
    ax1.set_ylabel(f'Success rate (%, {window}-ep avg)', fontsize=9)
    ax1.set_title('Success Rate During Training', fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.tick_params(labelsize=8)

    ax2.plot(log_df['episode'], rew_smooth, color='#e63946', linewidth=1.5)
    ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax2b = ax2.twinx()
    ax2b.plot(log_df['episode'], log_df['rho'],
              color='#2a9d8f', linewidth=1.0, linestyle=':', alpha=0.7)
    ax2b.set_ylabel('\u03c1 (exploration)', fontsize=8, color='#2a9d8f')
    ax2b.tick_params(labelsize=7, colors='#2a9d8f')
    ax2b.set_ylim(0, 1)
    ax2.set_xlabel('Training episode', fontsize=9)
    ax2.set_ylabel(f'Reward ({window}-ep avg)', fontsize=9)
    ax2.set_title('Reward During Training', fontsize=10)
    ax2.tick_params(labelsize=8)

    fig.tight_layout()
    fname = f'step4_learning_curves{"_wait" if "Wait" in title_suffix else ""}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure saved: {fname}")


def plot_protocol_comparison(summary_all, fname='step4_comparison.png'):
    """All protocols compared across TTL — 4 panels."""
    metrics = [
        ('success_rate',  'Success rate (%)',   True,  'Success Rate vs TTL'),
        ('mean_hops',     'Mean hops (H)',       False, 'Hop Count vs TTL'),
        ('mean_messages', 'Mean messages (M)',   False, 'Message Count vs TTL'),
        ('mean_utility',  'Mean utility (U_s)', False, 'Mission Utility vs TTL'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('MA-DTSR Step 4: Full Protocol Comparison', fontsize=12)

    for ax, (metric, ylabel, pct, title) in zip(axes.flatten(), metrics):
        for proto, grp in summary_all.groupby('protocol'):
            vals = grp[metric] * 100 if pct else grp[metric]
            ax.plot(grp['ttl'], vals,
                    color=_sty(proto,'color'), ls=_sty(proto,'ls'),
                    marker=_sty(proto,'marker'),
                    markersize=6, linewidth=2,
                    label=PROTOCOL_LABELS_4.get(proto, proto))
        ax.set_xlabel('TTL budget', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure saved: {fname}")


def plot_wait_analysis(df_wait, fname='step4_wait_analysis.png'):
    """
    Wait action analysis — what fraction of RL decisions were Wait,
    and does Wait improve success rate at low TTL?
    """
    if df_wait is None or df_wait.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('MA-DTSR: Wait Action Analysis', fontsize=11)

    # Panel 1: success rate comparison Forward-only vs Forward+Wait
    for proto in ['RL-MADTSR', 'RL-MADTSR-Wait']:
        sub = df_wait[df_wait['protocol'] == proto]
        if sub.empty:
            continue
        grp = sub.groupby('ttl')['success'].mean() * 100
        axes[0].plot(grp.index, grp.values,
                     color=_sty(proto,'color'),
                     ls=_sty(proto,'ls'),
                     marker=_sty(proto,'marker'),
                     label=PROTOCOL_LABELS_4.get(proto, proto),
                     linewidth=1.8)
    axes[0].set_xlabel('TTL budget', fontsize=9)
    axes[0].set_ylabel('Success rate (%)', fontsize=9)
    axes[0].set_title('Forward-Only vs Forward+Wait', fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 105)
    axes[0].tick_params(labelsize=8)

    # Panel 2: message efficiency comparison
    for proto in ['RL-MADTSR', 'RL-MADTSR-Wait']:
        sub = df_wait[df_wait['protocol'] == proto]
        if sub.empty:
            continue
        grp = sub.groupby('ttl')['messages'].mean()
        axes[1].plot(grp.index, grp.values,
                     color=_sty(proto,'color'),
                     ls=_sty(proto,'ls'),
                     marker=_sty(proto,'marker'),
                     label=PROTOCOL_LABELS_4.get(proto, proto),
                     linewidth=1.8)
    axes[1].set_xlabel('TTL budget', fontsize=9)
    axes[1].set_ylabel('Mean messages (M)', fontsize=9)
    axes[1].set_title('Message Efficiency', fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].tick_params(labelsize=8)

    fig.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure saved: {fname}")


# ============================================================================
# SECTION 10: Sanity Checks
# ============================================================================

def run_rl_sanity_checks(log_df, summary_rl, summary_heuristic,
                          use_wait=False):
    label = '(Forward+Wait)' if use_wait else '(Forward-Only)'
    print("=" * 65)
    print(f"  MA-DTSR Step 4 \u2014 RL Sanity Checks {label}")
    print("=" * 65)

    final_rho = log_df['rho'].iloc[-1]
    print(f"\n  [{'PASS' if final_rho < 0.85 else 'WARN'}] "
          f"rho decayed to {final_rho:.4f}")

    sr_1st = log_df['success'].iloc[:len(log_df)//2].mean()
    sr_2nd = log_df['success'].iloc[len(log_df)//2:].mean()
    ok     = sr_2nd >= sr_1st - 0.05
    print(f"  [{'PASS' if ok else 'WARN'}] "
          f"2nd-half SR ({sr_2nd*100:.1f}%) \u2265 "
          f"1st-half SR ({sr_1st*100:.1f}%) \u22125%")

    ttl_max = summary_rl['ttl'].max()
    rl_sr   = summary_rl[
        summary_rl['ttl']==ttl_max]['success_rate'].values
    h_sr    = summary_heuristic[
        summary_heuristic['ttl']==ttl_max]['success_rate'].values
    if len(rl_sr) > 0 and len(h_sr) > 0:
        ok2 = rl_sr[0] >= h_sr[0] - 0.05
        print(f"  [{'PASS' if ok2 else 'WARN'}] "
              f"RL SR ({rl_sr[0]*100:.1f}%) \u2265 "
              f"Heuristic SR ({h_sr[0]*100:.1f}%) at TTL={ttl_max}")

    print(f"\n  --- Performance by TTL ---")
    for _, row in summary_rl.sort_values('ttl').iterrows():
        print(f"    TTL={int(row['ttl'])}: "
              f"SR={row['success_rate']*100:.1f}%, "
              f"msgs={row['mean_messages']:.1f}")
    print("=" * 65)


# ============================================================================
# SECTION 11: Main
# ============================================================================

def main_step4(net, step2_module, step3_module,
               ttl_values=None,
               n_train_episodes=500,
               n_eval_episodes=200,
               epsilon=1.0,
               alpha=0.01,
               beta=2.0,
               rho_start=0.8,
               rho_end=0.05,
               rho_decay=0.995,
               lr=0.05,
               gamma_q=0.9,
               alpha_merge=0.3,
               run_wait=True):
    """
    Run Step 4: train and evaluate RL routers.

    When run_wait=True (default), runs both:
      - Forward-only RL router  (name: RL-MADTSR)
      - Forward+Wait RL router  (name: RL-MADTSR-Wait)
    and produces a Wait action analysis figure comparing the two.

    Parameters
    ----------
    run_wait : bool — also train and evaluate the Wait variant
    (all other params same as before)

    Returns
    -------
    df_all    : pd.DataFrame — all episode results
    summary   : pd.DataFrame — aggregated statistics
    routers   : dict — trained router objects
    """
    if ttl_values is None:
        ttl_values = [10, 20, 30, 40]

    print("MA-DTSR Simulation \u2014 Step 4: RL-Augmented Routing")
    print(f"Network    : N={len(net.agents)} agents, t={net.time:.0f}s")
    print(f"Training   : {n_train_episodes} episodes per TTL")
    print(f"Evaluation : {n_eval_episodes} episodes per TTL")
    print(f"Wait action: {'ENABLED' if run_wait else 'DISABLED'}")
    print(f"TTL range  : {ttl_values}")
    print()

    train_ttl = ttl_values[len(ttl_values) // 2]
    routers   = {}
    all_dfs   = []

    # ── Forward-only RL ───────────────────────────────────────────────────────
    print("Phase 1a: Training forward-only RL router...")
    rl_fwd = RLRouter(
        agents=net.agents, epsilon=epsilon, alpha=alpha,
        beta=beta, rho_start=rho_start, rho_end=rho_end,
        rho_decay=rho_decay, lr=lr, gamma_q=gamma_q,
        alpha_merge=alpha_merge, use_wait=False)
    log_fwd = train_rl_router(
        rl_fwd, net, step2_module, train_ttl,
        n_train_episodes, alpha)
    routers['RL-MADTSR'] = rl_fwd
    print()

    print("Phase 1b: Evaluating forward-only RL router...")
    df_fwd = evaluate_rl_router(
        rl_fwd, net, step2_module, ttl_values,
        n_eval_episodes, alpha)
    all_dfs.append(df_fwd)
    plot_learning_curves(log_fwd, title_suffix='(Forward-Only)')

    # ── Forward+Wait RL ───────────────────────────────────────────────────────
    if run_wait:
        print("\nPhase 2a: Training Forward+Wait RL router...")
        rl_wait = RLRouter(
            agents=net.agents, epsilon=epsilon, alpha=alpha,
            beta=beta, rho_start=rho_start, rho_end=rho_end,
            rho_decay=rho_decay, lr=lr, gamma_q=gamma_q,
            alpha_merge=alpha_merge, use_wait=True)
        log_wait = train_rl_router(
            rl_wait, net, step2_module, train_ttl,
            n_train_episodes, alpha, seed=789)
        routers['RL-MADTSR-Wait'] = rl_wait
        print()

        print("Phase 2b: Evaluating Forward+Wait RL router...")
        df_wait = evaluate_rl_router(
            rl_wait, net, step2_module, ttl_values,
            n_eval_episodes, alpha, seed=321)
        all_dfs.append(df_wait)
        plot_learning_curves(log_wait, title_suffix='(Forward+Wait)')

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\nPhase 3: Running baselines for comparison...")
    baselines = [
        step3_module.EpidemicRouter(epsilon=epsilon),
        step3_module.RandomWalkRouter(epsilon=epsilon),
        step3_module.HeuristicRouter(
            epsilon=epsilon, beta=beta,
            mode='softmin', alpha=alpha),
    ]
    df_base = step3_module.run_comparison(
        net, step2_module, baselines,
        ttl_values=ttl_values,
        n_episodes=n_eval_episodes,
        alpha=alpha)
    all_dfs.append(df_base)

    # ── Combine and summarise ─────────────────────────────────────────────────
    df_all  = pd.concat(all_dfs, ignore_index=True)
    summary = step3_module.summarise_results(df_all)

    print("\n--- Full Results Summary ---")
    print(summary[['protocol','ttl','success_rate',
                   'mean_hops','mean_messages',
                   'mean_utility']
          ].to_string(index=False, float_format='{:.3f}'.format))

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print()
    sum_h = summary[summary['protocol']=='Heuristic-MADTSR']
    sum_fwd = summary[summary['protocol']=='RL-MADTSR']
    run_rl_sanity_checks(log_fwd, sum_fwd, sum_h, use_wait=False)

    if run_wait and 'RL-MADTSR-Wait' in summary['protocol'].values:
        sum_wait = summary[summary['protocol']=='RL-MADTSR-Wait']
        run_rl_sanity_checks(log_wait, sum_wait, sum_h, use_wait=True)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_protocol_comparison(summary)
    if run_wait:
        df_rl_both = df_all[df_all['protocol'].str.startswith('RL')]
        plot_wait_analysis(df_rl_both)

    # ── Save ──────────────────────────────────────────────────────────────────
    df_all.to_csv('step4_raw_results.csv', index=False)
    summary.to_csv('step4_summary.csv', index=False)
    log_fwd.to_csv('step4_training_log_fwd.csv', index=False)
    if run_wait:
        log_wait.to_csv('step4_training_log_wait.csv', index=False)

    print("\nResults saved: step4_raw_results.csv, step4_summary.csv")
    return df_all, summary, routers


# ── Entry point guard ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    raise RuntimeError(
        "Import this module and call main_step4(net, step2, step3)."
    )
