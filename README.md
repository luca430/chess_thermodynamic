# Thermodynamic Chess

`thermo_chess` is a toy framework for studying decision making in chess through a thermodynamic/statistical-mechanics analogy.

The central idea is that a player does **not** choose a move directly from a handcrafted chess score. Instead, each player defines a **subjective probability measure over legal moves**, uses that measure to construct a shallow expected landscape, and then chooses the move that maximizes its expected subjective benefit after a bounded, entropy-adaptive, adversarial search.

The framework separates four objects that should not be confused:

1. **Universal static board evaluation** `E(B)`: a White-positive description of a board.
2. **Player-dependent move potential** `Phi_p(m,B)`: how attractive a move looks to player `p` according to that player's style.
3. **Subjective shallow landscape** `U_p(B)`: the expected static value of the next positions under the player's Boltzmann move distribution.
4. **Recursive search gain** `G_m`: the change in subjective landscape produced by a candidate move after subjective minimax search.

The code also decomposes changes between compatible subjective landscapes into finite **Q/W/A** terms analogous to redistribution (`Q`), change of observable values (`W`), and change of accessible support (`A`).

## Contents

- [1. Installation](#1-installation)
- [2. Conceptual overview](#2-conceptual-overview)
- [3. Universal static board evaluation `E(B)`](#3-universal-static-board-evaluation-eb)
- [4. Player-specific move features](#4-player-specific-move-features)
- [5. Game phase and effective style weights](#5-game-phase-and-effective-style-weights)
- [6. Subjective move potential](#6-subjective-move-potential)
- [7. Boltzmann move probability](#7-boltzmann-move-probability)
- [8. Entropy and effective number of moves](#8-entropy-and-effective-number-of-moves)
- [9. Shallow subjective landscape `U_p(B)`](#9-shallow-subjective-landscape-upb)
- [10. Preset player styles](#10-preset-player-styles)
- [11. Entropy-adaptive deepening](#11-entropy-adaptive-deepening)
- [12. Cycle depth: `cdepth`](#12-cycle-depth-cdepth)
- [13. Subjective minimax recursion](#13-subjective-minimax-recursion)
- [14. Candidate search gain `G_m`](#14-candidate-search-gain-gm)
- [15. Partial adaptive deepening](#15-partial-adaptive-deepening)
- [16. Finite Q/W/A decomposition](#16-finite-qwa-decomposition)
- [17. QWA for hypothetical searched candidates](#17-qwa-for-hypothetical-searched-candidates)
- [18. Realized full-cycle QWA during an actual game](#18-realized-full-cycle-qwa-during-an-actual-game)
- [19. Search diagnostics](#19-search-diagnostics)
- [20. Command-line workflow](#20-command-line-workflow)
- [21. Running simulations: `run_match.py`](#21-running-simulations-run_matchpy)
- [22. Saved simulation outputs](#22-saved-simulation-outputs)
- [23. Quick terminal inspection: `analyze_match.py`](#23-quick-terminal-inspection-analyze_matchpy)
- [24. Static response covariance analysis: `analyze_static_covariances.py`](#24-static-response-covariance-analysis-analyze_static_covariancespy)
- [25. Deep-search divergence analysis: `analyze_deep_search_divergence.py`](#25-deep-search-divergence-analysis-analyze_deep_search_divergencepy)
- [26. Visualizing saved matches: `view_match.py`](#26-visualizing-saved-matches-view_matchpy)
- [27. Search benchmark: `benchmark_adaptive.py`](#27-search-benchmark-benchmark_adaptivepy)
- [28. Move-feature scale diagnostics: `diagnose_feature_scales.py`](#28-move-feature-scale-diagnostics-diagnose_feature_scalespy)
- [29. Recommended end-to-end workflow](#29-recommended-end-to-end-workflow)
- [30. Module map](#30-module-map)
- [31. Core equations at a glance](#31-core-equations-at-a-glance)
- [Current scope](#current-scope)

---

## 1. Installation

The project is a Python package with command-line helper scripts in the repository root. The declared dependencies are:

```text
python-chess>=1.999
pytest>=8.0
```

A recent Python 3 environment is recommended.

From the repository root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade `pip` and install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run commands from the repository root so that the local `thermo_chess/` package and the default `data/...` paths resolve correctly.

Quick import test:

```bash
python -c "import thermo_chess; print('thermo_chess imported successfully')"
```

Run the test suite, when present, with:

```bash
pytest
```

## 2. Conceptual overview

Let `B` be a chess board and let `M(B)` be the set of legal moves.

For a given player `p`, the framework proceeds as

```text
board B
  |
  |-- universal board evaluation E(B)
  |
  |-- player-specific move features F_k(m,B)
  |       |
  |       -> subjective potential Phi_p(m,B)
  |                |
  |                -> Boltzmann probabilities P_p(m|B)
  |                         |
  |                         -> entropy S_p(B)
  |                         -> effective number N_eff
  |                         -> adaptive breadth K
  |
  -> shallow subjective landscape U_p(B)
           |
           -> cycle-depth subjective minimax search
                    |
                    -> endpoint value for candidate m
                    -> G_m
                    -> optional Q/W/A decomposition
```

White and Black use the same universal static board evaluator, but they may use different subjective styles and different inverse temperatures `beta`.

A critical design choice is:

> During a player's search, the entire hypothetical tree is evaluated using that player's own subjective potential.

White therefore does **not** switch to Black's style when imagining Black's replies. White can only reason using White's own model. Black behaves symmetrically with its own model.

The result is a **subjective minimax**: the alternating max/min structure is adversarial, but each player's internal tree is built from that player's own subjective measure.

---

# 3. Universal static board evaluation `E(B)`

The universal evaluator is defined in `thermo_chess/evaluation.py` and `thermo_chess/features.py`.

It is always **White-positive**:

- positive values favor White;
- negative values favor Black.

For non-terminal boards,

$$
E(B)
=\alpha_M\Delta M
+\alpha_P\Delta P
+\alpha_L\Delta L
+\alpha_C\Delta C
+\alpha_K\Delta K.
$$

The default evaluation weights are all one:

```python
EvaluationWeights(
    material=1.0,
    pawn_structure=1.0,
    mobility=1.0,
    center=1.0,
    king_safety=1.0,
    checkmate=10000.0,
)
```

Thus the default non-terminal value is simply the sum of the five White-minus-Black components.

## 3.1 Material

Piece values are

| Piece | Value |
|---|---:|
| Pawn | 1 |
| Knight | 3 |
| Bishop | 3 |
| Rook | 5 |
| Queen | 9 |
| King | 0 |

For color `c`,

$$
M_c(B)=\sum_t n_{c,t}v_t,
$$

and

$$
\Delta M=M_W-M_B.
$$

## 3.2 Pawn structure

For each side,

$$
P_c
=0.30\,N_{\rm passed}
+0.10\,N_{\rm connected}
-0.15\,N_{\rm isolated}
-0.15\,N_{\rm doubled}.
$$

Then

$$
\Delta P=P_W-P_B.
$$

The code defines:

- **isolated pawn**: no friendly pawn on either adjacent file;
- **connected pawn**: at least one friendly pawn on an adjacent file within one rank;
- **passed pawn**: no enemy pawn ahead on the same or adjacent files;
- **doubled pawn**: every pawn beyond the first on a file contributes one doubled-pawn count.

## 3.3 Mobility

For each color,

$$
L_c=0.03\,N_{\rm legal}(c),
$$

so

$$
\Delta L=L_W-L_B.
$$

The implementation evaluates legal moves for each color independently by temporarily setting the side to move.

## 3.4 Center control

The four central squares are

```text
d4, e4, d5, e5
```

and the extended center is

```text
c3 d3 e3 f3
c4          f4
c5          f5
c6 d6 e6 f6
```

The raw center-control score of color `c` is

$$
C_c^{\rm raw}
=N_{\rm attacked\ central}
+0.35\,N_{\rm attacked\ extended}.
$$

The pawn-equivalent score is

$$
C_c=0.10\,C_c^{\rm raw},
$$

and

$$
\Delta C=C_W-C_B.
$$

## 3.5 King safety

For a king of color `c`, let its king zone be the adjacent king-move squares.

The raw king-safety score is

$$
K_c^{\rm raw}
=N_{\rm shield}
-\frac{5}{6}N_{\rm enemy\ controlled\ zone}
-1.25N_{\rm unshielded\ files}.
$$

The final score is

$$
K_c=0.12\,K_c^{\rm raw},
$$

and

$$
\Delta K=K_W-K_B.
$$

`N_shield` counts friendly pawns in the king zone. `N_unshielded files` counts files around the king for which no friendly pawn exists forward of the king.

## 3.6 Terminal boards

The static evaluator overrides the ordinary feature sum at terminal positions:

- White checkmated: `E=-10000`;
- Black checkmated: `E=+10000`;
- stalemate: `0`;
- insufficient material: `0`;
- automatic 75-move draw: `0`;
- automatic fivefold repetition: `0`.

---

# 4. Player-specific move features

The **board evaluator** and the **player's move potential** are different objects.

The potential does not use the five static components directly. Instead each legal move is described, from the perspective of the mover, by six pawn-valued move features:

$$
\mathbf F(m,B)
=(F_M,F_C,F_D,F_{Ca},F_K,F_{KP}).
$$

These correspond to:

1. material;
2. center;
3. development;
4. castling;
5. king safety;
6. king pressure.

All are converted to approximately pawn-like units before being combined.

## 4.1 Material feature

For mover `p`,

$$
F_M(m,B)
=M_p(B_m)-M_p(B),
$$

where `M_p` is material balance from the mover's perspective.

A capture therefore produces a positive material feature for the capturing side.

## 4.2 Center feature

$$
F_C(m,B)
=0.10\left[C_p^{\rm raw}(B_m)-C_p^{\rm raw}(B)\right].
$$

## 4.3 Development feature

The initial minor-piece slots are `b1,c1,f1,g1` for White and `b8,c8,f8,g8` for Black.

A slot counts as developed if it no longer contains the expected original minor piece.

$$
F_D(m,B)
=0.15\,\Delta N_{\rm developed\ minor\ slots}.
$$

## 4.4 Castling feature

A castling move contributes

$$
+0.40.
$$

Losing a castling right without castling contributes

$$
-0.20
$$

for each lost right.

## 4.5 King-safety feature

$$
F_K(m,B)
=0.12\left[K_p^{\rm raw}(B_m)-K_p^{\rm raw}(B)\right].
$$

## 4.6 King-pressure feature

The code counts squares in the enemy king zone that are not occupied by an enemy piece and are not attacked by the moving side. This is interpreted as the enemy king's local freedom.

The move feature is

$$
F_{KP}(m,B)
=0.10\left[N_{\rm safe,enemy}(B)-N_{\rm safe,enemy}(B_m)\right].
$$

Thus reducing the opponent king's safe local squares gives positive king pressure.

---

# 5. Game phase and effective style weights

Each player has raw non-negative style coefficients

$$
\lambda_k.
$$

The game-phase parameter is based on remaining non-pawn material:

$$
g(B)
=1-\frac{M_{\rm nonpawn}(B)}{62},
$$

clamped to `[0,1]`.

Therefore:

- opening/full material: `g≈0`;
- late endgame: `g→1`.

The phase factors are

$$
f_M=1,
\qquad
f_{KP}=1,
$$

and

$$
f_C=f_D=f_{Ca}=f_K=1-g.
$$

The phase-effective style coefficients are normalized:

$$
\lambda^{\rm eff}_{p,k}(B)
=
\frac{\lambda_{p,k}f_k(g)}
{\sum_j\lambda_{p,j}f_j(g)}.
$$

Thus

$$
\sum_k\lambda^{\rm eff}_{p,k}=1.
$$

The phase modulation gradually removes center, development, castling, and king-safety preferences as non-pawn material disappears, while material and king pressure remain active.

---

# 6. Subjective move potential

For player `p`, legal move `m`, and board `B`,

$$
\boxed{
\Phi_p(m,B)
=
\sum_k\lambda^{\rm eff}_{p,k}(B)F_k(m,B)
}
$$

where the six `F_k` are the mover-relative features above.

This potential is **not** the universal board value `E(B_m)`.

`Phi` controls the subjective probability measure over moves; `E` supplies the observable whose expectation defines the shallow landscape.

The JSON output stores diagnostic information including:

- raw features;
- conversion scales;
- pawn-valued features;
- raw style lambdas;
- phase factors;
- effective lambdas;
- per-feature contributions to `Phi`;
- base potential;
- phase contribution;
- total potential.

---

# 7. Boltzmann move probability

The player converts potentials into a normalized probability distribution:

$$
\boxed{
P_p(m\mid B)
=
\frac{\exp[\beta_p\Phi_p(m,B)]}
{\sum_{m'\in\mathcal M(B)}\exp[\beta_p\Phi_p(m',B)]}
}
$$

where `beta_p > 0` is the player's inverse temperature.

The implementation uses a numerically stable softmax.

The optional thermodynamic notation is

$$
\beta=\frac{1}{\kappa T},
$$

with default

```python
KAPPA = 1.0
```

so

$$
T=\frac{1}{\kappa\beta}.
$$

Interpretation:

- larger `beta`: probability mass concentrates on moves with high subjective potential;
- smaller positive `beta`: the measure becomes flatter;
- `P_p` describes the player's subjective local move landscape; it is **not** itself the final deterministic move-selection rule.

---

# 8. Entropy and effective number of moves

For the move probabilities,

$$
\boxed{
S_p(B)
=-\sum_mP_p(m\mid B)\ln P_p(m\mid B)
}
$$

using natural logarithms.

The effective number of moves is

$$
\boxed{
N_{\rm eff}(B)=e^{S_p(B)}.
}
$$

This is the number of equally probable moves that would have the same entropy.

Examples:

- one move has probability almost one: `N_eff≈1`;
- `N` exactly equiprobable legal moves: `N_eff=N`.

---

# 9. Shallow subjective landscape `U_p(B)`

The shallow subjective landscape is

$$
\boxed{
U_p(B)
=
\sum_{m\in\mathcal M(B)}P_p(m\mid B)E(B_m).
}
$$

This is a complete normalized expectation over all legal moves.

Important distinctions:

- `E(B)` describes one board;
- `Phi_p(m,B)` describes how attractive one move is to player `p`;
- `P_p(m|B)` is the subjective measure induced by `Phi_p`;
- `U_p(B)` is the expected static value under that measure.

Because `E` is White-positive:

- White benefits from increasing `U`;
- Black benefits from decreasing `U`.

---

# 10. Preset player styles

`simulation.py` defines four presets.

| Strategy | material | center | development | castling | king safety | king pressure |
|---|---:|---:|---:|---:|---:|---:|
| `material_conservative` | 2.4 | 0.5 | 0.6 | 0.9 | 1.0 | 0.3 |
| `pressure_aggressive` | 0.9 | 1.1 | 0.6 | 0.4 | 0.4 | 1.8 |
| `tactical_attacker` | 1.0 | 1.4 | 0.5 | 0.3 | 0.5 | 2.2 |
| `positional_controller` | 1.2 | 1.8 | 0.9 | 0.8 | 1.1 | 0.6 |

These are raw coefficients. They are phase-modulated and normalized before entering `Phi`.

Custom styles can be supplied directly with `Style(...)`.

---

# 11. Entropy-adaptive deepening

The search does not recursively expand every legal move to the same depth.

At every searched board, the number of branches selected for recursive refinement is

$$
\boxed{
K
=
\min\left(
N_{\rm legal},
\max\left[1,\left\lceil cN_{\rm eff}\right\rceil\right]
\right)
}
$$

where `c = adaptive_c > 0`.

The selected branches are the `K` moves with the largest **original Boltzmann probabilities** `P_p(m|B)`.

The probabilities are not renormalized over this selected subset.

The role of `K` is computational/attentional:

> high-probability subjective moves receive deeper recursive analysis; other moves terminate at their current shallow landscape.

At the next recursive node, a new entropy and a new effective number are computed, generating a new breadth `K'`. Further levels similarly produce `K''`, etc.

Thus different branches can terminate at different effective depths.

## 11.1 Meaning of `adaptive_c`

Default:

```python
adaptive_c = 0.3
```

Smaller values deepen fewer branches. Larger values deepen more branches.

Because

$$
K\sim \lceil cN_{\rm eff}\rceil,
$$

high-entropy positions automatically receive broader search than strongly concentrated positions.

---

# 12. Cycle depth: `cdepth`

The public search horizon is measured in complete **move-response cycles**, not raw plies.

A cycle consists of a candidate action and the opponent response.

```text
cdepth = 1:  m -> r
cdepth = 2:  m -> r -> m' -> r'
cdepth = 3:  m -> r -> m' -> r' -> m'' -> r''
```

For a root candidate move `m`, the root move itself is already known when recursion starts. Therefore the number of additional recursive plies is

$$
\boxed{
n_{\rm recursive}=2\,\texttt{cdepth}-1.
}
$$

For example:

| `cdepth` | Recursive plies after root candidate | Total plies in a fully explored candidate trajectory |
|---:|---:|---:|
| 1 | 1 | 2 |
| 2 | 3 | 4 |
| 3 | 5 | 6 |

`cdepth` must be a positive integer.

---

# 13. Subjective minimax recursion

The probability measure defines the shallow landscape and adaptive breadth, but recursive backup is **not** a probability-weighted expectation.

Fix one search owner `p`. Define

$$
F_p(B,0)=U_p(B).
$$

At positive remaining recursive depth, each legal move `a` is assigned an endpoint value

$$
X_a=
\begin{cases}
F_p(B_a,n-1), & a\text{ selected among the adaptive top-}K,\\[4pt]
U_p(B_a), & a\text{ not selected for deeper refinement}.
\end{cases}
$$

The node backup is then

$$
\boxed{
F_p(B,n)=
\begin{cases}
\max_a X_a, & \text{White to move},\\[4pt]
\min_a X_a, & \text{Black to move}.
\end{cases}
}
$$

The sign convention comes from the White-positive static observable.

## 13.1 Why this is subjective minimax

A White search uses White's own

$$
\Phi_W,\quad P_W,\quad U_W
$$

at **every board in White's hypothetical tree**, including boards where Black is to move.

Likewise Black's tree is built entirely from

$$
\Phi_B,\quad P_B,\quad U_B.
$$

Therefore White is not predicting what Black subjectively wants according to Black's style. White asks:

> Given my own model of the game, what is the worst response that can occur?

Black performs the symmetric computation using Black's own subjective model.

The search is thus minimax-like in structure but player-dependent in its internal landscape.

---

# 14. Candidate search gain `G_m`

For a root position `B`, let candidate `m` lead through the subjective minimax search to search endpoint `B_search(m)`.

The candidate score is

$$
\boxed{
G_m^{\rm search}
=
U_p(B_{\rm search}(m))-U_p(B).
}
$$

In serialized output this is

```text
g_tilde
```

and the corresponding endpoint landscape is stored as

```text
terminal_u
```

with

$$
\boxed{
\texttt{g_tilde}
=
\texttt{terminal_u}-U_p(B).
}
$$

Actual move choice is deterministic:

### White

$$
\boxed{
m^*=\arg\max_m G_m^{\rm search}}
$$

### Black

$$
\boxed{
m^*=\arg\min_m G_m^{\rm search}}
$$

The move probabilities `P_p` therefore define subjective attention and the local landscape, but the final move is chosen using `G_m`.

---

# 15. Partial adaptive deepening

Adaptive search may terminate a principal variation before the requested cycle depth.

The code records both:

```text
requested_cycles = cdepth
deepened_cycles
```

`deepened_cycles` is the number of complete move-response cycles actually reached along the selected principal variation.

Examples for `cdepth=3`:

```text
Deepening: 0 / 3 cycles
Deepening: 1 / 3 cycles
Deepening: 2 / 3 cycles
Deepening: 3 / 3 cycles
```

It also records

```text
recursive_plies_used
principal_variation
search_endpoint_fen
thermodynamic_endpoint_fen
```

A branch may therefore have a valid search score while having no thermodynamic Q/W/A decomposition.

---

# 16. Finite Q/W/A decomposition

Consider two **complete normalized move landscapes** belonging to the same subjective player and the same side to move.

Let the old landscape have support `S_0`, probabilities `p_i`, and observables `E_i`, and the new landscape have support `S_1`, probabilities `p'_i`, and observables `E'_i`.

Here the observable associated with move `i` is the static value of the resulting board.

The landscape values are

$$
U_0=\sum_{i\in S_0}p_iE_i,
\qquad
U_1=\sum_{i\in S_1}p'_iE'_i.
$$

The total change is

$$
\Delta U=U_1-U_0.
$$

Let

$$
S_c=S_0\cap S_1
$$

be the common support.

The finite decomposition implemented in `thermodynamics.py` is

$$
\boxed{
\Delta U=\Delta Q+\Delta W+\Delta A.
}
$$

## 16.1 Redistribution term `Q`

$$
\boxed{
\Delta Q
=
\sum_{i\in S_c}
\frac{E_i+E'_i}{2}
\left(p'_i-p_i\right).
}
$$

Interpretation: `Q` measures the change due to **redistribution of probability mass** over moves that exist in both landscapes, using the midpoint observable value.

In the thermodynamic analogy, this is the heat-like contribution: the accessible common states remain present, but their statistical weights change.

## 16.2 Observable-change term `W`

$$
\boxed{
\Delta W
=
\sum_{i\in S_c}
\frac{p_i+p'_i}{2}
\left(E'_i-E_i\right).
}
$$

Interpretation: `W` measures how the values attached to common moves change while averaging over their old/new probability weights.

This is the work-like contribution.

## 16.3 Accessibility/support term `A`

Moves can disappear from or enter the legal-move support between landscapes. The accessibility term is

$$
\boxed{
\Delta A
=
\sum_{i\in S_1\setminus S_0}p'_iE'_i
-
\sum_{i\in S_0\setminus S_1}p_iE_i.
}
$$

Interpretation: `A` captures changes caused by **appearance and disappearance of accessible moves**.

## 16.4 Numerical decomposition error

The code records

$$
\epsilon
=
\Delta U-(\Delta Q+\Delta W+\Delta A).
$$

This should be close to zero up to floating-point error.

The code uses the actual independently calculated `delta_a`; `A` is not defined as the residual needed to force the identity.

---

# 17. QWA for hypothetical searched candidates

Q/W/A is only meaningful when the candidate's selected principal variation has completed at least one real move-response cycle and therefore returns to a board with the same side to move as the root.

For root board `B_0`, the search may produce a path such as

```text
B0 -> B1 -> B2 -> B3
```

where:

- `B0` and `B2` have the same side to move;
- `B3` has the opposite side to move.

Then

```text
search_endpoint = B3
thermodynamic_endpoint = B2
```

The search score is

$$
G_m^{\rm search}=U_p(B_3)-U_p(B_0),
$$

but the thermodynamic change is

$$
\boxed{
G_m^{\rm thermo}
=U_p(B_2)-U_p(B_0).
}
$$

The code stores this as

```text
thermo_g_tilde
```

and decomposes

$$
\boxed{
G_m^{\rm thermo}
=Q_m+W_m+A_m.
}
$$

Therefore `g_tilde` and `thermo_g_tilde` are not always identical.

They are identical for a fully deepened candidate whose search endpoint already lies on a completed cycle boundary.

## 17.1 No completed cycle

If

```text
deepened_cycles = 0
```

then the code intentionally stores

```text
thermo_g_tilde = null
q_tilde = null
w_tilde = null
a_tilde = null
```

because there is no same-turn landscape pair with a thermodynamic interpretation.

The viewer displays these unavailable quantities as `—`, not zero.

---

# 18. Realized full-cycle QWA during an actual game

The simulation logs a second, conceptually distinct QWA decomposition: the **realized full shallow same-player transition**.

Suppose White is to move at board `B_t`. After White's actual move and Black's actual response, White is to move again at `B_{t+2}`:

```text
B_t --White actual move--> B_{t+1} --Black actual response--> B_{t+2}
```

The simulation reconstructs White's complete shallow subjective landscapes at `B_t` and `B_{t+2}` using White's own style and computes

$$
\boxed{
U_W(B_{t+2})-U_W(B_t)
=
Q_W^{\rm real}+W_W^{\rm real}+A_W^{\rm real}.
}
$$

Black receives its own analogous decomposition between successive Black-to-move boards.

This is recorded with

```text
decomposition_type = "realized_full_shallow_same_player_transition"
```

and fields such as

```text
U_old
U_new
realized_delta_u
realized_delta_q
realized_delta_w
realized_delta_a
N_eff_old
N_eff_new
old_support_size
new_support_size
common_support_size
common_mass_old
common_mass_new
old_only_mass
new_only_mass
decomposition_error
```

This realized-cycle decomposition is different from the candidate QWA:

| Quantity | Candidate QWA | Realized full-cycle QWA |
|---|---|---|
| Path | hypothetical subjective-minimax principal variation | moves actually played |
| Endpoint | deepest completed cycle on candidate search path | next real board where same player is to move |
| Used for move choice? | no; `g_tilde` is used | no; retrospective diagnostic |
| Landscape | shallow subjective landscape | shallow subjective landscape |
| Identity | `thermo_g_tilde = q_tilde + w_tilde + a_tilde` | `realized_delta_u = realized_delta_q + realized_delta_w + realized_delta_a` |

The simulation also logs one-ply realized static changes for the actual action and response. These are changes of the universal static board value `E`, not the QWA decomposition itself.

---

# 19. Search diagnostics

The search records performance and structural diagnostics including:

- `nodes_evaluated`;
- `static_evaluations`;
- `recursive_nodes`;
- `maximum_depth_reached`;
- average adaptive `K`;
- cache hits;
- search elapsed time;
- requested cycle depth;
- requested recursive plies;
- branch probability masses;
- response entropy;
- response `N_eff`;
- response `K`;
- refinement rank;
- whether a response was selected for refinement;
- whether it was selected by the recursive max/min policy.

The detailed JSON is the main source for inspecting why a move received a particular search value.

---

# 20. Command-line workflow

The repository includes command-line scripts for the complete practical workflow:

```text
run_match.py
    -> data/games/{json,csv,pgn}/...

analyze_match.py
    -> quick terminal inspection of one saved ply

analyze_static_covariances.py
    -> data/analysis/covariance/<game>.csv
       [optional detailed JSON]

analyze_deep_search_divergence.py
    -> data/analysis/deep_search_divergence/<game>.csv

view_match.py
    -> browser-based interactive viewer

benchmark_adaptive.py
    -> search-performance diagnostics

diagnose_feature_scales.py
    -> move-feature scale diagnostics
```

All commands below assume that they are executed from the repository root, i.e. the directory containing `thermo_chess/` and the scripts.

---

# 21. Running simulations: `run_match.py`

The standard way to run a match is

```bash
python run_match.py
```

With no flags, the script uses:

```text
max_plies             = 80
seed                  = 1
beta_white            = 4.0
beta_black            = 4.0
kappa                 = 1.0
white_strategy        = material_conservative
black_strategy        = pressure_aggressive
cdepth                = 2
adaptive_c            = 0.3
viewer_workers        = 1
search_workers        = 1
parallel_min_branches = 8
name                  = automatically generated
```

By default, games are written below

```text
data/games/
```

and the script prints the paths of the generated CSV, JSON, and PGN files.

## 21.1 Simulation flags

### `--max-plies N`

Maximum number of half-moves played before the simulation stops.

Default:

```text
80
```

Example:

```bash
python run_match.py --max-plies 160
```

### `--seed N`

Random seed stored in the match configuration.

Default:

```text
1
```

Example:

```bash
python run_match.py --seed 42
```

The current move-selection rule is deterministic once the search values are known, but keeping a seed in the simulation configuration makes stochastic or randomized extensions reproducible.

### `--beta-white X`, `--beta-black X`

Inverse-temperature parameters for White's and Black's subjective Boltzmann distributions.

Defaults:

```text
beta_white = 4.0
beta_black = 4.0
```

The move distribution has the form

$$
P_p(m\mid B)
=\frac{\exp[\beta_p\Phi_p(m,B)]}
{\sum_{m'}\exp[\beta_p\Phi_p(m',B)]}.
$$

Larger `beta` concentrates probability more strongly on high-potential moves; smaller `beta` produces a flatter subjective landscape.

Example:

```bash
python run_match.py --beta-white 5.0 --beta-black 3.0
```

### `--kappa X`

Positive Boltzmann-like constant used to map the supplied inverse temperature to the reported thermodynamic temperature,

$$
T=\frac{1}{\kappa\beta}.
$$

Default:

```text
1.0
```

The CLI supplies `beta` directly. In the current implementation the softmax exponent is `beta * Phi`; changing `kappa` therefore changes the associated reported temperature but does not independently rescale the move probabilities when `beta` is explicitly provided.

Example:

```bash
python run_match.py --kappa 0.8
```

### `--white-strategy NAME`, `--black-strategy NAME`

Select the style preset used by each player.

The accepted values are whatever is exported by `thermo_chess.simulation.STRATEGY_NAMES`; in the current implementation these are:

```text
material_conservative
pressure_aggressive
tactical_attacker
positional_controller
```

Defaults:

```text
White: material_conservative
Black: pressure_aggressive
```

Example:

```bash
python run_match.py \
    --white-strategy material_conservative \
    --black-strategy positional_controller
```

### `--cdepth N`

Requested search depth measured in **complete move-response cycles**.

Default:

```text
2
```

The meaning is

```text
cdepth = 1:  m -> r
cdepth = 2:  m -> r -> m' -> r'
cdepth = 3:  m -> r -> m' -> r' -> m'' -> r''
```

For a root candidate already pushed on the board, the internal recursive search therefore receives

$$
n_{\rm recursive}=2\,\texttt{cdepth}-1
$$

additional plies.

Example:

```bash
python run_match.py --cdepth 3
```

`cdepth` must be at least 1.

### `--adaptive-c X`

Positive breadth coefficient controlling entropy-adaptive refinement.

Default:

```text
0.3
```

At each node,

$$
K
=\min\!\left(
N_{\rm legal},
\max\left[1,\left\lceil cN_{\rm eff}\right\rceil\right]
\right),
$$

where `c = adaptive_c` and

$$
N_{\rm eff}=e^S.
$$

Larger values recursively deepen more moves; smaller values concentrate computation on fewer high-probability branches.

Example:

```bash
python run_match.py --adaptive-c 0.5
```

### `--search-workers N|auto`

Number of worker processes available for root-branch search.

Default:

```text
1
```

Use serial execution:

```bash
python run_match.py --search-workers 1
```

Use the automatically selected CPU count:

```bash
python run_match.py --search-workers auto
```

### `--parallel-min-branches N`

Minimum number of root branches required before process-based search parallelism is activated.

Default:

```text
8
```

Example:

```bash
python run_match.py --search-workers 4 --parallel-min-branches 6
```

### `--viewer-workers N`

Compatibility option retained by the simulation interface.

Default:

```text
1
```

For normal saved-data viewer generation this option is deprecated and usually does not need to be changed.

### `--name NAME`

Explicit basename for generated output files.

Example:

```bash
python run_match.py --name material_vs_positional_c2
```

If omitted, the name is generated from the player strategies, inverse temperatures, and cycle depth.

### `--ignore-threefold`

By default the simulation stops on threefold repetition according to the simulation's repetition logic.

Use

```bash
python run_match.py --ignore-threefold
```

to continue through threefold repetition until another terminal condition or `--max-plies` is reached.

### `--allow-draw-claims`

By default the CLI configures the match to stop only on mate/stalemate apart from the separately controlled threefold rule and automatic terminal rules.

Use

```bash
python run_match.py --allow-draw-claims
```

to allow other claimable/automatic draw rules handled by the simulator to terminate the match as well.

## 21.2 Complete simulation example

```bash
python run_match.py \
    --white-strategy material_conservative \
    --black-strategy positional_controller \
    --beta-white 4.0 \
    --beta-black 4.0 \
    --kappa 1.0 \
    --cdepth 3 \
    --adaptive-c 0.3 \
    --search-workers auto \
    --parallel-min-branches 8 \
    --max-plies 120 \
    --name material_vs_positional_c3
```

The script prints paths similar to

```text
CSV:  data/games/csv/material_vs_positional_c3.csv
JSON: data/games/json/material_vs_positional_c3.json
PGN:  data/games/pgn/material_vs_positional_c3.pgn
```

---

# 22. Saved simulation outputs

Each match produces three complementary files.

## 22.1 PGN

```text
data/games/pgn/<match_name>.pgn
```

Contains the played chess game in standard PGN form.

## 22.2 CSV

```text
data/games/csv/<match_name>.csv
```

Contains compact per-ply diagnostics suitable for tabular analysis.

Typical fields include the played move, FEN before/after, static evaluation, current shallow landscape, search gain, cycle-deepening information, candidate/realized QWA information, entropy, `N_eff`, adaptive breadth, and search-performance diagnostics.

## 22.3 JSON

```text
data/games/json/<match_name>.json
```

This is the richest output and the main input for the analysis scripts and viewer.

It contains, for each ply, the current move landscape, full search result, candidate move features and probabilities, adaptive refinement metadata, saved responses, principal variations, search and thermodynamic endpoints, QWA diagnostics, and performance information.

When doing detailed post-hoc analysis, prefer the JSON file.

---

# 23. Quick terminal inspection: `analyze_match.py`

`analyze_match.py` prints the shallow move landscape for one played ply.

Basic usage:

```bash
python analyze_match.py data/games/json/<match>.json
```

If no JSON path is supplied, the script defaults to

```text
data/games/json/thermo_match.json
```

For the selected ply it prints, for each move:

- SAN move;
- subjective potential `phi`;
- `P(move|B)`;
- static value `E(B_move)`;
- mover-relative feature values:
  - material;
  - center;
  - development;
  - castling;
  - king safety;
  - king pressure.

## 23.1 Flags

### `--ply N`

Ply to inspect.

Default:

```text
1
```

Example:

```bash
python analyze_match.py data/games/json/example.json --ply 10
```

### `--sort FIELD`

Sort displayed moves by one of:

```text
probability
static_after
phi
```

Default:

```text
probability
```

Example:

```bash
python analyze_match.py data/games/json/example.json --ply 10 --sort phi
```

### `--limit N`

Maximum number of moves printed.

Default:

```text
20
```

Example:

```bash
python analyze_match.py data/games/json/example.json --ply 10 --limit 10
```

Combined example:

```bash
python analyze_match.py data/games/json/example.json \
    --ply 10 \
    --sort probability \
    --limit 15
```

---

# 24. Static response covariance analysis: `analyze_static_covariances.py`

This script analyzes the saved response ensembles of recursively deepened candidate moves.

For each candidate, define the vector of static evaluator components

$$
\mathbf X
=(M,P,L,C,K),
$$

corresponding to

```text
material
pawn_structure
mobility
center
king_safety
```

and response probabilities `p_r`.

The weighted mean is

$$
\mu_i=\sum_r p_r X_{r,i},
$$

and the covariance matrix is

$$
\boxed{
\Sigma_{ij}
=\sum_r p_r
(X_{r,i}-\mu_i)
(X_{r,j}-\mu_j).
}
$$

If the static evaluator uses weight vector

$$
\boldsymbol\alpha
=(\alpha_M,\alpha_P,\alpha_L,\alpha_C,\alpha_K),
$$

then the variance implied by component covariances is

$$
\boxed{
\operatorname{Var}[E]
=\boldsymbol\alpha^T\Sigma\boldsymbol\alpha.
}
$$

The script also computes the direct weighted variance of saved static values and stores a consistency error between the two calculations when available.

Only candidate moves with saved response data and static response components can be analyzed.

## 24.1 Basic usage

Analyze one game:

```bash
python analyze_static_covariances.py data/games/json/example.json
```

By default this writes

```text
data/analysis/covariance/example.csv
```

Analyze every JSON file in a directory:

```bash
python analyze_static_covariances.py data/games/json
```

Multiple files/directories can also be passed in one command.

## 24.2 Flags

### positional `inputs`

One or more JSON files or directories:

```bash
python analyze_static_covariances.py game1.json game2.json
```

or

```bash
python analyze_static_covariances.py data/games/json
```

### `--output PATH`

CSV output file or directory.

Default for each game:

```text
data/analysis/covariance/<game>.csv
```

Example for one game:

```bash
python analyze_static_covariances.py data/games/json/example.json \
    --output results/example_covariance.csv
```

For multiple input games, use a directory:

```bash
python analyze_static_covariances.py data/games/json \
    --output results/covariance
```

### `--json-output PATH`

Optional detailed JSON output containing mean component vectors, full covariance matrices, correlation matrices, and normalized response probabilities.

Example:

```bash
python analyze_static_covariances.py data/games/json/example.json \
    --json-output data/analysis/covariance/example_details.json
```

### `--ply N`

Print a readable covariance summary for one ply in addition to writing the analysis files.

Example:

```bash
python analyze_static_covariances.py data/games/json/example.json --ply 10
```

---

# 25. Deep-search divergence analysis: `analyze_deep_search_divergence.py`

This analysis compares the immediate static response-ensemble assessment with the deeper recursive search result.

It combines:

1. a covariance CSV produced by `analyze_static_covariances.py`;
2. the corresponding detailed game JSON.

For a candidate move `m`, let

$$
\langle E\rangle_m
$$

be the weighted mean static evaluation of its saved response ensemble and let

```text
terminal_u
```

be the deeper search value stored in the game JSON.

The script defines

$$
\boxed{
D_m
=\texttt{terminal\_u}-\langle E\rangle_m.
}
$$

and also stores

$$
|D_m|.
$$

Large `|D_m|` identifies candidates for which recursive search changes the immediate static assessment substantially.

## 25.1 Recommended workflow

First produce covariance files:

```bash
python analyze_static_covariances.py data/games/json
```

Then run divergence analysis:

```bash
python analyze_deep_search_divergence.py data/games/json
```

By default output is written to

```text
data/analysis/deep_search_divergence/<game>.csv
```

## 25.2 Flags

### positional `game_json`

One or more game JSON files or directories.

Example:

```bash
python analyze_deep_search_divergence.py data/games/json/example.json
```

or

```bash
python analyze_deep_search_divergence.py data/games/json
```

### `--covariance-csv PATH`

Explicit covariance CSV for a **single** game.

Example:

```bash
python analyze_deep_search_divergence.py data/games/json/example.json \
    --covariance-csv data/analysis/covariance/example.csv
```

This flag cannot be used when multiple game JSON files are being analyzed.

### `--covariance-dir PATH`

Directory containing covariance CSV files whose stems match the game JSON filenames.

Default:

```text
data/analysis/covariance
```

Example:

```bash
python analyze_deep_search_divergence.py data/games/json \
    --covariance-dir results/covariance
```

### `--output PATH`

Output CSV file or directory.

Default:

```text
data/analysis/deep_search_divergence/<game>.csv
```

For multiple games, use a directory.

### `--top N`

Print the `N` candidates with the largest absolute deep-search divergence.

Example:

```bash
python analyze_deep_search_divergence.py data/games/json/example.json --top 20
```

### `--ply N`

Print all analyzed candidate rows for one ply.

Example:

```bash
python analyze_deep_search_divergence.py data/games/json/example.json --ply 10
```

Combined example:

```bash
python analyze_deep_search_divergence.py data/games/json/example.json \
    --top 20 \
    --ply 10
```

---

# 26. Visualizing saved matches: `view_match.py`

The main browser-based viewer is launched with

```bash
python view_match.py
```

`view_match.py` delegates to the live viewer implemented in `thermo_chess.live_viewer`.

Default JSON directory:

```text
data/games/json
```

Default address:

```text
http://127.0.0.1:8765/
```

## 26.1 Viewer flags

### `--results-dir PATH`

Directory containing saved match JSON files.

Default:

```text
data/games/json
```

Example:

```bash
python view_match.py --results-dir experiments/run1/json
```

### `--host HOST`

Host interface used by the HTTP server.

Default:

```text
127.0.0.1
```

Example:

```bash
python view_match.py --host 0.0.0.0
```

Binding to `0.0.0.0` exposes the viewer on available network interfaces; use this only on an appropriate network.

### `--port N`

HTTP port.

Default:

```text
8765
```

Example:

```bash
python view_match.py --port 9000
```

### `--open`

Open the viewer automatically in the default browser.

Example:

```bash
python view_match.py --open
```

Combined example:

```bash
python view_match.py \
    --results-dir data/games/json \
    --host 127.0.0.1 \
    --port 8765 \
    --open
```

## 26.2 What to inspect in the viewer

For each ply and candidate move, the viewer exposes the quantities discussed in this README, including:

- static board evaluation `E`;
- shallow subjective landscape `U`;
- move potential `Phi`;
- move probability `P(m|B)`;
- entropy and `N_eff`;
- adaptive breadth `K`;
- search score `g_tilde`;
- requested and actually deepened cycles;
- principal variation;
- search endpoint and thermodynamic endpoint;
- `thermo_g_tilde`;
- Q/W/A when at least one complete cycle has been explored;
- realized full-cycle Q/W/A for the actual played trajectory where available;
- search-performance diagnostics.

The key distinction to remember is:

$$
G_m^{\rm search}
$$

is the quantity used to choose the move, while

$$
G_m^{\rm thermo}=Q_m+W_m+A_m
$$

refers to the deepest completed-cycle thermodynamic endpoint and can therefore differ from the search score for partially deepened candidates.

---

# 27. Search benchmark: `benchmark_adaptive.py`

This script measures adaptive-search diagnostics on a representative board position for cycle depths 1 through 4.

Run with defaults:

```bash
python benchmark_adaptive.py
```

It prints columns including:

```text
cdepth
seconds
nodes
static
recursive
avg_K
cache_hits
value
```

and also computes a two-ply full-expectation reference calculation.

This is primarily a performance/diagnostic tool rather than a match simulation.

## 27.1 Flags

### `--fen FEN`

Position to benchmark.

If omitted, the script uses its built-in representative middlegame FEN.

Example:

```bash
python benchmark_adaptive.py --fen "<FEN>"
```

### `--beta X`

Subjective inverse temperature.

Default:

```text
4.0
```

### `--adaptive-c X`

Adaptive breadth coefficient.

Default:

```text
0.3
```

Example:

```bash
python benchmark_adaptive.py --beta 5.0 --adaptive-c 0.5
```

---

# 28. Move-feature scale diagnostics: `diagnose_feature_scales.py`

This script samples legal positions generated by random play and measures the empirical scale of the mover-relative features entering the subjective move potential.

Run:

```bash
python diagnose_feature_scales.py
```

The output table reports, for every strategy feature:

- fraction of values that are nonzero;
- median nonzero absolute value;
- 75th percentile;
- 90th percentile;
- maximum;
- median within-position move range;
- 90th-percentile within-position move range.

This is useful when checking whether raw strategy coefficients act on similarly scaled feature variables.

## 28.1 Flags

### `--positions N`

Number of sampled positions.

Default:

```text
1000
```

### `--seed N`

Random seed used to generate sampled random-play positions.

Default:

```text
20261004
```

### `--max-random-plies N`

Maximum random-play length before a sampling game is restarted.

Default:

```text
80
```

Example:

```bash
python diagnose_feature_scales.py \
    --positions 5000 \
    --seed 123 \
    --max-random-plies 100
```

---

# 29. Recommended end-to-end workflow

## 29.1 Install

```bash
python -m venv .venv
```

Activate the environment and install the declared dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 29.2 Run a match

```bash
python run_match.py \
    --white-strategy material_conservative \
    --black-strategy positional_controller \
    --cdepth 2 \
    --adaptive-c 0.3 \
    --name example_match
```

## 29.3 Inspect a suspicious ply quickly

```bash
python analyze_match.py data/games/json/example_match.json \
    --ply 10 \
    --sort probability \
    --limit 20
```

## 29.4 Compute response covariances

```bash
python analyze_static_covariances.py data/games/json/example_match.json \
    --json-output data/analysis/covariance/example_match_details.json \
    --ply 10
```

This writes the standard CSV to

```text
data/analysis/covariance/example_match.csv
```

## 29.5 Compare shallow response ensembles with deep search

```bash
python analyze_deep_search_divergence.py data/games/json/example_match.json \
    --top 20 \
    --ply 10
```

## 29.6 Open the interactive viewer

```bash
python view_match.py --open
```

Then select `example_match` and inspect the played move, subjective landscape, adaptive refinement, principal variation, search gain, completed-cycle depth, Q/W/A, and realized full-cycle diagnostics.

## 29.7 Optional diagnostics

Benchmark adaptive search:

```bash
python benchmark_adaptive.py --beta 4.0 --adaptive-c 0.3
```

Inspect feature scales:

```bash
python diagnose_feature_scales.py --positions 1000
```

# 30. Module map

```text
thermo_chess/
├── evaluation.py      universal static board evaluator E(B)
├── features.py        board descriptors and mover-relative move features
├── metrics.py         entropy and effective-number utilities
├── measure.py         styles, potentials, Boltzmann probabilities, shallow U
├── search.py          adaptive subjective-minimax search and diagnostics
├── thermodynamics.py  finite Q/W/A decomposition
├── player.py          player object and deterministic move selection
├── simulation.py      match simulation and CSV/JSON/PGN logging
├── interactive.py     static HTML viewer generation
├── live_viewer.py     live HTTP viewer and its CLI
└── __init__.py        public package exports
```

Repository-level helper scripts:

```text
run_match.py                        run a simulation
analyze_match.py                    inspect one saved ply
analyze_static_covariances.py       response-component covariance analysis
analyze_deep_search_divergence.py   compare immediate response ensembles with deep search
view_match.py                       launch the browser viewer
benchmark_adaptive.py               benchmark adaptive search
diagnose_feature_scales.py          measure move-feature scales
```

---

# 31. Core equations at a glance

Static board evaluation:

$$
E(B)=\sum_j\alpha_j\Delta X_j(B).
$$

Subjective potential:

$$
\Phi_p(m,B)=\sum_k\lambda^{\rm eff}_{p,k}(B)F_k(m,B).
$$

Boltzmann measure:

$$
P_p(m|B)=\frac{e^{\beta_p\Phi_p(m,B)}}{\sum_{m'}e^{\beta_p\Phi_p(m',B)}}.
$$

Entropy:

$$
S_p=-\sum_mP_p\ln P_p.
$$

Effective number:

$$
N_{\rm eff}=e^{S_p}.
$$

Adaptive breadth:

$$
K=\min\left(N_{\rm legal},\max\left[1,\lceil cN_{\rm eff}\rceil\right]\right).
$$

Shallow subjective landscape:

$$
U_p(B)=\sum_mP_p(m|B)E(B_m).
$$

Cycle-depth conversion:

$$
n_{\rm recursive}=2\,\texttt{cdepth}-1.
$$

Subjective minimax backup:

$$
F_p(B,n)=
\begin{cases}
\max_aX_a,&B\text{ White to move},\\
\min_aX_a,&B\text{ Black to move},
\end{cases}
$$

with

$$
X_a=
\begin{cases}
F_p(B_a,n-1),&a\text{ deepened},\\
U_p(B_a),&a\text{ shallow fallback}.
\end{cases}
$$

Search gain:

$$
G_m^{\rm search}=U_p(B_{\rm search})-U_p(B_0).
$$

Thermodynamic candidate gain:

$$
G_m^{\rm thermo}=U_p(B_{\rm thermo})-U_p(B_0).
$$

Finite decomposition:

$$
G_m^{\rm thermo}=\Delta Q+\Delta W+\Delta A.
$$

---

## Current scope

This project is intentionally a toy research framework rather than a conventional chess engine. Its purpose is to make the agent's subjective measure, adaptive attention, recursive decision rule, and finite landscape decomposition explicit and inspectable.
