# Thermodynamic Toy Chess

`chess_thermodynamic` is an experimental chess simulator in which legal moves are assigned observer-dependent probabilities, while board positions are evaluated by a shared static evaluator. The simulator then combines these two ingredients through probability-weighted expectations.

The project is intended for studying how two observers with different strategic preferences can assign different importance to the same legal moves and, as a consequence, allocate computational effort differently when looking ahead.

This is **not** a conventional chess engine. It does not use minimax, alpha-beta pruning, opening books, or an external chess engine. Maxima and minima are used only at the final decision stage, when White selects the largest move score and Black selects the smallest one. All intermediate position values remain probability-weighted expectations.

---

## Contents

1. [Model overview](#1-model-overview)
2. [Notation and conventions](#2-notation-and-conventions)
3. [Observer-dependent move model](#3-observer-dependent-move-model)
4. [Universal board evaluation](#4-universal-board-evaluation)
5. [Entropy and adaptive search](#5-entropy-and-adaptive-search)
6. [Move selection](#6-move-selection)
7. [Thermodynamic diagnostics](#7-thermodynamic-diagnostics)
8. [Default parameters and strategy presets](#8-default-parameters-and-strategy-presets)
9. [Installation](#9-installation)
10. [Running simulations](#10-running-simulations)
11. [Viewer](#11-viewer)
12. [Saved data](#12-saved-data)
13. [Analysis, tests, and benchmark](#13-analysis-tests-and-benchmark)
14. [Python configuration](#14-python-configuration)
15. [Project structure](#15-project-structure)
16. [Performance and limitations](#16-performance-and-limitations)

---

# 1. Model Overview

The model separates three concepts.

1. **Universal board evaluation.**  
   A scalar function $E(B)$ evaluates a board position $B$. The same evaluator is used by both players.

2. **Observer-dependent move probabilities.**  
   Each player has a vector of strategic preferences. These preferences define a potential $\Phi_p(m,B)$ for every legal move $m$, which is converted into a probability distribution over moves.

3. **Adaptive expected-value search.**  
   The entropy of the move distribution determines how many branches receive refined evaluation and how deep the search is allowed to continue locally.

The global sign convention is

$$
E(B)>0 \quad\text{favors White},
\qquad
E(B)<0 \quad\text{favors Black}.
$$

The same sign convention is used for all expected board values derived from $E(B)$.

The players differ only through their observer-dependent move distributions. They do **not** use different material values, different board-evaluation formulas, or different sign conventions.

The rest of the README builds these ingredients in this order:

```text
board and move notation
        ↓
move features and style potential
        ↓
move probabilities
        ↓
universal board evaluation
        ↓
entropy and adaptive refinement
        ↓
expected-value decision rule
        ↓
Q/W/A and prediction diagnostics
```

---

# 2. Notation and Conventions

## 2.1 Board and move notation

Let

- $B$ be a chess position;
- $\mathcal M(B)$ be the set of legal moves in $B$;
- $m\in\mathcal M(B)$ be one legal move;
- $B_m$ be the board obtained after move $m$;
- $r$ be a legal response to $m$;
- $B_{mr}$ be the board after the sequence $m$ followed by $r$;
- $p$ denote the observer whose preferences are being used;
- $c\in\{\mathrm{White},\mathrm{Black}\}$ denote a chess color;
- $\bar c$ denote the opposite color.

A complete interaction cycle is

```text
player move -> opponent response
```

and therefore contains two plies.

The search parameter `cdepth` counts these complete interaction cycles.

## 2.2 Piece values

All material-like quantities are expressed in pawn units:

| Piece | Value |
|---|---:|
| Pawn | `1` |
| Knight | `3` |
| Bishop | `3` |
| Rook | `5` |
| Queen | `9` |
| King | `0` |

The king has value zero because checkmate is handled separately as a terminal condition.

## 2.3 Material notation

Let $n_t(B,c)$ be the number of pieces of type $t$ and color $c$ on board $B$.

Then the material owned by color $c$ is

$$
M(B,c)
=
\sum_{t\in\{P,N,B,R,Q,K\}}
v(t)\,n_t(B,c),
$$

where $v(t)$ is the piece value in the table above.

The mover-relative material balance is

$$
M_{\mathrm{rel}}(B,c)
=
M(B,c)-M(B,\bar c).
$$

This quantity is positive when color $c$ has more material than its opponent.

## 2.4 Mobility

Let

$$
L(B,c)
$$

be the number of legal moves available to color $c$ in board position $B$.

Operationally, the board is copied, the side to move is set to $c$, and legal moves are counted.

The value is zero for checkmate, stalemate, or insufficient-material positions.

## 2.5 Center control

Define the extended-center set

$$
\mathcal C_{\mathrm{ext}}
=
\{c3,d3,e3,f3,c4,f4,c5,f5,c6,d6,e6,f6\}.
$$

The attacked-center score of color $c$ is

$$
C(B,c)
=
\sum_{s\in\{d4,e4,d5,e5\}}
\mathbf 1[c\text{ attacks }s]
+
0.35
\sum_{s\in\mathcal C_{\mathrm{ext}}}
\mathbf 1[c\text{ attacks }s].
$$

This measures attacked squares, not piece occupancy.

## 2.6 King safety

Let $k_c$ be the square occupied by the king of color $c$. Define

- $A(B,c)$: number of enemy attackers of $k_c$;
- $D(B,c)$: number of friendly defenders of $k_c$;
- $P(B,c)$: number of king-adjacent squares attacked by the opponent.

The king-safety helper is

$$
K_{\mathrm{safe}}(B,c)
=
\frac{D(B,c)-1.5A(B,c)-0.25P(B,c)}{8}.
$$

If color $c$ has no king, the helper returns `-1`.

## 2.7 Exchange-aware exposure

Several later formulas require an estimate of how much material can be profitably captured on a square.

For a square $s$ occupied by an opposing non-king piece, let $\mathcal X(B,s,c)$ be the legal captures by color $c$ onto $s$.

Define the best profitable exchange gain recursively as

$$
G(B,s,c)
=
\max\left(
0,
\max_{m\in\mathcal X(B,s,c)}
\left[
v(\text{piece on }s)-G(B_m,s,\bar c)
\right]
\right).
$$

The outer zero allows the attacking side to stop the exchange whenever continuing would be unprofitable.

The total exposed material of color $c$ is

$$
X(B,c)
=
\sum_{s\text{ occupied by a non-king piece of }c}
G(B,s,\bar c).
$$

This is a local static-exchange-style calculation restricted to repeated captures on one square. It is not a general tactical search.

---

# 3. Observer-Dependent Move Model

This section defines how an observer assigns probabilities to legal moves. The board evaluator $E(B)$, which is independent of observer style, is introduced later in [Section 4](#4-universal-board-evaluation).

## 3.1 Style coefficients

Each observer $p$ has a vector of style coefficients

$$
\lambda_p
=
(\lambda_{p,1},\lambda_{p,2},\ldots).
$$

For a legal move $m$, the simulator computes mover-relative features

$$
F_k(m,B).
$$

The move potential is a weighted sum of these features:

$$
\Phi_p(m,B)
=
\sum_k \lambda_{p,k}(g)\,F_k(m,B),
$$

where $g\in[0,1]$ is the development-phase variable defined in [Section 3.4](#34-development-phase).

All move features are evaluated from the perspective of the side that actually makes the move. For example, gaining material is positive whether White or Black performs the capture.

An observer's style coefficients remain fixed throughout that observer's search, including nodes where the opponent is the side to move.

## 3.2 Move features

The model uses the following features:

| Feature | Meaning |
|---|---|
| `material` | Change in mover-relative material balance, in pawn units. |
| `preservation` | Reduction in the mover's exchange-aware exposed material. |
| `activity` | Change in own mobility minus half the opponent's mobility change, normalized by `30`. |
| `king_safety` | Change in the mover's static king-safety score. |
| `king_restriction` | Reduction in the opponent king's legal moves. |
| `king_pressure` | Increase in attacked squares around the opponent king. |
| `center` | Change in attacked center and extended-center squares, normalized by `6`. |
| `check` | `1` if the resulting position gives check, otherwise `0`. |
| `mate` | `1` if the move gives checkmate, otherwise `0`. |
| `promotion` | Promotion material gain normalized by `8`. |
| `castle_preserve` | Change in retained castling rights, with actual castling exempted. |
| `castle_deny` | Number of opponent castling rights removed. |
| `castle` | `1` if the move is castling, otherwise `0`. |

The exact formulas are defined below.

### Material

Let $c$ be the side to move before $m$. Then

$$
F_{\mathrm{material}}(m,B)
=
M_{\mathrm{rel}}(B_m,c)
-
M_{\mathrm{rel}}(B,c).
$$

### Preservation

$$
F_{\mathrm{preservation}}(m,B)
=
X(B,c)-X(B_m,c).
$$

A positive value means that the mover leaves less material profitably capturable after the move.

### Activity

$$
F_{\mathrm{activity}}(m,B)
=
\frac{
[L(B_m,c)-L(B,c)]
-
0.5[L(B_m,\bar c)-L(B,\bar c)]
}{30}.
$$

### King safety

$$
F_{\mathrm{king\_safety}}(m,B)
=
K_{\mathrm{safe}}(B_m,c)
-
K_{\mathrm{safe}}(B,c).
$$

### King restriction

Let $L_K(B,c)$ be the number of legal moves available specifically to the king of color $c$. Then

$$
F_{\mathrm{king\_restriction}}(m,B)
=
L_K(B,\bar c)
-
L_K(B_m,\bar c).
$$

### King pressure

Let $Q(B,c)$ be the number of squares adjacent to the opponent king that are attacked by color $c$. Then

$$
F_{\mathrm{king\_pressure}}(m,B)
=
Q(B_m,c)-Q(B,c).
$$

### Center control

$$
F_{\mathrm{center}}(m,B)
=
\frac{C(B_m,c)-C(B,c)}{6}.
$$

### Check

$$
F_{\mathrm{check}}(m,B)
=
\mathbf 1[B_m\text{ gives check}].
$$

### Mate

$$
F_{\mathrm{mate}}(m,B)
=
\mathbf 1[B_m\text{ is checkmate}].
$$

### Promotion

If move $m$ promotes a pawn to piece type $t$,

$$
F_{\mathrm{promotion}}(m,B)
=
\frac{v(t)-v(\mathrm{pawn})}{8}.
$$

For non-promotion moves this feature is zero.

Therefore, before multiplication by its style coefficient:

- queen promotion contributes `1.0`;
- rook promotion contributes `0.5`;
- bishop promotion contributes `0.25`;
- knight promotion contributes `0.25`.

### Castling rights

Let

$$
C_c(B)\in\{0,1,2\}
$$

be the number of castling rights retained by color $c$, counting kingside and queenside rights separately.

The castling-preservation feature is

$$
F_{\mathrm{castle\_preserve}}(m,B)
=
C_c(B_m)-C_c(B),
$$

except that it is set to zero when $m$ is itself castling.

Thus an ordinary king move usually gives `-2`, an original-rook move usually gives `-1`, and actually using the castling right is not treated as losing it.

The castling-denial feature is

$$
F_{\mathrm{castle\_deny}}(m,B)
=
C_{\bar c}(B)-C_{\bar c}(B_m).
$$

Actual castling is represented by

$$
F_{\mathrm{castle}}(m,B)
=
\mathbf 1[m\text{ is castling}].
$$

## 3.3 Base move potential

Ignoring the phase-dependent modification for the moment, the move potential is

$$
\Phi_{\mathrm{base}}(m,B)
=
\lambda_{\mathrm{material}}F_{\mathrm{material}}
+\lambda_{\mathrm{preservation}}F_{\mathrm{preservation}}
+\lambda_{\mathrm{activity}}F_{\mathrm{activity}}
+\lambda_{\mathrm{king\_safety}}F_{\mathrm{king\_safety}}
+\lambda_{\mathrm{king\_restriction}}F_{\mathrm{king\_restriction}}
+\lambda_{\mathrm{king\_pressure}}F_{\mathrm{king\_pressure}}
+\lambda_{\mathrm{center}}F_{\mathrm{center}}
+\lambda_{\mathrm{check}}F_{\mathrm{check}}
+\lambda_{\mathrm{mate}}F_{\mathrm{mate}}
+\lambda_{\mathrm{promotion}}F_{\mathrm{promotion}}
+\lambda_{\mathrm{castle\_preserve}}F_{\mathrm{castle\_preserve}}
+\lambda_{\mathrm{castle\_deny}}F_{\mathrm{castle\_deny}}
+\lambda_{\mathrm{castle}}F_{\mathrm{castle}}.
$$

The next subsection introduces the phase variable $g$ and the additional phase-dependent contribution.

## 3.4 Development phase

The phase variable is board-dependent and does not use the move number.

For each color $c$, define:

- $d_N(B,c)$: fraction of the four original minor-piece squares no longer occupied by that color's original minor-piece type;
- $d_P(B,c)$: analogous fraction for the two original central-pawn squares;
- $d_R(B,c)$: analogous fraction for the original rook and queen squares.

The relevant starting squares are:

```text
White minor pieces: b1 c1 f1 g1
Black minor pieces: b8 c8 f8 g8

White central pawns: d2 e2
Black central pawns: d7 e7

White rooks/queen: a1 d1 h1
Black rooks/queen: a8 d8 h8
```

For each color,

$$
d(B,c)
=
0.60\,d_N(B,c)
+
0.25\,d_P(B,c)
+
0.15\,d_R(B,c).
$$

The global phase is

$$
g(B)
=
\operatorname{clamp}_{[0,1]}
\left(
\frac{d(B,\mathrm{White})+d(B,\mathrm{Black})}{2}
\right).
$$

Three phase weights are then defined:

$$
w_D(g)=1-g,
\qquad
w_C(g)=4g(1-g),
\qquad
w_A(g)=g.
$$

They weight development, consolidation, and attacking tendencies respectively.

The corresponding phase features are

$$
F_D(m,B)
=
d(B_m,c)-d(B,c)
+
\max(0,F_{\mathrm{activity}}),
$$

$$
F_C(m,B)
=
F_{\mathrm{castle}}
+
F_{\mathrm{king\_safety}},
$$

and

$$
F_A(m,B)
=
F_{\mathrm{activity}}
+
F_{\mathrm{king\_pressure}}
+
F_{\mathrm{king\_restriction}}
+
F_{\mathrm{check}}.
$$

Let $\sigma\in[0,1]$ be the observer's `solidness`.

The unscaled phase contribution is

$$
\Phi_{\mathrm{phase}}
=
\lambda_D w_D F_D
+
\lambda_C w_C F_C
+
\lambda_A w_A F_A.
$$

Castling-right preservation receives a separate phase-dependent weight:

$$
w_{\mathrm{castle\_rights}}(g)=1-g,
$$

$$
\Phi_{\mathrm{castle\_preserve}}
=
\lambda_{\mathrm{CP}}
(1-g)
\left(1+\sigma w_C(g)\right)
F_{\mathrm{castle\_preserve}}.
$$

This makes wasted castling rights more costly in the opening and early middlegame, while the contribution fades as $g\to 1$.

The final move potential is

$$
\Phi_{\mathrm{total}}
=
\Phi_{\mathrm{base}}^*
+
\sigma\Phi_{\mathrm{phase}},
$$

where

$$
\Phi_{\mathrm{base}}^*
$$

is the base potential with the ordinary `castle_preserve` term replaced by the phase-dependent castling-preservation contribution above.

At `solidness = 0`, the additional phase term vanishes.

## 3.5 Move probabilities

Let

- $\kappa>0$ be the Boltzmann-like scale parameter;
- $T_p>0$ be the temperature of observer $p$;
- $\beta_p$ be the corresponding inverse temperature,

with

$$
\beta_p=\frac{1}{\kappa T_p}.
$$

The probability assigned by observer $p$ to legal move $m$ is

$$
P_p(m\mid B)
=
\frac{
\exp\left(\Phi_p(m,B)/(\kappa T_p)\right)
}{
\sum_{m'\in\mathcal M(B)}
\exp\left(\Phi_p(m',B)/(\kappa T_p)\right)
}.
$$

Equivalently, because $\beta_p=1/(\kappa T_p)$,

$$
P_p(m\mid B)
\propto
\exp\left(\beta_p\Phi_p(m,B)\right).
$$

The implementation evaluates this softmax in numerically stable form by subtracting the largest scaled potential before exponentiation.

Interpretation:

- larger `beta` means lower effective temperature and a more concentrated move distribution;
- smaller `beta` means higher effective temperature and a flatter move distribution;
- `kappa` rescales the relationship between `beta` and temperature;
- temperature affects the observer-dependent move probabilities, not the universal board evaluator.

The default value is

```text
kappa = 1.0
```

in pawn units.

---

# 4. Universal Board Evaluation

The previous section defined **which moves an observer considers likely or important**. This section defines the separate, observer-independent function used to evaluate a board.

## 4.1 Static evaluator

The universal evaluator is

$$
E(B)
=
\alpha_M\Delta M(B)
+
\alpha_{\mathrm{exposure}}\Delta H(B)
+
\alpha_{\mathrm{mob}}\Delta\mathrm{Mobility}(B)
+
\alpha_K\Delta K(B)
+
\alpha_C\Delta C(B).
$$

The components are defined below.

### Material difference

$$
\Delta M(B)
=
M(B,\mathrm{White})
-
M(B,\mathrm{Black}).
$$

### Exposure difference

Let the exposed material of side $a$ be

$$
H_a(B)
=
\sum_{s\text{ occupied by a non-king piece of }a}
R_s(B),
$$

where $R_s(B)$ is the exchange-aware profitable-capture exposure defined through the same local exchange logic used in $X(B,c)$.

The signed exposure difference is

$$
\Delta H(B)
=
H_{\mathrm{Black}}(B)
-
H_{\mathrm{White}}(B).
$$

Therefore White exposure lowers $E(B)$, while Black exposure raises it.

### Mobility difference

$$
\Delta\mathrm{Mobility}(B)
=
\frac{
L(B,\mathrm{White})
-
L(B,\mathrm{Black})
}{30}.
$$

### King-safety difference

$$
\Delta K(B)
=
K_{\mathrm{safe}}(B,\mathrm{White})
-
K_{\mathrm{safe}}(B,\mathrm{Black}).
$$

### Center-control difference

$$
\Delta C(B)
=
\frac{
C(B,\mathrm{White})
-
C(B,\mathrm{Black})
}{6}.
$$

## 4.2 Default evaluation weights

| Static component | Default |
|---|---:|
| Material | `1.0` |
| Exposure | `0.5` |
| Mobility | `0.25` |
| King safety | `0.6` |
| Center control | `0.35` |
| Checkmate magnitude | `10000.0` |

With these defaults,

$$
E(B)
=
\Delta M(B)
+
0.5\,\Delta H(B)
+
0.25\,\Delta\mathrm{Mobility}(B)
+
0.6\,\Delta K(B)
+
0.35\,\Delta C(B).
$$

The static evaluator does not directly use the observer-style coefficients defined in Section 3.

## 4.3 Terminal positions

If the side to move is checkmated, the evaluator returns the appropriate signed checkmate value.

The following positions evaluate to zero:

- stalemate;
- insufficient material;
- automatic 75-move draw;
- automatic fivefold repetition.

At a search terminal or a static depth-zero node, the search returns $E(B)$ directly.

---

# 5. Entropy and Adaptive Search

The probability distribution from Section 3 determines both the expectation over future positions and how computational effort is allocated.

## 5.1 Entropy

For observer $p$, the move-distribution entropy is

$$
S_p(B)
=
-\sum_{m\in\mathcal M(B)}
P_p(m\mid B)
\log P_p(m\mid B).
$$

The corresponding effective number of moves is

$$
N_{\mathrm{eff}}(B)
=
e^{S_p(B)}.
$$

Interpretation:

- concentrated move distributions have $N_{\mathrm{eff}}$ close to `1`;
- broad distributions have larger $N_{\mathrm{eff}}$.

## 5.2 Adaptive breadth

Let `adaptive_c` be the positive breadth parameter $c$.

At board $B$, the number of root moves selected for refined evaluation is

$$
K(B)
=
\min\left(
|\mathcal M(B)|,
\max\left(
1,
\left\lceil
cN_{\mathrm{eff}}(B)
\right\rceil
\right)
\right).
$$

For a selected root move $m$, let $\mathcal R(B_m)$ be the legal responses in $B_m$. The number of responses selected for refinement is

$$
K'_m
=
\min\left(
|\mathcal R(B_m)|,
\max\left(
1,
\left\lceil
cN_{\mathrm{eff}}(B_m)
\right\rceil
\right)
\right).
$$

The default is

```text
adaptive_c = 0.3
```

Only the selected branches receive extra computation. All legal moves and responses retain their original probability mass.

## 5.3 Refinement policy

The option `refinement_policy` determines **which** branches belong to the top-$K$ and top-$K'$ sets.

Available values are:

| Policy | Meaning |
|---|---|
| `static_eval` | Rank branches by the immediate universal evaluation $E$, using the objective of the side whose action is being ranked. |
| `probability` | Rank branches by the observer's move probability. |

The default is

```text
refinement_policy = static_eval
```

For `static_eval`:

- White-favored branches are ranked by larger $E$;
- Black-favored branches are ranked by smaller $E$.

The refinement policy changes only which branches receive extra computation. It does not modify or renormalize the move probabilities.

## 5.4 Entropy-based local depth cap

The entropy-derived depth cap is

$$
d_{\mathrm{eff}}(B)
=
\begin{cases}
4, & N_{\mathrm{eff}}(B)\le 4,\\
3, & 4<N_{\mathrm{eff}}(B)\le 8,\\
2, & 8<N_{\mathrm{eff}}(B)\le 15,\\
1, & N_{\mathrm{eff}}(B)>15.
\end{cases}
$$

If the remaining global cycle budget is $r$, the local search depth is

$$
d_{\mathrm{local}}(B,r)
=
\min(r,d_{\mathrm{eff}}(B)).
$$

The global parameter `cdepth` therefore acts as a hard upper bound, while entropy can reduce the depth locally.

## 5.5 Shallow same-player landscape

Before defining the recursive search, define the one-ply same-player expected value

$$
U_p^{1/2}(B)
=
\sum_{a\in\mathcal M(B)}
P_p(a\mid B)\,E(B_a).
$$

This is the expected static value after one move sampled from observer $p$'s move distribution.

It is called a "same-player landscape" because it is later compared with the analogous quantity after a complete move-response cycle, when the same player is again to move.

## 5.6 Recursive cycle expectation

For a remaining full-cycle depth $d\ge 1$, define

$$
V_p^{(d)}(B)
=
\sum_m
P_p(m\mid B)
\sum_r
P_p(r\mid B_m)
\widetilde V_{p,m,r}^{(d-1)}.
$$

For a complete branch

```text
B -> B_m -> B_mr
```

the branch value is

$$
\widetilde V_{p,m,r}^{(d-1)}
=
\begin{cases}
V_p^{(d-1)}(B_{mr}),
& m\in K,\ r\in K'_m,\ d-1>0,\\[4pt]
U_p^{1/2}(B_{mr}),
& m\in K,\ r\in K'_m,\ d-1=0,\\[4pt]
E(B_{mr}),
& \text{otherwise}.
\end{cases}
$$

Thus:

- selected branches receive refined evaluation;
- unselected branches fall back to the immediate static evaluator;
- no probability mass is discarded;
- no top-$K$ distribution is renormalized.

At `cdepth = 1`, a refined complete cycle ends in the shallow same-player landscape $U_p^{1/2}(B_{mr})$.

At `cdepth > 1`, a refined complete cycle can recurse into another complete interaction cycle.

At `cdepth = 0`, the adaptive value reduces to the static board evaluator $E(B)$.

This search remains an expectation, not minimax.

---

# 6. Move Selection

The previous section defined how future branch values are approximated. This section defines the quantity used to choose the move that is actually played.

For a candidate root move $m$, let

$$
V_m
=
\sum_r
P_p(r\mid B_m)\,
V_{m,r},
$$

where $V_{m,r}$ is the branch value assigned by the adaptive search.

The candidate's expected change relative to the current shallow landscape is

$$
\langle\Delta U^*\rangle_m
=
V_m-U_p^{1/2}(B).
$$

The simulator stores $V_m$ as

```text
expected_next_U
```

and the move-selection score as

```text
expected_delta_u_star
```

with

$$
\texttt{expected\_delta\_u\_star}
=
\texttt{expected\_next\_U}
-
U_p^{1/2}(B).
$$

Final move selection is

$$
m_W^*
=
\arg\max_m
\langle\Delta U^*\rangle_m
$$

for White, and

$$
m_B^*
=
\arg\min_m
\langle\Delta U^*\rangle_m
$$

for Black.

Because both players use the same sign convention for $E(B)$, White always prefers larger expected values and Black always prefers smaller expected values.

---

# 7. Thermodynamic Diagnostics

The simulator separates **predicted** starred quantities from **realized** bookkeeping.

Predicted quantities answer: what does the player expect to happen along the response branches it examined in detail?

Realized quantities answer: what actually changed between two consecutive decision states of the same player after the move-response cycle was played?

The diagnostics do not change move probabilities, branch ranking, adaptive refinement, or the move ultimately selected.

## 7.1 Predicted Branch Quantities

For a current decision state $B$, a candidate move $m$, a selected opponent response $r$, and the resulting same-player state $B_{mr}$, the branch diagnostic is

$$
\Delta U^*_{mr}
=
V_{m,r}-U_p^{1/2}(B).
$$

Here $V_{m,r}$ is the exact branch value used by the adaptive search for that refined response branch. At `cdepth = 1`, this is $U_p^{1/2}(B_{mr})`; at larger depths it can be a recursively refined adaptive value.

The decomposition compares the current shallow decision landscape

$$
U_p^{1/2}(B)=\sum_a p_a O_a
$$

with the adaptive next-state landscape whose expectation is exactly the branch value

$$
V_{m,r}=\sum_a p'_a O'_a.
$$

For common action labels $a\in\mathcal L_\cap$,

$$
\Delta Q^*_{mr}
=
\sum_{a\in\mathcal L_\cap}
\frac{O_a+O'_a}{2}(p'_a-p_a),
$$

$$
\Delta W^*_{mr}
=
\sum_{a\in\mathcal L_\cap}
\frac{p_a+p'_a}{2}(O'_a-O_a).
$$

If $\mathcal L_+$ contains newly available actions and $\mathcal L_-$ contains actions that disappeared,

$$
\Delta A^*_{mr}
=
\sum_{a\in\mathcal L_+}p'_aO'_a
-
\sum_{a\in\mathcal L_-}p_aO_a.
$$

The branch-level invariant is

$$
\Delta U^*_{mr}
=
\Delta Q^*_{mr}
+
\Delta W^*_{mr}
+
\Delta A^*_{mr}
$$

up to floating-point tolerance.

The JSON response fields are `branch_delta_u_star`, `branch_delta_q_star`, `branch_delta_w_star`, `branch_delta_a_star`, and `branch_decomposition_error`. Detailed branch diagnostics are stored only for selected/refined response branches.

## 7.2 Predicted Candidate Quantities

For candidate move $m$, the displayed predicted thermodynamic quantities are conditional averages over the selected response set $K'_m$.

Let

$$
\widehat P(r\mid m)
=
\frac{P_p(r\mid B_m)}
{\sum_{r'\in K'_m}P_p(r'\mid B_m)}
$$

for $r\in K'_m$. Then

$$
\langle\Delta U^*_m\rangle
=
\sum_{r\in K'_m}
\widehat P(r\mid m)\Delta U^*_{mr},
$$

and analogously

$$
\langle\Delta Q^*_m\rangle,
\qquad
\langle\Delta W^*_m\rangle,
\qquad
\langle\Delta A^*_m\rangle.
$$

Because these four quantities use the same refined response set and the same conditional probabilities,

$$
\langle\Delta U^*_m\rangle
=
\langle\Delta Q^*_m\rangle
+
\langle\Delta W^*_m\rangle
+
\langle\Delta A^*_m\rangle.
$$

These conditional predicted quantities are stored as `predicted_delta_u_star`, `predicted_delta_q_star`, `predicted_delta_w_star`, and `predicted_delta_a_star`. The refined response probability mass is stored as `predicted_refined_probability_mass`.

They are not the same concept as `expected_delta_u_star`, which remains the full move-selection score and can include unrefined responses through static fallback values.

## 7.3 Candidate Diagnostic Coverage

The option `candidate_thermo_mode` controls which candidate moves receive detailed predicted diagnostics.

| Mode | Meaning |
|---|---|
| `selected` | Compute detailed diagnostics only for the final chosen move. |
| `refined` | Compute them for root top-$K$ candidates and the final chosen move. |
| `all` | Compute them for every legal candidate. |

The default is

```text
candidate_thermo_mode = refined
```

Changing this option affects diagnostic cost only. It does not change the selected move.

## 7.4 Realized Cycle Quantities

A realized same-player transition compares two actual saved decision states of the same player. For White, this means one White-to-move position and the next White-to-move position two plies later; Black is analogous.

The simulator stores this as `adaptive same-player transition Q/W/A` and also exposes clearer realized aliases:

$$
\Delta U_{\mathrm{real}}
=
\Delta Q_{\mathrm{real}}
+
\Delta W_{\mathrm{real}}
+
\Delta A_{\mathrm{real}}.
$$

The JSON fields are `realized_delta_u`, `realized_delta_q`, `realized_delta_w`, and `realized_delta_a`, with older compatibility aliases such as `adaptive_same_player_delta_U` and `delta_U_tilde` retained.

The viewer presents realized bookkeeping as side-oriented same-player cycles.

White cycles have the form

$$
W_t \rightarrow B_t \rightarrow W_{t+1},
$$

for example `d4 -> d5 -> e4`. The first two moves are the completed move-response cycle, and the third move is the next White move selected at the new White decision state.

Black cycles have the form

$$
B_t \rightarrow W_{t+1} \rightarrow B_{t+1},
$$

for example `d5 -> e4 -> e5`. The two panels can therefore refer to overlapping played moves while describing different same-player transitions.

The simulator also records one-ply static board-value changes for the first two plies of each displayed cycle:

$$
\Delta U_{\mathrm{first}}=E(B_m)-E(B),
$$

$$
\Delta U_{\mathrm{second}}=E(B_{mr})-E(B_m).
$$

These are saved as `delta_u_first` and `delta_u_second` in the viewer-ready cycle payload, with source aliases `realized_action_delta_u` and `realized_response_delta_u` on the transition. They are simple realized static changes across the two plies, not Q/W/A decompositions.

Viewer states expose `white_realized_cycle` and `black_realized_cycle`. Each is present only when that side has a complete same-player cycle; incomplete cycles are left empty rather than fabricated.

## 7.5 Prediction Error

After a player chooses move $m$, the simulator stores that player's expected value for the subsequent response position.

After the opponent actually chooses response $r^*$, the prediction error is

$$
\epsilon
=
E(B_{m,r^*})
-
\widetilde U_p^{(D)}(B_m),
$$

where $\widetilde U_p^{(D)}(B_m)$ is the adaptive prediction previously made by the observer.

The error therefore compares the realized static board after the reply with the observer's earlier probability-weighted forecast.

---

# 8. Default Parameters and Strategy Presets

## 8.1 Simulation defaults

The command-line simulator uses the following defaults.

| Parameter | Default |
|---|---:|
| Maximum plies | `80` |
| Seed | `1` |
| White beta | `4.0` |
| Black beta | `4.0` |
| Kappa | `1.0` |
| White strategy | `material_conservative` |
| Black strategy | `activity_aggressive` |
| White solidness | preset value |
| Black solidness | preset value |
| Cycle depth (`cdepth`) | `1` |
| Adaptive breadth (`adaptive_c`) | `0.3` |
| Search workers | `1` |
| Parallel minimum branches | `8` |
| Candidate thermo mode | `refined` |
| Refinement policy | `static_eval` |
| Match name | generated automatically |
| Stop on checkmate | yes |
| Stop on stalemate | yes |
| Stop on insufficient material | yes |
| Stop on automatic fivefold repetition | yes |
| Stop on automatic 75-move draw | yes |
| Stop on actual threefold repetition | yes |
| Stop on claimable fifty-move draw | no |
| Stop on claimable threefold repetition | no |

When constructing `MatchConfig` directly in Python, `max_plies` defaults to `120`. The command-line launcher overrides this to `80`.

## 8.2 Built-in strategy presets

Four strategy presets are available from the command line.

| Style coefficient | `material_conservative` | `activity_aggressive` | `tactical_attacker` | `positional_controller` |
|---|---:|---:|---:|---:|
| `material` | `1.5` | `0.9` | `1.0` | `1.2` |
| `preservation` | `2.4` | `1.2` | `0.3` | `1.7` |
| `king_restriction` | `0.2` | `1.1` | `1.4` | `0.8` |
| `king_pressure` | `0.1` | `0.6` | `1.5` | `0.3` |
| `check` | `0.12` | `1.2` | `2.0` | `0.25` |
| `mate` | `80.0` | `80.0` | `80.0` | `80.0` |
| `activity` | `0.2` | `2.0` | `1.4` | `0.8` |
| `king_safety` | `0.8` | `0.35` | `0.8` | `1.1` |
| `center` | `0.2` | `1.2` | `1.4` | `1.8` |
| `promotion` | `0.8` | `0.8` | `1.0` | `0.8` |
| `castle_preserve` | `2.8` | `1.2` | `1.0` | `2.6` |
| `castle_deny` | `0.3` | `0.45` | `0.7` | `0.4` |
| `castle` | `1.0` | `0.55` | `0.4` | `0.9` |
| `development` | `0.8` | `0.5` | `0.4` | `0.9` |
| `phase_castle` | `0.8` | `0.4` | `0.3` | `0.9` |
| `phase_attack` | `0.4` | `1.1` | `1.4` | `0.6` |
| `solidness` | `0.75` | `0.35` | `0.25` | `0.8` |

`tactical_attacker` emphasizes forcing play near the opponent king and accepts more material exposure.

`positional_controller` emphasizes center control, preservation, king safety, and balanced activity.

Arbitrary coefficient combinations can be supplied through Python configuration.

---

# 9. Installation

Run all commands from the repository root.

The documented environment name is `chess`.

Create the environment and install dependencies with

```bash
conda create -n chess python pip
conda run -n chess pip install -r requirements.txt
```

The direct dependencies are:

```text
python-chess>=1.999
pytest>=8.0
```

The package source directory must be available on `PYTHONPATH`.

A simple import test is

```bash
conda run -n chess env PYTHONPATH=src python -c "import thermo_chess; print(thermo_chess.__name__)"
```

---

# 10. Running Simulations

## 10.1 Run with all defaults

```bash
conda run -n chess env PYTHONPATH=src python scripts/run_match.py
```

With the default match name, this writes

```text
data/results/thermo_match.csv
data/results/thermo_match.json
data/games/thermo_match.pgn
```

Files with the same match name are overwritten. Use `--name` to preserve previous runs.

## 10.2 Example with explicit options

```bash
conda run -n chess env PYTHONPATH=src python scripts/run_match.py \
  --name adaptive_cdepth1_c03 \
  --max-plies 120 \
  --seed 23 \
  --beta-white 5.5 \
  --beta-black 5.5 \
  --kappa 1.0 \
  --white-strategy tactical_attacker \
  --black-strategy positional_controller \
  --solidness-white 0.3 \
  --solidness-black 0.85 \
  --cdepth 1 \
  --adaptive-c 0.3 \
  --search-workers 1 \
  --parallel-min-branches 8 \
  --candidate-thermo-mode refined \
  --refinement-policy probability
```

## 10.3 Simulation command-line options

| Flag | Default | Meaning |
|---|---:|---|
| `--max-plies N` | `80` | Maximum number of half-moves before the simulation stops. |
| `--seed N` | `1` | Initializes Python's random generator. Current move selection is deterministic, so this mainly provides reproducibility for future stochastic extensions. |
| `--beta-white X` | `4.0` | White observer's inverse temperature. Larger values concentrate White's move probabilities. |
| `--beta-black X` | `4.0` | Black observer's inverse temperature. Larger values concentrate Black's move probabilities. |
| `--kappa X` | `1.0` | Boltzmann-like scale in pawn units. Effective temperature satisfies $T=1/(\kappa\beta)$. |
| `--white-strategy NAME` | `material_conservative` | White strategy preset. Choices: `material_conservative`, `activity_aggressive`, `tactical_attacker`, `positional_controller`. |
| `--black-strategy NAME` | `activity_aggressive` | Black strategy preset. Same four choices as White. |
| `--solidness-white X` | selected preset's value | Overrides White's `solidness`. Must satisfy `0 <= X <= 1`. |
| `--solidness-black X` | selected preset's value | Overrides Black's `solidness`. Must satisfy `0 <= X <= 1`. |
| `--cdepth N` | `1` | Maximum search depth in complete move-response cycles. Must satisfy `N >= 0`. |
| `--depth N` | none | Deprecated command-line synonym for `--cdepth`. |
| `--adaptive-c X` | `0.3` | Breadth coefficient in $K=\lceil cN_{\mathrm{eff}}\rceil$. Must be positive. |
| `--search-workers N` | `1` | Number of process workers for independent root branches. Use `1` for serial search or `auto` for CPU-based selection. |
| `--parallel-min-branches N` | `8` | Minimum number of root branches required before process-based parallelism is used. |
| `--candidate-thermo-mode MODE` | `refined` | Q/W/A diagnostic coverage. Choices: `selected`, `refined`, `all`. |
| `--refinement-policy POLICY` | `static_eval` | Branch-refinement ranking. Choices: `static_eval`, `probability`. |
| `--name TEXT` | generated | Base filename used for CSV, JSON, and PGN output. The generated name includes strategy names, beta values, effective solidness values, and `cdepth`. |
| `--ignore-threefold` | off | Do not stop when an actual threefold repetition occurs. |
| `--allow-draw-claims` | off | Also stop when a fifty-move or threefold-repetition draw is claimable. |
| `-h`, `--help` | n/a | Print command help and exit. |

Normal stopping conditions are:

- checkmate;
- stalemate;
- insufficient material;
- automatic fivefold repetition;
- automatic 75-move draw;
- actual threefold repetition;
- reaching `--max-plies`.

`--ignore-threefold` disables the actual-threefold stopping condition.

`--allow-draw-claims` additionally stops on claimable fifty-move and threefold-repetition draws.

Claimable draws are not treated as terminal by the search unless this simulation-level stopping option is enabled.

## 10.4 Search-cost note

`cdepth > 1` can be expensive, especially in low-entropy positions. Low entropy intentionally permits deeper local refinement.

The simulation also prepares both observers' viewer panels for every saved position, which adds additional search work.

---

# 11. Viewer

Start the dynamic viewer with

```bash
conda run -n chess env PYTHONPATH=src python scripts/view_match.py
```

The default address is

```text
http://127.0.0.1:8765/
```

## 11.1 Viewer options

| Flag | Default | Meaning |
|---|---:|---|
| `--results-dir PATH` | `data/results` | Directory containing match JSON files. |
| `--host HOST` | `127.0.0.1` | Interface on which the HTTP server listens. |
| `--port N` | `8765` | HTTP port. |
| `--open` | off | Ask Python to open the viewer URL in the default browser. |
| `-h`, `--help` | n/a | Print command help and exit. |

Example:

```bash
conda run -n chess env PYTHONPATH=src python scripts/view_match.py \
  --results-dir data/results \
  --host 127.0.0.1 \
  --port 8877 \
  --open
```

The server reads JSON files dynamically. The game list refreshes every ten seconds.

The viewer performs no chess search or expected-value calculation. It displays precomputed data contained in the saved JSON.

For each position, the viewer can display:

- board, FEN, side to move, move history, and static $E(B)$;
- configured `cdepth` and `adaptive_c`;
- adaptive expected value, entropy, $N_{\mathrm{eff}}$, $K$, response $K'$, and branch values;
- move ranks, probabilities, `expected_next_U`, and `expected_delta_u_star`;
- candidate Q/W/A diagnostics when they were requested by `candidate_thermo_mode`;
- aligned move and $E(B_m)$ columns for both observers;
- White and Black realized-cycle panels above their corresponding analysis sections;
- completed same-player transition diagnostics;
- old/new effective move counts, support changes, and common probability mass.

---

# 12. Saved Data

## 12.1 CSV

The CSV contains one row per played ply.

Important fields include:

| Field | Meaning |
|---|---|
| `E_before`, `E_after` | Universal static evaluations before and after the move. |
| `U_current` | Moving observer's shallow landscape $U_p^{1/2}(B)$. |
| `U_after_move` | Selected candidate's response-averaged `expected_next_U`. |
| `expected_delta_u_star` | Move-selection quantity `expected_next_U - U_current`. |
| `U_before`, `U_after` | Consecutive same-player adaptive expected values when such a transition exists. |
| `adaptive_same_player_delta_U` | Change in adaptive same-player expected value. |
| `adaptive_same_player_delta_Q` | Probability-redistribution contribution. |
| `adaptive_same_player_delta_W` | Observable-change contribution. |
| `adaptive_same_player_delta_A` | Action-accessibility contribution. |
| `decomposition_error` | Numerical residual between $\Delta U$ and $\Delta Q+\Delta W+\Delta A$. |
| `N_eff_before`, `N_eff_after` | Effective move counts for consecutive same-player states. |
| `entropy_current` | Root move-distribution entropy. |
| `N_eff` | Root effective number of moves. |
| `selected_depth` | Local entropy-limited `cdepth`. |
| `K_expanded` | Number of root branches selected for refinement. |
| `nodes_evaluated` | Adaptive value nodes computed. |
| `static_evaluations` | Uncached static evaluations. |
| `recursive_nodes` | Refined recursive child traversals. |
| `maximum_depth_reached` | Deepest search level reached. |
| `average_K` | Mean selected breadth across recursively expanded nodes. |
| `cache_hits` | Reused cached values and landscapes. |
| `search_elapsed_time` | Search time in seconds. |
| `prediction_error` | Realized static reply value minus the earlier adaptive prediction. |
| repetition fields | Twofold, threefold, fivefold, and draw-claim metadata. |

## 12.2 JSON

The JSON contains the complete simulation result, including:

- match and search configuration;
- serialized White and Black style coefficients;
- static evaluation weights;
- result, terminal reason, final FEN, and repetition metadata;
- one record per played ply;
- candidate scores and branch diagnostics;
- adaptive search summaries;
- same-player thermodynamic transitions;
- viewer-ready states for every saved position.

Candidate records also indicate whether a move belonged to the root top-$K$ refinement set.

When candidate thermodynamic diagnostics are enabled for that move, response-level data include response $N_{\mathrm{eff}}$, response $K'$, and Q/W/A-related records.

## 12.3 PGN

The PGN stores the played move sequence and standard result headers.

It does not contain the full thermodynamic or adaptive-search diagnostics. Use the CSV or JSON files for quantitative analysis.

---

# 13. Analysis, Tests, and Benchmark

## 13.1 Inspect a saved ply

```bash
conda run -n chess env PYTHONPATH=src python scripts/analyze_match.py \
  data/results/thermo_match.json \
  --ply 8 \
  --sort probability \
  --limit 20
```

Options:

| Argument | Default | Meaning |
|---|---:|---|
| `match_json` | `data/results/thermo_match.json` | JSON file to inspect. |
| `--ply N` | `1` | Played ply whose saved move landscape is printed. |
| `--sort MODE` | `probability` | Sort key. Choices: `probability`, `static_after`, `phi`. |
| `--limit N` | `20` | Maximum number of moves printed. |

The script reads saved results. It does not rerun adaptive search.

## 13.2 Run tests

```bash
conda run -n chess env PYTHONPATH=src pytest -q
```

The tests cover probability normalization, entropy, effective move counts, adaptive breadth and depth, full fallback probability mass, terminal handling, move selection, feature conventions, serialization, and viewer-ready output.

## 13.3 Optional benchmark

```bash
conda run -n chess env PYTHONPATH=src python scripts/benchmark_adaptive.py
```

Benchmark options:

| Flag | Default | Meaning |
|---|---:|---|
| `--fen FEN` | built-in middlegame position | Replace the benchmark board. Quote FEN strings in the shell. |
| `--beta X` | `4.0` | Inverse temperature used for every benchmark depth. |
| `--adaptive-c X` | `0.3` | Adaptive breadth coefficient used for every benchmark depth. |

The benchmark does not modify or regenerate saved match files.

---

# 14. Python Configuration

Command-line flags expose the common simulation controls. For custom style coefficients, evaluator weights, adaptive thresholds, and output directories, configure the simulation directly in Python.

```python
from pathlib import Path

from thermo_chess.evaluation import EvaluationWeights
from thermo_chess.measure import Style
from thermo_chess.simulation import MatchConfig, simulate_match

white_style = Style(
    material=1.8,
    preservation=2.0,
    activity=0.4,
    center=0.5,
    castle_preserve=0.7,
    castle=0.8,
    development=0.8,
    phase_castle=0.8,
    phase_attack=0.5,
    solidness=0.75,
)

black_style = Style(
    material=0.9,
    preservation=1.0,
    activity=2.2,
    king_pressure=0.8,
)

config = MatchConfig(
    max_plies=120,
    seed=23,
    beta_white=5.0,
    beta_black=5.0,
    white_strategy="material_conservative",
    black_strategy="activity_aggressive",
    white_solidness=0.75,
    black_solidness=0.35,
    cdepth=1,
    adaptive_c=0.3,
    refinement_policy="static_eval",
    depth4_max_neff=4.0,
    depth3_max_neff=8.0,
    depth2_max_neff=15.0,
    match_name="custom_adaptive_match",
    results_dir=Path("data/results"),
    games_dir=Path("data/games"),
)

simulate_match(
    white_style=white_style,
    black_style=black_style,
    eval_weights=EvaluationWeights(),
    config=config,
)
```

When a `Style` object is passed directly for a player, that object determines the player's coefficients. If no custom style object is supplied, the corresponding strategy preset in `MatchConfig` is used.

The documented validation rules are:

```text
cdepth >= 0
adaptive_c > 0
0 <= solidness <= 1
```

The adaptive-depth thresholds must be positive and strictly increasing:

```text
depth4_max_neff < depth3_max_neff < depth2_max_neff
```

The default threshold values are:

```text
depth4_max_neff = 4.0
depth3_max_neff = 8.0
depth2_max_neff = 15.0
```

---

# 15. Project Structure

```text
src/thermo_chess/
  evaluation.py      universal static evaluator E(B)
  features.py        board features, move features, exchange exposure
  measure.py         style potential, softmax, entropy, move landscapes
  metrics.py         entropy and effective-number utilities
  search.py          adaptive expected-value recursion and caching
  player.py          observer analysis and White-max/Black-min move choice
  simulation.py      match loop and CSV/JSON/PGN serialization
  live_viewer.py     dynamic HTTP viewer

scripts/
  run_match.py           simulation CLI
  view_match.py          viewer CLI
  analyze_match.py       saved-landscape inspection CLI
  benchmark_adaptive.py  optional adaptive-search benchmark

data/games/              generated PGN files
data/results/            generated CSV and JSON files
tests/                   mathematical and regression tests
match_viewer.html        legacy static viewer artifact
```

---

# 16. Performance and Limitations

- Adaptive search reduces tree growth but does not make large `cdepth` values inexpensive.
- Low-entropy positions are deliberately allowed to retain deeper search.
- Every legal root candidate contributes to the final expected-value calculation, even when only a subset receives refined evaluation.
- Viewer-state generation evaluates both observers at every saved position and therefore increases simulation time.
- The model is designed for interpretability and experimentation, not competitive chess strength.
- Static evaluation and move-style features are intentionally compact and hand-designed.
- Exchange exposure is a local capture-sequence calculation, not a general tactical engine.
- Search is CPU-bound.
- Process workers can parallelize independent root branches, but deep searches can still be expensive.
- JSON files can become large because they contain viewer-ready states and candidate diagnostics.

The central computational idea is

$$
\text{low entropy}
\Rightarrow
\text{few effective moves}
\Rightarrow
\text{narrower and potentially deeper refinement},
$$

whereas

$$
\text{high entropy}
\Rightarrow
\text{many comparable moves}
\Rightarrow
\text{broader but locally shallower refinement}.
$$

The observer-dependent move measure therefore controls both predicted behavior and the allocation of computational attention through the search tree.
