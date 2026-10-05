# Thermodynamic Toy Chess

`chess_thermodynamic` is an experimental chess simulator for studying decision making under limited computational resources.

The central idea is to separate three objects:

1. **How a board is objectively evaluated**, through a shared static evaluator $E(B)$.
2. **Which legal moves an observer considers attractive or likely**, through observer-dependent probabilities $P_p(B_m\mid B)$.
3. **How much of the future tree the observer actually explores**, through entropy-controlled branch truncation.

The model is not a conventional chess engine. It does not use minimax, alpha-beta pruning, opening books, or an external chess engine. Future positions are combined through probability-weighted expectations. White and Black use the same board-evaluation function and the same sign convention, but may assign different probabilities to the same moves because they have different strategic preferences.

The framework is designed to make the following question explicit:

> If a player cannot analyze every continuation, how does selective attention to only the most meaningful branches alter the move that appears best?

A low-probability continuation may contain a decisive tactical resource. Truncating that continuation makes the search cheaper, but can change the subjective ranking of the available moves. The simulator is built to expose this tradeoff.

---

## Contents

1. [Conceptual overview](#1-conceptual-overview)
2. [Board and move notation](#2-board-and-move-notation)
3. [Observer-dependent move probabilities](#3-observer-dependent-move-probabilities)
4. [Universal board evaluation](#4-universal-board-evaluation)
5. [The shallow decision landscape](#5-the-shallow-decision-landscape)
6. [Entropy and bounded attention](#6-entropy-and-bounded-attention)
7. [Prediction depth and the G quantity](#7-prediction-depth-and-the-g-quantity)
8. [Truncated prediction and G tilde](#8-truncated-prediction-and-g-tilde)
9. [Move selection](#9-move-selection)
10. [Q/W/A decomposition](#10-qwa-decomposition)
11. [Realized same-player Q/W/A](#11-realized-same-player-qwa)
12. [Move features and strategy presets](#12-move-features-and-strategy-presets)
13. [Configuration and defaults](#13-configuration-and-defaults)
14. [Installation](#14-installation)
15. [Running simulations](#15-running-simulations)
16. [Viewer](#16-viewer)
17. [Saved data](#17-saved-data)
18. [Analysis, tests, and benchmark](#18-analysis-tests-and-benchmark)
19. [Python configuration](#19-python-configuration)
20. [Project structure](#20-project-structure)
21. [Performance and limitations](#21-performance-and-limitations)

---

# 1. Conceptual Overview

Consider a chess position $B$.

The simulator assigns every legal move $m$ two different quantities:

- an **observer-dependent probability**
  $P_p(B_m\mid B)$, which describes how attractive or likely that move appears to observer $p$;

- an **observer-independent board value**
  $E(B_m)$, which evaluates the position reached after the move.

These quantities answer different questions.

The probability model asks:

> Which moves does this observer naturally pay attention to?

The static evaluator asks:

> How favorable is the resulting board, using the same objective convention for both players?

The two are combined through expectation values.

The global evaluation convention is

$$
E(B)>0 \quad\text{favors White},
\qquad
E(B)<0 \quad\text{favors Black}.
$$

White therefore prefers larger values and Black prefers smaller values.

The main flow of the model is

```text
legal moves
    ↓
observer-dependent move probabilities P
    ↓
full shallow decision landscape U^(1/2)
    ↓
Shannon entropy S and effective number of moves N_eff
    ↓
top-K branch truncation
    ↓
renormalized search probabilities P~
    ↓
future same-player landscapes
    ↓
bounded score G~_m^d
    ↓
move selection
    ↓
Q/W/A interpretation
```

The search horizon `depth` is fixed. Entropy does **not** change search depth. It only changes how many branches are retained at each intermediate decision node.

---

# 2. Board and Move Notation

Let

- $B$ be a chess position;
- $\mathcal M(B)$ be the set of legal moves in $B$;
- $m\in\mathcal M(B)$ be one legal move;
- $B_m$ be the board after move $m$;
- $r\in\mathcal M(B_m)$ be one legal response;
- $B_{mr}$ be the board after $m$ followed by $r$;
- $p$ denote the observer whose style coefficients are being used;
- $c\in\{\mathrm{White},\mathrm{Black}\}$ denote a chess color;
- $\bar c$ denote the opposite color.

A **ply** is one move by one side.

The search parameter

```text
depth
```

is the total prediction horizon in plies.

Only odd values are allowed:

```text
depth >= 3
depth is odd
```

The reason is conceptual: the framework compares the current player's decision landscape with a future decision landscape in which the **same player is again to move**.

For example,

```text
depth = 3

B
  -> B_m        current player chooses m
  -> B_mr       opponent responds r
  -> evaluate the current player's new shallow landscape
```

For `depth = 5`,

```text
B
  -> B_m
  -> B_mr
  -> B_mrm'
  -> B_mrm'r'
  -> evaluate the original player's new shallow landscape
```

Thus the terminal landscape is always associated with the same player who is deciding at $B$.

---

# 3. Observer-Dependent Move Probabilities

Each observer $p$ has a vector of non-negative strategic coefficients

$$
\lambda_p
=
(\lambda_{\mathrm{material}},
\lambda_{\mathrm{center}},
\lambda_{\mathrm{development}},
\lambda_{\mathrm{castling}},
\lambda_{\mathrm{king\_safety}},
\lambda_{\mathrm{king\_pressure}}).
$$

These coefficients do not evaluate the board directly. They determine how strongly the observer reacts to six **mover-relative move features**, each already expressed in pawn units:

$$
F_k(m,B).
$$

The six features are

$$
\boxed{
\mathcal F
=
\{
\mathrm{material},
\mathrm{center},
\mathrm{development},
\mathrm{castling},
\mathrm{king\_safety},
\mathrm{king\_pressure}
\}.
}
$$

The move potential is

$$
\boxed{
\Phi_p(m,B)
=
\sum_{k\in\mathcal F}
\lambda^{\mathrm{eff}}_{p,k}(B)\,
F_k(m,B),
}
$$

where the effective coefficients depend on the game phase and are normalized so that

$$
\sum_{k\in\mathcal F}
\lambda^{\mathrm{eff}}_{p,k}(B)=1.
$$

Therefore $\Phi_p$ is a weighted average of pawn-valued move features.

The probability assigned to move $m$ is

$$
P_p(B_m\mid B)
=
\frac{
\exp\!\left(\beta_p\Phi_p(m,B)\right)
}{
\sum_{m'\in\mathcal M(B)}
\exp\!\left(\beta_p\Phi_p(m',B)\right)
}.
$$

Equivalently,

$$
\beta_p=\frac{1}{\kappa T_p},
\qquad
P_p(B_m\mid B)
\propto
\exp\!\left(\frac{\Phi_p(m,B)}{\kappa T_p}\right).
$$

Interpretation:

- larger `beta` produces a more concentrated move distribution;
- smaller `beta` produces a flatter move distribution;
- `kappa` fixes the energy scale;
- the probability model is observer-dependent;
- the static evaluator $E(B)$ is shared by both players.

## 3.1 Mover-relative perspective

The same observer style is propagated throughout that observer's search tree, including nodes where the opponent is to move.

However, move features are always evaluated from the perspective of the side **actually making the move**.

For example, when Black is responding inside White's search, a material gain by Black, improved Black king safety, or increased pressure on the White king all contribute positively to Black's mover-relative features.

Branches are always ranked by descending probability. There is no manual reversal of the probability ordering for opponent moves.

## 3.2 Raw features, energy scales, and lambdas

The move potential keeps three concepts separate:

- $f_k(m,B)$: a raw chess descriptor;
- $s_k$: a fixed conversion scale into pawn-equivalent units;
- $F_k(m,B)=s_kf_k(m,B)$: the pawn-valued feature entering $\Phi$.

The scale $s_k$ ensures that heterogeneous chess descriptors are represented on comparable pawn-equivalent energy scales. The lambda coefficient does not perform this normalization; it represents strategic preference after the feature has already been converted to a common scale.

| metric | raw descriptor | conversion scale | final units |
|---|---|---:|---|
| material | relative material change | `s_M = 1` | pawns |
| center | change in weighted controlled center squares | `s_C = 0.10` pawns/control unit | pawns |
| development | change in developed minor slots | `s_D = 0.15` pawns/slot | pawns |
| castling | castle/right-loss events | `s_castle = 0.40`, `s_rights = 0.20` | pawns |
| king safety | change in raw structural safety | `s_K = 0.12` pawns/raw unit | pawns |
| king pressure | safe opponent-king squares removed | `s_P = 0.10` pawns/square | pawns |

In formula form:

$$
s_M=1,\quad
s_C=0.10,\quad
s_D=0.15,\quad
s_{\rm castle}=0.40,\quad
s_{\rm rights}=0.20,\quad
s_K=0.12,\quad
s_P=0.10.
$$

## 3.3 Move-feature definitions

Let $c$ be the side making move $m$, and $\bar c$ the opponent.

### Material

$$
f_{\mathrm{material}}(m,B)
=
M_{\mathrm{rel}}(B_m,c)
-
M_{\mathrm{rel}}(B,c).
$$

Since $s_M=1$, $F_{\mathrm{material}}=f_{\mathrm{material}}$.

### Center

$$
C_{\mathrm{raw}}(B,c)
=
\sum_{q\in\mathcal C}\mathbf 1[c\text{ attacks }q]
+
0.35\sum_{q\in\mathcal C_{\mathrm{ext}}}\mathbf 1[c\text{ attacks }q].
$$

$$
f_{\mathrm{center}}(m,B)
=
C_{\mathrm{raw}}(B_m,c)-C_{\mathrm{raw}}(B,c),
\qquad
F_{\mathrm{center}}=0.10f_{\mathrm{center}}.
$$

### Development

The original minor-piece squares are

```text
White: b1 c1 f1 g1
Black: b8 c8 f8 g8
```

Let $N_D(B,c)\in\{0,1,2,3,4\}$ count original minor-piece slots vacated. Then

$$
f_{\mathrm{development}}(m,B)
=
N_D(B_m,c)-N_D(B,c),
\qquad
F_{\mathrm{development}}=0.15f_{\mathrm{development}}.
$$

### Castling

Let $R_c(B)\in\{0,1,2\}$ be the number of retained castling rights. Then

$$
f_{\rm castle}=\mathbf 1[m\text{ is castling}],
\qquad
f_{\rm rights}=-L_R(m,B,c),
$$

where $L_R$ is the number of mover castling rights lost by the move, excluding rights consumed by castling itself. The single castling feature is

$$
F_{\mathrm{castling}}
=
0.40f_{\rm castle}
+
0.20f_{\rm rights}.
$$

### King safety

Raw structural king safety is

$$
K_{\rm raw}(B,c)
=
N_{\mathrm{shield}}
-
\frac{5}{6}N_{\mathrm{enemy}}
-
1.25N_{\mathrm{unshielded}}.
$$

Then

$$
f_{\mathrm{king\_safety}}(m,B)
=
K_{\rm raw}(B_m,c)-K_{\rm raw}(B,c),
\qquad
F_{\mathrm{king\_safety}}=0.12f_{\mathrm{king\_safety}}.
$$

### King pressure

Let $N_{\rm safe}(B,\bar c;c)$ be the number of geometrically adjacent opponent-king squares that lie on the board, are not occupied by a piece of color $\bar c$, and are not attacked by mover color $c$.

$$
f_{\mathrm{king\_pressure}}(m,B)
=
N_{\rm safe}(B,\bar c;c)
-
N_{\rm safe}(B_m,\bar c;c),
$$

$$
F_{\mathrm{king\_pressure}}
=
0.10f_{\mathrm{king\_pressure}}.
$$

A positive value means that the move reduces the opponent king's safe freedom.

## 3.4 Game phase and effective style coefficients

Let $M_{\mathrm{np}}(B)$ be the total non-pawn material on the board, counting knights, bishops, rooks, and queens for both colors. Its initial maximum is $62$.

Define

$$
g(B)
=
\operatorname{clamp}_{[0,1]}
\left(
1-\frac{M_{\mathrm{np}}(B)}{62}
\right).
$$

Thus $g\simeq0$ corresponds to the opening and $g\simeq1$ to a low-material late game.

The phase factors are

$$
w_{\mathrm{material}}=1,
\qquad
w_{\mathrm{king\_pressure}}=1,
$$

and

$$
w_{\mathrm{center}}
=
w_{\mathrm{development}}
=
w_{\mathrm{castling}}
=
w_{\mathrm{king\_safety}}
=
1-g.
$$

Given raw non-negative style coefficients $\lambda_{p,k}$, define

$$
\bar\lambda_{p,k}(B)
=
\lambda_{p,k}w_k(g(B)),
$$

and normalize them as

$$
\boxed{
\lambda^{\mathrm{eff}}_{p,k}(B)
=
\frac{
\bar\lambda_{p,k}(B)
}{
\sum_j\bar\lambda_{p,j}(B)
}.
}
$$

The final move potential is therefore

$$
\boxed{
\Phi_p(m,B)
=
\sum_k
\lambda^{\mathrm{eff}}_{p,k}(B)
s_kf_k(m,B).
}
$$

The implementation also stores raw coefficients, phase factors, normalized effective coefficients, feature contributions, the unmodulated base potential, the phase-induced correction, and the final potential for diagnostics.

---

# 4. Universal Board Evaluation

The move probabilities describe **which moves an observer considers attractive**. The board evaluator answers the separate question:

> How favorable is this board under the common White-positive convention?

The same static evaluator is used by both players.

The current evaluator is

$$
\boxed{
E(B)
=
\alpha_M\Delta M(B)
+
\alpha_P\Delta P(B)
+
\alpha_L\Delta L(B)
+
\alpha_C\Delta C(B)
+
\alpha_K\Delta K(B).
}
$$

The five metrics are material, pawn structure, mobility, center control, and king safety. All non-material descriptors are expressed in approximate pawn units.

The default weights are all equal to one:

$$
\alpha_M
=
\alpha_P
=
\alpha_L
=
\alpha_C
=
\alpha_K
=
1.
$$

Hence, by default,

$$
E(B)
=
\Delta M(B)
+
\Delta P(B)
+
\Delta L(B)
+
\Delta C(B)
+
\Delta K(B).
$$

The sign convention remains

$$
E(B)>0 \text{ favors White},
\qquad
E(B)<0 \text{ favors Black}.
$$

## 4.1 Piece values and material

| Piece | Value |
|---|---:|
| Pawn | `1` |
| Knight | `3` |
| Bishop | `3` |
| Rook | `5` |
| Queen | `9` |
| King | `0` |

Let

$$
M(B,c)=\sum_t v(t)n_t(B,c).
$$

Then

$$
\Delta M(B)
=
M(B,\mathrm{White})
-
M(B,\mathrm{Black}).
$$

## 4.2 Pawn structure

For each color $c$, count passed, connected, isolated, and doubled pawns.

Definitions:

- isolated: no friendly pawn exists on either adjacent file;
- doubled: if $n_f$ pawns occupy file $f$, then

$$
N_{\rm doubled}
=
\sum_f\max(0,n_f-1);
$$

- passed: no enemy pawn lies ahead of the pawn on its own file or either adjacent file. "Ahead" means toward increasing ranks for White and toward decreasing ranks for Black;
- connected: a friendly pawn exists on an adjacent file with rank difference at most one.

Connected pawns are counted pawn-by-pawn, not pair-by-pair.

Define

$$
P(B,c)
=
0.30N_{\mathrm{passed}}
+
0.10N_{\mathrm{connected}}
-
0.15N_{\mathrm{isolated}}
-
0.15N_{\mathrm{doubled}}.
$$

Then

$$
\Delta P(B)
=
P(B,\mathrm{White})
-
P(B,\mathrm{Black}).
$$

## 4.3 Mobility

Let $N_{\mathrm{legal}}(B,c)$ be the number of legal moves available to color $c$.

This is computed by treating $c$ as the side to move in the same piece configuration, without inheriting turn-specific en-passant rights from the original board.

Define

$$
L(B,c)
=
0.03N_{\mathrm{legal}}(B,c),
$$

so that

$$
\Delta L(B)
=
L(B,\mathrm{White})
-
L(B,\mathrm{Black}).
$$

## 4.4 Center control

Let

$$
\mathcal C=\{d4,e4,d5,e5\}
$$

and

$$
\mathcal C_{\mathrm{ext}}
=
\{c3,d3,e3,f3,c4,f4,c5,f5,c6,d6,e6,f6\}.
$$

Define

$$
C(B,c)
=
0.10
\sum_{s\in\mathcal C}
\mathbf 1[c\text{ attacks }s]
+
0.035
\sum_{s\in\mathcal C_{\mathrm{ext}}}
\mathbf 1[c\text{ attacks }s].
$$

Then

$$
\Delta C(B)
=
C(B,\mathrm{White})
-
C(B,\mathrm{Black}).
$$

This measures attacked squares rather than occupancy.

## 4.5 King safety

Let $Z(B,c)$ be the set of squares adjacent to king $c$.

Define:

- $N_{\mathrm{shield}}(B,c)$: adjacent friendly pawns;
- $N_{\mathrm{enemy}}(B,c)$: adjacent squares attacked by the opponent;
- $N_{\mathrm{unshielded}}(B,c)$: among the king's file and immediately adjacent files, count those containing no friendly pawn ahead of the king.

Then

$$
K(B,c)
=
0.12N_{\mathrm{shield}}
-
0.10N_{\mathrm{enemy}}
-
0.15N_{\mathrm{unshielded}}.
$$

The evaluator uses

$$
\Delta K(B)
=
K(B,\mathrm{White})
-
K(B,\mathrm{Black}).
$$

## 4.6 Default evaluation weights

| Static component | Default |
|---|---:|
| Material | `1.0` |
| Pawn structure | `1.0` |
| Mobility | `1.0` |
| Center control | `1.0` |
| King safety | `1.0` |
| Checkmate magnitude | `10000.0` |

The component weights can be changed through `EvaluationWeights`.

## 4.7 Terminal positions

If the side to move is checkmated, the evaluator returns the appropriately signed checkmate value.

With the default checkmate magnitude:

- White checkmated $\rightarrow -10000$;
- Black checkmated $\rightarrow +10000$.

The following positions evaluate to zero:

- stalemate;
- insufficient material;
- automatic 75-move draw;
- automatic fivefold repetition.

These terminal rules override the ordinary five-component evaluation.

---

# 5. The Shallow Decision Landscape

The first important expectation value is

$$
\boxed{
U_p^{1/2}(B)
=
\sum_{m\in\mathcal M(B)}
P_p(B_m\mid B)E(B_m)
}
$$

and uses **all legal moves**.

This quantity should not be interpreted simply as "the value of the current board." It is the observer's subjective value of the **decision landscape currently available**.

It combines:

- the quality of the positions reachable in one move;
- the observer's probability of considering or selecting each move.

In words:

> $U_p^{1/2}(B)$ asks: if I sample one of my currently available moves according to my own move distribution, what board value do I expect after making it?

### Short example

Suppose a player has three legal moves:

| move | probability | board value |
|---|---:|---:|
| `m1` | 0.60 | 2 |
| `m2` | 0.30 | 1 |
| `m3` | 0.10 | -4 |

Then

$$
U^{1/2}(B)
=
0.60(2)+0.30(1)+0.10(-4)
=
1.1.
$$

The value `1.1` summarizes the whole one-move landscape, not any single move.

This quantity becomes the reference point for the future-landscape score $G_m^d$.

---

# 6. Entropy and Bounded Attention

A player cannot usually explore every continuation equally deeply.

The simulator models limited attention through the entropy of the move distribution.

For observer $p$,

$$
S_p(B)
=
-\sum_{m\in\mathcal M(B)}
P_p(B_m\mid B)
\log P_p(B_m\mid B).
$$

Define the effective number of moves

$$
\boxed{
N_{\mathrm{eff}}(B)=e^{S_p(B)}.
}
$$

Interpretation:

- if one move dominates, $N_{\mathrm{eff}}$ is close to `1`;
- if many moves have comparable probabilities, $N_{\mathrm{eff}}$ is larger;
- for a perfectly uniform distribution over $K$ moves, $N_{\mathrm{eff}}=K$.

Thus $N_{\mathrm{eff}}$ measures how many moves are meaningfully represented by the probability distribution.

## 6.1 Adaptive breadth

Let

```text
adaptive_c = c > 0
```

be the breadth coefficient.

At any branching board $B$, retain

$$
\boxed{
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
}
$$

The retained set $\mathcal C(B)$ contains the top-$K(B)$ moves ranked by probability.

The same rule is applied recursively at every intermediate search node.

For example, if

$$
N_{\mathrm{eff}}=8.2,
\qquad
c=0.3,
$$

then

$$
K=\lceil 0.3\times8.2\rceil=3.
$$

Only the three highest-probability moves are explored beyond that node.

## 6.2 Renormalized search probabilities

The original probabilities of the retained branches no longer sum to one after truncation.

Therefore the bounded search uses

$$
\boxed{
\widetilde P(B_m\mid B)
=
\frac{
P(B_m\mid B)
}{
\sum_{a\in\mathcal C(B)}
P(B_a\mid B)
}
}
$$

for $m\in\mathcal C(B)$.

The quantity

$$
\sum_{a\in\mathcal C(B)}P(B_a\mid B)
$$

is the **retained probability mass** before renormalization.

The distinction is important:

- $P$ describes the observer's full move distribution;
- $\widetilde P$ describes the conditional distribution used by the bounded search after low-priority branches have been discarded.

At the root, legal moves outside the retained set are still reported with their original probability and immediate static evaluation, but:

- they do not receive a recursive $G$ score;
- they do not receive Q/W/A diagnostics;
- they cannot be selected as the played move.

---

# 7. Prediction Depth and the G Quantity

Before introducing truncation, it is useful to define the ideal full-search quantity.

For a candidate root move $m$, let $d$ be an odd ply horizon.

Define

$$
\boxed{
G_m^d
=
U^{(d-1)/2}(B_m)
-
U^{1/2}(B).
}
$$

Conceptually, $G_m^d$ asks:

> If I choose move $m$, how will my decision landscape have changed when it is my turn again after looking ahead $d$ plies?

This is why an odd horizon is required.

## 7.1 Depth 3 Case

For `depth = 3`,

$$
B
\to B_m
\to B_{mr},
$$

and the same player who moved at $B$ is again to move at $B_{mr}$.

The ideal full-search quantity is

$$
\boxed{
G_m^3
=
\sum_{r\in\mathcal M(B_m)}
P(B_{mr}\mid B_m)
U^{1/2}(B_{mr})
-
U^{1/2}(B).
}
$$

Interpretation:

- $G_m^3>0$: after choosing $m$ and allowing for the opponent's possible reply, the player's expected next-turn landscape is better than the landscape available now;
- $G_m^3<0$: choosing $m$ is expected to leave the player with a worse next-turn landscape.

### Queen-bait example

Suppose a move captures the opponent's queen.

Its immediate value

$$
E(B_m)
$$

may be very favorable.

But perhaps the capture is bait. After the opponent responds, the player may be close to checkmate and have very poor legal continuations.

Then it is possible to have

$$
E(B_m)\gg E(B)
$$

but

$$
G_m^3<0.
$$

The distinction is important: $E(B_m)$ describes the immediate board after the move, while $G_m^3$ describes the change in the player's **future decision landscape**.

## 7.2 Depth 5 Case

For `depth = 5`,

$$
B
\to B_m
\to B_{mr}
\to B_{mrm'}
\to B_{mrm'r'}.
$$

The terminal quantity is then

$$
U^{1/2}(B_{mrm'r'}).
$$

The expectation averages over every complete retained/full path with the corresponding product of conditional move probabilities.

More generally, an odd `depth` always ends at a position where the original player is again to move, and the terminal observable is the full shallow landscape $U^{1/2}$ of that position.

## 7.3 Full G Is Mainly A Conceptual Reference

Computing $G_m^d$ exactly requires retaining every legal branch through the whole horizon.

That defeats the purpose of modeling bounded computation.

The simulator therefore normally computes the truncated quantity

$$
\widetilde G_m^d,
$$

introduced next.

---

# 8. Truncated Prediction And G Tilde

The actual bounded agent recursively retains only the most probable branches.

For a refined root move $m$, define

$$
\boxed{
\widetilde G_m^d
=
\mathbb E_{\widetilde P}
\left[
U^{1/2}(B_{\mathrm{terminal}})
\mid m
\right]
-
U^{1/2}(B).
}
$$

The expectation is taken over the recursively truncated tree using the renormalized probabilities $\widetilde P$.

The terminal same-player landscape itself is **not truncated**:

$$
U^{1/2}(B_{\mathrm{terminal}})
=
\sum_{a\in\mathcal M(B_{\mathrm{terminal}})}
P(B_a\mid B_{\mathrm{terminal}})
E(B_a).
$$

This is deliberate. Truncation determines which future states are reached by the search; once a terminal same-player state is reached, the quantity being compared is the complete immediate decision landscape available there.

## 8.1 Explicit formula for `depth = 3`

Let $\mathcal C(B_m)$ be the retained opponent responses.

Then

$$
\boxed{
\widetilde G_m^3
=
\sum_{r\in\mathcal C(B_m)}
\widetilde P(B_{mr}\mid B_m)
U^{1/2}(B_{mr})
-
U^{1/2}(B).
}
$$

This is the quantity used for actual decision making.

## 8.2 Full versus bounded evaluation

In general,

$$
\widetilde G_m^d
\neq
G_m^d.
$$

They become equal only when no branch relevant to the requested horizon is discarded.

It is therefore possible that two moves satisfy

$$
\widetilde G_{m_1}^d
<
\widetilde G_{m_2}^d
$$

while

$$
G_{m_1}^d
>
G_{m_2}^d.
$$

This is one of the main phenomena the simulator is intended to study.

The bounded agent selects $m_2$, because according to the explored tree it looks better. A full search would have preferred $m_1$.

The mismatch is not numerical noise. It is a structural consequence of selective search.

### Tactical interpretation

Suppose a low-probability opponent response contains a decisive tactic.

If that response is discarded by the top-$K$ search, then its consequences do not contribute to $\widetilde G_m^d$.

The agent has therefore gained computational efficiency by ignoring a continuation, but this may make the selected move objectively worse than a move that would have been preferred under complete exploration.

The breadth parameter `adaptive_c` therefore controls a tradeoff:

```text
smaller adaptive_c
    -> fewer retained branches
    -> cheaper search
    -> greater risk of missing important low-probability continuations

larger adaptive_c
    -> more retained branches
    -> more expensive search
    -> bounded evaluation approaches full evaluation
```

No regret observable is currently implemented.

## 8.3 Terminal moves

If a candidate move terminates the game immediately, there is no future same-player decision landscape.

The simulator uses the boundary convention

$$
\boxed{
\widetilde G_m^d
=
E(B_m)-U^{1/2}(B).
}
$$

This should not be interpreted as an ordinary future-landscape comparison. The recursion has ended because the game itself has ended.

---

# 9. Move Selection

Only refined root moves receive $\widetilde G_m^d$.

White selects

$$
\boxed{
m_W^*
=
\arg\max_{m\in\mathcal C(B)}
\widetilde G_m^d
}
$$

and Black selects

$$
\boxed{
m_B^*
=
\arg\min_{m\in\mathcal C(B)}
\widetilde G_m^d.
}
$$

The canonical stored field is

```text
g_tilde
```

Unrefined root moves cannot be selected.

This is a bounded-rational decision rule:

1. the move probabilities determine which candidates receive attention;
2. the bounded search estimates how each refined move changes the future same-player landscape;
3. the player chooses the best refined move according to the global White-positive evaluation convention.

---

# 10. Q/W/A Decomposition

The Q/W/A decomposition explains **why** a decision landscape changes.

Consider two complete shallow landscapes,

$$
U
=
\sum_a p_aO_a,
\qquad
U'
=
\sum_a p'_aO'_a.
$$

In chess,

$$
O_a=E(B_a).
$$

The support can change because legal moves may appear or disappear.

Let

- $\mathcal L_\cap$: moves present in both landscapes;
- $\mathcal L_+$: newly available moves;
- $\mathcal L_-$: moves that disappeared.

Then

$$
\boxed{
\Delta U
=
\Delta Q+\Delta W+\Delta A.
}
$$

## 10.1 Probability Redistribution Q

$$
\boxed{
\Delta Q
=
\sum_{a\in\mathcal L_\cap}
\frac{O_a+O'_a}{2}
(p'_a-p_a).
}
$$

This term measures the effect of redistributing probability among moves that exist in both landscapes.

Interpretation:

> The moves are still available, but the observer's attention has shifted among them.

## 10.2 Observable Change W

$$
\boxed{
\Delta W
=
\sum_{a\in\mathcal L_\cap}
\frac{p_a+p'_a}{2}
(O'_a-O_a).
}
$$

This term measures how the evaluations associated with common moves have changed.

Interpretation:

> The same labeled move is still available, but playing it now leads to a different board value.

## 10.3 Accessibility Change A

$$
\boxed{
\Delta A
=
\sum_{a\in\mathcal L_+}p'_aO'_a
-
\sum_{a\in\mathcal L_-}p_aO_a.
}
$$

This term accounts for changes in the legal-action support.

Interpretation:

> Some possibilities have appeared and others have disappeared.

## 10.4 QWA for a refined candidate

For a refined root move $m$, every retained search path $\gamma$ ends in a terminal same-player landscape.

Let

$$
\Delta G_\gamma
=
U^{1/2}(B_{\gamma,\mathrm{terminal}})
-
U^{1/2}(B).
$$

For that path,

$$
\Delta G_\gamma
=
Q_\gamma+W_\gamma+A_\gamma.
$$

Let $\Gamma_m^{(d)}$ be the retained set of terminal paths conditional on root move $m$, and let

$$
\widetilde P(\gamma\mid m)
$$

be the product of renormalized conditional probabilities along path $\gamma$.

Then

$$
\boxed{
\widetilde G_m^d
=
\sum_{\gamma\in\Gamma_m^{(d)}}
\widetilde P(\gamma\mid m)
\Delta G_\gamma
}
$$

and

$$
\widetilde Q_m^d
=
\sum_{\gamma}
\widetilde P(\gamma\mid m)Q_\gamma,
$$

$$
\widetilde W_m^d
=
\sum_{\gamma}
\widetilde P(\gamma\mid m)W_\gamma,
$$

$$
\widetilde A_m^d
=
\sum_{\gamma}
\widetilde P(\gamma\mid m)A_\gamma.
$$

Therefore,

$$
\boxed{
\widetilde G_m^d
=
\widetilde Q_m^d
+
\widetilde W_m^d
+
\widetilde A_m^d.
}
$$

For `depth = 3`, each path $\gamma$ is simply one retained opponent response $r$.

The canonical stored candidate fields are

```text
g_tilde
q_tilde
w_tilde
a_tilde
```

and are present only for refined root moves.

---

# 11. Realized Same-Player Q/W/A

Predicted QWA and realized QWA answer different questions.

Predicted quantities ask:

> What change does the bounded search expect if I choose this candidate?

Realized quantities ask:

> What actually happened to my immediate decision landscape between two consecutive turns?

Consider two actual positions of the same player,

$$
B_t
\longrightarrow
B_{t+1},
$$

separated by one played move-response pair.

The realized change is

$$
\boxed{
\Delta U_{\mathrm{real}}
=
U^{1/2}(B_{t+1})
-
U^{1/2}(B_t).
}
$$

Both shallow landscapes are computed using **all legal moves** and the original normalized probabilities.

The same finite decomposition gives

$$
\boxed{
\Delta U_{\mathrm{real}}
=
\Delta Q_{\mathrm{real}}
+
\Delta W_{\mathrm{real}}
+
\Delta A_{\mathrm{real}}.
}
$$

These quantities do not depend on:

- search `depth`;
- `adaptive_c`;
- which branches were refined;
- truncated probabilities.

They describe an actual transition between two complete shallow landscapes encountered during play.

The stored fields are

```text
realized_delta_u
realized_delta_q
realized_delta_w
realized_delta_a
```

## 11.1 Same-player cycle examples

For White,

```text
White move at turn t
    -> Black response
    -> next White decision state
```

For Black,

```text
Black move at turn t
    -> White response
    -> next Black decision state
```

The viewer also records the simple static changes across the two individual plies:

$$
\Delta U_{\mathrm{first}}
=
E(B_m)-E(B),
$$

$$
\Delta U_{\mathrm{second}}
=
E(B_{mr})-E(B_m).
$$

These are not QWA decompositions. They are ordinary static board-value differences.

---

# 12. Move Features and Strategy Presets

The move probability model uses six raw descriptors converted to pawn-valued features:

| Feature | Raw descriptor | Pawn-valued feature |
|---|---|---|
| `material` | mover-relative material change | `F_M = f_M` |
| `center` | weighted center-control change | `F_C = 0.10 f_C` |
| `development` | developed original minor slots | `F_D = 0.15 f_D` |
| `castling` | castling event and rights-lost event | `F_R = 0.40 f_castle + 0.20 f_rights` |
| `king_safety` | raw structural king-safety change | `F_K = 0.12 f_K` |
| `king_pressure` | safe opponent-king squares removed | `F_P = 0.10 f_P` |

Equivalently:

$$
F_M=f_M,\qquad
F_C=0.10f_C,\qquad
F_D=0.15f_D,
$$

$$
F_R=0.40f_{\rm castle}+0.20f_{\rm rights},
\qquad
F_K=0.12f_K,\qquad
F_P=0.10f_P.
$$

They are combined through

$$
\Phi_p(m,B)
=
\sum_k
\lambda^{\mathrm{eff}}_{p,k}(B)
s_kf_k(m,B).
$$

Because the effective coefficients are normalized, changing a strategy changes the relative emphasis placed on the six already-scaled move features.

## 12.1 Built-in strategy presets

| Style coefficient | `material_conservative` | `pressure_aggressive` | `tactical_attacker` | `positional_controller` |
|---|---:|---:|---:|---:|
| `material` | `2.4` | `0.9` | `1.0` | `1.2` |
| `center` | `0.5` | `1.1` | `1.4` | `1.8` |
| `development` | `0.6` | `0.6` | `0.5` | `0.9` |
| `castling` | `0.9` | `0.4` | `0.3` | `0.8` |
| `king_safety` | `1.0` | `0.4` | `0.5` | `1.1` |
| `king_pressure` | `0.3` | `1.8` | `2.2` | `0.6` |

These are raw style coefficients. At each board they are multiplied by the phase factors defined in Section 3 and then normalized before entering $\Phi_p$.

`material_conservative` emphasizes material while retaining substantial king-safety and castling weight.

`pressure_aggressive` emphasizes king pressure and center control.

`tactical_attacker` gives the strongest raw emphasis to king pressure.

`positional_controller` emphasizes center control while keeping a relatively balanced contribution from material, development, castling, and king safety.

Arbitrary non-negative combinations can be supplied directly through `Style`.

---

# 13. Configuration and Defaults

## 13.1 Simulation defaults

| Parameter | Default |
|---|---:|
| Maximum plies | `80` |
| Seed | `1` |
| White beta | `4.0` |
| Black beta | `4.0` |
| Kappa | `1.0` |
| White strategy | `material_conservative` |
| Black strategy | `pressure_aggressive` |
| Search depth (`depth`) | `3` |
| Adaptive breadth (`adaptive_c`) | `0.3` |
| Search workers | `1` |
| Parallel minimum branches | `8` |
| Match name | generated automatically |
| Stop on checkmate | yes |
| Stop on stalemate | yes |
| Stop on insufficient material | yes |
| Stop on automatic fivefold repetition | yes |
| Stop on automatic 75-move draw | yes |
| Stop on actual threefold repetition | yes |
| Stop on claimable fifty-move draw | no |
| Stop on claimable threefold repetition | no |

When `MatchConfig` is constructed directly in Python, `max_plies` defaults to `120`. The CLI launcher overrides this to `80`.

## 13.2 Validation rules

```text
depth >= 3
depth is odd
adaptive_c > 0
```

---

# 14. Installation

Run commands from the repository root.

The documented environment name is `chess`.

```bash
conda create -n chess python pip
conda run -n chess pip install -r requirements.txt
```

Direct dependencies are

```text
python-chess>=1.999
pytest>=8.0
```

The package source directory must be on `PYTHONPATH`.

Import test:

```bash
conda run -n chess env PYTHONPATH=src python -c "import thermo_chess; print(thermo_chess.__name__)"
```

---

# 15. Running Simulations

## 15.1 Run with defaults

```bash
conda run -n chess env PYTHONPATH=src python scripts/run_match.py
```

The default match writes

```text
data/games/csv/thermo_match.csv
data/games/json/thermo_match.json
data/games/pgn/thermo_match.pgn
```

Files with the same match name are overwritten. Use `--name` to preserve previous runs.

## 15.2 Example with explicit options

```bash
conda run -n chess env PYTHONPATH=src python scripts/run_match.py \
  --name adaptive_depth3_c03 \
  --max-plies 120 \
  --seed 23 \
  --beta-white 5.5 \
  --beta-black 5.5 \
  --kappa 1.0 \
  --white-strategy tactical_attacker \
  --black-strategy positional_controller \
  --depth 3 \
  --adaptive-c 0.3 \
  --search-workers 1 \
  --parallel-min-branches 8
```

## 15.3 Command-line options

| Flag | Default | Meaning |
|---|---:|---|
| `--max-plies N` | `80` | Maximum number of half-moves before the simulation stops. |
| `--seed N` | `1` | Initializes Python's random generator. |
| `--beta-white X` | `4.0` | White observer's inverse temperature. |
| `--beta-black X` | `4.0` | Black observer's inverse temperature. |
| `--kappa X` | `1.0` | Boltzmann-like scale in pawn units. |
| `--white-strategy NAME` | `material_conservative` | White strategy preset. |
| `--black-strategy NAME` | `pressure_aggressive` | Black strategy preset. |
| `--depth N` | `3` | Fixed odd prediction horizon in plies. Must satisfy `N >= 3` and be odd. |
| `--adaptive-c X` | `0.3` | Breadth coefficient in the top-K rule. |
| `--search-workers N` | `1` | Number of process workers for independent root branches; `auto` is supported. |
| `--parallel-min-branches N` | `8` | Minimum number of root branches before process-based parallelism is used. |
| `--name TEXT` | generated | Base filename for CSV, JSON, and PGN outputs. |
| `--ignore-threefold` | off | Ignore actual threefold repetition as a stopping condition. |
| `--allow-draw-claims` | off | Also stop on claimable fifty-move or threefold draws. |
| `-h`, `--help` | n/a | Print help and exit. |

Normal stopping conditions include:

- checkmate;
- stalemate;
- insufficient material;
- automatic fivefold repetition;
- automatic 75-move draw;
- actual threefold repetition;
- reaching `--max-plies`.

`--ignore-threefold` disables the actual-threefold stopping condition.

`--allow-draw-claims` additionally enables claimable fifty-move and threefold-repetition stopping conditions.

## 15.4 Search cost

Search cost grows with both:

- the fixed odd horizon `depth`;
- the number of retained branches at intermediate nodes.

Entropy affects only the second quantity.

Low entropy generally gives

$$
N_{\mathrm{eff}}\text{ small}
\Rightarrow
K\text{ small}
\Rightarrow
\text{narrower search}.
$$

High entropy generally gives

$$
N_{\mathrm{eff}}\text{ large}
\Rightarrow
K\text{ large}
\Rightarrow
\text{broader search}.
$$

The depth remains fixed in both cases.

---

# 16. Viewer

Start the dynamic viewer with

```bash
conda run -n chess env PYTHONPATH=src python scripts/view_match.py
```

Default address:

```text
http://127.0.0.1:8765/
```

## 16.1 Viewer options

| Flag | Default | Meaning |
|---|---:|---|
| `--results-dir PATH` | `data/games/json` | Directory containing match JSON files. |
| `--host HOST` | `127.0.0.1` | Interface used by the HTTP server. |
| `--port N` | `8765` | HTTP port. |
| `--open` | off | Open the viewer URL in the default browser. |
| `-h`, `--help` | n/a | Print help and exit. |

Example:

```bash
conda run -n chess env PYTHONPATH=src python scripts/view_match.py \
  --results-dir data/games/json \
  --host 127.0.0.1 \
  --port 8877 \
  --open
```

The viewer performs no search itself. It displays precomputed JSON data.

For every legal move, the move table displays:

- move;
- $P(B_m\mid B)$;
- $E(B_m)$.

For refined root moves, it additionally displays:

- `g_tilde`;
- `q_tilde`;
- `w_tilde`;
- `a_tilde`.

Unrefined moves have no bounded recursive score.

The refined-move popup exposes deeper information such as:

- original probability;
- retained/conditional probability where relevant;
- immediate $E(B_m)$;
- configured `depth`;
- retained probability mass;
- retained continuations;
- terminal landscape values;
- Q/W/A diagnostics.

The viewer also displays realized same-player QWA panels based on the actual played trajectory.

---

# 17. Saved Data

## 17.1 CSV

The CSV contains one row per played ply.

Important fields include:

| Field | Meaning |
|---|---|
| `E_before`, `E_after` | Universal static evaluations before and after the played move. |
| `U_current` | Current full shallow landscape. |
| `terminal_u` | Selected candidate's bounded response/path-averaged terminal shallow value. |
| `g_tilde` | Canonical move-selection score for the selected refined move. |
| `U_before`, `U_after` | Consecutive full-shallow same-player values when a realized transition exists. |
| `realized_delta_u` | Realized full-shallow same-player change. |
| `realized_delta_q` | Probability-redistribution contribution. |
| `realized_delta_w` | Observable-change contribution. |
| `realized_delta_a` | Action-accessibility contribution. |
| `decomposition_error` | Numerical residual in the QWA identity. |
| `N_eff_before`, `N_eff_after` | Effective move counts for consecutive same-player states. |
| `entropy_current` | Root move-distribution entropy. |
| `N_eff` | Root effective number of moves. |
| `K_expanded` | Number of root candidates selected for refinement. |
| `nodes_evaluated` | Search nodes evaluated. |
| `static_evaluations` | Uncached static evaluations. |
| `recursive_nodes` | Refined recursive child traversals. |
| `maximum_depth_reached` | Deepest ply level reached by search. |
| `average_K` | Mean retained breadth across expanded nodes. |
| `cache_hits` | Reused cached values and landscapes. |
| `search_elapsed_time` | Search time in seconds. |
| `prediction_error` | Realized static reply value minus the earlier bounded prediction, where available. |

## 17.2 JSON

The JSON contains:

- match and search configuration;
- White and Black style coefficients;
- static evaluation weights;
- result and terminal metadata;
- one record per played ply;
- full root move landscapes;
- refined-candidate scores;
- retained branch/path diagnostics;
- Q/W/A decompositions;
- same-player realized transitions;
- viewer-ready states.

Candidate records indicate whether each move belongs to the root top-$K$ set.

For refined candidates, response/path diagnostics include retained probabilities, conditional probabilities, retained probability mass, terminal values, and Q/W/A information.

## 17.3 PGN

The PGN contains the played move sequence and standard result headers.

It does not contain the thermodynamic/search diagnostics. Use JSON or CSV for quantitative analysis.

---

# 18. Analysis, Tests, and Benchmark

## 18.1 Inspect a saved ply

```bash
conda run -n chess env PYTHONPATH=src python scripts/analyze_match.py \
  data/games/json/thermo_match.json \
  --ply 8 \
  --sort probability \
  --limit 20
```

Options:

| Argument | Default | Meaning |
|---|---|---|
| `match_json` | `data/games/json/thermo_match.json` | Match file to inspect. |
| `--ply N` | `1` | Saved ply to print. |
| `--sort MODE` | `probability` | Sort by `probability`, `static_after`, or `phi`. |
| `--limit N` | `20` | Maximum number of moves printed. |

The analysis script reads saved results and does not rerun search.

## 18.2 Static covariance analysis

For every deepened candidate move $m$, the retained opponent responses define a conditional ensemble

$$
\{B_{mr},P(r|m)\}.
$$

The covariance-analysis script computes the covariance matrix of the ordinary static-evaluation observables

$$
X=(M,P,L,C,K)
$$

as

$$
\Sigma_{ij}^{(m)}
=
\sum_rP(r|m)
\left(X_i-\langle X_i\rangle_m\right)
\left(X_j-\langle X_j\rangle_m\right).
$$

This covariance matrix concerns the observables entering the universal static board evaluator, not the observer-dependent move-potential features.

Analyze one game:

```bash
conda run -n chess env PYTHONPATH=src python scripts/analyze_static_covariances.py \
  data/games/json/thermo_match.json
```

This writes:

```text
data/analysis/covariance/thermo_match.csv
```

Analyze a directory of games:

```bash
conda run -n chess env PYTHONPATH=src python scripts/analyze_static_covariances.py \
  data/games/json \
  --json-output data/analysis/covariance
```

This writes one CSV per source game, and one optional JSON detail file per source game, under `data/analysis/covariance/`.

Inspect one ply in a readable diagnostic view:

```bash
conda run -n chess env PYTHONPATH=src python scripts/analyze_static_covariances.py \
  data/games/json/thermo_match.json \
  --ply 7
```

The CSV contains one row per deepened candidate move. It includes candidate identity, actual-move matching, response probability checks, component means, component variances, independent off-diagonal covariances, $\alpha^T\Sigma\alpha$, direct weighted variance of `static_value`, and their consistency error when terminal boards do not make that comparison unavailable.

## 18.3 Deep-search divergence analysis

The deep-search divergence script compares the immediate retained-response static value

$$
\langle E\rangle_m
=
\sum_r P(r|m)E(B_{mr})
$$

from the covariance CSV with the corresponding deep-search terminal value saved in the detailed game JSON. It computes

$$
\Delta_{\mathrm{deep}}(m)
=
U_{\mathrm{terminal}}(m)-\langle E\rangle_m.
$$

In the output CSV this is named `terminal_minus_mean_static`, with `abs_terminal_minus_mean_static` used for anomaly ranking. Large absolute values are not automatically errors; they identify candidate branches where the recursive search changes the assessment substantially relative to the immediate response ensemble.

Analyze one game:

```bash
conda run -n chess env PYTHONPATH=src python scripts/analyze_deep_search_divergence.py \
  data/games/json/thermo_match.json
```

This expects the matching covariance CSV at:

```text
data/analysis/covariance/thermo_match.csv
```

and writes:

```text
data/analysis/deep_search_divergence/thermo_match.csv
```

Inspect one ply:

```bash
conda run -n chess env PYTHONPATH=src python scripts/analyze_deep_search_divergence.py \
  data/games/json/thermo_match.json \
  --ply 7
```

Print the top 20 largest divergences:

```bash
conda run -n chess env PYTHONPATH=src python scripts/analyze_deep_search_divergence.py \
  data/games/json/thermo_match.json \
  --top 20
```

For multiple games, pass a directory of JSON files:

```bash
conda run -n chess env PYTHONPATH=src python scripts/analyze_deep_search_divergence.py \
  data/games/json
```

The script matches records by `(ply, candidate_uci)`, verifies SAN/probability/retained-response consistency where available, and fails clearly on ambiguous or missing deep-search records.

## 18.4 Run tests

```bash
conda run -n chess env PYTHONPATH=src pytest -q
```

The test suite covers the mathematical and serialization behavior of the model, including:

- probability normalization;
- entropy and $N_{\mathrm{eff}}$;
- entropy-controlled breadth;
- odd-depth validation;
- probability truncation and renormalization;
- terminal handling;
- move selection;
- feature conventions;
- QWA identities;
- realized same-player transitions;
- serialization;
- viewer-ready output.

## 18.5 Optional benchmark

```bash
conda run -n chess env PYTHONPATH=src python scripts/benchmark_adaptive.py
```

Benchmark options:

| Flag | Default | Meaning |
|---|---:|---|
| `--fen FEN` | built-in middlegame position | Replace the benchmark board. |
| `--beta X` | `4.0` | Inverse temperature. |
| `--adaptive-c X` | `0.3` | Breadth coefficient. |

The benchmark does not modify saved match files.

---

# 19. Python Configuration

Command-line flags expose common controls. Custom styles, evaluator weights, and output directories can be configured directly in Python.

```python
from pathlib import Path

from thermo_chess.evaluation import EvaluationWeights
from thermo_chess.measure import Style
from thermo_chess.simulation import MatchConfig, simulate_match

white_style = Style(
    material=1.8,
    center=0.5,
    development=0.8,
    castling=0.8,
    king_safety=1.0,
    king_pressure=0.4,
)

black_style = Style(
    material=0.9,
    center=1.1,
    development=0.6,
    castling=0.4,
    king_safety=0.4,
    king_pressure=1.8,
)

config = MatchConfig(
    max_plies=120,
    seed=23,
    beta_white=5.0,
    beta_black=5.0,
    white_strategy="material_conservative",
    black_strategy="pressure_aggressive",
    depth=3,
    adaptive_c=0.3,
    match_name="custom_adaptive_match",
    games_dir=Path("data/games"),
)

simulate_match(
    white_style=white_style,
    black_style=black_style,
    eval_weights=EvaluationWeights(),
    config=config,
)
```

When a `Style` object is passed directly, it determines that player's coefficients. Otherwise the corresponding strategy preset in `MatchConfig` is used.

---

# 20. Project Structure

```text
src/thermo_chess/
  evaluation.py      universal static evaluator E(B)
  features.py        pawn-scaled board descriptors and mover-relative move features
  measure.py         style potential, softmax, entropy, move landscapes
  metrics.py         entropy and effective-number utilities
  search.py          bounded probability-truncated recursive search
  player.py          observer analysis and White-max/Black-min move choice
  simulation.py      match loop and CSV/JSON/PGN serialization
  live_viewer.py     dynamic HTTP viewer

scripts/
  run_match.py                       simulation CLI
  view_match.py                      viewer CLI
  analyze_match.py                   saved-landscape inspection CLI
  analyze_static_covariances.py      static-component covariance analysis
  analyze_deep_search_divergence.py  deep-search divergence analysis
  benchmark_adaptive.py              optional search benchmark

data/games/pgn/          generated PGN files
data/games/csv/          generated compact CSV files
data/games/json/         generated detailed JSON files
data/analysis/covariance/ covariance-analysis outputs
data/analysis/deep_search_divergence/ deep-search divergence outputs
tests/                   mathematical and regression tests
match_viewer.html        legacy static viewer artifact
```

---

# 21. Performance and Limitations

The simulator is designed for interpretability rather than competitive chess strength.

Important limitations:

- the static evaluator is intentionally compact and hand-designed;
- move probabilities depend on hand-designed strategic features;
- bounded search can miss decisive low-probability branches;
- large odd `depth` values remain expensive;
- high-entropy nodes retain more branches and can be substantially more expensive than low-entropy nodes;
- unrefined root moves are excluded from final move selection;
- viewer-state generation evaluates both observers at saved positions and adds computational cost;
- JSON files can become large because they store detailed branch/path diagnostics;
- search is CPU-bound;
- process workers can parallelize independent root branches, but deep searches remain expensive.

The central bounded-attention mechanism is

$$
\boxed{
\text{low entropy}
\Rightarrow
N_{\mathrm{eff}}\text{ small}
\Rightarrow
K\text{ small}
\Rightarrow
\text{narrow search}
}
$$

and

$$
\boxed{
\text{high entropy}
\Rightarrow
N_{\mathrm{eff}}\text{ large}
\Rightarrow
K\text{ large}
\Rightarrow
\text{broad search}.
}
$$

The search horizon `depth` remains fixed.

The resulting framework separates two ideas that are usually mixed together in chess search:

1. **subjective importance**, encoded by move probabilities;
2. **computational attention**, encoded by entropy-based truncation.

This makes it possible to study not only which move an observer prefers, but also how limited computation can alter that preference.
