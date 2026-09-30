# Thermodynamic Toy Chess

`chess_thermodynamic` is an experimental chess simulator based on probability-weighted expected values, information entropy, and player-specific strategic styles. It is intended for studying how two observers can assign different importance to the same legal moves.

It is not a conventional chess engine. In particular, it does not use minimax, alpha-beta pruning, opening books, or an external engine. Max/min operations are used only to choose the final move and to decide which branches receive more computation. The value of a position remains a probability-weighted expectation.

## Contents

- [Conceptual overview](#conceptual-overview)
- [Mathematical framework](#mathematical-framework)
- [Exact potential feature formulas](#exact-potential-feature-formulas)
- [Exact board-evaluation formulas](#exact-board-evaluation-formulas)
- [Exact Cycle Recursive Expectation](#exact-cycle-recursive-expectation)
- [Adaptive cycle search](#adaptive-cycle-search)
- [Move choice and deltaU](#move-choice-and-deltau)
- [Heat, work, and accessibility](#heat-work-and-accessibility)
- [Default model parameters](#default-model-parameters)
- [Installation](#installation)
- [Running simulations](#running-simulations)
- [Viewer](#viewer)
- [Saved data](#saved-data)
- [Analysis and tests](#analysis-and-tests)
- [Python configuration](#python-configuration)
- [Project structure](#project-structure)
- [Performance and limitations](#performance-and-limitations)

## Conceptual Overview

The model separates three ideas:

1. **Universal board evaluation:** `E(B)` describes whether a board favors White or Black. Both players use the same evaluator.
2. **Observer-dependent move measure:** each player has style coefficients that produce a probability distribution over legal moves.
3. **Adaptive expected value:** the distribution's entropy controls how broadly and deeply future positions are explored.

The sign convention is global:

```math
E(B)>0 \quad\text{favors White},
\qquad
E(B)<0 \quad\text{favors Black}.
```

The same convention applies to adaptive expected values and `delta_u`. A positive value never changes meaning when the observer changes.

The players differ because they weight moves differently, not because they use different definitions of material or different signs for `E`.

## Mathematical Framework

### Board and move notation

Let:

- $B$ be a chess position;
- $\mathcal M(B)$ be the legal moves in that position;
- $B_m$ be the position after legal move $m$;
- $\lambda^{(0)}$ be an observer's baseline vector of style preferences;
- $\kappa$ be the Boltzmann-like constant, defaulting to `1.0` in pawn units;
- $T$ be that observer's temperature, with $\beta = 1/(\kappa T)$ kept as the backward-compatible inverse-temperature interface.

### Move features and style potential

For every legal move, the project computes features $F_k(m,B)$ from the perspective of the side actually making the move:

```math
\Phi_p(m,B)
=
\sum_k \lambda_{p,k}(g)F_k(m,B).
```

Features entering `Phi` are interpreted in pawn-equivalent units. Material and preservation are naturally in pawns; tactical and positional features are conservatively scaled to pawn fractions. Mate remains a separate dominant terminal feature. Baseline strategy coefficients are first phase-modulated into unnormalized weights and then renormalized so active non-terminal strategy weights sum to `1` at every game phase.

This mover-relative convention is important. For example, winning material is a positive move feature whether White or Black makes the capture. When an observer evaluates opponent moves deeper in the tree, the observer's same $\lambda$ is retained, while `move_features(board, move)` still describes the side that actually moves at that node.

The implemented move features are:

| Feature | Meaning |
|---|---|
| `material` | Change in mover-relative material balance, in pawn units. |
| `preservation` | Reduction in the mover's exchange-aware exposed material. |
| `activity` | Normalized change in own mobility minus half the opponent's mobility change. |
| `king_safety` | Change in the mover's static king-safety score. |
| `king_restriction` | Reduction in the opponent king's legal moves. |
| `king_pressure` | Increase in attacked squares around the opponent king. |
| `center` | Normalized change in attacked central and extended-central squares. |
| `check` | `1` if the resulting position checks the opponent, otherwise `0`. |
| `mate` | `1` if the move checkmates, otherwise `0`. |
| `promotion` | Promotion material gain, normalized by `8`. |
| `castle_preserve` | Change in the mover's retained castling rights; castling itself is exempt. |
| `castle_deny` | Number of opponent castling rights removed by the move. |
| `castle` | `1` when the move is castling, otherwise `0`. |

Piece values are pawn `1`, knight `3`, bishop `3`, rook `5`, queen `9`, and king `0`. Preservation uses a shallow legal-capture exchange calculation rather than raw attacker or defender counts.

### Shared helper quantities

The formulas below use the following helper quantities. Let $B$ be a board
position, $c\in\{\mathrm{White},\mathrm{Black}\}$ a color, $\bar c$ its
opponent, and $t$ a piece type. Define

```math
n_t(B,c)=\text{the number of pieces of type }t\text{ and color }c
\text{ present on board }B.
```

Here $t$ ranges over pawn, knight, bishop, rook, queen, and king. For example,
$n_{\mathrm{rook}}(B,\mathrm{White})=2$ when White has both rooks. Let $v(t)$
be the piece value above; in particular, $v(\mathrm{king})=0$, so the king is
included in the sum but contributes no material value.

Material owned by one color is

```math
M(B,c)=\sum_{t\in\{P,N,B,R,Q,K\}} v(t)\,n_t(B,c),
```

and mover-relative material balance is

```math
M_{\mathrm{rel}}(B,c)=M(B,c)-M(B,\bar c).
```

`legal_mobility(B, c)` copies the board, sets the side to move to $c$, and counts legal moves. Denote this count by $L(B,c)$. It returns zero for checkmate, stalemate, or insufficient-material positions.

The attacked-center score is

```math
C(B,c)
=
\sum_{s\in\{d4,e4,d5,e5\}}
\mathbf 1[c\text{ attacks }s]
+0.35
\sum_{s\in\mathcal C_{\mathrm{ext}}}
\mathbf 1[c\text{ attacks }s],
```

where

```math
\mathcal C_{\mathrm{ext}}
=
\{c3,d3,e3,f3,c4,f4,c5,f5,c6,d6,e6,f6\}.
```

This measures attacked squares, not piece occupancy.

For king square $k_c$, define

- $A(B,c)$: enemy attackers of $k_c$;
- $D(B,c)$: friendly defenders of $k_c$;
- $P(B,c)$: king-adjacent squares attacked by the enemy.

The king-safety helper is

```math
K_{\mathrm{safe}}(B,c)
=
\frac{D(B,c)-1.5A(B,c)-0.25P(B,c)}{8}.
```

If color $c$ has no king, this helper returns `-1`.

### Exchange exposure used by preservation

For a square $s$ occupied by an opposing non-king piece, the exchange routine considers only legal captures onto that square. If $G(B,s,c)$ is the best profitable gain available to attacker $c$, then conceptually

```math
G(B,s,c)
=
\max\left(
0,
\max_{m\in\mathcal X(B,s,c)}
\left[v(\text{piece on }s)-G(B_m,s,\bar c)\right]
\right),
```

where $\mathcal X(B,s,c)$ is the set of legal captures by $c$ onto $s$. The outer zero lets either side decline an unprofitable continuation.

The total exposed material for side $c$ is

```math
X(B,c)
=
\sum_{s\text{ occupied by a non-king piece of }c}
G(B,s,\bar c).
```

This is a local static-exchange-style calculation. It recursively follows captures on one square, but it is not a general board search.

## Exact Potential Feature Formulas

Let $c$ be `board.turn` before move $m$, and let $B_m$ be the resulting board.

The base move potential contains the original ten features plus three explicit
castling features:

```math
\Phi_\lambda(m,B)
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
```

### Material

Material is the change in own-minus-opponent material from the mover's perspective:

```math
\boxed{
F_{\mathrm{material}}(m,B)
=
M_{\mathrm{rel}}(B_m,c)-M_{\mathrm{rel}}(B,c).
}
```

### Preservation

Preservation is exposed material before the move minus exposed material afterward:

```math
\boxed{
F_{\mathrm{preservation}}(m,B)
=
X(B,c)-X(B_m,c).
}
```

A positive value means the mover leaves less material profitably capturable after the move. Capturing enemy material is represented separately by `material`.

### Activity

Activity combines the mover's mobility change with half the negative opponent mobility change:

```math
\boxed{
F_{\mathrm{activity}}(m,B)
=
\frac{
[L(B_m,c)-L(B,c)]
-0.5[L(B_m,\bar c)-L(B,\bar c)]
}{30}.
}
```

### King Safety

King safety is the change in the mover's king-safety helper:

```math
\boxed{
F_{\mathrm{king\_safety}}(m,B)
=
K_{\mathrm{safe}}(B_m,c)-K_{\mathrm{safe}}(B,c).
}
```

### King Restriction

Let $L_K(B,c)$ be the number of legal moves made by color $c$'s king. King restriction is the reduction in enemy king mobility:

```math
\boxed{
F_{\mathrm{king\_restriction}}(m,B)
=
L_K(B,\bar c)-L_K(B_m,\bar c).
}
```

### King Pressure

Let $Q(B,c)$ be the number of squares adjacent to the opponent king that are attacked by $c$. King pressure is its change:

```math
\boxed{
F_{\mathrm{king\_pressure}}(m,B)
=
Q(B_m,c)-Q(B,c).
}
```

### Center Control

Center control is the normalized change in attacked center score:

```math
\boxed{
F_{\mathrm{center}}(m,B)
=
\frac{C(B_m,c)-C(B,c)}{6}.
}
```

### Check

```math
\boxed{
F_{\mathrm{check}}(m,B)
=
\mathbf 1[B_m\text{ gives check}],
}
```

### Mate

```math
\boxed{
F_{\mathrm{mate}}(m,B)
=
\mathbf 1[B_m\text{ is checkmate}].
}
```

### Promotion

For a promotion to piece type $t$,

```math
\boxed{
F_{\mathrm{promotion}}(m,B)
=
\frac{v(t)-v(\mathrm{pawn})}{8},
}
```

and it is zero for a non-promotion. A queen promotion therefore contributes `1.0`, rook `0.5`, and bishop or knight `0.25` before multiplication by the style coefficient.

### Castling Rights

Let $C_c(B)\in\{0,1,2\}$ count the kingside and queenside castling rights
still held by color $c$. Each right has value one. Then

```math
\boxed{F_{\mathrm{castle\_preserve}}(m,B)=C_c(B_m)-C_c(B),}
```

except that this feature is set to zero when $m$ is itself castling. Thus an
ordinary king move usually scores `-2`, moving an original rook usually scores
`-1`, and legitimately using a castling right is not treated as losing it.

For the opponent's rights,

```math
\boxed{F_{\mathrm{castle\_deny}}(m,B)=C_{\bar c}(B)-C_{\bar c}(B_m),}
```

and actual castling is represented by

```math
\boxed{F_{\mathrm{castle}}(m,B)=\mathbf 1[m\text{ is castling}].}
```

### Development Phase

The phase is board-dependent and does not use the move number. For each color
$c$, let $d_N(B,c)$ be the fraction of its four original minor-piece squares
(`b1/c1/f1/g1` or `b8/c8/f8/g8`) no longer occupied by that color's original
minor-piece type. Define $d_P(B,c)$ analogously for the two original central
pawn squares (`d2/e2` or `d7/e7`), and $d_R(B,c)$ for the original rook and
queen squares (`a1/d1/h1` or `a8/d8/h8`). A moved or captured original piece
therefore advances this simple occupancy-based phase measure.

```math
d(B,c)=0.60d_N(B,c)+0.25d_P(B,c)+0.15d_R(B,c),
```

```math
\boxed{g(B)=\operatorname{clamp}_{[0,1]}
\left(\frac{d(B,\mathrm{White})+d(B,\mathrm{Black})}{2}\right).}
```

The phase weights are

```math
w_D(g)=1-g,\qquad w_C(g)=4g(1-g),\qquad w_A(g)=g.
```

Writing $F_{\mathrm{activity}}$, $F_{\mathrm{king\_safety}}$, and the other
terms as defined above, the three phase features are exactly

```math
\boxed{F_D(m,B)=d(B_m,c)-d(B,c)+\max(0,F_{\mathrm{activity}}),}
```

```math
\boxed{F_C(m,B)=F_{\mathrm{castle}}+F_{\mathrm{king\_safety}},}
```

```math
\boxed{F_A(m,B)=F_{\mathrm{activity}}+F_{\mathrm{king\_pressure}}
+F_{\mathrm{king\_restriction}}+F_{\mathrm{check}}.}
```

Castling-right preservation is weighted separately from the generic phase
potential. The raw feature remains $F_{\mathrm{castle\_preserve}}$, but its
effective contribution is

```math
w_{\mathrm{castle\_rights}}(g)=1-g,
```

```math
\boxed{\Phi_{\mathrm{castle\_preserve}}
=\lambda_{\mathrm{CP}}(1-g)(1+\sigma w_C(g))
F_{\mathrm{castle\_preserve}}.}
```

This makes wasted castling rights expensive in the opening and early
middlegame, with an additional consolidation-phase boost for high-solidness
players. The multiplier fades as $g$ approaches `1`. Since
$F_{\mathrm{castle\_preserve}}=0$ for actual castling, using the castling
right is not penalized.

For `solidness` $\sigma\in[0,1]$, the unscaled phase contribution and final
potential are

```math
\Phi_{\mathrm{phase}}
=\lambda_Dw_DF_D+\lambda_Cw_CF_C+\lambda_Aw_AF_A,
```

```math
\boxed{\Phi_{\mathrm{total}}=\Phi_{\mathrm{base}}^*
+\sigma\Phi_{\mathrm{phase}},
\qquad
\Phi_{\mathrm{base}}^*=\Phi_{\mathrm{base\ without\ CP}}
+\Phi_{\mathrm{castle\_preserve}}.}
```

At `solidness=0`, the phase mechanism contributes nothing and the potential is
exactly this adjusted non-phase base.
Candidate diagnostics save $g$, all three phase weights and features,
`castle_preserve_raw`, `castle_preserve_phase_weight`,
`castle_preserve_effective_weight`, `castle_preserve_contribution`,
`base_potential`, unscaled `phase_potential`, and `total_potential`.

### Boltzmann move distribution

The style potential induces a probability distribution:

```math
p_\lambda(m\mid B)
=
\frac{\exp\left(\Phi_\lambda(m,B)/(\kappa T)\right)}
{\sum_{m'\in\mathcal M(B)}
\exp\left(\Phi_\lambda(m',B)/(\kappa T)\right)}.
```

The implementation uses a numerically stable softmax by subtracting the largest scaled potential before exponentiation.

Interpretation of temperature:

- $\kappa=1.0$ by default, with pawns as the energy unit;
- the existing $\beta$ interface is preserved as $\beta=1/(\kappa T)$;
- smaller $\beta$ / larger $T$ produces a flatter distribution;
- larger $\beta$ / smaller $T$ concentrates probability on moves with high style potential;
- temperature changes the probability measure, not the static evaluator.

## Exact Board-Evaluation Formulas

### Universal static evaluation

The shared evaluator is

```math
E(B)
=
\alpha_M\Delta M
+\alpha_{\mathrm{exposure}}\Delta H
+\alpha_{\mathrm{mob}}\Delta\mathrm{Mobility}
+\alpha_K\Delta K
+\alpha_C\Delta C,
```

where material, mobility, king safety, and center are White-minus-Black features, while exposure is signed as Black exposed material minus White exposed material. The default weights are:

| Static component | Default |
|---|---:|
| Material | `1.0` |
| Exposure | `0.5` |
| Mobility | `0.25` |
| King safety | `0.6` |
| Center control | `0.35` |
| Checkmate magnitude | `10000.0` |

The board features passed to this evaluator are computed exactly as follows:

```math
\Delta M(B)=M(B,\mathrm{White})-M(B,\mathrm{Black}),
```

```math
H_a(B)
=
\sum_{s\text{ occupied by a non-king piece of }a} R_s(B),
\qquad
\Delta H(B)=H_{\mathrm{Black}}(B)-H_{\mathrm{White}}(B).
```

Here `R_s(B)` is the same exchange-aware profitable-capture exposure used by the preservation move feature. White exposed material decreases `E(B)`; Black exposed material increases it.

```math
\Delta\mathrm{Mobility}(B)
=
\frac{L(B,\mathrm{White})-L(B,\mathrm{Black})}{30},
```

```math
\Delta K(B)
=
K_{\mathrm{safe}}(B,\mathrm{White})
-K_{\mathrm{safe}}(B,\mathrm{Black}),
```

```math
\Delta C(B)
=
\frac{C(B,\mathrm{White})-C(B,\mathrm{Black})}{6}.
```

With default weights, a nonterminal board is therefore evaluated as

```math
E(B)
=
\Delta M(B)
+0.5\,\Delta H(B)
+0.25\,\Delta\mathrm{Mobility}(B)
+0.6\,\Delta K(B)
+0.35\,\Delta C(B).
```

Notice that the mobility and center differences are normalized by `30` and `6` before their evaluator weights are applied. The static exposure term uses the same exchange-aware helper as preservation, but it is a universal board feature rather than a style-dependent move feature. The static evaluator does not directly use preservation, activity, checks, king restriction, king pressure, or promotion; those belong to the observer-dependent move potential.

If the side to move is checkmated, `E(B)` receives the appropriate signed checkmate value. Stalemate, insufficient material, the automatic 75-move rule, and fivefold repetition evaluate to zero. At a search terminal or depth-zero node, the search returns `E(B)` directly and never applies a softmax to an empty move set.

## Exact Cycle Recursive Expectation

The configured recursive depth is `cdepth`, where one unit is a complete interaction cycle:

```text
player move -> opponent response
```

The depth-zero value is

```math
U_\lambda^{(0)}(B)=E(B).
```

For `cdepth = d > 0`, the focal observer first computes probabilities for all legal own moves and assigns each move a cycle-aware branch value `O_m`. The adaptive value is

```math
U_\lambda^{(d)}(B)
=
\sum_m p_\lambda(m\mid B)O_m^{(d)}(B).
```

For `cdepth = 1`, every legal own move receives an explicit expectation over all legal opponent replies:

```math
O_m^{(1)}(B)
=
\sum_r q_\lambda(r\mid B_m)E(B_{m,r}),
```

so

```math
U_\lambda^{(1)}(B)
=
\sum_m p_\lambda(m\mid B)
\sum_r q_\lambda(r\mid B_m)E(B_{m,r}).
```

This is not minimax. Every branch contributes according to its probability, and White/Black still choose moves by maximizing/minimizing the resulting branch values.

### Entropy and effective number of moves

For each nonterminal move distribution:

```math
S_\lambda(B)
=
-\sum_m p_\lambda(m\mid B)\log p_\lambda(m\mid B),
```

and

```math
N_{\mathrm{eff}}(B)=e^{S_\lambda(B)}.
```

`N_eff` is the number of equally likely moves that would have the same entropy. A concentrated distribution has `N_eff` near `1`; a broad distribution has a larger value.

## Adaptive Cycle Search

The implementation uses entropy for both adaptive breadth and the local cycle-depth cap.

### Adaptive breadth

At the focal player's board, the number of own moves selected for refinement is

```math
K(B)
=
\min\left(
|\mathcal M(B)|,
\max\left(1,\left\lceil cN_{\mathrm{eff}}(B)\right\rceil\right)
\right).
```

For each selected own move, the opponent-response distribution has its own independently computed

```math
K'_m
=
\min\left(
|\mathcal R(B_m)|,
\max\left(1,\left\lceil cN_{\mathrm{eff}}(B_m)\right\rceil\right)
\right).
```

The default is

```text
c = 0.3
```

Top-`K` and top-`K'` selection is controlled by `refinement_policy`. The default `static_eval` policy ranks immediate boards by the side-to-move objective: larger `E(B_m)` for White, smaller `E(B_m)` for Black. The alternative `probability` policy ranks by the observer's move probability. This choice selects which branches receive extra computation; it never changes or renormalizes the probability distribution. Use `--refinement-policy static_eval|probability` in `run_match.py`.

### Entropy-selected local cdepth

The local cycle-depth cap is still chosen from `N_eff`:

```math
d_{\mathrm{eff}}(B)=
\begin{cases}
4, & N_{\mathrm{eff}}(B)\le 4,\\
3, & 4<N_{\mathrm{eff}}(B)\le 8,\\
2, & 8<N_{\mathrm{eff}}(B)\le 15,\\
1, & N_{\mathrm{eff}}(B)>15.
\end{cases}
```

For remaining cycle budget `r`, the node uses

```math
d_{\mathrm{local}}(B,r)
=
\min\left(r,d_{\mathrm{eff}}(B)\right).
```

A recursive call consumes one unit only after a complete own-move/opponent-response cycle, so selected response branches continue with `d_local - 1`.

### Response Evaluation

For every legal own move, the opponent-response probabilities use the same observer style and beta as root move probabilities:

```math
q_\lambda(r\mid B_m)
=
\frac{\exp\left(\Phi_\lambda(r,B_m)/(\kappa T)\right)}
{\sum_{r'}\exp\left(\Phi_\lambda(r',B_m)/(\kappa T)\right)}.
```

The response move features remain mover-relative, so a response made by Black is described from Black's mover perspective, but the style weights are still the focal observer's weights. Static `E(B_{m,r})` does not enter this response softmax.

For every legal own move, non-recursed probability mass is evaluated at the response board:

```math
O_m^{(d)}
=
\sum_r q_{r\mid m}V_{m,r}^{(d)},
```

where selected response branches recurse to `U^{(d-1)}(B_{m,r})` when `d > 1`, and non-recursed branches use `E(B_{m,r})`. Even non-top-`K` own moves receive the full shallow response expectation. The original probability mass is preserved; neither top-`K` nor top-`K'` is renormalized.

### Terminal positions

If a candidate move directly produces a terminal board, its branch value is `E(B_m)`. If a response reaches a terminal board, the response value is `E(B_{m,r})`. For thermodynamic bookkeeping only, a terminal same-player state is represented as an absorbing pseudo-branch `__terminal__` with probability `1`.

### Caching and Stored Results

Each observer search caches:

- static evaluations by complete position state;
- move landscapes, including features, probabilities, entropy, and `N_eff`;
- adaptive expected values by position and remaining `cdepth`;
- selected branch metadata, including `K`, response `K'`, branch values, and selected local `cdepth`;
- the serializable `SearchResult` for an evaluated decision state.

`SearchResult` is the single source of truth for an actual decision. It stores `U`, entropy, `N_eff`, `K`, selected depth, every move probability, every branch value `O_m`, candidate `delta_u`, and response diagnostics that were actually computed. Move choice, CSV/JSON logging, thermodynamic cycles, and the normal viewer all consume this stored result instead of rerunning search.

Caches belong to a fixed style, beta, and `cdepth`, so values from different observers cannot mix. Process workers use process-local caches; those caches are not shared with the parent process, but each worker reuses its own cache while evaluating its assigned branch.

### Observer convention

One adaptive search object belongs to one observer. Its style $\lambda$ and $\beta$ are used at every node, including nodes where the opponent is to move. Move features remain mover-relative. This distinction lets White and Black produce different expectations for the same board without changing the universal evaluator.

## Move Choice and deltaU

For every legal candidate `m` from current board `B`, the player uses the cycle-aware branch value

```math
O_m^{(d)}(B).
```

The current-board value is

```math
U_\lambda^{(d)}(B)
=
\sum_m p_\lambda(m\mid B)O_m^{(d)}(B).
```

The candidate thermodynamic diagnostic is the predicted change in the same player's next decision landscape. For candidate `m`, the model first computes the subjective opponent-response distribution `q(r | B_m)` from the observer's fixed style. The response move features are computed from the responding mover's perspective, but no extra side-to-move static-evaluation term is added to the probability.

Response probabilities weight the expectation, but the `K_response` replies selected for deeper recursive work are ranked by the configured `refinement_policy`. With the default `static_eval`, White-to-move replies prefer larger `E(B_{m,r})` and Black-to-move replies prefer smaller `E(B_{m,r})`. With `probability`, replies are ranked by `q(r | B_m)` for both sides. The response breadth is adaptive:

```math
K_{\mathrm{response}}
=
\min\left(
N_{\mathrm{legal\ replies}},
\max\left(1,\left\lceil c\exp(S_{\mathrm{reply}})\right\rceil\right)
\right),
```

where `c` is the configured adaptive breadth coefficient and `S_reply` is the entropy of `q`. Let `R_K(m)` be the top-`K_response` replies under the configured refinement policy. These replies are marked as refinement-eligible diagnostics, but candidate scoring keeps the full subjective response distribution and does not renormalize it.

After a candidate move and response, one complete cycle has been consumed. The future state is therefore valued with one fewer cycle. At candidate remaining depth zero, the future-state value is the static board value `E(B_{m,r})`. When remaining depth is positive, selected response branches recurse into the next complete cycle.

For candidate depth `d`, the stored predicted next value and candidate change are

```math
\operatorname{expected\_next\_U}(m)
=
\sum_r q(r\mid B_m)U_{\lambda,\mathrm{future}}^{(d-1)}(B_{m,r}),
```

```math
\Delta U_{\mathrm{candidate}}(m;B)
=
\operatorname{expected\_next\_U}(m)-U_\lambda^{(d)}(B).
```

Each response-specific transition compares `B` with `B_mr`, where the same player is again to move, and is decomposed with the finite midpoint formulas. At remaining depth zero, `U_future(B_mr)` is the static board value `E(B_mr)`; it does not add another same-player decision expansion. The candidate row stores the full-response probability average:

```math
\Delta U_m\approx\Delta Q_m+\Delta W_m+\Delta A_m.
```

Fields named `candidate_delta_u`, `delta_u`, `candidate_delta_q`, `candidate_delta_w`, `candidate_delta_a`, `expected_next_U`, `U_current`, and `U_after_move` follow this full-response predicted-candidate definition. `response_K` is the selected reply count for refinement diagnostics, and stored response probabilities remain the original `q(r|B_m)`. `branch_value` remains the internal `O_m` term used in `U(B)=sum p_m O_m`.

For performance, candidate scoring is split from detailed thermodynamic diagnostics. The `candidate_thermo_mode` option controls which candidates receive response-level `delta_Q`, `delta_W`, and `delta_A` records:

| Mode | Meaning |
|---|---|
| `selected` | Only the final chosen move is enriched. Fastest mode. |
| `refined` | Root top-`K` candidates plus the final chosen move are enriched. This is the default. |
| `all` | Every legal candidate is enriched for debugging/research. |

Non-enriched candidates still have `expected_next_U` and `candidate_delta_u`, but their detailed `candidate_delta_q`, `candidate_delta_w`, `candidate_delta_a`, and response records may be null/empty. Detailed enrichment is diagnostic and does not change the selected move. The command-line flag is `--candidate-thermo-mode selected|refined|all`.

Final move selection is:

```math
m_W^*=\arg\max_m \Delta U_{\mathrm{candidate}}(m;B),
```

```math
m_B^*=\arg\min_m \Delta U_{\mathrm{candidate}}(m;B).
```

The viewer ranks each observer's estimate of the side-to-move's best action:

- on White's turn, both panels rank larger `deltaU` first;
- on Black's turn, both panels rank smaller `deltaU` first;
- each panel may assign `#1` to a different move because the observers use different probability measures.

The shared table columns contain the move and $E(B_m)$. A light-grey outline marks White observer's `#1`, a black outline marks Black observer's `#1`, and light blue marks the move actually played.

### Prediction error

After a player moves, the simulator stores that player's expected value for the resulting reply position. After the opponent replies, it records

```math
\epsilon
=
E(B_{m,r^*})
-
\widetilde U_\lambda^{(D)}(B_m).
```

This compares the realized static board after the reply with the observer's earlier probability-weighted prediction.

## Heat, Work, and Accessibility

The transition accounting compares the same observer on consecutive turns by
that same player, not adjacent plies. For White, a transition compares one
White-to-move position with the next White-to-move position two plies later;
Black is handled analogously. The first turn for each player has no previous
same-player state and therefore no thermodynamic delta.

For each same-player state, the accounting reuses the exact cycle-aware branch
values already produced by that player's adaptive evaluator. For each legal
move $m$, the stored observable $O_m(B)$ is the same branch value used for
move choice. Consequently,

```math
\widetilde U(B)=\sum_m p_mO_m
```

is the same adaptive value used by the player, not a separate diagnostic
search. Each saved branch records its UCI identity, probability,
`adaptive_branch_value`, `was_deepened`, and `depth_used`.

For the common support $\mathcal L_\cap$ of two same-player positions, heat and
work use the finite midpoint formulas

```math
\Delta Q=\sum_{m\in\mathcal L_\cap}
\frac{O_m+O'_m}{2}(p'_m-p_m),
```

```math
\Delta W=\sum_{m\in\mathcal L_\cap}
\frac{p_m+p'_m}{2}(O'_m-O_m).
```

Moves that appear or disappear contribute

```math
\Delta A=\sum_{m\in\mathcal L_+}p'_mO'_m
-\sum_{m\in\mathcal L_-}p_mO_m.
```

Thus $\Delta U=\Delta Q+\Delta W+\Delta A$ to floating-point tolerance.
At configured `cdepth = 0`, adaptive evaluation directly returns the static
fallback. At terminal same-player states, thermodynamic bookkeeping uses an
absorbing pseudo-branch `__terminal__` with probability `1`, so the identity is
preserved without adding a pseudo-move to ordinary chess move generation.

## Default Model Parameters

### CLI simulation defaults

| Parameter | Default |
|---|---:|
| Maximum plies | `80` |
| Seed | `1` |
| White beta | `4.0` |
| Black beta | `4.0` |
| Kappa | `1.0` |
| White strategy | `material_conservative` |
| Black strategy | `activity_aggressive` |
| Cycle depth (`cdepth`) | `1` |
| Adaptive breadth `c` | `0.3` |
| Candidate thermo mode | `refined` |
| Refinement policy | `static_eval` |
| Match name | Generated from strategies, beta, solidness, and `cdepth` |
| Stop on checkmate/stalemate | Yes |
| Stop on insufficient material | Yes |
| Stop on automatic fivefold/75-move draws | Yes |
| Stop on actual threefold repetition | Yes |

`MatchConfig` itself defaults to `max_plies=120`; the `run_match.py` command-line launcher intentionally overrides that with `80`.

### Built-in strategy presets

The command-line simulator provides four fixed presets. The first two remain the defaults, while the tactical and positional presets provide additional matchups.

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

`tactical_attacker` favors forcing play around the enemy king and accepts more material exposure. `positional_controller` favors center control, preservation, king safety, and balanced activity. Presets can be selected through CLI flags; arbitrary coefficient combinations remain available through Python.

## Installation

All project commands should run in the `chess` conda environment.

Create the environment if needed:

```bash
conda create -n chess python pip
conda run -n chess pip install -r requirements.txt
```

The direct dependencies are `python-chess>=1.999` and `pytest>=8.0`.

Commands need the package source directory on `PYTHONPATH`:

```bash
conda run -n chess env PYTHONPATH=src python -c "import thermo_chess; print(thermo_chess.__name__)"
```

Run commands from the repository root.

## Running Simulations

### Run with all CLI defaults

```bash
conda run -n chess env PYTHONPATH=src python scripts/run_match.py
```

This creates:

```text
data/results/thermo_match.csv
data/results/thermo_match.json
data/games/thermo_match.pgn
```

Files with the same match name are overwritten. Use a distinct `--name` to preserve earlier games.

### Example with explicit search parameters

```bash
conda run -n chess env PYTHONPATH=src python scripts/run_match.py \
  --name adaptive_cdepth1_c03 \
  --max-plies 120 \
  --seed 23 \
  --beta-white 5.5 \
  --beta-black 5.5 \
  --white-strategy tactical_attacker \
  --black-strategy positional_controller \
  --solidness-white 0.3 \
  --solidness-black 0.85 \
  --cdepth 1 \
  --adaptive-c 0.3 \
  --refinement-policy probability
```

Use `--refinement-policy probability` to deepen the most probable root `K` moves and response `K'` replies. Omit it, or pass `--refinement-policy static_eval`, to keep the default side-aware immediate-evaluation ordering. With `cdepth=1`, this changes the saved/decomposed K/K' branches but normally not the played moves, because no selected response can recurse into another complete cycle. To make the policy affect decisions, use `--cdepth 2` or higher and positions whose entropy-selected local depth exceeds `1`.

### Simulation flags

| Flag | Default | Effect |
|---|---:|---|
| `--max-plies N` | `80` | Safety cap on the number of half-moves. |
| `--seed N` | `1` | Initializes Python's random generator. Current move selection is deterministic, so this is mainly a reproducibility hook for future stochastic behavior. |
| `--beta-white X` | `4.0` | White observer's inverse temperature. Larger values concentrate White's move probabilities. |
| `--beta-black X` | `4.0` | Black observer's inverse temperature. Larger values concentrate Black's move probabilities. |
| `--kappa X` | `1.0` | Boltzmann-like constant in pawn units. The effective temperature is `1 / (kappa * beta)`. |
| `--white-strategy NAME` | `material_conservative` | White preset. Choices: `material_conservative`, `activity_aggressive`, `tactical_attacker`, `positional_controller`. |
| `--black-strategy NAME` | `activity_aggressive` | Black preset. Uses the same four choices. |
| `--solidness-white X` | Preset value | Override White's solidness $\sigma$. Must be between `0` and `1`. |
| `--solidness-black X` | Preset value | Override Black's solidness $\sigma$. Must be between `0` and `1`. |
| `--cdepth N` | `1` | Cycle depth. One unit is own move plus opponent response. At `0`, values reduce to static `E(B)`. |
| `--depth N` | None | Deprecated alias for `--cdepth`. |
| `--adaptive-c X` | `0.3` | Breadth fraction in $K=\lceil cN_{\mathrm{eff}}\rceil$. Must be positive. `K` is still clipped to legal moves and at least one. |
| `--search-workers N` | `1` | Process workers for independent root own-move branches. Use `1` for fully serial search or `auto` for CPU-based worker count. |
| `--parallel-min-branches N` | `8` | Minimum root branch count before worker parallelism activates. |
| `--candidate-thermo-mode MODE` | `refined` | Detailed Q/W/A diagnostics. Choices: `selected`, `refined`, `all`. |
| `--refinement-policy POLICY` | `static_eval` | Selects which root `K` moves and response `K'` replies are deepened. Choices: `static_eval`, `probability`. At `cdepth=1`, this affects diagnostics but not move choice. |
| `--name TEXT` | Generated | Base filename for CSV, JSON, and PGN outputs. By default it includes both strategy names, beta values, effective solidness values, and `cdepth`. |
| `--ignore-threefold` | Off | Continue through actual threefold repetition instead of stopping there. |
| `--allow-draw-claims` | Off | Also stop when a fifty-move or threefold-repetition draw can be claimed. Automatic draw rules still stop regardless of this flag. |
| `-h`, `--help` | | Print command help. |

Normal stopping conditions are checkmate, stalemate, insufficient material, automatic fivefold repetition, automatic 75-move draw, actual threefold repetition, or `--max-plies`. Use `--ignore-threefold` to continue through actual threefold repetition. Use `--allow-draw-claims` to also stop on claimable fifty-move and threefold-repetition draws; merely claimable draws are not treated as terminal by the search itself.

`cdepth > 1` can still be expensive in low-entropy positions because those positions are explicitly allowed to retain deeper search. The simulator also precomputes both observers' viewer panels at every saved position.

## Viewer

Start the dynamic viewer:

```bash
conda run -n chess env PYTHONPATH=src python scripts/view_match.py
```

Default address:

```text
http://127.0.0.1:8765/
```

### Viewer flags

| Flag | Default | Effect |
|---|---:|---|
| `--results-dir PATH` | `data/results` | Directory containing match JSON files. |
| `--host HOST` | `127.0.0.1` | Interface on which the HTTP server listens. |
| `--port N` | `8765` | HTTP port. Use another value if the default port is occupied. |
| `--open` | Off | Ask Python to open the viewer URL in the default browser. |
| `-h`, `--help` | | Print command help. |

Example on another port:

```bash
conda run -n chess env PYTHONPATH=src python scripts/view_match.py \
  --results-dir data/results \
  --host 127.0.0.1 \
  --port 8877 \
  --open
```

The server reads JSON files dynamically. The game list refreshes every ten seconds, and the browser requests raw saved data rather than rebuilding a static HTML bundle. A game appears after simulation serialization completes.

The viewer performs no chess or expected-value calculations. New JSON files should contain `viewer_states`; older files without them are marked as requiring regeneration.

For each position the viewer shows:

- board, FEN, side to move, move history, and static `E(B)`;
- configured `cdepth` and `c`;
- adaptive `U(B)`, entropy, $N_{\mathrm{eff}}$, `K`, response `K'`, and branch values for the actual decision side;
- actual-side ranks, probabilities, full-response expected next `U`, candidate `deltaU`, and candidate `deltaQ/deltaW/deltaA`; the opposite counterfactual side is marked unavailable unless generated separately;
- shared move and $E(B_m)$ columns aligned across both observers;
- completed same-player cycle cards for realized game-level cycle $\Delta U$, $\Delta Q$, $\Delta W$, and $\Delta A$;
- old/new effective move counts, old/new `K`, common support data, and common probability mass.

## Saved Data

### CSV

The CSV contains one row per played ply. Important fields include:

| Field | Meaning |
|---|---|
| `E_before`, `E_after` | Universal static evaluations before and after the move. |
| `U_current` | Moving observer's adaptive value of the current board. |
| `U_after_move` | Selected candidate's full-response predicted next value `expected_next_U`. |
| `delta_u`, `candidate_delta_u` | Full-response predicted next value minus current adaptive value. |
| `U_before`, `U_after`, `delta_U`, `cycle_delta_U` | Completed same-player cycle values, aligned to the later decision state. Null for the first state of each player. |
| `delta_Q`, `delta_W`, `delta_A` | Heat/work/accessibility decomposition of the completed same-player cycle. Null until that player has a previous decision state. |
| `decomposition_error` | `delta_U - (delta_Q + delta_W + delta_A)`. |
| `N_eff_before`, `N_eff_after` | Effective move counts for the current state and, when present, the completed same-player cycle. |
| `adaptive_cdepth`, `adaptive_depth_before`, `adaptive_depth_after` | Selected local cycle-depth diagnostics. Legacy names are retained for compatibility. |
| support and mass fields | Old, new, common, old-only, and new-only support diagnostics. |
| `entropy_current` | Root move-distribution entropy. |
| `N_eff` | Root effective number of moves. |
| `selected_depth` | Entropy-selected local `cdepth` after applying the global cap. |
| `K_expanded` | Number of root branches selected for deeper recursion. |
| `nodes_evaluated` | Adaptive value nodes computed for the saved analysis. |
| `static_evaluations` | Uncached static evaluations. |
| `recursive_nodes` | Selected recursive child traversals. |
| `maximum_depth_reached` | Deepest level actually reached. |
| `average_K` | Mean selected breadth over recursively expanded nodes. |
| `cache_hits` | Reused cached values and landscapes. |
| `search_elapsed_time` | Search time in seconds. |
| `prediction_error` | Realized static reply value minus the earlier predicted adaptive value. |
| repetition fields | Twofold, threefold, fivefold, and claim metadata. |

### JSON

The JSON is the complete result and contains:

- `config`: match, search, threshold, directory, and stopping configuration;
- `white_style`, `black_style`: serialized style coefficients;
- `evaluation_weights`: serialized static feature coefficients;
- result, terminal reason, final FEN, and repetition metadata;
- `plies`: row data, candidate scores, the actual decision `search_result`, and exact adaptive branch observations for completed same-player cycles;
- `thermodynamic_transitions`: same-player heat/work/accessibility records aligned to the later turn;
- `viewer_states`: ready-to-render panels for every board position, including the completed same-player cycle transition visible at that state when available.

Candidate diagnostics include whether each move belonged to the root top-$K$ set, plus `response_N_eff`, `response_K`, and, when that candidate was enriched by `candidate_thermo_mode`, the response records used for candidate `deltaQ/deltaW/deltaA`. `viewer_states` duplicate stored search-result summary values intentionally so browser navigation requires no model computation.

### PGN

The PGN contains the played move sequence and result headers. It does not contain the full thermodynamic diagnostics; use JSON or CSV for analysis.

## Analysis and Tests

### Inspect one saved ply

```bash
conda run -n chess env PYTHONPATH=src python scripts/analyze_match.py \
  data/results/thermo_match.json \
  --ply 8 \
  --sort probability \
  --limit 20
```

Analysis options:

| Argument | Default | Effect |
|---|---:|---|
| `match_json` | `data/results/thermo_match.json` | JSON file to inspect. |
| `--ply N` | `1` | Played ply whose current shallow landscape is printed. |
| `--sort probability` | `probability` | Sort by `probability`, `static_after`, or `phi`. |
| `--limit N` | `20` | Maximum moves printed. |

The analysis script prints the saved shallow move landscape. It does not rerun adaptive search.

### Run tests

```bash
conda run -n chess env PYTHONPATH=src pytest -q
```

The tests cover probability normalization, entropy, effective move counts, adaptive breadth and depth, full fallback probability mass, terminal handling, move selection, feature conventions, logging, and viewer-ready serialization.

### Optional benchmark

```bash
conda run -n chess env PYTHONPATH=src python scripts/benchmark_adaptive.py
```

Benchmark options:

| Flag | Default | Effect |
|---|---:|---|
| `--fen FEN` | Built-in middlegame | Replace the representative benchmark position. Quote FEN strings in the shell. |
| `--beta X` | `4.0` | Probability-distribution beta used for every benchmark depth. |
| `--adaptive-c X` | `0.3` | Adaptive breadth parameter used for every benchmark depth. |

The script is a legacy benchmark for adaptive depth aliases and does not change or regenerate match files.

## Python Configuration

CLI flags expose the common simulation controls. Styles, threshold values, evaluator weights, and output directories can be configured directly in Python:

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

When `white_style` or `black_style` is passed directly, that custom object overrides the corresponding preset in `MatchConfig`. Omit the custom style arguments to use `white_strategy` and `black_strategy`.

Validation rules:

- `cdepth >= 0`;
- `adaptive_c > 0`;
- adaptive cdepth thresholds must be positive and strictly increasing.
- `0 <= solidness <= 1`.

## Project Structure

```text
src/thermo_chess/
  evaluation.py      universal static evaluator E(B)
  features.py        board features, move features, exchange exposure
  measure.py         style potential, softmax, entropy, move landscapes
  metrics.py         entropy and effective-number utilities
  search.py          cached adaptive expected-value recursion
  player.py          observer analysis and White-max/Black-min choice
  simulation.py      match loop and CSV/JSON/PGN serialization
  live_viewer.py     dynamic HTTP viewer and browser application

scripts/
  run_match.py             simulation CLI
  view_match.py            viewer CLI
  analyze_match.py         saved-landscape inspection CLI
  benchmark_adaptive.py    optional cdepth benchmark

data/games/          generated PGN files
data/results/        generated CSV and JSON files
tests/               mathematical and regression tests
match_viewer.html   legacy static viewer artifact
```

## Performance and Limitations

- Adaptive search reduces tree growth but does not make high `cdepth` free. Low-entropy nodes are deliberately allowed to search more deeply.
- Every legal root candidate receives an adaptive value before final move selection. A complete decision is therefore more expensive than one call to `U(B)`.
- Saving viewer-ready panels evaluates both observers at every match position. This increases simulation time but makes later browsing fast.
- The model is designed for interpretability and experimentation, not competitive playing strength.
- Static evaluation and move-style features are intentionally compact and hand-designed.
- The local exchange-exposure feature is a shallow capture-sequence calculation, not a general tactical engine.
- Search is CPU-bound. Viewer panel generation can use process workers, but high `cdepth` can still be expensive.
- JSON files can become large because they contain complete viewer states and candidate diagnostics.

The main scientific interpretation is:

```math
\text{low entropy}
\Rightarrow
\text{few effective moves}
\Rightarrow
\text{narrower and potentially deeper computation},
```

while

```math
\text{high entropy}
\Rightarrow
\text{many comparable moves}
\Rightarrow
\text{broader but locally shallower computation}.
```

The observer's own move measure therefore controls not only predicted behavior, but also how computational attention is allocated through the tree.
