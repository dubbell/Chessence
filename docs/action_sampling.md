# Action Sampling Algorithm

Actions consist of two items:
- `select`: square that a piece will be moved from
- `target`: square that the piece will be moved to

These are both indices in the `[0,63]` range, each representing a corresponding coordinate on the 8x8 chess board. The actor produces a selection and target distribution respectively, but all moves will never be possible in any state. Filtering is therefore required, and is applied with a 64x64 move matrix `M` where `M[i,j]` represents the possibility of moving a piece from square `i` to square `j`. The following algorithm computes `select` and `target`, taking filtering and minibatches (batch size $B$) into account.

Inputs:
- $M \in \mathbb R^{B\times 64 \times 64}$ (move matrix filters)
- $p_s \in \mathbb R^{B \times 64}$ (`select` distributions)
- $p_t \in \mathbb R^{B \times 64}$ (`target` distributions)
```
1: s_filter = M.any(axis=2)                # given select, if any target is available
2: s_filtered = s_filter * p_s             # multiply by 0 in p_s if select has no targets
3: select = s_filtered.argmax(axis=1)      # select = highest in filtered p_s
4: t_filter = M[arange(B), select]         # available targets for the select in each batch item
5: t_filtered = t_filter * p_t             # multiply by 0 in p_t if target not available
6: target = t_filtered.argmax(axis=1)      # target = highest in filtered p_t
7: return: 
      select, 
      target, 
      log(p_s[select]) + log(p_t[target])  # log likelihood
```