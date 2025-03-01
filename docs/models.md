### Simple critic model
- In given state, calculate possible moves and get resulting states if those moves are made.
- Compute state values for these following states, and choose the action that leads to the state with the highest value.
- This will likely be difficult for pretraining however.
  - Could instead do preference-based RL, and pretrain on pairs of available moves, giving preference to the one that is chosen by the player.
  - Same would be done in RL finetuning, by giving preference to the moves that get made if they lead to win.
  - Sounds inefficient, however.