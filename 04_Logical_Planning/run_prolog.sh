swipl -q -g "
  forall(member(Q,[can_move(a,b),can_move(a,c),valid_move(a,b),valid_move(b,c),valid_move(a,c),reduce_speed,valid_path([a,b,c]),valid_path([a,c])]),
         ( (call(Q) -> R = true ; R = false), format('?- ~w.  ~w~n',[Q,R]) ))" -t halt planner.pl
