connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

can_move(X,Y) :- connected(X,Y).
valid_move(X,Y) :- connected(X,Y).

% Task 8
wet_road.
slippery :- wet_road.
reduce_speed :- slippery.

% Extra: check a whole plan of moves
valid_path([_]).
valid_path([X,Y|T]) :- valid_move(X,Y), valid_path([Y|T]).
