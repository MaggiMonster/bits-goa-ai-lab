# Lab 04: Written answers

## Task 0
I = {At(Robot,A), At(Package,A)}; G = {At(Package,C)}.
Actions: Move(x,y) for A-B, B-A, B-C, C-B (pre At(Robot,x); eff -At(Robot,x), +At(Robot,y)); PickUp(Package,l) (pre At(Robot,l), At(Package,l); eff -At(Package,l), +Holding(Package)); Drop(Package,l) (pre At(Robot,l), Holding(Package); eff -Holding(Package), +At(Package,l)).
PickUp(Package,A) is applicable in I (both preconditions hold); Drop(Package,C) is not (robot not at C, not holding). Listed is not the same as applicable: all preconditions must hold.

## Task 1: hand plan
S0 At(Robot,A), At(Package,A) -> PickUp(Package,A) -> S1 At(Robot,A), Holding(Package) -> Move(A,B) -> S2 At(Robot,B), Holding(Package) -> Move(B,C) -> S3 At(Robot,C), Holding(Package) -> Drop(Package,C) -> S4 At(Robot,C), At(Package,C) |= G.
Note: the lab's hint sequence with PickUp(Package,B) is invalid, since the package stays at A.

## Task 3 tests
A solvable: plan above, verified by re-execution. B no PickUp: "No plan found". C: with an extra Move(A,C) the robot reaches C alone but the package goal still needs PickUp/Drop (plan PickUp, Move(A,C), Drop; without PickUp no plan); robot-at-C goal is solved by Move(A,C) alone with the package left at A.

## Task 4: logic and search
Flow: state -> check preconditions -> apply effects (remove negatives, add positives) -> successor -> search over alternatives -> goal test. Logic determines what is possible; search determines what to try.

## Task 5: trust
The independently executed transitions are more trustworthy than an LLM's explanation: they are computed mechanically from the action definitions, while a generated explanation can sound right and be wrong. A generated explanation is not an independent verification.

## Tasks 6-8 (Prolog)
can_move(a,b) true (fact connected(a,b) + rule); can_move(a,c) fails (no connected(a,c) and no rule derives it, not provable, which is weaker than provably false); the rule is Connected(X,Y) -> CanMove(X,Y) written head :- body. valid_move(a,b), valid_move(b,c) succeed, valid_move(a,c) fails, so a proposed Move(a,c) is not supported by the warehouse knowledge. reduce_speed: wet_road (fact) => slippery (rule) => reduce_speed (rule) => conclusion.

## Reflection questions
1. Specifying preconditions and effects first defines correctness and stops the LLM inventing semantics.
2. Without precondition checks the robot could Drop something it is not holding or PickUp at B when the package is at A.
3. A plan that looks reasonable may skip a step; validity needs every precondition true when the action runs and S_n |= G.
4. The LLM wrote the state representation, apply/applicable, BFS, plan printing, tests and the Prolog file.
5. Independent checks: action definitions, re-execution of the plan, the impossible/irrelevant/invalid cases.
6. In S |= Pre(a), effect application, S_n |= G, and the Prolog rules.
7. Planning is search over states with logic defining the transitions; BFS is the same algorithm as in the search module.

### Optional reflection
Fact = unconditional statement; rule = head derived from body. A query asks whether something follows from the knowledge base. An independent Prolog check stops a generator validating itself and catches plausible but wrong LLM plans.
