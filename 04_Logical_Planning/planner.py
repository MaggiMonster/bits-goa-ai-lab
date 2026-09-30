"""STRIPS-style planner with BFS for the warehouse robot (locations A, B, C).

State   : frozenset of ground propositions, e.g. ("At","Robot","A").
Action  : name, positive/negative preconditions, positive/negative effects.
Applicable(S, a): pos_pre <= S and neg_pre disjoint from S        (logic: S |= Pre(a))
Apply(S, a)     : (S - neg_eff) | pos_eff                          (effects)
Search  : breadth-first over states; goal reached when goal <= S.
Assumptions: closed-world (facts not in the set are false); the package is the only
movable object; Move is allowed only along the connected edges A-B and B-C.
"""
from collections import deque
from collections import namedtuple

Action = namedtuple("Action", "name pos_pre neg_pre pos_eff neg_eff")

LOCS = ["A", "B", "C"]
EDGES = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]


def at(x, l): return ("At", x, l)


def make_actions(include_pickup=True, extra_edges=()):
    acts = []
    for x, y in list(EDGES) + list(extra_edges):
        acts.append(Action(f"Move({x},{y})", {at("Robot", x)}, set(), {at("Robot", y)}, {at("Robot", x)}))
    for l in LOCS:
        if include_pickup:
            acts.append(Action(f"PickUp(Package,{l})", {at("Robot", l), at("Package", l)}, set(),
                               {("Holding", "Package")}, {at("Package", l)}))
        acts.append(Action(f"Drop(Package,{l})", {at("Robot", l), ("Holding", "Package")}, set(),
                           {at("Package", l)}, {("Holding", "Package")}))
    return acts


def applicable(state, a):
    return a.pos_pre <= state and not (a.neg_pre & state)


def apply_action(state, a):
    return frozenset((state - a.neg_eff) | a.pos_eff)


def bfs_plan(init, goal, actions):
    """Return (plan, states) or None. plan = list of action names; states = S0..Sn."""
    init = frozenset(init)
    frontier = deque([init])
    parent = {init: None}
    while frontier:
        s = frontier.popleft()
        if set(goal) <= s:
            plan, states = [], [s]
            while parent[s] is not None:
                s, name = parent[s]
                plan.append(name)
                states.append(s)
            return plan[::-1], states[::-1]
        for a in actions:
            if applicable(s, a):
                n = apply_action(s, a)
                if n not in parent:
                    parent[n] = (s, a.name)
                    frontier.append(n)
    return None


def fmt(state):
    return ", ".join(sorted(f"{p}({','.join(args)})" for p, *args in state))


def show(title, init, goal, actions):
    print(f"--- {title}")
    print("Initial:", fmt(init)); print("Goal   :", fmt(goal))
    res = bfs_plan(init, goal, actions)
    if res is None:
        print("No plan found")
        return None
    plan, states = res
    print("Plan:", " -> ".join(plan))
    for i, s in enumerate(states):
        print(f"  S{i}: {fmt(s)}")
    return res


def verify(init, goal, plan, actions):
    """Independent re-execution of a plan: every precondition checked, goal checked."""
    by_name = {a.name: a for a in actions}
    s = frozenset(init)
    for name in plan:
        a = by_name[name]
        if not applicable(s, a):
            return False
        s = apply_action(s, a)
    return set(goal) <= s


I = {at("Robot", "A"), at("Package", "A")}
G = {at("Package", "C")}

if __name__ == "__main__":
    acts = make_actions()
    print("Initially applicable actions:", [a.name for a in acts if applicable(frozenset(I), a)])
    r = show("Test A: solvable warehouse problem", I, G, acts)
    print("Plan verified independently:", verify(I, G, r[0], acts))
    print()
    show("Test B: no PickUp action", I, G, make_actions(include_pickup=False))
    print()
    # Test C: extra move-only action (A->C shortcut for the robot alone); robot at C must not count as package at C
    acts_c = make_actions(extra_edges=[("A", "C")])
    r = show("Test C1: goal = Package at C, with extra Move(A,C) shortcut", I, G, acts_c)
    print("Plan verified independently:", verify(I, G, r[0], acts_c))
    print()
    show("Test C2: goal = Robot at C (package need not move)", I, {at("Robot", "C")}, acts_c)
    print()
    # Test C3: robot-only moves cannot satisfy a package goal when PickUp is removed
    show("Test C3: extra Move(A,C), no PickUp, goal Package at C", I, G, make_actions(include_pickup=False, extra_edges=[("A", "C")]))
    # extra: a proposed invalid plan must be rejected
    bad = ["Move(A,B)", "Move(B,C)", "Drop(Package,C)"]
    print("\nInvalid plan (robot never picked up package) verifies as:", verify(I, G, bad, acts))
