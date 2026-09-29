import random
import copy
from evaluator import Timetable, TimetableEvaluator

def move_event(timetable: Timetable, eidx: int, room_id: str = None, period: int = None):
    problem = timetable.problem
    if room_id is None:
        room_id = random.choice(list(problem.rooms.keys()))
    if period is None:
        period = random.randint(0, problem.total_periods - 1)
    timetable.assign(eidx, room_id, period)

def swap_events(timetable: Timetable, e1: int, e2: int):
    a1 = timetable.get_assignment(e1)
    a2 = timetable.get_assignment(e2)
    if a1 and a2:
        timetable.assign(e1, a2[0], a2[1])
        timetable.assign(e2, a1[0], a1[1])

def kempe_chain_perturbation(timetable: Timetable, donor1: Timetable, donor2: Timetable,
                             evaluator: TimetableEvaluator, max_evals: int = None):
    problem = timetable.problem
    # Pick a random event (preferably one with conflicts if any exist)
    eidx = random.randint(0, problem.num_events - 1)
    donors = [donor1, donor2]
    donor = random.choice(donors)
    if donor and eidx in donor.assignments:
        timetable.assign(eidx, donor.assignments[eidx][0], donor.assignments[eidx][1])
    else:
        move_event(timetable, eidx)
    if max_evals is None or evaluator.evaluation_count < max_evals:
        evaluator.evaluate(timetable)

def slot_swap_local_search(timetable: Timetable, best_timetable: Timetable, max_swaps: int,
                           evaluator: TimetableEvaluator, max_evals: int = None):
    problem = timetable.problem
    for _ in range(max_swaps):
        if max_evals is not None and evaluator.evaluation_count >= max_evals:
            break
        e1 = random.randint(0, problem.num_events - 1)
        e2 = random.randint(0, problem.num_events - 1)
        if e1 == e2:
            continue
        a1 = timetable.get_assignment(e1)
        a2 = timetable.get_assignment(e2)
        if a1 is None or a2 is None:
            continue

        old_fit = timetable.fitness
        old_hard = timetable.hard_violations
        old_soft = timetable.soft_penalty
        if old_fit is None:
            if max_evals is not None and evaluator.evaluation_count >= max_evals:
                break
            old_fit, _, _ = evaluator.evaluate(timetable)
            old_hard = timetable.hard_violations
            old_soft = timetable.soft_penalty

        swap_events(timetable, e1, e2)
        if max_evals is not None and evaluator.evaluation_count >= max_evals:
            swap_events(timetable, e1, e2)
            timetable.fitness = old_fit
            timetable.hard_violations = old_hard
            timetable.soft_penalty = old_soft
            break
        new_fit, new_hard, _ = evaluator.evaluate(timetable)

        if new_hard > 0 or new_fit >= old_fit:
            # Revert
            swap_events(timetable, e1, e2)
            timetable.fitness = old_fit
            timetable.hard_violations = old_hard
            timetable.soft_penalty = old_soft
    return True

def greedy_repair(timetable: Timetable, evaluator: TimetableEvaluator, max_evals: int = None,
                  passes: int = 5, samples_per_conflict: int = 20):
    problem = timetable.problem
    # Ensure all events have assignments
    for eidx in range(problem.num_events):
        if eidx not in timetable.assignments:
            rid = random.choice(list(problem.rooms.keys()))
            period = random.randint(0, problem.total_periods - 1)
            timetable.assign(eidx, rid, period)

    for _ in range(passes):
        if max_evals is not None and evaluator.evaluation_count >= max_evals:
            break
        evaluator.evaluate(timetable)
        if timetable.hard_violations == 0:
            break

        # Build conflict maps
        room_period = {}
        teacher_period = {}
        for eidx, (rid, period) in timetable.assignments.items():
            room_period.setdefault((rid, period), []).append(eidx)
            teacher = problem.event_teacher[eidx]
            teacher_period.setdefault((teacher, period), []).append(eidx)

        conflicts = set()
        for key, evts in room_period.items():
            if len(evts) > 1:
                conflicts.update(evts)
        for key, evts in teacher_period.items():
            if len(evts) > 1:
                conflicts.update(evts)

        if not conflicts:
            break

        for eidx in conflicts:
            if max_evals is not None and evaluator.evaluation_count >= max_evals:
                break
            current = timetable.get_assignment(eidx)
            best_move = None
            best_hard = float('inf')
            for _ in range(samples_per_conflict):
                if max_evals is not None and evaluator.evaluation_count >= max_evals:
                    break
                rid = random.choice(list(problem.rooms.keys()))
                period = random.randint(0, problem.total_periods - 1)
                timetable.assign(eidx, rid, period)
                fit, hard, _ = evaluator.evaluate(timetable)
                if hard < best_hard:
                    best_hard = hard
                    best_move = (rid, period)
            if best_move:
                timetable.assign(eidx, best_move[0], best_move[1])
            else:
                timetable.assign(eidx, current[0], current[1])

    if timetable.fitness is None and (max_evals is None or evaluator.evaluation_count < max_evals):
        evaluator.evaluate(timetable)
