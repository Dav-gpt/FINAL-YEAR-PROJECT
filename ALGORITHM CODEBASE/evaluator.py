import copy
import random
from typing import Dict, List, Set, Tuple
from problem import ProblemInstance

class Timetable:
    def __init__(self, problem: ProblemInstance, assignments: Dict[int, Tuple[str, int]] = None):
        self.problem = problem
        self.assignments = assignments or {}
        self.fitness = None
        self.hard_violations = None
        self.soft_penalty = None

    def copy(self):
        t = Timetable(self.problem, copy.deepcopy(self.assignments))
        t.fitness = self.fitness
        t.hard_violations = self.hard_violations
        t.soft_penalty = self.soft_penalty
        return t

    def assign(self, eidx: int, room_id: str, period: int):
        self.assignments[eidx] = (room_id, period)
        self.fitness = None

    def get_assignment(self, eidx: int):
        return self.assignments.get(eidx, None)

    def random_initialize(self):
        for eidx in range(self.problem.num_events):
            rid = random.choice(list(self.problem.rooms.keys()))
            period = random.randint(0, self.problem.total_periods - 1)
            self.assign(eidx, rid, period)

class TimetableEvaluator:
    def __init__(self, problem: ProblemInstance, hard_weight: float = 1e6):
        self.problem = problem
        self.hard_weight = hard_weight
        self.evaluation_count = 0

    def reset_count(self):
        self.evaluation_count = 0

    def evaluate(self, timetable: Timetable):
        if timetable.fitness is not None:
            return timetable.fitness, timetable.hard_violations, timetable.soft_penalty

        self.evaluation_count += 1

        hard = 0
        soft = 0

        # Penalty for any unassigned event
        for eidx in range(self.problem.num_events):
            if eidx not in timetable.assignments:
                hard += 100
                timetable.assign(eidx, list(self.problem.rooms.keys())[0], 0)

        # Build conflict maps
        room_period_events: Dict[Tuple[str, int], List[int]] = {}
        teacher_period_events: Dict[Tuple[str, int], List[int]] = {}
        curriculum_period_events: Dict[Tuple[str, int], List[int]] = {}
        course_period_events: Dict[Tuple[str, int], List[int]] = {}

        for eidx, (rid, period) in timetable.assignments.items():
            room_period_events.setdefault((rid, period), []).append(eidx)
            teacher = self.problem.event_teacher[eidx]
            teacher_period_events.setdefault((teacher, period), []).append(eidx)
            for curid in self.problem.event_curricula[eidx]:
                curriculum_period_events.setdefault((curid, period), []).append(eidx)
            cid = self.problem.event_course[eidx]
            course_period_events.setdefault((cid, period), []).append(eidx)

        # Hard 1: Room double-booking
        for events in room_period_events.values():
            if len(events) > 1: hard += len(events) - 1

        # Hard 2: Teacher conflict
        for events in teacher_period_events.values():
            if len(events) > 1: hard += len(events) - 1

        # Hard 3: Curriculum conflict
        for events in curriculum_period_events.values():
            if len(events) > 1: hard += len(events) - 1

        # Hard 4: Same course, same period (multiple lectures)
        for events in course_period_events.values():
            if len(events) > 1: hard += len(events) - 1

        # Hard 5: Room capacity
        for eidx, (rid, period) in timetable.assignments.items():
            room = self.problem.rooms[rid]
            course = self.problem.courses[self.problem.event_course[eidx]]
            if room.capacity < course.students: hard += 1

        # Hard 6: Period constraints (unavailability)
        for eidx, (rid, period) in timetable.assignments.items():
            cid = self.problem.event_course[eidx]
            banned = self.problem.period_constraints.get(cid, [])
            if period in banned: hard += 1

        # Hard 7: Room constraints
        for eidx, (rid, period) in timetable.assignments.items():
            cid = self.problem.event_course[eidx]
            required = self.problem.room_constraints.get(cid, [])
            if required and rid not in required: hard += 1

        # Soft 1: Room capacity wastage
        for eidx, (rid, period) in timetable.assignments.items():
            room = self.problem.rooms[rid]
            course = self.problem.courses[self.problem.event_course[eidx]]
            if room.capacity > course.students:
                soft += (room.capacity - course.students) // 10

        # Soft 2: Minimum working days
        course_days: Dict[str, Set[int]] = {}
        for eidx, (rid, period) in timetable.assignments.items():
            cid = self.problem.event_course[eidx]
            day = period // self.problem.periods_per_day
            course_days.setdefault(cid, set()).add(day)
        for cid, c in self.problem.courses.items():
            if cid in course_days:
                days_used = len(course_days[cid])
                target = min(c.lectures, 3)
                if days_used < target: soft += (target - days_used) * 5
            else:
                soft += c.lectures * 5

        # Soft 3: Curriculum compactness (gaps)
        for curid, cur in self.problem.curricula.items():
            cur_periods = []
            for cid in cur.courses:
                if cid in self.problem.course_events:
                    for eidx in self.problem.course_events[cid]:
                        if eidx in timetable.assignments:
                            cur_periods.append(timetable.assignments[eidx][1])
            if len(cur_periods) > 1:
                cur_periods.sort()
                for i in range(len(cur_periods) - 1):
                    gap = cur_periods[i+1] - cur_periods[i]
                    if gap > 1: soft += (gap - 1) * 2

        # Soft 4: Room stability
        for cid, c in self.problem.courses.items():
            rooms_used = set()
            if cid in self.problem.course_events:
                for eidx in self.problem.course_events[cid]:
                    if eidx in timetable.assignments:
                        rooms_used.add(timetable.assignments[eidx][0])
                if len(rooms_used) > 1:
                    soft += (len(rooms_used) - 1) * 3

        fitness = hard * self.hard_weight + soft
        timetable.fitness = fitness
        timetable.hard_violations = hard
        timetable.soft_penalty = soft
        return fitness, hard, soft
