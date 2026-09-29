import random
from typing import Dict, List, Set, Tuple

class Room:
    def __init__(self, rid: str, capacity: int):
        self.id = rid
        self.capacity = capacity

class Course:
    def __init__(self, cid: str, teacher: str, lectures: int, students: int):
        self.id = cid
        self.teacher = teacher
        self.lectures = lectures
        self.students = students

class Student:
    def __init__(self, sid: str, courses: List[str]):
        self.id = sid
        self.courses = courses

class Curriculum:
    def __init__(self, curid: str, courses: List[str]):
        self.id = curid
        self.courses = courses

class ProblemInstance:
    def __init__(self, name: str, rooms: Dict[str, Room], courses: Dict[str, Course],
                 students: Dict[str, Student], curricula: Dict[str, Curriculum],
                 period_constraints: Dict[str, List[int]] = None,
                 room_constraints: Dict[str, List[str]] = None,
                 days: int = 5, periods_per_day: int = 9):
        self.name = name
        self.rooms = rooms
        self.courses = courses
        self.students = students
        self.curricula = curricula
        self.period_constraints = period_constraints or {}
        self.room_constraints = room_constraints or {}
        self.days = days
        self.periods_per_day = periods_per_day
        self.total_periods = days * periods_per_day

        # Flatten events: one per lecture occurrence
        self.events: List[Tuple[str, int]] = []
        self.event_course: List[str] = []
        for cid, c in self.courses.items():
            for lec in range(c.lectures):
                self.events.append((cid, lec))
                self.event_course.append(cid)
        self.num_events = len(self.events)

        # Course -> event indices
        self.course_events: Dict[str, List[int]] = {}
        for i, (cid, _) in enumerate(self.events):
            self.course_events.setdefault(cid, []).append(i)

        # Teacher per event
        self.event_teacher = [self.courses[cid].teacher for cid in self.event_course]

        # Students per event (set of student ids)
        self.event_students: List[Set[str]] = [set() for _ in range(self.num_events)]
        for sid, s in self.students.items():
            for cid in s.courses:
                if cid in self.course_events:
                    for eidx in self.course_events[cid]:
                        self.event_students[eidx].add(sid)

        # Curricula per event
        self.event_curricula: List[Set[str]] = [set() for _ in range(self.num_events)]
        for curid, cur in self.curricula.items():
            for cid in cur.courses:
                if cid in self.course_events:
                    for eidx in self.course_events[cid]:
                        self.event_curricula[eidx].add(curid)