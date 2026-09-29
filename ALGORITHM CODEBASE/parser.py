import numpy as np
from typing import Dict
from problem import Room, Course, Student, Curriculum, ProblemInstance

def parse_itc2007(filepath: str) -> ProblemInstance:
    """
    Parser for ITC-2007 Track 2 (.tim) Post-Enrolment Course Timetabling instances.
    Format: num_events, num_rooms, num_features, num_students, then 7 data blocks.
    """
    with open(filepath, 'r') as f:
        tokens = f.read().split()
    
    if not tokens:
        raise ValueError(f"Empty file: {filepath}")
    
    # Header
    num_events = int(tokens[0])
    num_rooms = int(tokens[1])
    num_features = int(tokens[2])
    num_students = int(tokens[3])
    num_timeslots = 45  # Standard for ITC-2007 Track 2
    
    ptr = 4
    
    # Room capacities (Size: R)
    room_capacities = [int(x) for x in tokens[ptr : ptr + num_rooms]]
    ptr += num_rooms
    
    # Student-event enrollment (Size: S x E)
    student_events = np.array([int(x) for x in tokens[ptr : ptr + num_students * num_events]]).reshape(num_students, num_events)
    ptr += num_students * num_events
    
    # Event-feature requirements (Size: E x F)
    event_features = np.array([int(x) for x in tokens[ptr : ptr + num_events * num_features]]).reshape(num_events, num_features)
    ptr += num_events * num_features
    
    # Room-feature provided (Size: R x F)
    room_features = np.array([int(x) for x in tokens[ptr : ptr + num_rooms * num_features]]).reshape(num_rooms, num_features)
    ptr += num_rooms * num_features
    
    # Event availability (Size: E x 45)
    event_availability = np.array([int(x) for x in tokens[ptr : ptr + num_events * num_timeslots]]).reshape(num_events, num_timeslots)
    ptr += num_events * num_timeslots
    
    # Event-event relationships (Size: E x E)
    event_relationships = np.array([int(x) for x in tokens[ptr : ptr + num_events * num_events]]).reshape(num_events, num_events)
    
    # Build problem structures
    rooms: Dict[str, Room] = {}
    for r in range(num_rooms):
        rooms[f"R{r}"] = Room(f"R{r}", capacity=room_capacities[r])
    
    courses: Dict[str, Course] = {}
    for e in range(num_events):
        courses[f"E{e}"] = Course(f"E{e}", teacher=f"T{e}", lectures=1, students=np.sum(student_events[:, e]))
    
    students: Dict[str, Student] = {}
    for s in range(num_students):
        enrolled_courses = [f"E{e}" for e in range(num_events) if student_events[s, e] == 1]
        students[f"S{s}"] = Student(f"S{s}", enrolled_courses)
    
    # Create curricula from event relationships (group related events)
    curricula: Dict[str, Curriculum] = {}
    processed = set()
    curriculum_count = 0
    for e in range(num_events):
        if e in processed:
            continue
        # Find all events related to event e
        related = [str(f"E{e}")]
        for other in range(num_events):
            if event_relationships[e, other] == 1 and other != e:
                related.append(f"E{other}")
                processed.add(other)
        processed.add(e)
        curricula[f"CUR{curriculum_count}"] = Curriculum(f"CUR{curriculum_count}", related)
        curriculum_count += 1
    
    # Period constraints: events unavailable in certain timeslots
    period_constraints: Dict[str, list] = {}
    for e in range(num_events):
        unavailable = [t for t in range(num_timeslots) if event_availability[e, t] == 0]
        if unavailable:
            period_constraints[f"E{e}"] = unavailable
    
    # Room constraints: events requiring specific room features
    room_constraints: Dict[str, list] = {}
    for e in range(num_events):
        required_features = [f for f in range(num_features) if event_features[e, f] == 1]
        if required_features:
            # Find rooms that provide these features
            suitable_rooms = []
            for r in range(num_rooms):
                if all(room_features[r, f] == 1 for f in required_features):
                    suitable_rooms.append(f"R{r}")
            if suitable_rooms:
                room_constraints[f"E{e}"] = suitable_rooms
    
    # ITC-2007 Track 2 uses 5 days (9 timeslots per day)
    days = 5
    periods_per_day = 9
    
    instance_name = filepath.split('\\')[-1].replace('.tim', '')
    
    return ProblemInstance(instance_name, rooms, courses, students, curricula,
                          period_constraints, room_constraints, days, periods_per_day)

def parse_json(filepath: str) -> ProblemInstance:
    """
    Parser for JSON-formatted UCTP instances.
    Supports list or dict representation for rooms, courses, students, curricula.
    """
    import json
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)
    
    name = data.get('name', 'JSON_Instance')
    days = int(data.get('days', 5))
    periods_per_day = int(data.get('periods_per_day', 9))
    
    rooms = {}
    rooms_data = data.get('rooms', [])
    if isinstance(rooms_data, list):
        for r in rooms_data:
            rid = str(r['id'])
            rooms[rid] = Room(rid, capacity=int(r['capacity']))
    elif isinstance(rooms_data, dict):
        for rid, r in rooms_data.items():
            rooms[rid] = Room(rid, capacity=int(r.get('capacity', r) if isinstance(r, dict) else r))
            
    courses = {}
    courses_data = data.get('courses', [])
    if isinstance(courses_data, list):
        for c in courses_data:
            cid = str(c['id'])
            courses[cid] = Course(
                cid,
                teacher=str(c.get('teacher', f"T_{cid}")),
                lectures=int(c.get('lectures', 1)),
                students=int(c.get('students', 0))
            )
    elif isinstance(courses_data, dict):
        for cid, c in courses_data.items():
            if isinstance(c, dict):
                courses[cid] = Course(
                    cid,
                    teacher=str(c.get('teacher', f"T_{cid}")),
                    lectures=int(c.get('lectures', 1)),
                    students=int(c.get('students', 0))
                )
            else:
                courses[cid] = Course(cid, f"T_{cid}", 1, int(c))
                
    students = {}
    students_data = data.get('students', [])
    if isinstance(students_data, list):
        for s in students_data:
            sid = str(s['id'])
            students[sid] = Student(sid, [str(x) for x in s.get('courses', [])])
    elif isinstance(students_data, dict):
        for sid, s in students_data.items():
            courses_list = s.get('courses', s) if isinstance(s, dict) else s
            students[sid] = Student(sid, [str(x) for x in courses_list])
            
    curricula = {}
    curricula_data = data.get('curricula', [])
    if isinstance(curricula_data, list):
        for cur in curricula_data:
            curid = str(cur['id'])
            curricula[curid] = Curriculum(curid, [str(x) for x in cur.get('courses', [])])
    elif isinstance(curricula_data, dict):
        for curid, cur in curricula_data.items():
            courses_list = cur.get('courses', cur) if isinstance(cur, dict) else cur
            curricula[curid] = Curriculum(curid, [str(x) for x in courses_list])
            
    period_constraints = {}
    for cid, banned in data.get('period_constraints', {}).items():
        period_constraints[str(cid)] = [int(p) for p in banned]
        
    room_constraints = {}
    for cid, allowed in data.get('room_constraints', {}).items():
        room_constraints[str(cid)] = [str(r) for r in allowed]
        
    return ProblemInstance(name, rooms, courses, students, curricula,
                           period_constraints, room_constraints, days, periods_per_day)

def parse_csv(filepath: str) -> ProblemInstance:
    """
    Parser for CSV-formatted UCTP instances.
    Each row starts with the type: 'Room', 'Course', 'Student', 'Curriculum',
    'PeriodConstraint', 'RoomConstraint', or 'Settings'.
    """
    import csv
    rooms = {}
    courses = {}
    students = {}
    curricula = {}
    period_constraints = {}
    room_constraints = {}
    days = 5
    periods_per_day = 9
    name = filepath.split('\\')[-1].split('/')[-1].replace('.csv', '')
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].strip() or row[0].strip().startswith('#'):
                continue
            
            row_type = row[0].strip().lower()
            if row_type == 'room':
                # Room, id, capacity
                if len(row) >= 3:
                    rid = row[1].strip()
                    cap = int(row[2].strip())
                    rooms[rid] = Room(rid, cap)
            elif row_type == 'course':
                # Course, id, teacher, lectures, students
                if len(row) >= 5:
                    cid = row[1].strip()
                    teacher = row[2].strip()
                    lectures = int(row[3].strip())
                    students_count = int(row[4].strip())
                    courses[cid] = Course(cid, teacher, lectures, students_count)
            elif row_type == 'student':
                # Student, id, course1;course2;...
                if len(row) >= 3:
                    sid = row[1].strip()
                    c_list = [x.strip() for x in row[2].split(';') if x.strip()]
                    students[sid] = Student(sid, c_list)
            elif row_type == 'curriculum':
                # Curriculum, id, course1;course2;...
                if len(row) >= 3:
                    curid = row[1].strip()
                    c_list = [x.strip() for x in row[2].split(';') if x.strip()]
                    curricula[curid] = Curriculum(curid, c_list)
            elif row_type == 'periodconstraint':
                # PeriodConstraint, course_id, banned_period1;banned_period2;...
                if len(row) >= 3:
                    cid = row[1].strip()
                    banned = [int(x.strip()) for x in row[2].split(';') if x.strip()]
                    period_constraints[cid] = banned
            elif row_type == 'roomconstraint':
                # RoomConstraint, course_id, allowed_room1;allowed_room2;...
                if len(row) >= 3:
                    cid = row[1].strip()
                    allowed = [x.strip() for x in row[2].split(';') if x.strip()]
                    room_constraints[cid] = allowed
            elif row_type == 'settings':
                # Settings, days, periods_per_day
                if len(row) >= 3:
                    days = int(row[1].strip())
                    periods_per_day = int(row[2].strip())
                    
    return ProblemInstance(name, rooms, courses, students, curricula, 
                           period_constraints, room_constraints, days, periods_per_day)

def parse_file(filepath: str) -> ProblemInstance:
    """
    Unified parser selector based on file extension.
    """
    lower_path = filepath.lower()
    if lower_path.endswith('.tim'):
        return parse_itc2007(filepath)
    elif lower_path.endswith('.json'):
        return parse_json(filepath)
    elif lower_path.endswith('.csv'):
        return parse_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format for {filepath}. Must be .tim, .json, or .csv")