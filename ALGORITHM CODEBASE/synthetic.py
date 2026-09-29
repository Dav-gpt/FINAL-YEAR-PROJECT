from problem import Room, Course, Student, Curriculum, ProblemInstance

def generate_synthetic_instance(name="synthetic-small", num_courses=20, num_rooms=5,
                                num_students=50, num_curricula=5):
    import random
    random.seed(42)

    rooms = {}
    for i in range(num_rooms):
        rid = f"R{i}"
        rooms[rid] = Room(rid, capacity=random.choice([30, 50, 70, 100]))

    courses = {}
    teachers = [f"T{i}" for i in range(num_courses // 2 + 1)]
    for i in range(num_courses):
        cid = f"C{i}"
        courses[cid] = Course(cid, teacher=random.choice(teachers),
                              lectures=random.choice([1, 2, 3]),
                              students=random.choice([20, 40, 60]))

    # Students enroll in 3-5 courses
    course_ids = list(courses.keys())
    students_dict = {}
    for i in range(num_students):
        sid = f"S{i}"
        enrolled = random.sample(course_ids, k=random.randint(3, 5))
        students_dict[sid] = Student(sid, enrolled)

    # Curricula group 3-4 courses
    curricula = {}
    for i in range(num_curricula):
        curid = f"CUR{i}"
        curricula[curid] = Curriculum(curid, random.sample(course_ids, k=random.randint(3, 4)))

    return ProblemInstance(name, rooms, courses, students_dict, curricula)