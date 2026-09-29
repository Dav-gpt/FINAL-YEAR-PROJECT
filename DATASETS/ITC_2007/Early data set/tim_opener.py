import numpy as np

class TimParser:
    """
    Parser for ITC-2007 Track 2 (.tim) Post-Enrolment Course Timetabling instances.
    """
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            # Flatten all tokens into a list for sequential processing
            tokens = f.read().split()
        
        if not tokens:
            raise ValueError("The file is empty.")

        # 1. Header Information
        self.num_events = int(tokens[0])
        self.num_rooms = int(tokens[1])
        self.num_features = int(tokens[2])
        self.num_students = int(tokens[3])
        self.num_timeslots = 45 # Standard for ITC-2007 Track 2
        
        ptr = 4
        
        # 2. Room Capacities (Size: R)
        self.room_capacities = [int(x) for x in tokens[ptr : ptr + self.num_rooms]]
        ptr += self.num_rooms
        
        # 3. Student-Event Enrollment (Size: S x E)
        # 1 if student s is enrolled in event e
        self.student_events = self._get_matrix(tokens, ptr, self.num_students, self.num_events)
        ptr += self.num_students * self.num_events
        
        # 4. Event-Feature Requirements (Size: E x F)
        # 1 if event e requires feature f
        self.event_features = self._get_matrix(tokens, ptr, self.num_events, self.num_features)
        ptr += self.num_events * self.num_features
        
        # 5. Room-Feature Provided (Size: R x F)
        # 1 if room r provides feature f
        self.room_features = self._get_matrix(tokens, ptr, self.num_rooms, self.num_features)
        ptr += self.num_rooms * self.num_features
        
        # 6. Event Availability (Size: E x 45)
        # 1 if event e can be scheduled in timeslot t
        self.event_availability = self._get_matrix(tokens, ptr, self.num_events, self.num_timeslots)
        ptr += self.num_events * self.num_timeslots
        
        # 7. Event-Event Relationships (Size: E x E)
        # Defines precedence or grouping constraints
        self.event_relationships = self._get_matrix(tokens, ptr, self.num_events, self.num_events)
        ptr += self.num_events * self.num_events

    def _get_matrix(self, tokens, start, rows, cols):
        """Helper to slice tokens and reshape into a 2D numpy array."""
        data = [int(x) for x in tokens[start : start + rows * cols]]
        return np.array(data).reshape(rows, cols)

    def get_conflicts(self):
        """
        Derives the Event-Event conflict matrix based on student enrollments.
        Two events conflict if they share at least one student.
        """
        # (E x S) * (S x E) -> (E x E) matrix where entry > 0 means conflict
        return (self.student_events.T @ self.student_events) > 0

    def summary(self):
        """Returns a summary of the parsed data."""
        return {
            "Metadata": {
                "Events": self.num_events,
                "Rooms": self.num_rooms,
                "Features": self.num_features,
                "Students": self.num_students
            },
            "Statistics": {
                "Enrollment Density": f"{np.mean(self.student_events)*100:.2f}%",
                "Availability Density": f"{np.mean(self.event_availability)*100:.2f}%",
                "Room Capacity Range": (min(self.room_capacities), max(self.room_capacities))
            }
        }

# Usage Example:
parser = TimParser('comp-2007-2-1 (1).tim')
print(parser.summary())