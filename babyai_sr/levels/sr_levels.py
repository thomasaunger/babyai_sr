from babyai.levels.levelgen import *


class Level_GoToObjUnlocked(RoomGridLevel):
    """
    Go to an object, inside another room behind an unlocked door with no distractors
    """
    
    def __init__(self, seed=None):
        super().__init__(
            num_rows=2,
            num_cols=2,
            room_size=5,
            max_steps=64,
            seed=seed
        )
    
    def gen_mission(self):
        door_color = self._rand_color()
        
        # Add a door of color door_color connecting starting room to another room
        self.add_door(0, 0, door_idx=0, color=door_color, locked=False)
        
        self.place_agent(0, 0)
        
        obj, _ = self.add_object(1, 0, kind=self._rand_elem(["ball", "box"]),  color=self._rand_color())
        
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToObjLocked(RoomGridLevel):
    """
    Go to an object, inside another room behind a locked door with no distractors
    """
    
    def __init__(self, seed=None):
        super().__init__(
            num_rows=2,
            num_cols=2,
            room_size=5,
            max_steps=64,
            seed=seed
        )
    
    def gen_mission(self):
        door_color = self._rand_color()
        
        # Add a door of color door_color connecting starting room to another room
        self.add_door(0, 0, door_idx=0, color=door_color, locked=True)
        
        # Add a key of color door_color in the starting room
        self.add_object(0, 0, kind="key", color=door_color)
        
        self.place_agent(0, 0)
        
        obj, _ = self.add_object(1, 0, kind=self._rand_elem(["ball", "box"]),  color=self._rand_color())
        
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToObjLocked_ambiguous(RoomGridLevel):
    """
    Go to an object, inside another room behind a locked door with one distractor
    """
    
    def __init__(self, seed=None):
        super().__init__(
            num_rows=2,
            num_cols=2,
            room_size=5,
            max_steps=64,
            seed=seed
        )
    
    def gen_mission(self):
        door_color = self._rand_color()
        
        # Add a door of color door_color connecting starting room to another room
        self.add_door(0, 0, door_idx=0, color=door_color, locked=True)
        
        # Add a key of color door_color in the starting room
        self.add_object(0, 0, kind="key", color=door_color)
        
        self.place_agent(0, 0)
        
        objs = []
        for _ in range(2):
            obj, _ = self.add_object(1, 0, kind=self._rand_elem(["ball", "box"]),  color=self._rand_color())
            objs.append(obj)
        
        obj = self._rand_elem(objs)
        
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

for name, level in list(globals().items()):
    if name.startswith('Level_'):
        level.is_bonus = True

# Register the levels in this file
register_levels(__name__, globals())
