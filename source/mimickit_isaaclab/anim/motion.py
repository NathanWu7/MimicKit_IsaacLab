import enum
import pickle

class LoopMode(enum.Enum):
    CLAMP = 0
    WRAP = 1

def load_motion(file):
    with open(file, "rb") as filestream:
        in_dict = pickle.load(filestream)

        loop_mode_val = in_dict["loop_mode"]
        fps = in_dict["fps"]
        frames = in_dict["frames"]
        
        loop_mode = LoopMode(loop_mode_val)

        motion_data = Motion(loop_mode=loop_mode,
                             fps=fps,
                             frames=frames)
    return motion_data

class Motion():
    def __init__(self, loop_mode, fps, frames):
        self.loop_mode = loop_mode
        self.fps = fps
        self.frames = frames
        return

    def save(self, out_file):
        with open(out_file, "wb") as out_f:
            out_dict = {
                "loop_mode": self.loop_mode.value,
                "fps": self.fps,
                "frames": self.frames
            }
            pickle.dump(out_dict, out_f)
        return

    def get_length(self):
        num_frames = self.frames.shape[0]
        motion_len = float(num_frames - 1) / self.fps
        return motion_len