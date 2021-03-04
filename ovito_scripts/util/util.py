#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions that are useful for more than one postprocessing script"""

__author__ = "Wolfram Georg Nöhring"
__email__ = "wolfram.noehring@imtek.uni-freiburg.de"

xyz_to_num = {"x": 0, "y":1, "z": 2, "norm": slice(0, 3), "xy": slice(0, 2), "xz": slice(0, 3, 2), "yz": slice(1, 3)}
xyz_to_num["yx"] = xyz_to_num["xy"]
xyz_to_num["zx"] = xyz_to_num["xz"]
xyz_to_num["zy"] = xyz_to_num["yz"]

def parse_frame_range(string):
    """Parse string specifying frame range

    Parameters
    ----------
    string : string
        Range specification. Valid formats are `start`, `start:end`, and
        `start:end:increment`. In the first case, the list contains at most one
        frame. A dash followed by a comma-separated list may be appended;
        frames in this list will be ignored. E.g. `0:10:2-2,4` gives `0,6,8,10`.

    Returns
    -------
    frames : set

    Examples
    --------
    >>> parse_frame_range("0")
    [0]
    >>> parse_frame_range("0:10")
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> parse_frame_range("0:10:2")
    [0, 2, 4, 6, 8, 10]
    >>> parse_frame_range("0:10:2-2,4")
    [0, 6, 8, 10]
    """
    components = string.split("-")
    words = components[0].split(":")
    if len(words) == 1:  # one frame only
        number = int(words[0])
        frames = set()
        frames.add(number)
    elif len(words) == 2:  # start and end
        start = int(words[0])
        end = int(words[1]) + 1
        frames = set(range(start, end))
    elif len(words) == 3:
        start = int(words[0])
        end = int(words[1]) + 1
        increment = int(words[2])
        frames = set(range(start, end, increment))
    else:
        raise ValueError(f"could not interpret {components[0]} as a number range")
    if len(components) > 1:
        words = components[1].split(",")
        numbers = [int(i) for i in words]
        skipped_frames = set(numbers)
    else:
        skipped_frames = set()
    frames = list(frames.difference(skipped_frames))
    frames.sort()
    return frames
