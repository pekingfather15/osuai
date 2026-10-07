"""Where replays and maps come from.

Every imported replay is stored as data/replays-<source>/<id>.osr with a <id>.meta file
{"map": "<path of its .osu file, relative to data/>"}. Only the .osu files of a
beatmapset are kept, never the music or the images.
"""
