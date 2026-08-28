# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class BaseAudioProcessor:
    def __init__(self):
        pass

    def extract_features(self, wav, fs):
        pass

    def duration_to_frames(self, duration):
        """duration in s, estimated frames"""
        frame_rate = 10

        num_frames = duration * 1000 // frame_rate
        return num_frames
