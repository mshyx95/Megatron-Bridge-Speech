# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io

def bytes2wav(byte_array):
    """convert byte array to wave data"""
    byte_stream = io.BytesIO(byte_array)
    return sf.read(byte_stream)
