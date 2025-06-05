import os

ZP_INT8 = int(os.getenv("ZP_INT8", '0'))
SIGNED_KV = int(os.getenv("SIGNED_KV", '0'))
ZP_CLAMP = int(os.getenv("ZP_CLAMP", '1'))
SCALE_NO_UPCAST = int(os.getenv("SCALE_NO_UPCAST", '0'))