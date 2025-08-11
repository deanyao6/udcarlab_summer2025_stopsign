import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/media/nvidia/AGXDavos/Github/dean_stopsign/src/py_pubsub/install/py_pubsub'
