# song2vec/core.py

from .submodule.c_extension.song2vec_c import Song2Vec

# If you have a pure Python fallback, you can conditionally import
# try:
#     from .submodule.c_extension.song2vec_c import Song2Vec
# except ImportError:
#     from .submodule.song2vec import Song2Vec
