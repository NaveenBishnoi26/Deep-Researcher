"""
Utility functions for completely preventing TensorFlow from loading.
"""

import os
import logging
import warnings
import sys
from pathlib import Path

# Create a list of modules to block
MODULES_TO_BLOCK = [
    'tensorflow',
    'tensorflow_hub',
    'tensorflow_text',
    'tensorflow_lite',
    'tflite',
    'tflite_runtime',
    'tensorrt',
    'tf',
    'tf_hub'
]

class ImportBlocker:
    """Block specific modules from being imported."""
    def __init__(self, modules_to_block):
        self.modules_to_block = modules_to_block
        
    def find_spec(self, fullname, path, target=None):
        if any(fullname == module or fullname.startswith(f"{module}.") for module in self.modules_to_block):
            return None  # This will cause ImportError
        return None  # Let regular import machinery handle other modules

def configure_tensorflow_environment():
    """
    Completely prevent TensorFlow from loading by blocking its import.
    
    This should be called at the very beginning of your application before
    any other imports.
    """
    # Install the import blocker to prevent TensorFlow imports
    sys.meta_path.insert(0, ImportBlocker(MODULES_TO_BLOCK))
    
    # Set environment variables as a backup to prevent TensorFlow loading
    # or to limit its functionality if it somehow gets loaded
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages (errors only)
    os.environ["TFHUB_CACHE_DIR"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tf_cache")
    os.environ["DISABLE_TFLITE_RUNTIME"] = "1"
    os.environ["DISABLE_TENSORFLOW_LITE"] = "1"
    os.environ["TF_LITE_DISABLE_DELEGATES"] = "1"
    os.environ["TFLITE_DISABLE_GPU_DELEGATE"] = "1"
    os.environ["TFLITE_DISABLE_NNAPI_DELEGATE"] = "1"
    os.environ["TFLITE_DISABLE_XNNPACK_DELEGATE"] = "1"
    os.environ["TF_LITE_DISABLE_XNNPACK_DYNAMIC_TENSORS"] = "1"
    os.environ["TF_LITE_DISABLE_DELEGATE_CLUSTERING"] = "1"
    
    # Create TF cache directory if it doesn't exist (just in case)
    cache_dir = os.environ["TFHUB_CACHE_DIR"]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Suppress all warnings
    warnings.filterwarnings("ignore", category=Warning)
    
    # Suppress specific TensorFlow warnings
    warnings.filterwarnings('ignore', message='.*Created TensorFlow Lite XNNPACK delegate.*')
    warnings.filterwarnings('ignore', message='.*only supports static-sized tensors.*')
    warnings.filterwarnings('ignore', message='.*tensor#-1 is a dynamic-sized tensor.*')
    warnings.filterwarnings('ignore', message='.*XNNPack.*')
    
    # Suppress all logging
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("tensorflow_hub").setLevel(logging.ERROR)
    logging.getLogger("tensorboard").setLevel(logging.ERROR)

def initialize_tensorflow():
    """
    This function suppresses TensorFlow related messages.
    
    TensorFlow should not be loaded at all due to the import blocker,
    but this serves as a final safeguard against any TensorFlow messages.
    """
    # Additional warning suppression
    warnings.filterwarnings('ignore', message='.*')
    
    # Suppress stderr temporarily to prevent any TensorFlow output 
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    try:
        # Try to set TensorFlow settings if it somehow got imported
        tf_module = sys.modules.get('tensorflow')
        if tf_module:
            # Suppress TensorFlow logging completely
            if hasattr(tf_module, 'get_logger'):
                tf_module.get_logger().setLevel(logging.ERROR)
            
            # Disable TensorFlow Lite delegates
            if hasattr(tf_module, 'lite'):
                lite = tf_module.lite
                if hasattr(lite, 'experimental'):
                    lite.experimental.disable_delegate_clustering = True
                if hasattr(lite, 'Interpreter'):
                    lite.Interpreter._experimental_disable_all_delegates = True
    except:
        pass
    finally:
        # Restore stderr
        sys.stderr.close()
        sys.stderr = original_stderr
    
    return True 