import gc
import threading
import time
import torch

from deepseek_vl import load_model


class ModelManager:
    def __init__(self, timeout=600):
        self.model = None
        self.last_access = None
        self.timeout = timeout  # seconds
        self.lock = threading.Lock()
        self._start_monitor()

    def _start_monitor(self):
        def monitor():
            while True:
                time.sleep(30)  # check every 30 seconds
                with self.lock:
                    if self.model and time.time() - self.last_access > self.timeout:
                        print("[ModelManager] Unloading model due to inactivity.")
                        self._unload_model()

        t = threading.Thread(target=monitor, daemon=True)
        t.start()

    def _unload_model(self):
        self.model = None
        self.last_access = None
        torch.cuda.empty_cache()
        gc.collect()

    def get_model(self):
        with self.lock:
            if self.model is None:
                print("[ModelManager] Loading model...")
                self.model = load_model()
            self.last_access = time.time()
            return self.model

    def force_unload(self):
        with self.lock:
            print("[ModelManager] Force unloading model.")
            self._unload_model()
