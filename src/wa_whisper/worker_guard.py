"""Exit an orphaned inference worker even while CUDA is processing a request."""
import os
import threading
import time


def watch_parent(parent_pid):
    def wait_for_parent():
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x00100000, False, parent_pid)
            if not handle:
                os._exit(1)
            try:
                while kernel.WaitForSingleObject(handle, 500) == 258:
                    pass
            finally:
                kernel.CloseHandle(handle)
        else:
            while os.getppid() == parent_pid:
                time.sleep(0.5)
        os._exit(1)
    threading.Thread(target=wait_for_parent, daemon=True, name="whisper-parent-guard").start()
