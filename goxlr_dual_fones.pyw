"""
GoXLR Dual Fones
----------------
Pega o áudio do "Broadcast Stream Mix" (GoXLR) e toca em DOIS fones ao mesmo
tempo, substituindo o "Escutar este dispositivo" do Windows (que só aceita 1).

Requisitos: Windows + Python 3.9+  ->  pip install sounddevice numpy
"""
import ctypes
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

APP_NAME = "GoXLR Dual Fones"
if getattr(sys, "frozen", False):  # rodando como .exe: salva em %APPDATA%
    BASE = Path(os.environ.get("APPDATA", str(Path.home()))) / "GoXLR Dual Fones"
    BASE.mkdir(parents=True, exist_ok=True)
else:
    BASE = Path(__file__).resolve().parent
CONFIG = BASE / "goxlr_dual_fones.json"
INPUT_HINT = "Broadcast Stream Mix"


# --------------------------------------------------------------------------
# Buffer circular (entrada -> saída), tolera pequenas diferenças de clock
# --------------------------------------------------------------------------
class Ring:
    def __init__(self, sr, channels=2, max_ms=200, keep_ms=80, prime_ms=40):
        self.ch = channels
        self.size = int(sr)  # 1 segundo de capacidade
        self.buf = np.zeros((self.size, channels), dtype=np.float32)
        self.r = 0
        self.w = 0
        self.count = 0
        self.max_lat = int(sr * max_ms / 1000)
        self.keep = int(sr * keep_ms / 1000)
        self.prime = int(sr * prime_ms / 1000)
        self.primed = False
        self.lock = threading.Lock()

    def write(self, data):
        n = len(data)
        with self.lock:
            if n >= self.size:
                data = data[-self.size:]
                n = self.size
            over = self.count + n - self.size
            if over > 0:  # estouro: descarta o mais antigo
                self.r = (self.r + over) % self.size
                self.count -= over
            first = min(n, self.size - self.w)
            self.buf[self.w:self.w + first] = data[:first]
            if n > first:
                self.buf[:n - first] = data[first:]
            self.w = (self.w + n) % self.size
            self.count += n

    def read(self, n):
        out = np.zeros((n, self.ch), dtype=np.float32)
        with self.lock:
            if self.count > self.max_lat:  # atrasou demais: pula pra frente
                drop = self.count - self.keep
                self.r = (self.r + drop) % self.size
                self.count -= drop
            if not self.primed:
                if self.count < self.prime:
                    return out
                self.primed = True
            m = min(n, self.count)
            first = min(m, self.size - self.r)
            out[:first] = self.buf[self.r:self.r + first]
            if m > first:
                out[first:m] = self.buf[:m - first]
            self.r = (self.r + m) % self.size
            self.count -= m
            if m < n:
                self.primed = False
        return out


# --------------------------------------------------------------------------
# Utilidades de dispositivo
# --------------------------------------------------------------------------
def wasapi_index(sd):
    for i, h in enumerate(sd.query_hostapis()):
        if "WASAPI" in h["name"]:
            return i
    raise RuntimeError("WASAPI não disponível (este programa é para Windows).")


def list_devices(sd):
    wi = wasapi_index(sd)
    ins, outs = [], []
    for d in sd.query_devices():
        if d["hostapi"] != wi:
            continue
        if d["max_input_channels"] > 0:
            ins.append(d["name"])
        if d["max_output_channels"] > 0:
            outs.append(d["name"])
    return ins, outs


def find_device(sd, wi, wanted, want_input, hint=None):
    key = "max_input_channels" if want_input else "max_output_channels"
    cands = [(i, d) for i, d in enumerate(sd.query_devices())
             if d["hostapi"] == wi and d[key] > 0]
    for i, d in cands:
        if d["name"] == wanted:
            return i, d
    if hint:
        for i, d in cands:
            if hint.lower() in d["name"].lower():
                return i, d
    return None, None


# --------------------------------------------------------------------------
# Worker: 1 processo por fone (independentes; um cair não afeta o outro)
# --------------------------------------------------------------------------
def worker(slot, in_name, out_name, volume, stop, status_q):
    import sounddevice as sd

    def say(msg):
        try:
            status_q.put((slot, msg))
        except Exception:
            pass

    first = True
    while not stop.is_set():
        if not first:  # re-escaneia dispositivos (fone Bluetooth que voltou)
            try:
                sd._terminate()
                sd._initialize()
            except Exception:
                pass
        first = False
        ins = outs = None
        try:
            wi = wasapi_index(sd)
            i_idx, i_dev = find_device(sd, wi, in_name, True, INPUT_HINT)
            o_idx, o_dev = find_device(sd, wi, out_name, False)
            if i_idx is None:
                say("Entrada não encontrada (GoXLR ligada?)")
                stop.wait(3)
                continue
            if o_idx is None:
                say("Fone não encontrado — ligue/conecte o fone")
                stop.wait(3)
                continue

            sr = int(i_dev["default_samplerate"])
            ch_in = max(1, min(2, i_dev["max_input_channels"]))
            ch_out = max(1, min(2, o_dev["max_output_channels"]))
            ring = Ring(sr, 2)
            last = [time.time()]
            try:
                extra = sd.WasapiSettings(auto_convert=True)
            except TypeError:
                extra = None

            def in_cb(indata, frames, t, status, ch_in=ch_in, ring=ring):
                if ch_in == 1:
                    indata = np.repeat(indata, 2, axis=1)
                ring.write(indata)

            def out_cb(outdata, frames, t, status, ch_out=ch_out, ring=ring, last=last):
                last[0] = time.time()
                data = ring.read(frames)
                if ch_out == 1:
                    data = data.mean(axis=1, keepdims=True)
                outdata[:] = np.clip(data * volume.value, -1.0, 1.0)

            ins = sd.InputStream(device=i_idx, channels=ch_in, samplerate=sr,
                                 dtype="float32", latency="low",
                                 extra_settings=extra, callback=in_cb)
            outs = sd.OutputStream(device=o_idx, channels=ch_out, samplerate=sr,
                                   dtype="float32", latency="high",
                                   extra_settings=extra, callback=out_cb)
            ins.start()
            outs.start()
            say("Ativo ✔")

            while not stop.is_set():
                stop.wait(0.5)
                if not (ins.active and outs.active):
                    break
                if time.time() - last[0] > 3:  # saída travou
                    break
            if not stop.is_set():
                say("Desconectou — reconectando…")
        except Exception as e:  # noqa: BLE001
            say(f"Erro: {str(e)[:80]}")
        finally:
            for s in (ins, outs):
                if s is not None:
                    try:
                        s.abort()
                        s.close()
                    except Exception:
                        pass
        if not stop.is_set():
            stop.wait(2)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def load_config():
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg):
    try:
        CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Bandeja, iniciar com o Windows, instância única
# --------------------------------------------------------------------------
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_MUTEX = None


def startup_command():
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --minimized'
    pyw = Path(sys.executable).with_name("pythonw.exe")
    exe = pyw if pyw.exists() else Path(sys.executable)
    return f'"{exe}" "{Path(__file__).resolve()}" --minimized'


def startup_enabled():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, APP_NAME)
        return True
    except Exception:
        return False


def set_startup(on):
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
        if on:
            winreg.SetValueEx(k, APP_NAME, 0, winreg.REG_SZ, startup_command())
        else:
            try:
                winreg.DeleteValue(k, APP_NAME)
            except FileNotFoundError:
                pass


def already_running():
    global _MUTEX
    try:
        k = ctypes.windll.kernel32
        _MUTEX = k.CreateMutexW(None, False, "GoXLRDualFones_SingleInstance")
        return k.GetLastError() == 183  # ERROR_ALREADY_EXISTS
    except Exception:
        return False


def make_icon_image():
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((2, 2, 62, 62), radius=14, fill=(30, 30, 36, 255))
    d.arc((12, 10, 52, 50), 180, 360, fill=(255, 255, 255, 255), width=4)
    d.ellipse((9, 26, 26, 48), fill=(83, 252, 24, 255))
    d.ellipse((38, 26, 55, 48), fill=(83, 252, 24, 255))
    return img


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
class Slot:
    def __init__(self, tk, cfg, n):
        self.name = tk.StringVar(value=cfg.get(f"out{n}", ""))
        self.enabled = tk.BooleanVar(value=cfg.get(f"on{n}", True))
        self.vol = tk.DoubleVar(value=cfg.get(f"vol{n}", 100))
        self.status = tk.StringVar(value="Parado")
        self.volume = mp.Value("d", self.vol.get() / 100.0, lock=False)
        self.proc = None
        self.stop = None


class App:
    def __init__(self, root, sd, minimized=False):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk, self.sd, self.root = tk, ttk, sd, root
        self.cfg = load_config()
        self.status_q = mp.Queue()
        self.running = False
        self.cmd_q = queue.Queue()
        self.notified = False
        self.tray = None
        self.startup = tk.BooleanVar(value=startup_enabled())
        self.slots = [Slot(tk, self.cfg, 1), Slot(tk, self.cfg, 2)]

        root.title(APP_NAME)
        root.resizable(False, False)
        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(root, padding=10)
        frm.grid()

        ins, outs = list_devices(sd)
        default_in = self.cfg.get("input") or next(
            (n for n in ins if INPUT_HINT.lower() in n.lower()), ins[0] if ins else "")
        self.input_var = tk.StringVar(value=default_in)

        ttk.Label(frm, text="Entrada (GoXLR):").grid(row=0, column=0, sticky="w", **pad)
        self.cb_in = ttk.Combobox(frm, textvariable=self.input_var, values=ins,
                                  width=58, state="readonly")
        self.cb_in.grid(row=0, column=1, columnspan=2, sticky="w", **pad)
        self.cb_in.bind("<<ComboboxSelected>>", lambda e: self.restart_all())

        self.cb_out = []
        for i, s in enumerate(self.slots):
            r = 1 + i * 2
            ttk.Label(frm, text=f"Fone {i + 1}:").grid(row=r, column=0, sticky="w", **pad)
            cb = ttk.Combobox(frm, textvariable=s.name, values=outs, width=58, state="readonly")
            cb.grid(row=r, column=1, sticky="w", **pad)
            cb.bind("<<ComboboxSelected>>", lambda e, i=i: self.restart_slot(i))
            self.cb_out.append(cb)
            ttk.Checkbutton(frm, text="Ativo", variable=s.enabled,
                            command=self.sync).grid(row=r, column=2, **pad)
            ttk.Label(frm, text="Volume:").grid(row=r + 1, column=0, sticky="e", **pad)
            ttk.Scale(frm, from_=0, to=100, variable=s.vol, length=300,
                      command=lambda v, s=s: self.set_vol(s)).grid(row=r + 1, column=1, sticky="w", **pad)
            ttk.Label(frm, textvariable=s.status, width=36).grid(row=r + 1, column=2, sticky="w", **pad)

        bar = ttk.Frame(frm)
        bar.grid(row=5, column=0, columnspan=3, pady=8)
        self.btn = ttk.Button(bar, text="▶  Ligar", width=16, command=self.toggle)
        self.btn.grid(row=0, column=0, padx=6)
        ttk.Button(bar, text="↻  Atualizar lista", command=self.refresh).grid(row=0, column=1, padx=6)
        self.auto = tk.BooleanVar(value=self.cfg.get("autostart", False))
        ttk.Checkbutton(bar, text="Ligar automaticamente ao abrir",
                        variable=self.auto).grid(row=0, column=2, padx=6)
        ttk.Checkbutton(bar, text="Iniciar com o Windows (já ligado, escondido na bandeja)",
                        variable=self.startup, command=self.on_startup).grid(
            row=1, column=0, columnspan=3, pady=(8, 0))

        tips = ("Dicas:\n"
                "• Desative \"Escutar este dispositivo\" do Broadcast Stream Mix no Windows\n"
                "  (senão você ouve em dobro).\n"
                "• Em fones Bluetooth escolha a opção \"Stereo\", não \"Hands-Free\".\n"
                "• Fone novo/religado que não aparece? Clique em \"Atualizar lista\".\n"
                "• O X da janela manda o programa para a bandeja (perto do relógio).\n"
                "  Para fechar de vez: ícone da bandeja → Sair.")
        ttk.Label(frm, text=tips, foreground="#555", justify="left").grid(
            row=6, column=0, columnspan=3, sticky="w", **pad)

        self.setup_tray()
        if startup_enabled():
            try:
                set_startup(True)  # atualiza o caminho caso o .exe tenha sido movido
            except Exception:
                pass
        root.protocol("WM_DELETE_WINDOW", self.hide)
        root.after(300, self.poll)
        if minimized:
            if self.tray is not None:
                root.withdraw()
            root.after(600, self.toggle)  # iniciado pelo Windows: já ativo
        elif self.auto.get():
            root.after(600, self.toggle)

    # ---- controle ----
    def set_vol(self, s):
        s.volume.value = max(0.0, min(1.0, s.vol.get() / 100.0))

    def toggle(self):
        self.running = not self.running
        self.btn.config(text="■  Desligar" if self.running else "▶  Ligar")
        self.sync()

    def sync(self):
        for i, s in enumerate(self.slots):
            want = self.running and s.enabled.get() and s.name.get()
            alive = s.proc is not None and s.proc.is_alive()
            if want and not alive:
                self.start_slot(i)
            elif not want and alive:
                self.stop_slot(i)
            elif not want:
                s.status.set("Parado" if self.running is False or not s.enabled.get() else "Escolha um fone")
        self.save()

    def start_slot(self, i):
        s = self.slots[i]
        s.stop = mp.Event()
        s.proc = mp.Process(target=worker, daemon=True,
                            args=(i, self.input_var.get(), s.name.get(),
                                  s.volume, s.stop, self.status_q))
        s.proc.start()
        s.status.set("Iniciando…")

    def stop_slot(self, i):
        s = self.slots[i]
        if s.proc is None:
            return
        s.stop.set()
        s.proc.join(2)
        if s.proc.is_alive():
            s.proc.terminate()
        s.proc = None
        s.status.set("Parado")

    def restart_slot(self, i):
        if self.slots[i].proc is not None:
            self.stop_slot(i)
        self.sync()

    def restart_all(self):
        for i in range(2):
            self.stop_slot(i)
        self.sync()

    def refresh(self):
        for i in range(2):
            self.stop_slot(i)
        try:
            self.sd._terminate()
            self.sd._initialize()
        except Exception:
            pass
        ins, outs = list_devices(self.sd)
        self.cb_in["values"] = ins
        for cb in self.cb_out:
            cb["values"] = outs
        self.sync()

    def poll(self):
        try:
            while True:
                slot, msg = self.status_q.get_nowait()
                if self.slots[slot].proc is not None:
                    self.slots[slot].status.set(msg)
        except Exception:
            pass
        try:
            while True:
                cmd = self.cmd_q.get_nowait()
                if cmd == "show":
                    self.show()
                elif cmd == "toggle":
                    self.toggle()
                elif cmd == "quit":
                    self.quit()
                    return
        except queue.Empty:
            pass
        self.root.after(300, self.poll)

    # ---- bandeja ----
    def setup_tray(self):
        try:
            import pystray
            menu = pystray.Menu(
                pystray.MenuItem("Abrir", lambda i, it: self.cmd_q.put("show"), default=True),
                pystray.MenuItem(lambda it: "Desligar" if self.running else "Ligar",
                                 lambda i, it: self.cmd_q.put("toggle")),
                pystray.MenuItem("Sair", lambda i, it: self.cmd_q.put("quit")),
            )
            self.tray = pystray.Icon(APP_NAME, make_icon_image(), APP_NAME, menu)
            self.tray.run_detached()
        except Exception:
            self.tray = None  # sem bandeja: o X fecha o programa

    def show(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def hide(self):
        if self.tray is None:
            self.quit()
            return
        self.root.withdraw()
        if not self.notified:
            self.notified = True
            try:
                self.tray.notify("Continua funcionando aqui na bandeja. "
                                 "Clique com o botão direito → Sair para fechar.", APP_NAME)
            except Exception:
                pass

    def on_startup(self):
        try:
            set_startup(self.startup.get())
        except Exception as e:  # noqa: BLE001
            self.startup.set(not self.startup.get())
            from tkinter import messagebox
            messagebox.showerror(APP_NAME, f"Não consegui alterar a inicialização com o Windows:\n{e}")
        self.save()

    def save(self):
        save_config({
            "input": self.input_var.get(),
            "out1": self.slots[0].name.get(), "out2": self.slots[1].name.get(),
            "on1": self.slots[0].enabled.get(), "on2": self.slots[1].enabled.get(),
            "vol1": round(self.slots[0].vol.get()), "vol2": round(self.slots[1].vol.get()),
            "autostart": self.auto.get(),
        })

    def quit(self):
        self.running = False
        for i in range(2):
            self.stop_slot(i)
        self.save()
        if self.tray is not None:
            try:
                self.tray.stop()
            except Exception:
                pass
        self.root.destroy()


def run_gui():
    import tkinter as tk
    from tkinter import messagebox
    minimized = "--minimized" in sys.argv
    if already_running():
        if not minimized:
            r = tk.Tk()
            r.withdraw()
            messagebox.showinfo(APP_NAME, "O programa já está aberto.\n"
                                "Procure o ícone na bandeja do Windows (perto do relógio).")
        return
    import sounddevice as sd
    root = tk.Tk()
    App(root, sd, minimized=minimized)
    root.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    run_gui()
