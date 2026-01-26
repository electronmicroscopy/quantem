"""
Visual regression test - screenshots the actual Show4DSTEM widget in Jupyter.

This is a manual test (not run in CI) because it:
- Requires real data files
- Takes ~1-2 minutes
- Needs JupyterLab + Chromium

Usage:
    playwright install chromium  # one-time setup
    python widget/tests/visual_check.py
"""

import subprocess
import time
import json
import tempfile
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright, Page

OUTPUT_DIR = Path(__file__).parent / "screenshots"
OUTPUT_DIR.mkdir(exist_ok=True)

# GPU/memory cleanup to prepend to each test
CLEANUP = """
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch, gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
"""

# Real data test cases
REAL_TESTS = [
    ("arina_george", CLEANUP + """
from quantem.core.io.file_readers import read_4dstem
from quantem.widget import Show4DSTEM
dataset = read_4dstem('/home/bobleesj/data/geroge/gold-10/gold_10_master.h5', file_type='arina')
widget = Show4DSTEM(dataset)
widget
"""),
    ("arina_ncem", CLEANUP + """
from quantem.core.io.file_readers import read_4dstem
from quantem.widget import Show4DSTEM
dataset = read_4dstem('/home/bobleesj/data/251115_ncem_arina_steph/lamella_2_002_master.h5', file_type='arina')
widget = Show4DSTEM(dataset)
widget
"""),
    ("rect_scan", CLEANUP + """
import h5py
from quantem.widget import Show4DSTEM
with h5py.File('/home/bobleesj/data/ptycho_MoS2_bin2.h5', 'r') as f:
    data = f['4DSTEM/datacube/data'][:]
print(f'Data shape: {data.shape}')
widget = Show4DSTEM(data)
widget
"""),
    ("legacy_h5", CLEANUP + """
from quantem.core.io.file_readers import read_emdfile_to_4dstem
from quantem.widget import Show4DSTEM
dataset = read_emdfile_to_4dstem('/home/bobleesj/data/ptycho_gold_data_2024.h5')
widget = Show4DSTEM(dataset)
widget
"""),
]


def create_notebook(path: Path, code: str):
    """Create a Jupyter notebook with the given code."""
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(path, 'w') as f:
        json.dump(notebook, f)


def run_cell(page: Page, notebook_name: str):
    """Execute notebook cell and wait for completion."""
    # Click tab to ensure focus
    try:
        page.click(f'.lm-TabBar-tab[data-id*="{notebook_name}"]', timeout=3000)
        page.wait_for_timeout(500)
    except Exception:
        pass

    # Click cell editor and run
    try:
        page.click('.jp-NotebookPanel:not(.lm-mod-hidden) .jp-Cell .cm-content', timeout=3000)
    except Exception:
        page.click('.jp-NotebookPanel:not(.lm-mod-hidden) .jp-Cell', timeout=3000)
    page.wait_for_timeout(300)
    page.keyboard.press('Shift+Enter')
    print("  Executing cell...")

    # Wait for idle
    try:
        page.wait_for_selector('.jp-NotebookPanel:not(.lm-mod-hidden) .jp-Notebook-ExecutionIndicator[data-status="idle"]', timeout=30000)
    except Exception:
        page.wait_for_timeout(10000)


def wait_for_widget(page: Page, timeout: int = 60000) -> bool:
    """Wait for widget canvas to render."""
    try:
        page.wait_for_selector('.jp-NotebookPanel:not(.lm-mod-hidden) .jp-OutputArea-output canvas', timeout=timeout)
        print("  Widget rendered")
        page.wait_for_timeout(2000)
        return True
    except Exception:
        print("  Warning: Widget not detected")
        return False


def run_tests(tests: list, port: int = 18888):
    """Run screenshot tests for given test cases."""
    temp_dir = Path(tempfile.mkdtemp())
    jupyter_proc = None

    try:
        print("=" * 60)
        print(f"Show4DSTEM Screenshot Tests ({len(tests)} tests)")
        print(f"Output: {OUTPUT_DIR}")
        print("=" * 60)

        print("\nStarting JupyterLab...")
        jupyter_proc = subprocess.Popen(
            ["jupyter", "lab", "--no-browser", f"--port={port}",
             f"--notebook-dir={temp_dir}", "--ServerApp.token=", "--ServerApp.password="],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        time.sleep(8)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            for name, code in tests:
                print(f"\n[TEST] {name}")
                create_notebook(temp_dir / f"{name}.ipynb", code.strip())

                context = browser.new_context(viewport={"width": 1600, "height": 1200})
                page = context.new_page()
                page.goto(f"http://localhost:{port}/lab/tree/{name}.ipynb")

                # Wait for JupyterLab ready
                try:
                    page.wait_for_selector('.jp-NotebookPanel:not(.lm-mod-hidden)', timeout=30000)
                    page.wait_for_selector('.jp-NotebookPanel:not(.lm-mod-hidden) .jp-Notebook-ExecutionIndicator[data-status="idle"]', timeout=30000)
                except Exception:
                    page.wait_for_timeout(8000)

                run_cell(page, name)
                widget_found = wait_for_widget(page)

                if not widget_found:
                    page.screenshot(path=str(OUTPUT_DIR / f"{name}_debug.png"), full_page=True)

                # Scroll to widget and capture
                page.evaluate("document.querySelector('.jp-OutputArea-output canvas')?.scrollIntoView({block: 'center'})")
                page.wait_for_timeout(1000)
                page.screenshot(path=str(OUTPUT_DIR / f"{name}.png"), full_page=True)
                print(f"  Saved: {name}.png")

                context.close()

            browser.close()

        print("\n" + "=" * 60)
        print(f"Done! Screenshots in: {OUTPUT_DIR}")
        print("=" * 60)

    finally:
        if jupyter_proc:
            jupyter_proc.terminate()
            try:
                jupyter_proc.wait(timeout=5)
            except Exception:
                jupyter_proc.kill()
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_tests(REAL_TESTS)
