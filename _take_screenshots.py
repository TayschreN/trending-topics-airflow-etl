import sys
import os
import time
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

# Start dash as a subprocess
dash_proc = subprocess.Popen(
    [sys.executable, os.path.join(ROOT, "dashboard", "app.py")],
    cwd=ROOT,
)
print("Dash server starting on http://localhost:8050")

time.sleep(8)

from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.edge.service import Service

options = Options()
options.add_argument("--headless")
options.add_argument("--window-size=1400,900")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

service = Service(EdgeChromiumDriverManager().install())
driver = webdriver.Edge(service=service, options=options)
wait = WebDriverWait(driver, 20)

screenshots_dir = os.path.join(ROOT, "dashboard", "screenshots")
os.makedirs(screenshots_dir, exist_ok=True)

try:
    print("Navigating to dashboard...")
    driver.get("http://localhost:8050")
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
    time.sleep(6)

    print("Taking full page screenshot...")
    driver.save_screenshot(os.path.join(screenshots_dir, "dashboard_full.png"))
    print("  done: dashboard_full.png")

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    driver.save_screenshot(os.path.join(screenshots_dir, "dashboard_bottom.png"))
    print("  done: dashboard_bottom.png")

    print("All screenshots captured successfully!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    driver.quit()
    dash_proc.terminate()
    dash_proc.wait()
