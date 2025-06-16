import sys
import traceback
from src.AWSFactory import (
    ScaledEmergentSpacetime, AdSGeometryAnalyzer, AdSGeometryAnalyzer3D,
    AdSGeometryAnalyzer6Q, EmergentSpacetime, page_curve_demo, page_curve_mi_demo
)
from braket.devices import LocalSimulator

logfile = open("awsfactory_output.log", "w")

def log(msg):
    print(msg)
    logfile.write(msg + "\n")

try:
    log("Running ScaledEmergentSpacetime...")
    ses = ScaledEmergentSpacetime(device=LocalSimulator())
    ses.run_entropy_area_experiment()
    log("ScaledEmergentSpacetime done.\n")
except Exception as e:
    log("Error in ScaledEmergentSpacetime:")
    log(traceback.format_exc())

try:
    log("Running AdSGeometryAnalyzer...")
    aga = AdSGeometryAnalyzer(device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1")
    aga.run_experiment()
    log("AdSGeometryAnalyzer done.\n")
except Exception as e:
    log("Error in AdSGeometryAnalyzer:")
    log(traceback.format_exc())

try:
    log("Running AdSGeometryAnalyzer3D...")
    ag3d = AdSGeometryAnalyzer3D(use_local=True)
    ag3d.run()
    log("AdSGeometryAnalyzer3D done.\n")
except Exception as e:
    log("Error in AdSGeometryAnalyzer3D:")
    log(traceback.format_exc())

try:
    log("Running AdSGeometryAnalyzer6Q...")
    ag6q = AdSGeometryAnalyzer6Q(n_qubits=6, timesteps=5, mode="flat")
    ag6q.run()
    log("AdSGeometryAnalyzer6Q done.\n")
except Exception as e:
    log("Error in AdSGeometryAnalyzer6Q:")
    log(traceback.format_exc())

try:
    log("Running EmergentSpacetime...")
    es = EmergentSpacetime(device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1")
    es.run()
    log("EmergentSpacetime done.\n")
except Exception as e:
    log("Error in EmergentSpacetime:")
    log(traceback.format_exc())

try:
    log("Running page_curve_demo...")
    page_curve_demo()
    log("page_curve_demo done.\n")
except Exception as e:
    log("Error in page_curve_demo:")
    log(traceback.format_exc())

try:
    log("Running page_curve_mi_demo...")
    page_curve_mi_demo()
    log("page_curve_mi_demo done.\n")
except Exception as e:
    log("Error in page_curve_mi_demo:")
    log(traceback.format_exc())

logfile.close() 