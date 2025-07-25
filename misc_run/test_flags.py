#!/usr/bin/env python3

import argparse

def test_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_qubits", type=int, default=7, help="Number of qubits")
    p.add_argument("--shots", type=int, default=1000, help="Number of shots")
    p.add_argument("--curvature", type=float, default=1.0, help="Curvature")
    p.add_argument("--timesteps", type=int, default=1, help="Number of timesteps")
    p.add_argument("--device", type=str, default="simulator", help="Device")
    p.add_argument("--solve_regge", action="store_true", help="Enable Regge calculus solver")
    p.add_argument("--fast", action="store_true", help="Fast mode: skip expensive computations")
    p.add_argument("--lorentzian", action="store_true", help="Enable Lorentzian mode")
    p.add_argument("--alpha", type=float, default=0.1, help="Alpha parameter")
    p.add_argument("--edge_floor", type=float, default=0.001, help="Edge floor")
    
    # Test the exact command line arguments you're using
    test_args = [
        "--num_qubits", "7", 
        "--shots", "10", 
        "--curvature", "12", 
        "--timesteps", "2", 
        "--device", "simulator", 
        "--solve_regge", 
        "--lorentzian", 
        "--alpha", "0.025", 
        "--edge_floor", "0.05"
    ]
    
    args = p.parse_args(test_args)
    
    print("🔍 DEBUG: Flag parsing test")
    print(f"🔍 DEBUG: solve_regge = {args.solve_regge}")
    print(f"🔍 DEBUG: fast = {args.fast}")
    print(f"🔍 DEBUG: lorentzian = {args.lorentzian}")
    print(f"🔍 DEBUG: timesteps = {args.timesteps}")
    print(f"🔍 DEBUG: Condition check: {args.solve_regge and not args.fast}")
    
    if args.solve_regge and not args.fast:
        print("✅ Regge solver should run!")
    else:
        print("❌ Regge solver will NOT run!")
        if not args.solve_regge:
            print("   - solve_regge is False")
        if args.fast:
            print("   - fast is True")

if __name__ == "__main__":
    test_args() 