"""
Q-DROP CHRONOS-7: Temporal Dilation CLI
Calculate how time slows down when you travel near the speed of light.
Based on Einstein's Special Relativity.
"""
import math

C = 299792458 # Speed of light in m/s

def calculate_dilation():
    print("=======================================")
    print("⏳ CHRONOS-7 TEMPORAL DILATION ENGINE ⏳")
    print("=======================================")
    print("Developed by Q-Drop Labs.\n")
    
    try:
        v_fraction = float(input("Enter your speed as a fraction of light speed (e.g. 0.99 for 99%): "))
    except ValueError:
        print("Error: Invalid input.")
        return

    if v_fraction >= 1.0:
        print("\n⚠️ CRITICAL: Speed exceeds light speed. Causality violation detected. Interstellar patrol dispatched.")
        return
    elif v_fraction < 0:
        print("\nError: Speed cannot be negative in this universe.")
        return

    v = v_fraction * C
    
    # Lorentz Factor formula: γ = 1 / √(1 - v²/c²)
    gamma = 1 / math.sqrt(1 - (v**2 / C**2))
    
    print(f"\n🚀 Velocity: {v_fraction * 100}% of light speed.")
    print(f"📐 Lorentz Factor (γ): {gamma:.4f}")
    print(f"⏱️  Time Dilation: 1 hour for you equals {gamma:.2f} hours on Earth.")
    
    if v_fraction > 0.9:
        print("\nWARNING: Approaching relativistic speeds. Mass increase detected.")
    
    print("\n-- Want to simulate black holes and warp drives? --")
    print("Access the full 100-project source code: https://github.com/aminechabane/Q-Drop-Marketplace")

if __name__ == "__main__":
    calculate_dilation()
