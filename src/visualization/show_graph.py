import matplotlib.pyplot as plt

# === Your Real Data from the Test ===
charge_levels = [
    1.57, 1.8055, 2.07632, 2.38777, 2.74594, 3.15783, 3.63151, 4.17623, 4.80267, 5.52307,
    6.35153, 7.30425, 8.39989, 9.65988, 11.10886, 12.77519, 14.69146, 16.89518, 19.42946, 22.34388,
    25.69546, 29.54978, 33.98225, 39.07959, 44.94153, 51.68276, 59.43517, 68.35044, 78.60301, 90.39346
]

prob_heads = [
    0.46875, 0.4375, 0.46875, 0.5, 0.46875, 0.5, 0.5, 0.40625, 0.5625, 0.4375,
    0.5625, 0.4375, 0.5, 0.4375, 0.46875, 0.53125, 0.40625, 0.53125, 0.4375, 0.5625,
    0.5, 0.34375, 0.4375, 0.40625, 0.53125, 0.5, 0.59375, 0.59375, 0.40625, 0.5
]# === Plotting ===
plt.figure(figsize=(10, 6))
plt.plot(charge_levels, prob_heads, 'o-', color='blue', label='P(Heads)')
plt.axhline(0.5, color='gray', linestyle='--', label='Fair Coin (0.5)')

# === Labels & Titles ===
plt.title('Charge Injection vs. Probability of Heads (QRNG Test)')
plt.xlabel('Charge Injection Level')
plt.ylabel('Probability of Heads')
plt.legend()
plt.grid(True)

# === Show Plot ===
plt.show()
