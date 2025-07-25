import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

# Usage: python animate_curvature_quantities.py <path_to_results_json>
def main():
    if len(sys.argv) < 2:
        print("Usage: python animate_curvature_quantities.py <path_to_results_json>")
        sys.exit(1)
    results_path = sys.argv[1]
    if not os.path.isfile(results_path):
        print(f"File not found: {results_path}")
        sys.exit(1)
    with open(results_path, 'r') as f:
        data = json.load(f)
    out_dir = os.path.dirname(results_path)

    # 1. Animate Ricci scalar vs. time
    einstein = data.get('einstein_analysis_per_timestep', None)
    if einstein and isinstance(einstein, list) and any(e for e in einstein if e):
        ricci = [e['ricci_scalar'] if e and 'ricci_scalar' in e else np.nan for e in einstein]
        timesteps = list(range(1, len(ricci)+1))
        fig, ax = plt.subplots()
        ax.set_xlim(1, len(ricci))
        ax.set_ylim(np.nanmin(ricci)-0.1, np.nanmax(ricci)+0.1)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Ricci Scalar')
        ax.set_title('Ricci Scalar vs. Time')
        line, = ax.plot([], [], 'o-', color='purple')
        def update(frame):
            line.set_data(timesteps[:frame+1], ricci[:frame+1])
            return line,
        ani = animation.FuncAnimation(fig, update, frames=len(ricci), blit=True, repeat=False)
        gif_path = os.path.join(out_dir, 'ricci_scalar_vs_time.gif')
        ani.save(gif_path, writer='pillow', fps=1)
        plt.close()
        print(f"Saved: {gif_path}")
    else:
        print("No einstein_analysis_per_timestep or Ricci scalar data found.")

    # 2. Animate emergent metric tensor (heatmap per timestep)
    if einstein and isinstance(einstein, list) and any(e for e in einstein if e and 'metric_tensor' in e):
        metric_tensors = [np.array(e['metric_tensor']) if e and 'metric_tensor' in e else None for e in einstein]
        fig, ax = plt.subplots()
        vmin = min(np.nanmin(m) for m in metric_tensors if m is not None)
        vmax = max(np.nanmax(m) for m in metric_tensors if m is not None)
        cax = ax.imshow(metric_tensors[0], vmin=vmin, vmax=vmax, cmap=cm.viridis)
        fig.colorbar(cax)
        ax.set_title('Emergent Metric Tensor')
        def update_metric(frame):
            ax.clear()
            m = metric_tensors[frame]
            im = ax.imshow(m, vmin=vmin, vmax=vmax, cmap=cm.viridis)
            ax.set_title(f'Metric Tensor (Timestep {frame+1})')
            for (i, j), val in np.ndenumerate(m):
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='w', fontsize=10)
            return [im]
        ani_metric = animation.FuncAnimation(fig, update_metric, frames=len(metric_tensors), blit=False, repeat=False)
        gif_path = os.path.join(out_dir, 'metric_tensor_animation.gif')
        ani_metric.save(gif_path, writer='pillow', fps=1)
        plt.close()
        print(f"Saved: {gif_path}")
    else:
        print("No metric tensor data found in einstein_analysis_per_timestep.")

    # 3. Animate Lorentzian embedding (3D scatter per timestep)
    lorentz = data.get('lorentzian_embedding', None)
    num_qubits = data['spec']['num_qubits'] if 'spec' in data and 'num_qubits' in data['spec'] else 7
    timesteps = data['spec']['timesteps'] if 'spec' in data and 'timesteps' in data['spec'] else 4
    if lorentz and isinstance(lorentz, list) and len(lorentz) == num_qubits * timesteps:
        lorentz = np.array(lorentz)
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        def update_lorentz(frame):
            ax.clear()
            start = frame * num_qubits
            end = (frame+1) * num_qubits
            coords = lorentz[start:end]
            ax.scatter(coords[:,0], coords[:,1], coords[:,2], c='b', s=60)
            for idx, (x, y, z) in enumerate(coords):
                ax.text(x, y, z, str(idx), fontsize=10)
            ax.set_title(f'Lorentzian Embedding (Timestep {frame+1})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(np.nanmin(lorentz[:,0]), np.nanmax(lorentz[:,0]))
            ax.set_ylim(np.nanmin(lorentz[:,1]), np.nanmax(lorentz[:,1]))
            ax.set_zlim(np.nanmin(lorentz[:,2]), np.nanmax(lorentz[:,2]))
            return []
        ani_lorentz = animation.FuncAnimation(fig, update_lorentz, frames=timesteps, blit=False, repeat=False)
        gif_path = os.path.join(out_dir, 'lorentzian_embedding_animation.gif')
        ani_lorentz.save(gif_path, writer='pillow', fps=1)
        plt.close()
        print(f"Saved: {gif_path}")
    else:
        print("No Lorentzian embedding data found or shape mismatch.")

if __name__ == '__main__':
    main() 