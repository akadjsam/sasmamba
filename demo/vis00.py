import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D




#
def show_pose3D_36M(sample, output_dir='./', sample_idx=0):
    """
    Visualize and save 3D human poses.

    Args:
        sample (torch.Tensor): Input tensor of shape [B, T, 17, 3], where:
                               B: batch size,
                               T: sequence length,
                               17: number of keypoints,
                               3: (x, y, z) coordinates.
        output_dir (str): Directory to save the visualized 3D poses.
        sample_idx (int): Index of the sample in the batch to visualize.
    """
    sample = sample.cpu().numpy()
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = "#3498db"  # Left-side joints (blue)
    rcolor = "#e74c3c"  # Right-side joints (red)
    save_dir = os.path.join(output_dir, f'3D_action_{sample_idx}')
    os.makedirs(save_dir, exist_ok=True)

    kps = sample[0]  # Shape: [T, 17, 3]
    # rotation and translation
    # rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    # rot = np.array(rot, dtype='float32')
    # action = camera_to_world(action, R=rot, t=0)
    # action[:, 2] -= np.min(action[:, 2])
    # max_value = np.max(action)
    # action /= max_value


    # kps shape : [17, 3]
    save_path = os.path.join(save_dir, f'{i:04d}_3D.png')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for j, c in enumerate(connections):
        start = kps[c[0]]
        end = kps[c[1]]
        # Draw connections between joints
        ax.plot(
            [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
            c=lcolor if LR[j] else rcolor, linewidth=2
        )

    # Draw individual joints
    ax.scatter(kps[:, 0], kps[:, 1], kps[:, 2], c='black', s=20)

    # Set axis labels and view

    ax.view_init(elev=102, azim=150, roll=60)  # Adjust the view angle

    # Set limits to maintain a uniform scale
    max_range = np.array([kps[:, 0].max() - kps[:, 0].min(),
                          kps[:, 1].max() - kps[:, 1].min(),
                          kps[:, 2].max() - kps[:, 2].min()]).max() / 2.0
    mid_x = (kps[:, 0].max() + kps[:, 0].min()) * 0.5
    mid_y = (kps[:, 1].max() + kps[:, 1].min()) * 0.5
    mid_z = (kps[:, 2].max() + kps[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


    # Get rid of the panes (actually, make them white)
    # Set the pane color to white
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Set the X pane to white
    ax.yaxis.set_pane_color((0.4, 0.4, 0.4, 0.4))  # Set the Y pane to white
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Set the Z pane to white if needed

    # Hide axis lines (completely remove them)
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Transparent X-axis line
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Transparent Y-axis line
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Transparent Z-axis line

    # Hide gridlines if they exist
    ax.grid(False)

    # Remove axes by hiding the ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Optionally, remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    plt.show()

sample = np.load('3D.npy')
show_pose3D_36M(sample, output_dir='./', sample_idx=0)