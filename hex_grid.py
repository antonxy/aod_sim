import numpy as np

def hex_grid_positions(x_dist, tiles_x, tiles_y):
    """
          o           o

    o           o           o -
                              │
          o           o       │
                              │
    o           o           o -
                              │
          o           o       │tiles_y
                              │
    o           o           o -
                              │
          o           o       │
                              │
    o──────────►o           o -
       x_dist

    |───────────|───────────|
           tiles_x
    """
    off_y = np.sin(np.deg2rad(60)) * x_dist
    off_x = np.cos(np.deg2rad(60)) * x_dist

    P = np.zeros((tiles_x * tiles_y * 2, 2))

    i = 0;
    for tx in range(tiles_x):
       for ty in range(tiles_y):
            P[i, :] = np.array([tx * x_dist, ty * off_y * 2])
            P[i + 1, :] = P[i, :] + np.array([off_x, off_y])
            i = i + 2
    return P


def projection_hex_pattern_deg(step_size_deg, steps_x, steps_y, orientation_rad=0.0, aspect_ratio=1.0):
    y_tweak = aspect_ratio

    positions = hex_grid_positions(step_size_deg, steps_x, steps_y)

    # Center positions around 0
    positions = positions - np.mean(positions, axis=0)

    # Apply aspect ratio
    positions[:, 1] = positions[:, 1] * y_tweak

    num_steps = positions.shape[0]

    print(f"{num_steps} scan positions in hex grid")

    # Apply orientation
    c, s = np.cos(orientation_rad), np.sin(orientation_rad)
    R = np.array(((c, -s), (s, c)))
    positions = np.dot(positions, R.T)


    # Shift by 1/7 period
    shift_pos = hex_grid_positions(step_size_deg, 1, 2)
    shift_pos = shift_pos[3, :]

    shifts = shift_pos[np.newaxis, :] * (np.arange(7) / 7)[:, np.newaxis]

    positions_shifted = positions[np.newaxis, :, :] + shifts[:, np.newaxis, :]

    num_steps = num_steps * 7;

    print(f"hex grid shifted 7 times, {num_steps} scan positions total");

    return positions_shifted


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    P = projection_hex_pattern_deg(0.1, 10, 5, orientation_rad = 0.1)
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.set_aspect(1.0)
    for i in range(7):
        plt.scatter(P[i, :, 0], P[i, :, 1])
    plt.show()
