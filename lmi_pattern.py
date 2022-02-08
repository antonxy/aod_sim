import numpy as np

def lmi_pattern_deg(steps_x, steps_y, multiscan_x, multiscan_y, distance_x, distance_y, orientation_rad=0.0):

    # multiscan
    positions_x = np.linspace(0, distance_x, multiscan_x, endpoint=False)
    positions_y = np.linspace(0, distance_y, multiscan_y, endpoint=False)

    xx, yy = np.meshgrid(positions_x, positions_y)
    positions = np.stack((xx, yy), axis=-1)
    positions = positions.reshape((-1, 2))

    # Center positions around 0
    positions = positions - np.mean(positions, axis=0)

    # Shift
    shift_positions_x = np.linspace(0, distance_x / multiscan_x, steps_x, endpoint=False)
    shift_positions_y = np.linspace(0, distance_y / multiscan_y, steps_y, endpoint=False)

    xx, yy = np.meshgrid(shift_positions_x, shift_positions_y)
    shift_positions = np.stack((xx, yy), axis=-1)
    shift_positions = shift_positions.reshape((-1, 2))

    positions_shifted = positions[np.newaxis, :, :] + shift_positions[:, np.newaxis, :]

    # Apply orientation
    c, s = np.cos(orientation_rad), np.sin(orientation_rad)
    R = np.array(((c, -s), (s, c)))
    positions_shifted = np.dot(positions_shifted, R.T)

    return positions_shifted
