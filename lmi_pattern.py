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

def line_lmi_pattern_deg(steps, multiscan, grating_dot_distances, distance_between_gratings, orientation_rad=0.0):

    patterns = []
    for grating_orientation_deg, grating_dot_distance in zip([0, 120, 240], grating_dot_distances):
        # multiscan
        positions = np.linspace(0, grating_dot_distance, multiscan, endpoint=False)

        # Center positions around grating
        grating_y_offset = 0.5*distance_between_gratings / np.sin(np.deg2rad(60))
        positions = positions - np.mean(positions) + grating_y_offset

        # Shift
        shift_positions = np.linspace(0, grating_dot_distance / multiscan, steps, endpoint=False)
        positions_shifted = positions[np.newaxis, :] + shift_positions[:, np.newaxis]

        # Add x coordinate (= 0)
        positions_with_x = np.zeros(positions_shifted.shape + (2, ))
        positions_with_x[:, :, 1] = positions_shifted

        # Apply orientation
        orientation_rad_rot = orientation_rad + np.deg2rad(grating_orientation_deg)
        c, s = np.cos(orientation_rad_rot), np.sin(orientation_rad_rot)
        R = np.array(((c, -s), (s, c)))
        positions_rotated = np.dot(positions_with_x, R.T)
        patterns.append(positions_rotated)

    return np.concatenate(patterns, axis=0)
