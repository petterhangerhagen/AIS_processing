import numpy as np
import matplotlib.pyplot as plt


def rotate(vec, ang):
    """
    :param vec: 2D vector
    :param ang: angle in radians
    :return: input vector rotated by the input angle
    """
    # r_mat = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    r_mat = np.array([[np.sin(ang), np.cos(ang)], [np.cos(ang), -np.sin(ang)]])
    rot_vec = np.dot(r_mat, vec)
    return rot_vec

vessel = [-4.76581838, -31.64542414, -2.71490918, -1.7297862, -0.05288967]
obst = [8.00633926, -47.60644207, 1.79709, 2.29542542, -0.52849236]

vessel_ang_vec = [np.sin(vessel[2]), np.cos(vessel[2])]
obst_ang_vec = [np.sin(obst[2]), np.cos(obst[2])]

vs = rotate(vessel[3:5], vessel[2])

phi_min = 112.5
idk = [-400,-300]
print(f"norm idk: {np.linalg.norm(idk)}")

los = np.empty(2)
los[0] = obst[0] - vessel[0]
los[1] = obst[1] - vessel[1]
los = los / np.linalg.norm(los)
print(np.dot(vs, los))
print(np.cos(np.deg2rad(phi_min))*np.linalg.norm(vs))

# if np.dot(vs, los) < np.cos(self.phi_OT_min) * np.linalg.norm(vs):


fig, ax = plt.subplots(figsize=(11, 7.166666))
ax.plot(vessel[0], vessel[1], color="blue", marker='o', zorder=2)
# ax.plot([0, vessel_ang_vec[0]], [0, vessel_ang_vec[1]], color="blue", zorder=2)
ax.plot([vessel[0], vessel[0] + vessel_ang_vec[0]], [vessel[1], vessel[1] + vessel_ang_vec[1]], color="blue", zorder=2)
ax.plot([vessel[0], vessel[0] + vessel[3]], [vessel[1], vessel[1] + vessel[4]], color="green", zorder=2)

ax.plot([vessel[0], vessel[0] + los[0]], [vessel[1], vessel[1] + los[1]], color="black", zorder=2)
ax.plot([vessel[0], vessel[0] + np.sin(np.deg2rad(phi_min))], [vessel[1], vessel[1] + np.cos(np.deg2rad(phi_min))], color="yellow", zorder=2)

ax.plot(obst[0], obst[1], color="red", marker='o', zorder=2)
ax.plot([obst[0], obst[0] + obst_ang_vec[0]], [obst[1], obst[1] + obst_ang_vec[1]], color="red",linewidth=2, zorder=2)
ax.plot([obst[0], obst[0] + obst[3]], [obst[1], obst[1] + obst[4]], color="green", zorder=2)


# ax.plot([0, obst_ang_vec[0]], [0, obst_ang_vec[1]], color="red", zorder=2)
# ax.scatter(vessel[0], vessel[1], color="blue", marker='o', zorder=2)
# ax.scatter(obst[0], obst[1], color="red", marker='o', zorder=2)
plt.show()