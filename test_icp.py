from utils.ICP import icp, icp2
from tf import Tf2D
import cv2

import matplotlib.pyplot as plt
import numpy as np

def polar2xy(data):
    return np.array([ [-r*np.sin(np.radians(theta)), r*np.cos(np.radians(theta))] for theta, r in data ])

def sample(data, npoints):
    n = data.shape[0]
    assert npoints <= n
    return np.take(data, np.linspace(0, n, npoints, endpoint=False, dtype=int), axis=0)

def run_icp(coord1, coord2, guess=None):

    coord1 = np.asarray(coord1)
    coord2 = np.asarray(coord2)

    T, e, I, iter = icp2(coord1, coord2, init_pose=guess.matrix, max_iterations=100, num_closest=300)

    error = np.mean(e[I])
    print(round(error, 2), iter)

    tf = Tf2D.from_matrix(T)
    #tf = guess

    #coord1_new = tf(coord1)

    print(tf)

    #plt.scatter(coord1_new[:, 0], coord1_new[:, 1], color="red", marker=".")
    #plt.scatter(coord2[:, 0], coord2[:, 1], color="blue", marker=".")

    #plt.show()
    return tf, error

def test():
    
    data1 = np.load("data1.npy")
    data2 = np.load("data2.npy")
    
    coord1 = polar2xy(data1)
    coord2 = polar2xy(data2)

    guess = Tf2D((-100, 0), np.radians(7))
    
    run_icp(coord1, coord2)

def main():
    
    from pirobot_lib.lidar import Lidar # type:ignore
    from time import sleep
    from occupancy_grid import OccupancyGrid
    
    l = Lidar(binary=None, addr="tcp://myraspberrypi.local:5556")
    l.start()
    
    sleep(1)
    
    last = None
    pose = Tf2D.identity() # X: avanti, Y: destra
    pose_list = np.array([])
    ptcloud = np.array([])
    scans = []
    
    grid = OccupancyGrid(30)
    
    plt.ion()
    fig, ax = plt.subplots()
    
    try:
        while True:
            
            #input("Press ENTER to add a data point: ")
            l.clear_input()
            ok, data = l.get_coord()
            
            if not ok: break
            
            data = np.asarray(data)
            
            if last is not None:
                tf, error = run_icp(data, ptcloud, guess=pose)
                #pose = tf @ pose
                pose = tf
                
                pose_list = np.concatenate((pose_list, [pose]))

                print("Pose:", pose)
                
                if error < 8.0:
                    ptcloud = np.vstack((ptcloud, pose(data)))
                    scans.append(data)
                    grid.update(data, pose)
            
            else:
                ptcloud = data
                pose_list = np.array([pose])
                scans.append(data)
                grid.update(data, pose)
            
            last = data
            
            #ax.clear()

            #ax.scatter(ptcloud[:, 0], ptcloud[:, 1], color="blue", marker=".")
            #ax.scatter([pose.tx for pose in pose_list], [pose.ty for pose in pose_list], color="red", marker=".")
        
            
            map_origin, map = grid.get_map()
            map_pose = grid.get_map_point(pose.translation)
            
            map = cv2.cvtColor(map, cv2.COLOR_GRAY2RGB)
            cv2.circle(map, map_pose[::-1], 3, (255, 0, 0), -1)
            
            ax.imshow(cv2.flip(map, 0), aspect="equal")#, cmap="plasma")
            
            plt.draw()
            plt.pause(0.1)
    
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")
    
    finally:
        #np.savez("poses.npz", pose_list)
        #np.savez("scans.npz", *scans)
        ...
        
if __name__ == "__main__":
    main()