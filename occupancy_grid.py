import numpy as np
from numba import jit
from tf import Tf2D
import cv2

# Questo è il cuore della tua logica, ottimizzato per Numba.
# Accetta solo tipi semplici (array, int, float).
@jit(nopython=True)
def _numba_update_core(log_odds_map, scan, origin, resolution, max_size, l_occ, l_free, get_cells_on_line_jit):
    """
    Funzione core di aggiornamento della griglia di occupazione.
    Progettata per essere chiamata da Numba in modalità nopython.
    """
    # Ritorna i limiti aggiornati
    lim_x = np.array([max_size, 0])
    lim_y = np.array([max_size, 0])

    # Converti le coordinate del punto di origine in indici di griglia
    origin_grid_x = int(origin[0] / resolution + max_size / 2)
    origin_grid_y = int(origin[1] / resolution + max_size / 2)

    # Assicurati che l'origine sia nei limiti, altrimenti non possiamo fare il ray-casting
    if not (0 <= origin_grid_x < max_size) and (0 <= origin_grid_y < max_size):
        return lim_x, lim_y

    # Loop su ogni punto nello scan
    for i in range(scan.shape[0]):
        point = scan[i]
        
        hit_grid_x = int(point[0] / resolution + max_size / 2)
        hit_grid_y = int(point[1] / resolution + max_size / 2)
            
        # Mark free cells using a ray-casting algorithm
        cells_on_path = get_cells_on_line_jit(origin_grid_x, origin_grid_y, hit_grid_x, hit_grid_y)
        
        for j in range(len(cells_on_path)-1):
            cx, cy = cells_on_path[j]
            #if (cx, cy) != (hit_grid_x, hit_grid_y):
            if 0 <= cx < max_size and 0 <= cy < max_size:
                log_odds_map[cx, cy] += l_free
        
        # Mark occupied cell
        if (0 <= hit_grid_x < max_size) and (0 <= hit_grid_y < max_size):
            log_odds_map[hit_grid_x, hit_grid_y] += l_occ
            
            # Aggiorna i limiti
            if hit_grid_x < lim_x[0]: lim_x[0] = hit_grid_x
            if hit_grid_x + 1 > lim_x[1]: lim_x[1] = hit_grid_x + 1
            if hit_grid_y < lim_y[0]: lim_y[0] = hit_grid_y
            if hit_grid_y + 1 > lim_y[1]: lim_y[1] = hit_grid_y + 1
    
    return lim_x, lim_y


# Funzione Bresenham, già compatibile con Numba.
@jit(nopython=True)
def get_cells_on_line(x0, y0, x1, y1):
    points = []
    is_steep = abs(y1 - y0) > abs(x1 - x0)
    
    if is_steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        
    swapped = False
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        swapped = True
    
    dx = x1 - x0
    dy = abs(y1 - y0)
    
    error = dx / 2.0
    ystep = 1 if y0 < y1 else -1
    y = y0
    
    for x in range(x0, x1 + 1):
        if is_steep:
            points.append((y, x))
        else:
            points.append((x, y))
            
        error -= dy
        if error < 0:
            y += ystep
            error += dx
    
    if swapped:
        points.reverse()
        
    return points

L_occ = np.log(0.9 / 0.1)
L_free = np.log(0.1 / 0.9)
L_prior = np.log(0.5 / 0.5)

def create_slice(lim, margin, max_size):
    start = lim[0] - margin
    stop  = lim[1] + margin
    
    if start<0:
        start=0
    if stop>=max_size:
        stop = max_size-1
    
    return slice(start, stop, 1)

class OccupancyGrid:
    
    def __init__(self, px_per_unit=10.0, max_size=10000, margin=20):
        
        self.resolution = float(px_per_unit)
        self.max_size = int(max_size)
        self.margin = int(margin)
        
        self.log_odds_map = np.full((max_size, max_size), L_prior)
        self.lim_x = [max_size//2, max_size//2 + 1]
        self.lim_y = [max_size//2, max_size//2 + 1]
        
        self.sl_x = create_slice(self.lim_x, margin, max_size)
        self.sl_y = create_slice(self.lim_y, margin, max_size)
        
        self.map = None
        self.map_origin = None
    
    # Questo è il wrapper di alto livello. Non è necessario decorarlo con Numba.
    # Gestisce tipi complessi come `pose` e prepara i dati per la funzione JIT.
    def update(self, scan: np.ndarray, pose: Tf2D | None = None):
        
        # Gestione del caso None prima di passare a Numba.
        if pose is None:
            # Se la posa non è fornita, assumiamo che lo scan sia già in coordinate del mondo
            origin = np.array([0.0, 0.0])
        else:
            # Altrimenti, trasforma i punti dello scan
            scan = pose.apply_npoints(scan)
            origin = pose.translation

        self.map = None
        
        # Chiama la funzione JIT con tipi compatibili
        lim_x_new, lim_y_new = _numba_update_core(
            self.log_odds_map, 
            np.asarray(scan), 
            origin, 
            self.resolution, 
            self.max_size, 
            L_occ, 
            L_free,
            get_cells_on_line
        )

        # Aggiorna i limiti della mappa
        if lim_x_new[0] < self.lim_x[0]: self.lim_x[0] = lim_x_new[0]
        if lim_x_new[1] > self.lim_x[1]: self.lim_x[1] = lim_x_new[1]
        
        if (self.sl_x.start > self.lim_x[0]) or (self.sl_x.stop < self.lim_x[1]):
            self.sl_x = create_slice(self.lim_x, self.margin, self.max_size)
        
        if lim_y_new[0] < self.lim_y[0]: self.lim_y[0] = lim_y_new[0]
        if lim_y_new[1] > self.lim_y[1]: self.lim_y[1] = lim_y_new[1]
        
        if (self.sl_y.start > self.lim_y[0]) or (self.sl_y.stop < self.lim_y[1]):
            self.sl_y = create_slice(self.lim_y, self.margin, self.max_size)
    
    def get_map(self):
        if self.map is not None:
            return self.map_origin, self.map
        
        prob_grid = 1 / (1 + np.exp(-self.log_odds_map[self.sl_x, self.sl_y]))
        
        self.map = (prob_grid * 255).astype(np.uint8)
        self.map_origin = np.array([self.max_size//2 - self.sl_x.start, self.max_size//2 - self.sl_y.start])
        
        return self.map_origin, self.map
    
    def get_binary(self, thresh=150, type=cv2.THRESH_BINARY):
        map_origin, map = self.get_map()
        _, binary = cv2.threshold(map, thresh, 255, type)
        
        return map_origin, binary
    
    def get_map_point(self, point):
        px = int(point[0] / self.resolution + self.map_origin[0])
        py = int(point[1] / self.resolution + self.map_origin[1])
        return np.array([px, py])

# Le funzioni seguenti sono corrette e non richiedono modifiche
# per la compatibilità con Numba.
def test():
    
    npz = np.load("scans.npz")
    scans = [npz[f] for f in npz.files]
    npz.close()
    
        
    npz = np.load("poses.npz", allow_pickle=True)
    poses = npz["arr_0"]
    npz.close()
    
    grid = OccupancyGrid(30)
    
    for scan, pose in zip(scans, poses):
        grid.update(scan, pose)
    
    cv2.imwrite("grid.jpg", cv2.bitwise_not(grid.get_map()[1]))

if __name__ == "__main__":
    test()