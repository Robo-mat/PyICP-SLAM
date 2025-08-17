import numpy as np
from collections.abc import Iterable
from numbers import Number

np.set_printoptions(precision=2, suppress=True)

class Tf2D:
    
    def __init__(self, translation: Iterable, rotation: Number):
        self.translation= np.asarray(translation, dtype=np.float32)
        assert self.translation.shape == (2,)
        
        self.rotation = rotation
        
        self._matrix = None
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        
        assert matrix.shape == (3,3)
        
        translation = (matrix @ np.array([0,0,1]).T)[:2]
        rotation = np.arctan2(matrix[1, 0], matrix[0,0])
        
        obj = cls(translation, rotation)
        obj._matrix = matrix
        return obj
    
    def _create_matrix(self):
        
        s, c = np.sin(self.rotation), np.cos(self.rotation)
        tx, ty = self.translation
        
        self._matrix = np.array([
                [c, -s, tx],
                [s,  c, ty],
                [0.0,  0.0, 1.0]
            ], dtype=np.float32)
    
    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._create_matrix()
        return self._matrix

    @property
    def tx(self) -> Number:
        return self.translation[0]

    @property
    def ty(self) -> Number:
        return self.translation[1]
    
    @property
    def theta(self) -> Number:
        return self.rotation
    
    @property
    def translation_tf(self) -> "Tf2D":
        return Tf2D(self.translation, 0.0)
    
    @property
    def rotation_tf(self) -> "Tf2D":
        return Tf2D((0,0), self.rotation)
    
    @property
    def inverse(self):
        inv_matrix = np.linalg.inv(self.matrix)
        return Tf2D.from_matrix(inv_matrix)

    def apply(self, point: Iterable) -> np.ndarray:
        point = np.asarray([point])
        assert point.shape == (1,2)
        
        p_homo = np.vstack((point.T, [1]))
        return (self.matrix @ p_homo)[:2].T[0]
    
    def apply_npoints(self, points: Iterable):
        points = np.asarray(points)
        assert points.ndim == 2 and points.shape[1] == 2
        n = points.shape[0]
        
        p_homo = np.vstack((points.T, np.ones(n)))

        return (self.matrix @ p_homo)[:2].T
    
    def chain(self, t: "Tf2D"):
        return Tf2D.from_matrix(self.matrix @ t.matrix)
    
    def translate(self, tx: Number, ty: Number):
        return Tf2D((self.tx+tx, self.ty+ty), self.rotation)
    
    def rotate(self, theta: Number):
        return Tf2D(self.translation, self.rotation+theta)

    def __call__(self, points):
        points = np.asarray(points)
        if points.ndim == 1:
            return self.apply(points)
        return self.apply_npoints(points)
    
    def __matmul__(self, other):
        if isinstance(other, Tf2D):
            return self.chain(other)
        
        return self.matrix @ other

    def __repr__(self):
        return f"Tf2D(tx={self.tx:.2f}, ty={self.ty:.2f}, theta={self.theta:.2f})"
    
    def __eq__(self, other):
        return isinstance(other, self.__class__) and np.all(np.isclose([self.tx, self.ty, self.theta], [other.tx, other.ty, other.theta]))

def test():
    pi = np.pi
    
    tx, ty = 0.1, 0.2
    rot = pi/2
    
    tf = Tf2D((tx, ty), rot)
    
    p = (2, 1)
    
    tf_rotonly = tf.rotation_tf
    tf_transonly = tf.translation_tf
    
    allinone = tf(p)
    rot_trans = tf_transonly(tf_rotonly(p))
    trans_rot = tf_rotonly(tf_transonly(p))
    
    assert np.array_equal(allinone, rot_trans) #prima rotazione poi translazione
    
    trans_rot_chain = tf_rotonly @ tf_transonly
    rot_trans_chain = tf_transonly @ tf_rotonly
    
    assert rot_trans_chain == tf
    
    inv = tf.inverse
    assert np.array_equal(p, (inv @ tf)(p))
    assert np.array_equal(p, (tf @ inv)(p))

if __name__ == "__main__":
    test()