
import numpy as np

# Function to create a 3D basis for any defining normal vector (to an arbitrary hyperplane)
# Returns the basis as column vectors
def normal_vec_to_basis(norm_vec: np.ndarray) -> np.ndarray:
    """Creates a 3D basis for any defining normal vector (to an arbitrary hyperplane).
    
    Args:
        norm_vec (np.ndarray): The normal vector.

    Returns:
        np.ndarray: The basis as column vectors.
    """
    # first normalize the vector
    norm_vec = np.array(norm_vec, dtype=float).squeeze()
    norm_vec = norm_vec / np.linalg.norm(norm_vec)

    # Helper function for simple basis with unitary elements
    def simple_basis(order):
        ret = np.eye(3)
        idx = (np.arange(3) + order) % 3
        return ret[:,idx]

    # Try to project [1, 0, 0]
    if (proj := np.dot([1.0, 0, 0], norm_vec)):
        # Use this vector to create an orthogonal component
        rej = np.array([1.0, 0, 0]) - (norm_vec * proj)
        # Case for normal vector of [1, 0, 0]
        if np.linalg.norm(rej) == 0:
            return simple_basis(1)
    # If not, try to project [0, 1, 0] and do the same
    elif (proj := np.dot([0, 1.0, 0], norm_vec)):
        rej = np.array([0, 1.0, 0]) - (norm_vec * proj)
        # Case for normal vector of [0, 1, 0]
        if np.linalg.norm(rej) == 0:
            return simple_basis(2)
    else:
        # Otherwise, we are dealing with normal vector of [0, 0, 1],
        # so just create the identity as the basis
        return simple_basis(3)
    
    # Find a third orthogonal vector
    cross = np.cross(rej, norm_vec)
    # Just for simplicity, we treat the cross as x, the rej as y, and the vec as z
    # in order to keep a properly left-handed basis
    cross = cross / np.linalg.norm(cross)
    rej = rej / np.linalg.norm(rej)
    return np.column_stack((cross, rej, norm_vec))