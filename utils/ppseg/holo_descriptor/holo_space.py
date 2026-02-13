import numpy as np
import pymesh
from sklearn.cluster import DBSCAN


class HoloSpace:
    """HoloSpace class for estimating the fragment-binding (holo) space
    from the deep-learning semantic segmentation prediction of protein
    surface.

    Args:
        ply_path (str, required): Path to the PLY file (protein surface mesh).
            Required `pred` attribute.
        num_of_frags (int): Number of fragments to extract. (default: ``6``)
        q (float): Quantile value for fragment extraction. (default: ``1.0``,
            maximum value)
        corrected (bool): Whether to use corrected volume calculation.
            (default: ``True``)

    Attributes:
        mesh (pymesh.Mesh): The loaded mesh.
        pocket_frags (list): List of pocket fragments.
        pocket (pymesh.Mesh): The combined pocket mesh.

    Example:
        .. code:: python

            from holo_space import HoloSpace

            # Initialize HoloSpace with a PLY file and parameters
            holo_space = HoloSpace("path/to/mesh.ply", num_of_frags=6, q=1.0)

            # Get the pocket volume
            pocket_volume = holo_space.get_pocket_volume()
            pocket_frag_volumes = holo_space.get_pocket_frag_volumes()

    """

    def __init__(self, ply_path, num_of_frags=6, q=1.00, corrected=True):
        """Initializes the HoloSpace object.

        Args:
            ply_path (str): Path to the PLY file (protein surface mesh).
            num_of_frags (int): Number of fragments to extract. (default: ``6``)
            q (float): Quantile value for fragment extraction. (default: ``1.0``)
            corrected (bool): Whether to use corrected volume calculation.
                (default: True)


        """
        self.ply_path = ply_path
        self.mesh = pymesh.load_mesh(ply_path)
        self.num_of_frags = num_of_frags
        self.q = q
        self.corrected = corrected

        self.pocket_frags = get_pocket_frags(self.mesh, num_of_frags, q)
        self.pocket = combine_pocket_frags(self.pocket_frags)

    def get_pocket_volume(self):
        """Returns the volume of the pocket."""
        return self.pocket.volume

    def get_pocket_frag_volumes(self):
        """Returns the volumes of the pocket fragments."""
        self.pocket_frag_volumes = pocket_frag_volume(self.pocket_frags, self.corrected)
        return self.pocket_frag_volumes

    def save_pocket(self, path):
        """Saves the pocket mesh to the specified path.

        Args:
            path (str): Path to save the pocket mesh.
        """
        pymesh.save_mesh(path, self.pocket, ascii=True, use_float=True)


def pocket_frag_volume(pocket_frags, corrected=False):
    """Corrected volume is calculated by subtracting the top 2 (with others) average
    intersection volume

    Args:
        pocket_frags (list): List of pocket fragments.
        corrected (bool): Whether to use corrected volume calculation.

    Returns:
        list: List of pocket fragment volumes.
    """
    if corrected:
        # This is an approximation of the corrected volume of the pocket fragment
        # Consider one fragment usually has 2 nearby fragments, there are 2 intersection
        # volumes with other fragments
        # thus, the corrected volume is calculated by subtracting the
        # top 2 (with others) average intersection volume
        pIoV = pairwise_intersection_volume(pocket_frags)
        top_2_avg = top_k_average(pIoV=pIoV, k=2)
        pocket_frag_volumes = []
        for i, pocket_frag in enumerate(pocket_frags):
            pocket_frag_vol = pocket_frag.volume - top_2_avg[i]
            pocket_frag_volumes.append(pocket_frag_vol)

    else:
        pocket_frag_volumes = []
        for i in range(len(pocket_frags)):
            pocket_frag_volumes.append(pocket_frags[i].volume)

    return pocket_frag_volumes


def top_k_average(pIoV, k=2):
    """Calculate the average of the top k values in each row of pIoV.

    Args:
        pIoV (numpy.ndarray): Pairwise intersection volume matrix.
        k (int): Number of top values to average. (default: ``2``)

    Returns:
        list: List of average values for each row.
    """
    # for each row in pIoV, find the top k (except for itself) of the value
    # average the top k values
    top_k_avg = []
    for i in range(pIoV.shape[0]):
        row = pIoV[i]
        top_k = np.sort(row)[-k - 1 : -1]
        top_k_avg.append(np.mean(top_k))
    return top_k_avg


def pairwise_intersection_volume(mesh_list):
    """Calculate the pairwise intersection volume of a list of meshes.

    Args:
        mesh_list (list): List of meshes.

    Returns:
        numpy.ndarray: Pairwise intersection volume matrix.
    """
    # Create a square matrix to store the intersection volumes
    pIoV = np.zeros((len(mesh_list), len(mesh_list)))

    # Calculate the pairwise intersection volume for mesh_list
    for i in range(len(mesh_list)):
        for j in range(i, len(mesh_list)):
            intersection = pymesh.boolean(
                mesh_list[i], mesh_list[j], operation="intersection"
            )
            pIoV[i, j] = intersection.volume

    pIoV = pIoV + pIoV.T
    return pIoV


def combine_pocket_frags(pocket_frags):
    """Combine the pocket fragments into a single pocket mesh.

    Args:
        pocket_frags (list): List of pocket fragments.

    Returns:
        pymesh.Mesh: Combined pocket mesh.
    """
    for i in range(len(pocket_frags)):
        if i == 0:
            pocket = pocket_frags[i]
        else:
            pocket = pymesh.boolean(pocket, pocket_frags[i], operation="union")
    return pocket


def extract_pocket(mesh, num_of_frags=6, q=1.00):
    """Extract the pocket from the mesh.

    Args:
        mesh (pymesh.Mesh): The input mesh.
        num_of_frags (int): Number of fragments to extract. (default: ``6``)
        q (float): Quantile value for fragment extraction. (default: ``1.0``)

    Returns:
        pymesh.Mesh: The extracted pocket mesh.
    """
    pocket_frags = get_pocket_frags(mesh, num_of_frags, q=q)
    pocket = combine_pocket_frags(pocket_frags)

    return pocket


def get_pocket_frags(mesh, num_of_frags=6, q=1.00):
    """Get the pocket fragments (raw HoloSpace for each fragment) from the mesh.

    Args:
        mesh (pymesh.Mesh): The input mesh.
        num_of_frags (int): Number of fragments to extract. (default: ``6``)
        q (float): Quantile value for fragment extraction. (default: ``1.0``)

    Returns:
        list: List of pocket fragments.
    """
    pocket_frags = []
    for i in range(1, num_of_frags + 1, 1):
        pocket_frag = extract_fragment_based_pocket(
            mesh, func=np.quantile, label=i, q=q
        )
        pocket_frags.append(pocket_frag)
    return pocket_frags


def extract_fragment_based_pocket(mesh, func, label, **kwargs):
    """Extract the pocket-based on the fragment (raw HoloSpace).

    Args:
        mesh (pymesh.Mesh): The input mesh.
        func (function): Function to calculate the radius.
        label (int): The fragment label.
        **kwargs: Additional arguments for the function.

    Returns:
        pymesh.Mesh: The extracted pocket mesh.
    """
    class_vertices = get_class_vertices(mesh, label)

    if len(class_vertices) <= 3:
        pass
    else:
        # clustering the points with DBSCAN
        # eps = 5, because the labelled points is defined within 5Å near the fragment
        # the predicted points is highly likely to be within 5Å near the fragment
        # thus, the max distance to be considered as the same cluster should be less
        # than 5Å (radius)
        # additional, the min_samples is set to one third of the labelled points
        clustering = DBSCAN(min_samples=int(len(class_vertices) / 3), eps=5, n_jobs=1)
        clustering.fit(class_vertices)
        class_vertices = class_vertices[clustering.labels_ != -1]

    center = calculate_center(class_vertices)
    radius = radius_thres_from_distrib(class_vertices, center, func=func, **kwargs)

    sphere = pymesh.generate_icosphere(radius, center, refinement_order=2)
    pocket_by_frag = pymesh.boolean(sphere, mesh, operation="difference")
    return pocket_by_frag


def get_class_vertices(mesh, label=0, return_mask=False, feature="pred"):
    """Get the vertices of a specific class from the mesh.

    Args:
        mesh (pymesh.Mesh): The input mesh.
        label (int): The class label. (default: ``0``)
        return_mask (bool): Whether to return the mask. (default: ``False``)
        feature (str): The feature name. (default: ``pred``)

    Returns:
        numpy.ndarray | (numpy.ndarray, numpy.ndarray): The vertices of the \
            specified class. If `return_mask` is ``True``, also returns the mask \
            (second element).
    """
    features = mesh.get_attribute(f"vertex_{feature}")
    mask = features == label
    vertices = mesh.vertices[mask]
    if return_mask:
        return vertices, mask
    else:
        return vertices


def calculate_center(class_vertices):
    """Calculate the center of the class vertices.

    Args:
        class_vertices (numpy.ndarray): The vertices of the specified class.

    Returns:
        numpy.ndarray: The center of the class vertices.
    """
    if len(class_vertices) == 0:
        return np.array([0, 0, 0])
    else:
        center = np.mean(class_vertices, axis=0)
        return center


def radius_thres_from_distrib(class_vertices, center, func=np.max, **kwargs):
    """Calculate the radius threshold from the distribution of class vertices.

    Args:
        class_vertices (numpy.ndarray): The vertices of the specified class.
        center (numpy.ndarray): The center of the class vertices.
        func (function): Function to calculate the radius. (default: ``np.max``)
        **kwargs: Additional arguments for the function.

    Returns:
        float: The radius threshold. ``0.0`` if class_vertices is empty.
    """
    if len(class_vertices) == 0:
        return 0.0
    else:
        radius = func(np.linalg.norm(class_vertices - center, axis=1), **kwargs)
        return radius
