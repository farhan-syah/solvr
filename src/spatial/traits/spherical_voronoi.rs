//! Spherical Voronoi diagram trait.
//!
//! Computes the Voronoi diagram on the surface of a sphere from a set of
//! generator points. Uses 3D convex hull as dual: each hull face's circumcenter
//! (projected to the sphere) becomes a Voronoi vertex.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Spherical Voronoi diagram result.
#[derive(Debug, Clone)]
pub struct SphericalVoronoi<R: Runtime> {
    /// Generator points on the sphere `[n, 3]`.
    pub points: Tensor<R>,

    /// Center of the sphere `[3]`.
    pub center: Tensor<R>,

    /// Radius of the sphere.
    pub radius: f64,

    /// Voronoi vertices on the sphere surface `[n_vertices, 3]`.
    /// These are circumcenters of hull faces, projected onto the sphere.
    pub vertices: Tensor<R>,

    /// Region vertex indices in CSR format.
    /// For point i, its region vertices are at
    /// `regions_indices[regions_indptr[i]..regions_indptr[i+1]]`.
    pub regions_indices: Tensor<R>,

    /// Region index pointers `[n+1]` (I64).
    pub regions_indptr: Tensor<R>,
}

/// Algorithmic contract for spherical Voronoi diagram operations.
pub trait SphericalVoronoiAlgorithms<R: Runtime> {
    /// Compute the spherical Voronoi diagram.
    ///
    /// # Arguments
    ///
    /// * `points` - Generator points on the sphere `[n, 3]`, must have n >= 3
    /// * `radius` - Sphere radius (positive)
    /// * `center` - Sphere center `[3]` (or None for origin)
    ///
    /// # Returns
    ///
    /// `SphericalVoronoi` with vertices on the sphere and regions for each point.
    fn spherical_voronoi(
        &self,
        points: &Tensor<R>,
        radius: f64,
        center: Option<&Tensor<R>>,
    ) -> Result<SphericalVoronoi<R>>;

    /// Sort region vertices in counter-clockwise order when viewed from outside.
    ///
    /// Modifies regions so vertices are ordered for polygon rendering / area computation.
    fn spherical_voronoi_sort_regions(
        &self,
        sv: &SphericalVoronoi<R>,
    ) -> Result<SphericalVoronoi<R>>;

    /// Compute the area of each Voronoi region on the sphere.
    ///
    /// # Returns
    ///
    /// Tensor `[n]` with the spherical area of each region. Sum equals 4*pi*r^2.
    fn spherical_voronoi_region_areas(&self, sv: &SphericalVoronoi<R>) -> Result<Tensor<R>>;
}
