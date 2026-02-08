//! CUDA implementation of spherical Voronoi diagram.

use crate::spatial::impl_generic::{
    spherical_voronoi_impl, spherical_voronoi_region_areas_impl,
    spherical_voronoi_sort_regions_impl,
};
use crate::spatial::traits::spherical_voronoi::{SphericalVoronoi, SphericalVoronoiAlgorithms};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl SphericalVoronoiAlgorithms<CudaRuntime> for CudaClient {
    fn spherical_voronoi(
        &self,
        points: &Tensor<CudaRuntime>,
        radius: f64,
        center: Option<&Tensor<CudaRuntime>>,
    ) -> Result<SphericalVoronoi<CudaRuntime>> {
        spherical_voronoi_impl(self, points, radius, center)
    }

    fn spherical_voronoi_sort_regions(
        &self,
        sv: &SphericalVoronoi<CudaRuntime>,
    ) -> Result<SphericalVoronoi<CudaRuntime>> {
        spherical_voronoi_sort_regions_impl(self, sv)
    }

    fn spherical_voronoi_region_areas(
        &self,
        sv: &SphericalVoronoi<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        spherical_voronoi_region_areas_impl(self, sv)
    }
}
