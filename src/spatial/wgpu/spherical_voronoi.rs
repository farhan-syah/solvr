//! WebGPU implementation of spherical Voronoi diagram.

use crate::spatial::impl_generic::{
    spherical_voronoi_impl, spherical_voronoi_region_areas_impl,
    spherical_voronoi_sort_regions_impl,
};
use crate::spatial::traits::spherical_voronoi::{SphericalVoronoi, SphericalVoronoiAlgorithms};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl SphericalVoronoiAlgorithms<WgpuRuntime> for WgpuClient {
    fn spherical_voronoi(
        &self,
        points: &Tensor<WgpuRuntime>,
        radius: f64,
        center: Option<&Tensor<WgpuRuntime>>,
    ) -> Result<SphericalVoronoi<WgpuRuntime>> {
        spherical_voronoi_impl(self, points, radius, center)
    }

    fn spherical_voronoi_sort_regions(
        &self,
        sv: &SphericalVoronoi<WgpuRuntime>,
    ) -> Result<SphericalVoronoi<WgpuRuntime>> {
        spherical_voronoi_sort_regions_impl(self, sv)
    }

    fn spherical_voronoi_region_areas(
        &self,
        sv: &SphericalVoronoi<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        spherical_voronoi_region_areas_impl(self, sv)
    }
}
