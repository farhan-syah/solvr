//! CPU implementation of spatial algorithms.
//!
//! This module implements the spatial algorithm traits for CPU
//! by delegating to the generic implementations in `impl_generic/`.

mod balltree;
mod convex_hull;
mod delaunay;
mod distance;
mod distance_transform;
mod halfspace_intersection;
mod kdtree;
mod mesh;
mod procrustes;
mod rotation;
mod spherical_voronoi;
mod voronoi;
