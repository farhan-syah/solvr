//! Spatial algorithm traits.
//!
//! This module defines the algorithmic contracts for spatial operations.
//! Each trait represents a logical group of related algorithms.

pub mod balltree;
pub mod convex_hull;
pub mod delaunay;
pub mod distance;
pub mod distance_transform;
pub mod halfspace_intersection;
pub mod kdtree;
pub mod mesh;
pub mod procrustes;
pub mod rotation;
pub mod spherical_voronoi;
pub mod voronoi;

pub use balltree::{BallTree, BallTreeAlgorithms, BallTreeOptions};
pub use convex_hull::{ConvexHull, ConvexHullAlgorithms};
pub use delaunay::{Delaunay, DelaunayAlgorithms};
pub use distance::{DistanceAlgorithms, DistanceMetric};
pub use distance_transform::{DistanceTransformAlgorithms, DistanceTransformMetric};
pub use halfspace_intersection::{HalfspaceIntersection, HalfspaceIntersectionAlgorithms};
pub use kdtree::{KDTree, KDTreeAlgorithms, KDTreeOptions, KNNResult, RadiusResult};
pub use mesh::{Mesh, MeshAlgorithms, SimplificationMethod, SmoothingMethod};
pub use procrustes::{ProcrustesAlgorithms, ProcrustesResult};
pub use rotation::{EulerOrder, Rotation, RotationAlgorithms};
pub use spherical_voronoi::{SphericalVoronoi, SphericalVoronoiAlgorithms};
pub use voronoi::{Voronoi, VoronoiAlgorithms};
