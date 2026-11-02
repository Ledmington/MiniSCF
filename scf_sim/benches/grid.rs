//! Cube-file grids on water/6-31G*: every MO and the electrostatic potential on a few
//! central x-slabs of the grid `rhf_sim` writes.

mod common;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const NUM_SLABS: usize = 4;

fn grid(c: &mut Criterion) {
    let water = common::water();
    let mut group = c.benchmark_group("water_6-31gstar_grid");
    group.sample_size(20);
    group.throughput(Throughput::Elements(water.num_grid_points(NUM_SLABS) as u64));
    group.bench_function("orbitals", |b| {
        b.iter(|| water.orbitals_on_slabs(NUM_SLABS))
    });
    group.bench_function("electrostatic_potential", |b| {
        b.iter(|| water.electrostatic_potential_on_slabs(NUM_SLABS))
    });
    group.finish();
}

criterion_group!(benches, grid);
criterion_main!(benches);
