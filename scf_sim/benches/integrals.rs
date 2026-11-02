//! Gaussian integrals: the ERIs of four shells by angular momentum, the Boys function, and the full
//! one- and two-electron matrices of water/6-31G*.

mod common;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use scf_sim::bench_support::{AngularMomentum, ShellQuartet, boys_all};
use std::hint::black_box;

fn electron_repulsion_quartets(c: &mut Criterion) {
    let mut group = c.benchmark_group("eri_shell_quartet");
    // The f quartet takes about 0.2 s per iteration.
    group.sample_size(20);
    for (name, l) in [
        ("s", AngularMomentum::S),
        ("p", AngularMomentum::P),
        ("d", AngularMomentum::D),
        ("f", AngularMomentum::F),
    ] {
        let quartet = ShellQuartet::new(l);
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| black_box(&quartet).electron_repulsion_tensor())
        });
    }
    group.finish();
}

fn boys_function(c: &mut Criterion) {
    let mut group = c.benchmark_group("boys_all");
    let mut out = [0.0; 13];
    // Power series, intermediate range and large-argument closed form.
    for t in [0.5, 12.0, 60.0] {
        group.bench_function(BenchmarkId::new("n_max=12", t), |b| {
            b.iter(|| boys_all(12, black_box(t), &mut out))
        });
    }
    group.finish();
}

fn molecular_integrals(c: &mut Criterion) {
    let water = common::water();
    let mut group = c.benchmark_group("water_6-31gstar");
    group.sample_size(20);
    group.bench_function("one_electron_matrices", |b| {
        b.iter(|| water.one_electron_matrices())
    });
    group.bench_function("eri_tensor", |b| {
        b.iter(|| water.electron_repulsion_tensor())
    });
    group.finish();
}

criterion_group!(
    benches,
    electron_repulsion_quartets,
    boys_function,
    molecular_integrals
);
criterion_main!(benches);
